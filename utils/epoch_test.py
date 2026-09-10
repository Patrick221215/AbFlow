#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Shared full-RAbD test engine for in-Trainer and standalone evaluation.

Design goals
------------
1. Run the *real* generation path: ``model.sample(...)`` followed by the
   project's existing ``cal_metrics.py``.  No proxy test loss is introduced.
2. Support one model snapshot on multiple DDP ranks by assigning complete
   logical test batches to ranks.  A logical batch is never split across ranks.
3. Make standalone and in-Trainer evaluation use the same test protocol through
   batch-keyed deterministic RNG seeds.  This avoids world-size-dependent random
   streams from ``torch.randn``, ``torch.randint``, ``torch.multinomial`` and
   ``torch.rand`` inside AbFlow sampling.
4. Never feed test metrics back into optimization/model selection.

This module does not modify model parameters, losses, samplers or scientific
configuration.  The Trainer wrapper is responsible for applying EMA and for
saving/restoring the training RNG state around this evaluator.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from data.dataset import E2EDataset
from data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex


V207_EXPLICIT_EPOCH_TEST_CDR_CONTRACT = True

METRIC_NAMES = [
    "AAR H3",
    "CAAR H3",
    "AAR",
    "CAAR",
    "RMSD(CA) aligned",
    "RMSD(CA) CDRH3",
    "RMSD(CA) CDRH3 aligned",
    "TMscore",
    "LDDT",
    "DockQ",
]
METRIC_RE = re.compile(
    r"^(?P<name>" + "|".join(re.escape(x) for x in METRIC_NAMES)
    + r"):\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
DOCKQ_PROP_RE = re.compile(
    r"proportion of DockQ above 0\.23:\s*"
    r"(?P<p23>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
    r"0\.49:\s*(?P<p49>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
    r"0\.8:\s*(?P<p80>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

TB_METRICS = {
    "AAR_mean": "Test/AAR",
    "CAAR_mean": "Test/CAAR",
    "RMSDCA_aligned_mean": "Test/RMSD_CA_aligned",
    "RMSDCA_CDRH3_mean": "Test/H3_raw_RMSD",
    "RMSDCA_CDRH3_aligned_mean": "Test/H3_aligned_RMSD",
    "TMscore_mean": "Test/TMscore",
    "LDDT_mean": "Test/lDDT",
    "DockQ_mean": "Test/DockQ",
    "DockQ_above_0.23": "Test/DockQ_above_0.23",
    "DockQ_above_0.49": "Test/DockQ_above_0.49",
    "DockQ_above_0.8": "Test/DockQ_above_0.8",
}


def _metric_key(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def parse_cal_metrics_output(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in text.splitlines():
        s = line.strip()
        m = METRIC_RE.match(s)
        if m:
            out[f"{_metric_key(m.group('name'))}_mean"] = float(m.group("value"))
            continue
        dm = DOCKQ_PROP_RE.search(s)
        if dm:
            out["DockQ_above_0.23"] = float(dm.group("p23"))
            out["DockQ_above_0.49"] = float(dm.group("p49"))
            out["DockQ_above_0.8"] = float(dm.group("p80"))
    return out


def dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return 0, 1


def dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_logical_batch(base_seed: int, logical_batch_id: int) -> int:
    """Use a world-size-independent seed for one complete logical batch.

    The same logical batch receives the same random stream whether it is run in
    standalone world_size=1 or assigned to another DDP rank in world_size>1.
    """
    seed = int(base_seed) + int(logical_batch_id)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def logical_batches(n_items: int, batch_size: int) -> List[List[int]]:
    batch_size = max(1, int(batch_size))
    return [
        list(range(start, min(start + batch_size, int(n_items))))
        for start in range(0, int(n_items), batch_size)
    ]


def assigned_logical_batches(
    n_items: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> List[Tuple[int, List[int]]]:
    batches = logical_batches(n_items, batch_size)
    return [
        (bid, indices)
        for bid, indices in enumerate(batches)
        if bid % max(1, int(world_size)) == int(rank)
    ]


def to_cplx(ori_cplx, ab_x, ab_s) -> AgAbComplex:
    """Exact structural conversion used by the project's generate.py."""
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)
        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(residue, coord, _id=(len(chain), " ")))

    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
        res.id = ori_res.id

    antibody = Protein(
        ori_cplx.pdb_id,
        {
            ori_cplx.heavy_chain: heavy_chain,
            ori_cplx.light_chain: light_chain,
        },
    )
    cplx = AgAbComplex(
        ori_cplx.antigen,
        antibody,
        ori_cplx.heavy_chain,
        ori_cplx.light_chain,
        skip_epitope_cal=True,
        skip_validity_check=True,
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


def _to_device(batch: dict, device: torch.device) -> dict:
    for key in list(batch.keys()):
        value = batch[key]
        if hasattr(value, "to"):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def _split_graph_outputs(batch: dict, X: torch.Tensor, S: torch.Tensor):
    X_cpu = X.detach().cpu().tolist()
    S_cpu = S.detach().cpu().tolist()

    if "bid" in batch:
        batch_id = batch["bid"].detach().cpu()
    else:
        lengths = batch["lengths"].detach().cpu()
        batch_id = torch.zeros_like(batch["S"].detach().cpu())
        if lengths.numel() > 1:
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)

    X_list: List[list] = []
    S_list: List[list] = []
    cur_bid = -1
    for i, bid_tensor in enumerate(batch_id):
        bid = int(bid_tensor.item())
        if bid != cur_bid:
            cur_bid = bid
            X_list.append([])
            S_list.append([])
        X_list[-1].append(X_cpu[i])
        S_list[-1].append(S_cpu[i])
    return X_list, S_list


def normalize_test_cdr(cdr_type) -> List[str]:
    """Normalize the formal evaluation CDR identity without consulting model state."""
    if cdr_type is None:
        raw = str(os.environ.get("ABFLOW_EPOCH_TEST_CDR", "H3") or "H3")
        values = [x.strip().upper() for x in raw.split(",") if x.strip()]
    elif isinstance(cdr_type, str):
        values = [x.strip().upper() for x in cdr_type.split(",") if x.strip()]
    else:
        values = [str(x).strip().upper() for x in cdr_type if str(x).strip()]
    if values != ["H3"]:
        raise RuntimeError(f"Formal RAbD Test requires cdr=['H3']; got {values!r}")
    return values


@dataclass
class GenerationResult:
    summary_file: Optional[str]
    records: List[dict]
    rank: int
    world_size: int


def generate_distributed(
    *,
    model,
    dataset: E2EDataset,
    device: torch.device,
    save_dir: str,
    batch_size: int = 20,
    n_steps: int = 10,
    base_seed: int = 2023,
    show_sample_progress: bool = False,
    cdr_type=None,
) -> GenerationResult:
    """Generate the full test set exactly once across the current DDP world."""
    rank, world_size = dist_info()
    formal_cdr = normalize_test_cdr(cdr_type)
    save_dir = os.path.abspath(save_dir)
    rank_dir = os.path.join(save_dir, f"rank_{rank:02d}")
    os.makedirs(rank_dir, exist_ok=True)

    local_records: List[dict] = []
    assignments = assigned_logical_batches(
        len(dataset), batch_size, rank=rank, world_size=world_size
    )

    local_error = ""
    try:
        for logical_batch_id, global_indices in assignments:
            seed_logical_batch(base_seed, logical_batch_id)
            items = [dataset[i] for i in global_indices]
            batch = dataset.collate_fn(items)
            batch = _to_device(batch, device)
            batch.pop("xloss_mask", None)

            with torch.no_grad():
                X, S, _ = model.sample(
                    **batch,
                    n_steps=int(n_steps),
                    show_progress=bool(show_sample_progress and rank == 0),
                )

            X_list, S_list = _split_graph_outputs(batch, X, S)
            if len(X_list) != len(global_indices):
                raise RuntimeError(
                    "Generated graph count does not match logical test batch: "
                    f"generated={len(X_list)} expected={len(global_indices)} "
                    f"logical_batch_id={logical_batch_id}."
                )

            for local_i, global_i in enumerate(global_indices):
                ori_cplx = dataset.data[global_i]
                cplx = to_cplx(ori_cplx, X_list[local_i], S_list[local_i])
                pdb_id = cplx.get_id().split("(")[0]
                # Prefix by global index to make filenames collision-proof while
                # preserving the biological pdb id in summary metadata.
                stem = f"{global_i:04d}_{pdb_id}"
                mod_pdb = os.path.join(rank_dir, stem + ".pdb")
                ref_pdb = os.path.join(rank_dir, stem + "_original.pdb")
                cplx.to_pdb(mod_pdb)
                ori_cplx.to_pdb(ref_pdb)
                local_records.append(
                    {
                        "_global_index": int(global_i),
                        "_logical_batch_id": int(logical_batch_id),
                        "mod_pdb": mod_pdb,
                        "ref_pdb": ref_pdb,
                        "H": cplx.heavy_chain,
                        "L": cplx.light_chain,
                        "A": cplx.antigen.get_chain_names(),
                        "cdr_type": formal_cdr,
                        "pdb": pdb_id,
                        "pmetric": None,
                    }
                )
    except Exception as exc:
        local_error = f"rank={rank}: {type(exc).__name__}: {exc}"

    if world_size > 1:
        error_list: List[Optional[str]] = [None for _ in range(world_size)]
        dist.all_gather_object(error_list, local_error)
        errors = [x for x in error_list if x]
    else:
        errors = [local_error] if local_error else []
    if errors:
        raise RuntimeError("Distributed epoch-test generation failed: " + " | ".join(errors))

    if world_size > 1:
        gathered: List[Optional[List[dict]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_records)
        all_records = [r for part in gathered if part is not None for r in part]
    else:
        all_records = local_records

    summary_file: Optional[str] = None
    if rank == 0:
        all_records.sort(key=lambda x: int(x["_global_index"]))
        expected = list(range(len(dataset)))
        actual = [int(x["_global_index"]) for x in all_records]
        if actual != expected:
            raise RuntimeError(
                "Distributed test coverage is not exact. "
                f"expected={expected[:5]}...{expected[-5:] if expected else []}, "
                f"actual={actual[:5]}...{actual[-5:] if actual else []}"
            )

        os.makedirs(save_dir, exist_ok=True)
        summary_file = os.path.join(save_dir, "summary.json")
        with open(summary_file, "w", encoding="utf-8") as fout:
            for item in all_records:
                public_item = {
                    k: v for k, v in item.items() if not k.startswith("_")
                }
                fout.write(json.dumps(public_item, ensure_ascii=False) + "\n")

        plan = {
            "protocol": "logical_batch_seeded_v1",
            "base_seed": int(base_seed),
            "logical_batch_size": int(batch_size),
            "n_steps": int(n_steps),
            "n_items": int(len(dataset)),
            "world_size": int(world_size),
            "cdr_type": formal_cdr,
            "logical_batches": logical_batches(len(dataset), batch_size),
        }
        with open(os.path.join(save_dir, "test_protocol.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

    dist_barrier()
    return GenerationResult(
        summary_file=summary_file,
        records=all_records if rank == 0 else [],
        rank=rank,
        world_size=world_size,
    )


def validate_summary_contract(summary_file: str, expected_cdr=None) -> int:
    """Cheap fail-fast validation before launching the external metric program."""
    expected = normalize_test_cdr(expected_cdr)
    records = []
    with open(summary_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("mod_pdb", "ref_pdb", "H", "L", "A", "cdr_type", "pdb"):
                if key not in item:
                    raise RuntimeError(f"summary.json line {lineno} missing key {key!r}")
            if normalize_test_cdr(item.get("cdr_type")) != expected:
                raise RuntimeError(
                    f"summary.json line {lineno} cdr_type={item.get('cdr_type')!r}; expected {expected!r}"
                )
            for key in ("mod_pdb", "ref_pdb"):
                if not os.path.isfile(item[key]):
                    raise FileNotFoundError(f"summary.json line {lineno} missing file: {item[key]}")
            records.append(item)
    if not records:
        raise RuntimeError("summary.json contains zero generated complexes")
    return len(records)


def run_cal_metrics_rank0(
    *,
    summary_file: Optional[str],
    save_dir: str,
    project_root: str,
    num_workers: int = 8,
    cdr_type=None,
) -> Dict[str, float]:
    """Run the project's original cal_metrics.py on rank 0 and parse output."""
    rank, world_size = dist_info()
    payload: List[object] = [None]

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        try:
            if not summary_file or not os.path.isfile(summary_file):
                raise FileNotFoundError(f"summary.json not found: {summary_file}")
            cal_metrics = os.path.join(os.path.abspath(project_root), "cal_metrics.py")
            if not os.path.isfile(cal_metrics):
                raise FileNotFoundError(f"cal_metrics.py not found: {cal_metrics}")

            n_records = validate_summary_contract(summary_file, expected_cdr=cdr_type)
            env = os.environ.copy()
            env["OPENMM_CPU_THREADS"] = "1"

            def _run_metric_process(workers: int):
                cmd = [
                    sys.executable, cal_metrics,
                    "--test_set", os.path.abspath(summary_file),
                    "--num_workers", str(int(workers)),
                ]
                return subprocess.run(
                    cmd, cwd=os.path.abspath(project_root), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )

            requested_workers = max(1, int(num_workers))
            proc = _run_metric_process(requested_workers)
            log_text = proc.stdout or ""
            log_path = os.path.join(save_dir, "cal_metrics.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(
                    f"[V207MetricContract] records={n_records} cdr={normalize_test_cdr(cdr_type)} "
                    f"workers={requested_workers}\n"
                )
                f.write(log_text)

            # Same scientific evaluator, execution-only fallback. Some metric stacks
            # fail under multiprocessing/OpenMM yet are deterministic with one worker.
            if int(proc.returncode) != 0 and requested_workers != 1:
                retry = _run_metric_process(1)
                retry_text = retry.stdout or ""
                retry_path = os.path.join(save_dir, "cal_metrics_retry_worker1.log")
                with open(retry_path, "w", encoding="utf-8") as f:
                    f.write(retry_text)
                if int(retry.returncode) == 0:
                    proc, log_text, log_path = retry, retry_text, retry_path
                else:
                    tail0 = "\n".join(log_text.splitlines()[-40:])
                    tail1 = "\n".join(retry_text.splitlines()[-40:])
                    raise RuntimeError(
                        "cal_metrics.py failed with requested workers and worker=1. "
                        f"requested_tail=\n{tail0}\nworker1_tail=\n{tail1}"
                    )

            metrics = parse_cal_metrics_output(log_text)
            result = {
                "returncode": int(proc.returncode),
                "metrics": metrics,
                "log": log_path,
                "error": "",
                "records": int(n_records),
                "cdr_type": normalize_test_cdr(cdr_type),
            }
        except Exception as exc:
            result = {
                "returncode": -999,
                "metrics": {},
                "log": os.path.join(save_dir, "cal_metrics.log"),
                "error": f"{type(exc).__name__}: {exc}",
            }
            with open(result["log"], "a", encoding="utf-8") as f:
                f.write("\n[EpochTestError] " + result["error"] + "\n")

        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        payload[0] = result

    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)

    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("Failed to broadcast test metric result from rank 0.")
    if int(result.get("returncode", 1)) != 0:
        raise RuntimeError(
            "cal_metrics.py failed. See: " + str(result.get("log", "<unknown>"))
        )
    return dict(result.get("metrics", {}))


def cleanup_structures(save_dir: str) -> None:
    """Delete generated PDB/summary artifacts after metrics, keeping logs/json."""
    rank, _ = dist_info()
    dist_barrier()
    if rank == 0:
        root = Path(save_dir)
        for p in root.glob("rank_*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        summary = root / "summary.json"
        if summary.exists():
            summary.unlink()
    dist_barrier()


def append_epoch_metrics(
    *,
    root_dir: str,
    epoch: int,
    global_step: int,
    metrics: Dict[str, float],
    protocol: str = "logical_batch_seeded_v1",
) -> None:
    rank, _ = dist_info()
    if rank != 0:
        return
    os.makedirs(root_dir, exist_ok=True)
    record = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "protocol": protocol,
        **{k: float(v) for k, v in metrics.items()},
    }
    with open(os.path.join(root_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp = os.path.join(root_dir, ".latest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(root_dir, "latest.json"))
