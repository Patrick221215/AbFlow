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
5. Fail before PDB metrics if sampling changes any framework/antigen residue;
   H3 is the only legal sequence-design region for the V206 test contract.

This module does not modify model parameters, losses, samplers or scientific
configuration.  The Trainer wrapper is responsible for applying EMA and for
saving/restoring the training RNG state around this evaluator.
"""
from __future__ import annotations

import json
import math
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


MODEL_OUTPUT_INVALID_MARKER = "[ModelOutputInvalid]"

# Bio.PDB writes Cartesian coordinates with the PDB fixed-width 8.3 format.
# Values outside this interval can remain finite tensors while overflowing the
# coordinate columns, after which PDBParser reports "Invalid or missing
# coordinate(s)".  This is a model-output failure, not an evaluator bug.
PDB_COORD_MIN = -999.999
PDB_COORD_MAX = 9999.999


def _validate_generated_coordinates_for_pdb(
    X: torch.Tensor, logical_batch_id: int
) -> None:
    """Validate model coordinates before converting them into a PDB artifact.

    This check is evaluation-only.  It does not clamp, rescale or otherwise
    alter generated structures; invalid model outputs remain invalid evidence.
    """
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a tensor, got {type(X).__name__}")

    finite = torch.isfinite(X)
    if not bool(finite.all().item()):
        bad = int((~finite).sum().item())
        raise RuntimeError(
            f"{MODEL_OUTPUT_INVALID_MARKER} generated coordinates contain "
            f"{bad} non-finite values in logical_batch_id={logical_batch_id}."
        )

    x32 = X.detach().float()
    xmin = float(x32.min().item()) if x32.numel() else 0.0
    xmax = float(x32.max().item()) if x32.numel() else 0.0
    if xmin < PDB_COORD_MIN or xmax > PDB_COORD_MAX:
        raise RuntimeError(
            f"{MODEL_OUTPUT_INVALID_MARKER} generated coordinates exceed the "
            "PDB 8.3 Cartesian field range before serialization: "
            f"logical_batch_id={logical_batch_id} min={xmin:.6g} "
            f"max={xmax:.6g} allowed=[{PDB_COORD_MIN},{PDB_COORD_MAX}]."
        )

# These fields are produced for both the single-CDR and whole-antibody branches
# of the original ``cal_metrics.py``.  A zero return code without all of them is
# not a valid Test result (for example, a truncated worker log must not be
# accepted as a successful evaluation).
CORE_METRIC_KEYS = {
    "AAR_mean",
    "CAAR_mean",
    "RMSDCA_aligned_mean",
    "RMSDCA_CDRH3_mean",
    "RMSDCA_CDRH3_aligned_mean",
    "TMscore_mean",
    "LDDT_mean",
    "DockQ_mean",
    "DockQ_above_0.23",
    "DockQ_above_0.49",
    "DockQ_above_0.8",
}
CDR_NAME_RE = re.compile(r"^[HL][123]$")


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


def _normalise_cdr_type(value):
    """Return a cal_metrics-compatible CDR selection or ``None``.

    Older checkpoints sometimes stored ``cdr_type=None`` while keeping the
    actual design region in ``model.paratope``.  Treating that checkpoint as a
    whole-antibody task silently inflates AAR because unchanged framework
    residues dominate the average.  This normaliser only accepts canonical CDR
    names and therefore cannot turn an arbitrary paratope definition into a CDR
    task by accident.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [x.strip().upper() for x in re.split(r"[,\s]+", value) if x.strip()]
    elif isinstance(value, (list, tuple, set)):
        parts = [str(x).strip().upper() for x in value if str(x).strip()]
    else:
        return None
    if not parts or any(CDR_NAME_RE.fullmatch(x) is None for x in parts):
        return None
    parts = list(dict.fromkeys(parts))
    return parts[0] if len(parts) == 1 else parts


def resolve_eval_cdr_type(model):
    """Resolve the metric region without changing model sampling semantics."""
    env_value = os.environ.get("ABFLOW_EPOCH_TEST_CDR", "").strip()
    if env_value:
        resolved = _normalise_cdr_type(env_value)
        if resolved is None:
            raise ValueError(
                "ABFLOW_EPOCH_TEST_CDR must contain canonical CDR names "
                f"(H1/H2/H3/L1/L2/L3), got {env_value!r}."
            )
        return resolved, "env"

    resolved = _normalise_cdr_type(getattr(model, "cdr_type", None))
    if resolved is not None:
        return resolved, "model.cdr_type"

    resolved = _normalise_cdr_type(getattr(model, "paratope", None))
    if resolved is not None:
        return resolved, "model.paratope_fallback"
    return None, "whole_antibody"


def _atomic_write_json(path: str, payload: object) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write("\n")
    os.replace(tmp, path)


def _log_tail(text: str, max_lines: int = 80) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join(lines[-max(1, int(max_lines)):])


def _metric_validation_error(metrics: Dict[str, float]) -> str:
    missing = sorted(CORE_METRIC_KEYS.difference(metrics))
    nonfinite = sorted(
        key for key, value in metrics.items()
        if not math.isfinite(float(value))
    )
    problems = []
    if missing:
        problems.append("missing=" + ",".join(missing))
    if nonfinite:
        problems.append("nonfinite=" + ",".join(nonfinite))
    return "; ".join(problems)


def _validate_summary_file(summary_file: str, save_dir: str) -> int:
    required = {"mod_pdb", "ref_pdb", "H", "L", "A", "cdr_type", "pdb"}
    records = []
    with open(summary_file, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid JSON in summary line {line_no}: {exc}"
                ) from exc
            missing = sorted(required.difference(item))
            if missing:
                raise ValueError(
                    f"summary line {line_no} misses fields: {','.join(missing)}"
                )
            records.append(item)
    if not records:
        raise ValueError("summary.json contains no evaluation records.")

    seen_mod = set()
    missing_files = []
    for item in records:
        mod = os.path.realpath(str(item["mod_pdb"]))
        if mod in seen_mod:
            raise ValueError(f"Duplicate generated structure in summary: {mod}")
        seen_mod.add(mod)
        for field in ("mod_pdb", "ref_pdb"):
            path = str(item[field])
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                missing_files.append(path)
    if missing_files:
        raise FileNotFoundError(
            "summary.json references missing/empty PDB files: "
            + ", ".join(missing_files[:5])
        )

    protocol_path = os.path.join(save_dir, "test_protocol.json")
    if os.path.isfile(protocol_path):
        with open(protocol_path, "r", encoding="utf-8") as fin:
            protocol = json.load(fin)
        expected = protocol.get("n_items")
        if expected is not None and int(expected) != len(records):
            raise ValueError(
                "summary/protocol coverage mismatch: "
                f"summary={len(records)} protocol_n_items={expected}."
            )
    return len(records)


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
) -> GenerationResult:
    """Generate the full test set exactly once across the current DDP world."""
    rank, world_size = dist_info()
    save_dir = os.path.abspath(save_dir)
    rank_dir = os.path.join(save_dir, f"rank_{rank:02d}")
    os.makedirs(rank_dir, exist_ok=True)

    local_records: List[dict] = []
    assignments = assigned_logical_batches(
        len(dataset), batch_size, rank=rank, world_size=world_size
    )
    eval_cdr_type, eval_cdr_source = resolve_eval_cdr_type(model)
    if rank == 0:
        print(
            "[EpochTestGeneration] "
            f"n_items={len(dataset)} logical_batch_size={int(batch_size)} "
            f"logical_batches={len(logical_batches(len(dataset), batch_size))} "
            f"world_size={world_size} eval_cdr={eval_cdr_type} "
            f"cdr_source={eval_cdr_source}",
            flush=True,
        )

    local_error = ""
    try:
        for logical_batch_id, global_indices in assignments:
            seed_logical_batch(base_seed, logical_batch_id)
            items = [dataset[i] for i in global_indices]
            batch = dataset.collate_fn(items)
            batch = _to_device(batch, device)
            if "S" not in batch or "paratope_mask" not in batch:
                raise KeyError(
                    "Epoch-test batch must contain S and paratope_mask for the "
                    "V206 framework-sequence invariant."
                )
            input_S = batch["S"].detach().clone()
            design_mask = batch["paratope_mask"].detach().bool().clone()

            with torch.no_grad():
                X, S, _ = model.sample(
                    **batch,
                    n_steps=int(n_steps),
                    show_progress=bool(show_sample_progress and rank == 0),
                )

            if not torch.is_tensor(X) or not torch.is_tensor(S):
                raise TypeError(
                    "model.sample() must return tensor X and S outputs; "
                    f"got X={type(X).__name__}, S={type(S).__name__}."
                )
            if S.shape != input_S.shape or design_mask.shape != input_S.shape:
                raise ValueError(
                    "Generated/input sequence mask shape mismatch: "
                    f"generated={tuple(S.shape)} input={tuple(input_S.shape)} "
                    f"paratope={tuple(design_mask.shape)}."
                )
            framework_changed = (S != input_S) & ~design_mask
            if bool(framework_changed.any().item()):
                first = int(
                    framework_changed.nonzero(as_tuple=False)[0].item()
                )
                raise RuntimeError(
                    "[V206FrameworkSequenceFAIL] model.sample() changed "
                    "sequence outside H3 before PDB writing: "
                    f"logical_batch_id={logical_batch_id} "
                    f"flat_residue_index={first} "
                    f"changed={int(framework_changed.sum().item())}."
                )
            _validate_generated_coordinates_for_pdb(
                X, logical_batch_id=logical_batch_id
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
                        "cdr_type": eval_cdr_type,
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
    finalise_payload: List[object] = [None]
    if rank == 0:
        try:
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
            summary_tmp = summary_file + ".tmp"
            with open(summary_tmp, "w", encoding="utf-8") as fout:
                for item in all_records:
                    public_item = {
                        k: v for k, v in item.items() if not k.startswith("_")
                    }
                    fout.write(json.dumps(public_item, ensure_ascii=False) + "\n")
            os.replace(summary_tmp, summary_file)

            missing_files = []
            for item in all_records:
                for field in ("mod_pdb", "ref_pdb"):
                    path = str(item[field])
                    if not os.path.isfile(path) or os.path.getsize(path) == 0:
                        missing_files.append(path)
            if missing_files:
                raise RuntimeError(
                    "Generated summary references missing/empty PDB files: "
                    + ", ".join(missing_files[:5])
                )

            plan = {
                "protocol": "logical_batch_seeded_v2",
                "base_seed": int(base_seed),
                "logical_batch_size": int(batch_size),
                "n_steps": int(n_steps),
                "n_items": int(len(dataset)),
                "world_size": int(world_size),
                "logical_batches": logical_batches(len(dataset), batch_size),
                "eval_cdr_type": eval_cdr_type,
                "eval_cdr_source": eval_cdr_source,
            }
            _atomic_write_json(os.path.join(save_dir, "test_protocol.json"), plan)
            finalise_payload[0] = {"summary_file": summary_file, "error": ""}
        except Exception as exc:
            finalise_payload[0] = {
                "summary_file": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    if world_size > 1:
        dist.broadcast_object_list(finalise_payload, src=0)
    finalise_result = finalise_payload[0]
    if not isinstance(finalise_result, dict):
        raise RuntimeError("Failed to broadcast epoch-test finalisation result.")
    if finalise_result.get("error"):
        raise RuntimeError(
            "Distributed epoch-test finalisation failed: "
            + str(finalise_result["error"])
        )
    summary_file = (
        str(finalise_result["summary_file"]) if rank == 0 else None
    )
    return GenerationResult(
        summary_file=summary_file,
        records=all_records if rank == 0 else [],
        rank=rank,
        world_size=world_size,
    )


def run_cal_metrics_rank0(
    *,
    summary_file: Optional[str],
    save_dir: str,
    project_root: str,
    num_workers: int = 8,
) -> Dict[str, float]:
    """Run the original metrics, retrying serially only after a failed parallel run.

    The retry does not change metric definitions or the evaluated structures.  It
    removes only ``cal_metrics.py`` worker concurrency, which is a known failure
    mode for external TM-score/DockQ subprocesses.  We never average a partial
    subset: success requires every core metric to be present and finite.
    """
    rank, world_size = dist_info()
    payload: List[object] = [None]

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        try:
            if not summary_file or not os.path.isfile(summary_file):
                raise FileNotFoundError(f"summary.json not found: {summary_file}")
            n_records = _validate_summary_file(summary_file, save_dir)
            cal_metrics = os.path.join(os.path.abspath(project_root), "cal_metrics.py")
            if not os.path.isfile(cal_metrics):
                raise FileNotFoundError(f"cal_metrics.py not found: {cal_metrics}")

            env = os.environ.copy()
            env["OPENMM_CPU_THREADS"] = "1"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")

            attempts = []
            attempt_outputs = []
            worker_plan = [max(1, int(num_workers))]
            if worker_plan[0] > 1:
                worker_plan.append(1)
            successful = None
            for attempt_index, workers in enumerate(worker_plan):
                per_sample_tmp = os.path.join(
                    save_dir, f"per_sample_metrics.workers_{workers}.pkl"
                )
                cmd = [
                    sys.executable,
                    cal_metrics,
                    "--test_set",
                    os.path.abspath(summary_file),
                    "--metrics_path",
                    per_sample_tmp,
                    "--num_workers",
                    str(workers),
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=os.path.abspath(project_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                attempt_text = proc.stdout or ""
                attempt_log_path = os.path.join(
                    save_dir,
                    f"cal_metrics.attempt_{attempt_index + 1}.workers_{workers}.log",
                )
                with open(attempt_log_path, "w", encoding="utf-8") as fout:
                    fout.write(attempt_text)
                attempt_outputs.append(attempt_text)
                attempt_metrics = parse_cal_metrics_output(attempt_text)
                validation_error = _metric_validation_error(attempt_metrics)
                attempt = {
                    "attempt": int(attempt_index + 1),
                    "workers": int(workers),
                    "returncode": int(proc.returncode),
                    "metrics": attempt_metrics,
                    "validation_error": validation_error,
                    "log": attempt_log_path,
                    "log_tail": _log_tail(attempt_text),
                    "per_sample_metrics": per_sample_tmp,
                }
                attempts.append(attempt)
                if proc.returncode == 0 and not validation_error:
                    successful = (attempt, attempt_text)
                    if os.path.isfile(per_sample_tmp):
                        os.replace(
                            per_sample_tmp,
                            os.path.join(save_dir, "per_sample_metrics.pkl"),
                        )
                    break

            log_sections = []
            for attempt, attempt_text in zip(attempts, attempt_outputs):
                log_sections.append(
                    "[EpochTestMetricAttempt] "
                    f"attempt={attempt['attempt']} workers={attempt['workers']} "
                    f"returncode={attempt['returncode']} "
                    f"validation_error={attempt['validation_error'] or 'none'}\n"
                    f"{attempt_text.rstrip()}"
                )
            log_text = "\n\n".join(log_sections) + "\n"
            log_path = os.path.join(save_dir, "cal_metrics.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_text)

            if successful is None:
                last = attempts[-1]
                result = {
                    "returncode": int(last["returncode"] or 1),
                    "metrics": {},
                    "log": log_path,
                    "attempts": attempts,
                    "error": (
                        "All cal_metrics.py attempts failed; last attempt: "
                        + (last["validation_error"] or "non-zero return code")
                    ),
                }
            else:
                success_attempt, _ = successful
                result = {
                    "returncode": 0,
                    "metrics": dict(success_attempt["metrics"]),
                    "log": log_path,
                    "attempts": attempts,
                    "workers_used": int(success_attempt["workers"]),
                    "n_records": int(n_records),
                    "serial_retry_used": bool(success_attempt["workers"] == 1 and int(num_workers) > 1),
                    "error": "",
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

        _atomic_write_json(os.path.join(save_dir, "metrics.json"), result)
        payload[0] = result

    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)

    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("Failed to broadcast test metric result from rank 0.")
    if int(result.get("returncode", 1)) != 0:
        detail = str(result.get("error", "")).strip()
        attempts = result.get("attempts") or []
        tail = ""
        if attempts and isinstance(attempts[-1], dict):
            tail = str(attempts[-1].get("log_tail", "")).strip()
        raise RuntimeError(
            "cal_metrics.py failed after parallel/serial exact retries. "
            + detail
            + "\nSee: " + str(result.get("log", "<unknown>"))
            + ("\n--- cal_metrics tail ---\n" + tail if tail else "")
        )
    if rank == 0:
        print(
            "[EpochTestMetrics] "
            f"n_records={result.get('n_records')} "
            f"workers_used={result.get('workers_used')} "
            f"serial_retry={result.get('serial_retry_used', False)} "
            f"log={result.get('log')}",
            flush=True,
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
    protocol: str = "logical_batch_seeded_v2",
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
