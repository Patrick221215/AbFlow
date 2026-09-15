#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
AbFlow training entry with reproducibility recording.

This version keeps the original AbFlow training behavior, but adds an
ABJE-style run directory and snapshot system:
  - cfg_runtime.json: all args + git/env/python/torch/runtime information
  - command.txt: exact command line
  - env_abflow.json: important experiment environment variables
  - data_manifest.json: file size/mtime/sha256 for train/valid/pep/surface files
  - config_snapshot/: safe copies of small JSON/TXT/SH/PY config-like files
  - code_snapshot/: train script + model/trainer source files when available

The goal is to make every DTM/ScoreFM ablation reproducible.
"""

import os, sys
import re
import json
import time
import socket
import shutil
import hashlib
import inspect
import platform
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

########### Import your packages below ##########
from data.dataset import E2EDataset, VOCAB
from trainer import TrainConfig




class CostBalancedDistributedSampler(Sampler):
    """DDP sampler that preserves each global batch but balances memory cost.

    Ordinary ``DistributedSampler`` shuffles globally and then takes every
    ``world_size``-th sample per rank.  With variable-length antibody complexes
    this can put the same expensive samples on local-rank 0 every epoch/run.

    V236 uses a *per-GPU/local batch* contract.  ``local_batch_size`` is fixed by
    JSON and the effective global optimizer batch is therefore
    ``local_batch_size * world_size``.  This keeps the memory/sample count of
    each rank invariant when the number of GPUs changes, while still balancing
    each optimizer-step sample set across the active ranks.

    For R28-R30, dense Triangle operations scale approximately as O(L_rel^3),
    where L_rel is the complete antibody plus the dataset-defined epitope.
    The cost proxy is whole_residue_count + relational_token_count^3.
    """
    def __init__(self, dataset, local_batch_size, num_replicas, rank, shuffle=True, seed=0):
        self.dataset = dataset
        self.local_batch_size = int(local_batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_replicas <= 0 or not (0 <= self.rank < self.num_replicas):
            raise ValueError('invalid distributed sampler rank/world_size')
        if self.local_batch_size <= 0:
            raise ValueError('local_batch_size must be positive')
        self.global_batch_size = self.local_batch_size * self.num_replicas
        self.num_samples = (len(dataset) + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        self.costs = self._estimate_costs()

    def _estimate_costs(self):
        # The formal RAbD runs use one in-memory processed part.  Avoid calling
        # dataset.__getitem__ here because that would regenerate templates and
        # reload proposal/surface data merely to schedule a batch.
        if not (hasattr(self.dataset, 'data') and hasattr(self.dataset, 'idx_mapping')):
            return [1.0] * len(self.dataset)
        if hasattr(self.dataset, 'file_names') and len(self.dataset.file_names) != 1:
            return [1.0] * len(self.dataset)
        out = []
        try:
            for logical_idx in range(len(self.dataset)):
                raw_idx = int(self.dataset.idx_mapping[logical_idx])
                item = self.dataset.data[raw_idx]
                h = item.get_heavy_chain()
                l = item.get_light_chain()
                if getattr(self.dataset, 'full_antigen', False):
                    ag_obj = item.get_antigen()
                    ag_n = 0
                    for chain_name in ag_obj.get_chain_names():
                        ag_n += len(ag_obj.get_chain(chain_name))
                else:
                    ag_n = len(item.get_epitope())
                antibody_n = int(len(h) + len(l))
                whole_n = int(ag_n + antibody_n + 3)
                relational_n = int(antibody_n + ag_n)
                out.append(float(whole_n + relational_n ** 3))
            return out
        except Exception as exc:
            print_log(f'[DDPBatchBalance][WARN] cost estimation fallback to uniform: {exc}', level='WARN')
            return [1.0] * len(self.dataset)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices += (indices * ((padding + len(indices) - 1) // len(indices)))[:padding]
        else:
            indices = indices[:self.total_size]

        rank_indices = []
        for start in range(0, self.total_size, self.global_batch_size):
            global_batch = indices[start:start + self.global_batch_size]
            if not global_batch:
                continue
            if len(global_batch) % self.num_replicas != 0:
                raise RuntimeError('balanced DDP final batch is not divisible by world_size')
            target = len(global_batch) // self.num_replicas
            buckets = [[] for _ in range(self.num_replicas)]
            loads = [0.0 for _ in range(self.num_replicas)]
            # Largest-first greedy partition with an exact sample-count cap.
            for idx in sorted(global_batch, key=lambda x: self.costs[x], reverse=True):
                candidates = [r for r in range(self.num_replicas) if len(buckets[r]) < target]
                r = min(candidates, key=lambda rr: (loads[rr], rr))
                buckets[r].append(idx)
                loads[r] += self.costs[idx]
            rank_indices.extend(buckets[self.rank])
        if len(rank_indices) != self.num_samples:
            raise RuntimeError(
                f'balanced sampler produced {len(rank_indices)} samples, expected {self.num_samples}'
            )
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples

# ============================================================
# 0. Recording helpers
# ============================================================

def _is_main_rank(local_rank: int) -> bool:
    return local_rank in (-1, 0)


def _safe_makedirs(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _run_cmd(cmd, cwd=None):
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"<failed: {' '.join(cmd)} | {repr(e)}>"


def _sha256_file(path: str, chunk_size: int = 1024 * 1024):
    if not path or (not os.path.isfile(path)):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _file_manifest(path: str):
    """Return a JSON-serializable manifest for one file path."""
    if not path:
        return {"path": path, "exists": False, "reason": "empty"}
    p = os.path.abspath(path)
    if not os.path.exists(p):
        return {"path": p, "exists": False, "reason": "not_found"}
    if not os.path.isfile(p):
        return {"path": p, "exists": True, "is_file": False}
    st = os.stat(p)
    return {
        "path": p,
        "exists": True,
        "is_file": True,
        "size_bytes": int(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256": _sha256_file(p),
    }


def _safe_copy_file(src: str, dst_dir: str, max_bytes: int, note_list=None):
    """Copy small files into a snapshot directory; only record large files."""
    if not src:
        return None
    src_abs = os.path.abspath(src)
    if not os.path.isfile(src_abs):
        return None

    st = os.stat(src_abs)
    record = _file_manifest(src_abs)
    record["copied"] = False
    record["copied_to"] = None

    if st.st_size <= max_bytes:
        _safe_makedirs(dst_dir)
        dst = os.path.join(dst_dir, os.path.basename(src_abs))
        # Avoid accidental overwrite if two files share the same basename.
        if os.path.exists(dst):
            stem = Path(src_abs).stem
            suffix = Path(src_abs).suffix
            digest = (record.get("sha256") or "unknown")[:8]
            dst = os.path.join(dst_dir, f"{stem}.{digest}{suffix}")
        try:
            shutil.copy2(src_abs, dst)
            record["copied"] = True
            record["copied_to"] = dst
        except Exception as e:
            record["copy_error"] = repr(e)
    else:
        record["skip_copy_reason"] = f"file larger than snapshot_max_bytes={max_bytes}"

    if note_list is not None:
        note_list.append(record)
    return record


def _snapshot_source(obj, dst_dir: str, copied_records: list, max_bytes: int):
    """Copy the source file of a class/function/module if Python can locate it."""
    try:
        src = inspect.getsourcefile(obj)
    except Exception:
        src = None
    if src and os.path.isfile(src):
        _safe_copy_file(src, dst_dir, max_bytes=max_bytes, note_list=copied_records)


def _collect_env(prefixes=None):
    if prefixes is None:
        prefixes = [
            "ABFLOW_", "CUDA", "NCCL", "MASTER_", "WORLD_", "LOCAL_", "RANK",
            "OMP_", "OPENMM_", "PYTHON", "CONDA", "MAMBA", "MICROMAMBA"
        ]
    env = {}
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in prefixes):
            env[k] = v
    return dict(sorted(env.items()))


def _collect_git_info(project_dir: str):
    return {
        "project_dir": os.path.abspath(project_dir),
        "commit": _run_cmd(["git", "rev-parse", "HEAD"], cwd=project_dir),
        "branch": _run_cmd(["git", "branch", "--show-current"], cwd=project_dir),
        "status_short": _run_cmd(["git", "status", "--short"], cwd=project_dir),
        "remote": _run_cmd(["git", "remote", "-v"], cwd=project_dir),
    }


def _make_run_dir(args, local_rank: int):
    """Create/resolve run directory while preserving old AbFlow behavior by default.

    Original AbFlow uses args.save_dir directly. To avoid breaking existing scripts,
    this function uses args.save_dir as the run directory unless --auto_run_name or
    --run_name is provided.
    """
    save_root = os.path.abspath(args.save_dir)
    if args.auto_run_name:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_root, run_name)
    elif args.run_name:
        run_dir = os.path.join(save_root, args.run_name)
    else:
        run_dir = save_root

    if _is_main_rank(local_rank):
        _safe_makedirs(run_dir)
    return run_dir


def _list_version_dirs(save_root: str):
    """Return version-like directories created by the original AbFlow Trainer.

    AbFlow's trainer usually creates a subdirectory such as version=7 or
    version_7 under the configured save_dir. The exact spelling may vary across
    code versions, so this function accepts several common forms.
    """
    root = os.path.abspath(save_root)
    if not os.path.isdir(root):
        return set()
    version_dirs = set()
    pat = re.compile(r"^version([=_-]?\d+)?$", re.IGNORECASE)
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and pat.match(name):
            version_dirs.add(os.path.abspath(path))
    return version_dirs


def _looks_like_version_dir(path: str, save_root: str, must_exist: bool = False):
    """Return True if path looks like the concrete Trainer version directory.

    Important: the original AbFlow Trainer sets config.save_dir to
    <save_root>/version_N during Trainer.__init__, but it creates the directory
    later in Trainer.train(). Therefore we must NOT require the directory to
    already exist when inferring from trainer.config.save_dir.
    """
    if not path:
        return False
    p = os.path.abspath(path)
    root = os.path.abspath(save_root)
    base = os.path.basename(p)
    ok = (
        p.startswith(root + os.sep)
        and p != root
        and re.match(r"^version([=_-]?\d+)?$", base, flags=re.IGNORECASE) is not None
    )
    if must_exist:
        ok = ok and os.path.isdir(p)
    return ok


def _candidate_paths_from_attr(value):
    """Convert a trainer attribute value into plausible run-directory paths."""
    if value is None:
        return []
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str):
        return []
    value = os.path.abspath(value)
    cands = [value]
    # If the attribute points to a file or a subfolder such as checkpoints/logs,
    # its parent may be the actual version directory.
    cands.append(os.path.dirname(value))
    cands.append(os.path.dirname(os.path.dirname(value)))
    # keep order while deduplicating
    out = []
    seen = set()
    for c in cands:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _infer_trainer_run_dir(trainer, save_root: str, before_version_dirs=None):
    """Infer the real version directory created by the original AbFlow Trainer.

    The original AbFlow Trainer does this in Trainer.__init__:
        self.version = self._get_version()
        self.config.save_dir = os.path.join(self.config.save_dir, f"version_{self.version}")
        self.model_dir = os.path.join(self.config.save_dir, "checkpoint")

    The directory itself is created later in Trainer.train(). Therefore the
    correct source of truth is trainer.config.save_dir, not the latest existing
    version directory on disk. Relying on existing directories causes records to
    be written into version_{n-1}.
    """
    root = os.path.abspath(save_root)
    before_version_dirs = before_version_dirs or set()

    # 1) Highest priority: original AbFlow Trainer mutates config.save_dir to
    # the concrete version directory before the directory exists.
    cfg = getattr(trainer, "config", None)
    if cfg is not None and hasattr(cfg, "save_dir"):
        cand = os.path.abspath(getattr(cfg, "save_dir"))
        if _looks_like_version_dir(cand, root, must_exist=False):
            return cand

    # 2) model_dir usually points to <version_dir>/checkpoint. Its parent is the
    # version directory. This also may not exist yet.
    model_dir = getattr(trainer, "model_dir", None)
    if model_dir is not None:
        cand = os.path.abspath(os.path.dirname(str(model_dir)))
        if _looks_like_version_dir(cand, root, must_exist=False):
            return cand

    # 3) Other explicit attributes, if available.
    objects = [trainer]
    for attr in ("config", "cfg", "train_config"):
        obj = getattr(trainer, attr, None)
        if obj is not None:
            objects.append(obj)

    attr_names = [
        "version_dir", "run_dir", "save_root", "save_dir", "log_dir",
        "ckpt_dir", "model_dir", "checkpoint_dir", "tensorboard_dir"
    ]
    for obj in objects:
        for attr in attr_names:
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                except Exception:
                    continue
                for cand in _candidate_paths_from_attr(val):
                    if _looks_like_version_dir(cand, root, must_exist=False):
                        return os.path.abspath(cand)

    # 4) If a new version directory appeared after Trainer construction, use it.
    after = _list_version_dirs(root)
    new_dirs = sorted(after - set(before_version_dirs), key=lambda x: os.path.getmtime(x))
    if new_dirs:
        return os.path.abspath(new_dirs[-1])

    # 5) Last resort: latest existing version directory. This should rarely be
    # used because it can point to version_{n-1}; keep it only as fallback.
    if after:
        latest = sorted(after, key=lambda x: os.path.getmtime(x))[-1]
        return os.path.abspath(latest)

    return root


def setup_experiment_record(args, *, local_rank: int, rank: int, world_size: int, stage: str = "pre_build", record_root: str = None):
    """Write ABJE-style reproducibility records into the actual trainer run directory.

    This is deliberately independent of the Trainer implementation. The trainer
    only sees args.save_dir, which we set to the resolved run_dir before building
    TrainConfig.
    """
    if not _is_main_rank(local_rank):
        return None

    run_dir = os.path.abspath(record_root or getattr(args, "actual_run_dir", args.save_dir))
    _safe_makedirs(run_dir)

    record_dir = os.path.join(run_dir, "record")
    snap_dir = os.path.join(record_dir, "config_snapshot")
    code_dir = os.path.join(record_dir, "code_snapshot")
    _safe_makedirs(record_dir)
    _safe_makedirs(snap_dir)
    _safe_makedirs(code_dir)

    project_dir = os.getcwd()
    max_bytes = int(float(args.snapshot_max_mb) * 1024 * 1024)

    # 1) exact command
    command_text = " ".join([sys.executable] + sys.argv)
    with open(os.path.join(record_dir, "command.txt"), "w", encoding="utf-8") as fw:
        fw.write(command_text + "\n")

    # 2) environment variables
    env_abflow = _collect_env()
    with open(os.path.join(record_dir, "env_abflow.json"), "w", encoding="utf-8") as fw:
        json.dump(env_abflow, fw, indent=2, ensure_ascii=False)

    # 3) input file manifests and safe snapshots
    input_paths = {
        "train_set": args.train_set,
        "valid_set": args.valid_set,
        "train_pep": args.train_pep,
        "valid_pep": args.valid_pep,
        "train_surf": args.train_surf,
        "valid_surf": args.valid_surf,
    }
    manifests = {}
    copied_inputs = []
    if not args.no_snapshot:
        for key, path in input_paths.items():
            manifests[key] = _file_manifest(path) if path else {"path": path, "exists": False, "reason": "empty"}
            # Copy small config-like files; large dataset/pkl files are only hashed.
            if path and Path(path).suffix.lower() in {".json", ".jsonl", ".txt", ".idx", ".csv", ".tsv", ".yaml", ".yml"}:
                _safe_copy_file(path, snap_dir, max_bytes=max_bytes, note_list=copied_inputs)
    else:
        for key, path in input_paths.items():
            manifests[key] = _file_manifest(path) if path else {"path": path, "exists": False, "reason": "empty"}

    with open(os.path.join(record_dir, "data_manifest.json"), "w", encoding="utf-8") as fw:
        json.dump({"inputs": manifests, "copied_inputs": copied_inputs}, fw, indent=2, ensure_ascii=False)

    # 4) code snapshot: this train file
    copied_code = []
    if not args.no_snapshot:
        try:
            _safe_copy_file(__file__, code_dir, max_bytes=max_bytes, note_list=copied_code)
        except Exception:
            pass

    # 5) runtime config
    runtime = {
        "stage": stage,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "command": command_text,
        "args": vars(args),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        "git": _collect_git_info(project_dir),
        "env_abflow": env_abflow,
        "code_snapshot": copied_code,
    }
    with open(os.path.join(run_dir, "cfg_runtime.json"), "w", encoding="utf-8") as fw:
        json.dump(runtime, fw, indent=2, ensure_ascii=False)

    # Also keep a copy inside record/ for clarity.
    with open(os.path.join(record_dir, "cfg_runtime.json"), "w", encoding="utf-8") as fw:
        json.dump(runtime, fw, indent=2, ensure_ascii=False)

    return record_dir


def finalize_code_snapshot(args, model=None, trainer_cls=None):
    """After model/trainer construction, snapshot their source files if possible."""
    local_rank = getattr(args, "local_rank", -1)
    if not _is_main_rank(local_rank) or args.no_snapshot:
        return
    run_dir = os.path.abspath(getattr(args, "actual_run_dir", args.save_dir))
    code_dir = os.path.join(run_dir, "record", "code_snapshot")
    _safe_makedirs(code_dir)
    max_bytes = int(float(args.snapshot_max_mb) * 1024 * 1024)
    records = []
    try:
        if model is not None:
            _snapshot_source(model.__class__, code_dir, records, max_bytes=max_bytes)
    except Exception as e:
        records.append({"source": "model", "error": repr(e)})
    try:
        if trainer_cls is not None:
            _snapshot_source(trainer_cls, code_dir, records, max_bytes=max_bytes)
    except Exception as e:
        records.append({"source": "trainer", "error": repr(e)})
    try:
        _snapshot_source(TrainConfig, code_dir, records, max_bytes=max_bytes)
    except Exception as e:
        records.append({"source": "TrainConfig", "error": repr(e)})

    with open(os.path.join(run_dir, "record", "code_manifest.json"), "w", encoding="utf-8") as fw:
        json.dump(records, fw, indent=2, ensure_ascii=False)


# ============================================================
# 1. Arguments
# ============================================================

def _resolve_project_path(project_root, value):
    if value in (None, ""):
        return ""
    value = str(value)
    return value if os.path.isabs(value) else os.path.abspath(os.path.join(project_root, value))

def _apply_trainer_runtime_from_config(cfg, config_path):
    """Bridge JSON runtime/evaluation settings into the existing Trainer API.

    The V203 Trainer intentionally consumes runtime infrastructure through
    ``ABFLOW_*`` variables.  Scientific model parameters never pass through this
    bridge.  Paths and sampling settings keep JSON as their single authority.
    """
    project_root = os.path.abspath(
        os.environ.get("ABFLOW_PROJECT_ROOT") or os.path.dirname(__file__)
    )
    os.environ["ABFLOW_PROJECT_ROOT"] = project_root

    runtime = cfg["runtime"]
    ddp = runtime.get("ddp", {})
    os.environ["ABFLOW_DDP_FIND_UNUSED_PARAMETERS"] = (
        "on" if ddp.get("find_unused_parameters", False) else "off"
    )
    os.environ["ABFLOW_DDP_STATIC_GRAPH"] = (
        "on" if ddp.get("static_graph", True) else "off"
    )
    os.environ["ABFLOW_DDP_VALIDATION"] = "on"

    # Formal project protocol: every epoch is Train -> Val -> observational Test.
    # Sampling hyperparameters are shared with standalone generation instead of
    # being duplicated under a second epoch-test configuration tree.
    data_test = cfg["data"]["test"]
    generation = cfg["generation"]
    evaluation = cfg.get("evaluation", {})
    os.environ["ABFLOW_EPOCH_TEST"] = "on"
    os.environ["ABFLOW_EPOCH_TEST_INTERVAL"] = "1"
    os.environ["ABFLOW_EPOCH_TEST_JSON"] = _resolve_project_path(
        project_root, data_test["set"]
    )
    os.environ["ABFLOW_EPOCH_TEST_PEP"] = _resolve_project_path(
        project_root, data_test.get("pep")
    )
    os.environ["ABFLOW_EPOCH_TEST_SURF"] = _resolve_project_path(
        project_root, data_test.get("surface")
    )
    os.environ["ABFLOW_EPOCH_TEST_BATCH_SIZE"] = str(int(generation["batch_size"]))
    os.environ["ABFLOW_EPOCH_TEST_N_STEPS"] = str(int(generation["n_steps"]))
    os.environ["ABFLOW_EPOCH_TEST_BASE_SEED"] = str(int(generation["seed"]))
    os.environ["ABFLOW_EPOCH_TEST_SHOW_SAMPLE_PROGRESS"] = (
        "on" if generation.get("show_sample_progress", False) else "off"
    )
    os.environ["ABFLOW_EPOCH_TEST_METRIC_WORKERS"] = str(
        int(evaluation.get("metric_workers", 8))
    )
    os.environ["ABFLOW_EPOCH_TEST_KEEP_STRUCTURES"] = (
        "on" if evaluation.get("keep_structures", False) else "off"
    )
    # V203 formal failure contract has two deliberately different authorities:
    #   1) infrastructure/protocol failures are always fail-fast;
    #   2) finite-but-invalid model outputs are observational and are recorded
    #      without terminating training.
    # Keep these explicit here so direct train.py invocation and launcher-based
    # invocation have identical semantics.
    os.environ["ABFLOW_EPOCH_TEST_FAIL_FAST"] = "on"
    os.environ["ABFLOW_EPOCH_TEST_MODEL_INVALID_POLICY"] = "record_and_continue"

    # Legacy evaluation.failure_policy is retained for compatibility with
    # downstream metric/evaluation code, but it must not weaken the V203
    # infrastructure fail-fast contract above.
    failure_policy = str(
        evaluation.get("failure_policy", "record_and_continue")
    ).strip().lower()
    if failure_policy not in {"abort", "record_and_continue"}:
        raise ValueError(
            "evaluation.failure_policy must be 'abort' or "
            f"'record_and_continue', got {failure_policy!r}"
        )
    os.environ["ABFLOW_EPOCH_TEST_FAILURE_POLICY"] = failure_policy

    logging_cfg = cfg["training"]["logging"]
    os.environ["ABFLOW_SCI_LOG_FIRST_STEPS"] = str(
        int(logging_cfg.get("science_first_steps", 3))
    )
    os.environ["ABFLOW_SCI_LOG_INTERVAL"] = str(
        int(logging_cfg.get("science_interval", 0))
    )
    os.environ["ABFLOW_GRAD_FINITE_GUARD_STEPS"] = str(
        int(logging_cfg.get("runtime_guard_steps", 2))
    )
    os.environ["ABFLOW_TRAIN_LOSS_OUTLIER_THRESHOLD"] = str(
        float(logging_cfg.get("train_loss_outlier_threshold", 1.0e4))
    )


def _runtime_local_gpu_count():
    """Resolve local CUDA worker count from torchrun/launcher, never JSON.

    GPU placement is an execution resource decision.  The scientific JSON must
    not encode physical GPU ids.  ``torchrun`` exports LOCAL_WORLD_SIZE/WORLD_SIZE;
    direct single-process fallback uses CUDA_VISIBLE_DEVICES.
    """
    for key in ("LOCAL_WORLD_SIZE", "WORLD_SIZE", "ABFLOW_NPROC_PER_NODE"):
        raw = os.environ.get(key)
        if raw:
            try:
                value = int(raw)
                if value > 0:
                    return value
            except ValueError:
                pass
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible:
        ids = [x.strip() for x in visible.split(',') if x.strip()]
        if ids:
            return len(ids)
    return 1


def _runtime_resume_checkpoint(schedule_cfg):
    """CLI/launcher resume overrides JSON; JSON fallback remains compatible."""
    env_value = str(os.environ.get("ABFLOW_RESUME_CHECKPOINT", "") or "").strip()
    if env_value:
        return env_value
    return schedule_cfg.get("resume_checkpoint", "") or ""


def _namespace_from_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data = cfg["data"]
    task = data["task"]
    tr = cfg["training"]
    opt = tr["optimizer"]
    sched = tr["schedule"]
    loader = tr["loader"]
    precision = tr["precision"]
    ema = tr["ema"]
    logging_cfg = tr["logging"]
    snapshot = tr["snapshot"]

    model = cfg["model"]
    arch = model["architecture"]
    model_task = model["task"]
    runtime = cfg["runtime"]

    _apply_trainer_runtime_from_config(cfg, config_path)

    return argparse.Namespace(
        config=config_path,

        train_set=data["train"]["set"],
        valid_set=data["valid"]["set"],
        train_pep=data["train"].get("pep"),
        valid_pep=data["valid"].get("pep"),
        train_surf=data["train"].get("surface"),
        valid_surf=data["valid"].get("surface"),
        cdr=task["cdr"],
        paratope=task["paratope"],

        lr=opt["lr"],
        final_lr=opt["final_lr"],
        warmup=opt["warmup"],
        max_epoch=sched["max_epoch"],
        grad_clip=opt["grad_clip"],
        save_dir=tr["output_dir"],
        # V236: train/validation batch size is per GPU/rank.  Keep a legacy
        # fallback so old configs remain readable, but formal R72/R73 configs
        # use ``per_gpu_batch_size`` explicitly.
        batch_size=loader.get("per_gpu_batch_size", loader.get("batch_size")),
        patience=sched["patience"],
        save_topk=sched["save_topk"],
        shuffle=loader["shuffle"],
        num_workers=loader["num_workers"],
        prefetch_factor=loader["prefetch_factor"],
        valid_num_workers=loader["valid_num_workers"],
        valid_prefetch_factor=loader["valid_prefetch_factor"],
        valid_persistent_workers=loader["valid_persistent_workers"],
        amp=precision["amp"],
        amp_dtype=precision["amp_dtype"],
        log_interval=logging_cfg["log_interval"],
        tqdm_mininterval=logging_cfg["tqdm_mininterval"],
        allow_tf32=precision["allow_tf32"],
        save_interval=sched["save_interval"],
        resume_checkpoint=_runtime_resume_checkpoint(sched),
        use_ema=ema["enabled"],
        ema_decay=ema["decay"],

        run_name=None,
        auto_run_name=False,
        snapshot_max_mb=snapshot["max_mb"],
        no_snapshot=not snapshot["enabled"],

        gpus=list(range(_runtime_local_gpu_count())),
        local_rank=-1,

        model_type=model["type"],
        embed_dim=arch["embed_dim"],
        hidden_size=arch["hidden_size"],
        k_neighbors=arch["k_neighbors"],
        n_layers=arch["n_layers"],
        iter_round=arch["iter_round"],
        num_verts=task["num_verts"],
        dropout=arch["dropout"],
        relative_position=arch["relative_position"],

        seq_warmup=model_task["seq_warmup"],
        pep_seq=model_task["pep_seq"],
        pep_struct=model_task["pep_struct"],
        struct_only=model_task["struct_only"],
        bind_dist_cutoff=model_task["bind_dist_cutoff"],
        no_pred_edge_dist=not model_task["pred_edge_dist"],
        backbone_only=model_task["backbone_only"],
        fix_channel_weights=model_task["fix_channel_weights"],
        no_memory=not model_task["keep_memory"],

        experiment=cfg.get("experiment", {}),
        model_config=model,
        loss=cfg["loss"],
        runtime=runtime,
        generation=cfg.get("generation", {}),
        evaluation=cfg.get("evaluation", {}),
    )


def parse():
    parser = argparse.ArgumentParser(description="AbFlow training")
    parser.add_argument("--config", required=True, help="Experiment JSON")
    cli = parser.parse_args()
    return _namespace_from_config(cli.config)


# ============================================================
# 2. Main training entry
# ============================================================

def main(args):
    # import ipdb; ipdb.set_trace()
    ########### DDP and run directory setup ###########
    os.environ.setdefault('NCCL_TIMEOUT', '30')

    if bool(getattr(args, "allow_tf32", False)) and torch.cuda.is_available():
        # Ampere/A6000 speed path.  TF32 affects matmul/conv kernels, not tensor
        # semantics or model architecture. It is safe to disable from config if
        # exact FP32 reproducibility is needed.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Robust DDP detection. This keeps old behavior for single GPU while also
    # working with torchrun, where WORLD_SIZE/LOCAL_RANK/RANK are set by torch.
    world_size = int(os.environ.get('WORLD_SIZE', str(len(args.gpus) if len(args.gpus) > 1 else 1)))
    is_ddp = len(args.gpus) > 1 or world_size > 1

    if is_ddp:
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        rank = int(os.environ.get('RANK', args.local_rank))
        torch.cuda.set_device(args.local_rank)
        if not torch.distributed.is_initialized():
            # env:// is the safest for torchrun. If you use the old launcher,
            # LOCAL_RANK/WORLD_SIZE/RANK are also normally provided.
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
        train_sampler = None  # will be built after dataset is loaded
    else:
        args.local_rank = -1
        rank = 0
        world_size = 1
        train_sampler = None

    # Resolve run directory and redirect args.save_dir to it before TrainConfig.
    run_dir = _make_run_dir(args, args.local_rank)
    args.save_dir = run_dir

    if is_ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # The original AbFlow Trainer creates a concrete subdirectory such as
    # version=7 under save_dir. We record the set before Trainer construction,
    # then write records after Trainer has created/inferred that version dir.
    version_dirs_before = _list_version_dirs(run_dir)

    ########### load your train / valid set ###########
    if _is_main_rank(args.local_rank):
        print_log(
            "[FormalRun] "
            f"experiment={args.experiment.get('id', '<unknown>')} "
            f"protocol={args.experiment.get('protocol', '<unknown>')} "
            f"train={args.train_set} valid={args.valid_set} "
            f"test={os.environ['ABFLOW_EPOCH_TEST_JSON']} "
            f"cdr={','.join(args.cdr)} paratope={','.join(args.paratope)} "
            f"per_gpu_train_val_batch={args.batch_size} "
            f"effective_global_train_batch={int(args.batch_size) * int(world_size)} "
            f"epochs={args.max_epoch} ckpt=validation test=observation_only"
        )

    train_set = E2EDataset(args.train_set, pep_file=args.train_pep, surf_file=args.train_surf,
                           cdr=args.cdr, paratope=args.paratope, num_verts=args.num_verts)
    valid_set = E2EDataset(args.valid_set, pep_file=args.valid_pep, surf_file=args.valid_surf,
                           cdr=args.cdr, paratope=args.paratope, num_verts=args.num_verts)

    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(**vars(args))

    if args.model_type == 'AbFlow':
        from trainer import AbFlowTrainer as Trainer
        from models import AbFlowModel
        model = AbFlowModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), args.num_verts, VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers,
                   dropout=args.dropout,
                   pep_seq=args.pep_seq,
                   pep_struct=args.pep_struct,
                   struct_only=args.struct_only,
                   iter_round=args.iter_round,
                   backbone_only=args.backbone_only,
                   fix_channel_weights=args.fix_channel_weights,
                   pred_edge_dist=not args.no_pred_edge_dist,
                   keep_memory=not args.no_memory,
                   cdr_type=args.cdr, paratope=args.paratope,
                   relative_position=args.relative_position,
                   model_config=args.model_config,
                   loss_config=args.loss)
    elif args.model_type == 'AbFlowStruct':
        from trainer import AbFlowTrainer as Trainer
        from models import AbFlowStructModel
        model = AbFlowStructModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), args.num_verts, VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   iter_round=args.iter_round,
                   backbone_only=args.backbone_only,
                   fix_channel_weights=args.fix_channel_weights,
                   pred_edge_dist=not args.no_pred_edge_dist,
                   keep_memory=not args.no_memory,
                   cdr_type=args.cdr, paratope=args.paratope)
    elif args.model_type == 'AbFlowOpt':
        from trainer import AbFlowOptTrainer as Trainer
        from models import AbFlowOptModel
        model = AbFlowOptModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   fix_atom_weights=args.fix_channel_weights, cdr_type=args.cdr)
    else:
        raise NotImplementedError(f'model {args.model_type} not implemented')

    # V236 batch-size contract:
    #   JSON training.loader.per_gpu_batch_size = samples owned by each GPU/rank.
    # GPU count is a runtime resource choice supplied by torchrun/launcher and is
    # NOT constrained by the scientific config.  Therefore:
    #   effective_global_train_batch = per_gpu_batch_size * world_size.
    # Validation keeps the same per-rank/logical batch size, while formal Test
    # remains controlled independently by generation.batch_size.
    local_batch_size = int(args.batch_size)
    if local_batch_size <= 0:
        raise ValueError(f"per-GPU batch size must be positive, got {local_batch_size}")
    effective_global_batch_size = local_batch_size * max(1, world_size)

    # One optimizer step consumes one local batch on every active rank.
    step_per_epoch = (
        len(train_set) + effective_global_batch_size - 1
    ) // effective_global_batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)
    config.add_parameter(effective_global_batch_size=effective_global_batch_size)

    if is_ddp:
        use_cost_balance = bool(
            args.runtime.get("ddp", {}).get("cost_balanced", False)
        )
        if use_cost_balance:
            train_sampler = CostBalancedDistributedSampler(
                train_set, local_batch_size=local_batch_size,
                num_replicas=world_size, rank=rank, shuffle=args.shuffle,
                seed=0,
            )
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set, shuffle=args.shuffle
            )
    else:
        train_sampler = None

    # TrainConfig and both DataLoaders use the exact same per-GPU batch size.
    args.batch_size = local_batch_size
    config.batch_size = local_batch_size
    config.local_rank = args.local_rank


    # DataLoader settings.  GPU under-utilization is often caused by the GPU
    # waiting for CPU collation / host-to-device transfer.  pin_memory,
    # persistent_workers and prefetch_factor keep batches ready in the background.
    pin_memory = torch.cuda.is_available() and args.gpus[0] != -1

    def _make_loader_kwargs(num_workers, prefetch_factor, persistent_workers):
        num_workers = int(num_workers)
        kwargs = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if num_workers > 0:
            kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
            kwargs["persistent_workers"] = bool(persistent_workers)

        return kwargs


    train_loader_kwargs = _make_loader_kwargs(
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True,
    )

    # Formal validation must keep every validation epoch and every metric.
    # The only change here is resource scheduling: use fewer workers and no
    # persistent validation workers by default to avoid a memory spike when
    # train_loader workers are still alive.
    valid_loader_kwargs = _make_loader_kwargs(
        num_workers=args.valid_num_workers,
        prefetch_factor=args.valid_prefetch_factor,
        persistent_workers=args.valid_persistent_workers,
    )

    if _is_main_rank(args.local_rank):
        print_log(
            "[DDPData] "
            f"world={world_size} per_gpu_batch={args.batch_size} "
            f"effective_global_train_batch={effective_global_batch_size} "
            f"steps_per_epoch={step_per_epoch} "
            f"cost_balanced={int(bool(is_ddp and args.runtime.get('ddp', {}).get('cost_balanced', False)))} "
            f"workers=train:{args.num_workers}/valid:{args.valid_num_workers}"
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(args.shuffle and train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
        **train_loader_kwargs,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **valid_loader_kwargs,
    )
    
    # import ipdb; ipdb.set_trace()
    trainer = Trainer(model, train_loader, valid_loader, config)

    # Now the original Trainer should have created or exposed its concrete
    # version directory. Put record/ under that directory, e.g. version=7/record.
    actual_run_dir = _infer_trainer_run_dir(trainer, args.save_dir, version_dirs_before)
    args.actual_run_dir = actual_run_dir
    setup_experiment_record(
        args,
        local_rank=args.local_rank,
        rank=rank,
        world_size=world_size,
        stage="post_trainer_init",
        record_root=actual_run_dir,
    )
    finalize_code_snapshot(args, model=model, trainer_cls=Trainer)

    trainer.train(args.gpus, args.local_rank)

    if is_ddp and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = parse()
    main(args)
