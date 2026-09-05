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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

########### Import your packages below ##########
from data.dataset import E2EDataset, VOCAB
from trainer import TrainConfig


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

def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')
    parser.add_argument('--train_pep', type=str, default=None, help='path to train pep set')
    parser.add_argument('--valid_pep', type=str, default=None, help='path to valid pep set')
    parser.add_argument('--train_surf', type=str, default=None, help='path to train surf set')
    parser.add_argument('--valid_surf', type=str, default=None, help='path to valid surf set')
    parser.add_argument('--cdr', type=str, default=None, nargs='+', help='cdr to generate, L1/2/3, H1/2/3,(can be list, e.g., L3 H3) None for all including framework')
    parser.add_argument('--paratope', type=str, default='H3', nargs='+', help='cdrs to use as paratope')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='exponential decay from lr to final_lr')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model, logs and experiment records')
    parser.add_argument('--batch_size', type=int, required=True, help='per-GPU micro-batch size; effective global batch = batch_size * world_size')
    parser.add_argument('--patience', type=int, default=1000, help='patience before early stopping (set with a large number to turn off early stopping)')
    parser.add_argument('--save_topk', type=int, default=10, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        '--prefetch_factor',
        type=int,
        default=4,
        help='Train DataLoader prefetch factor when num_workers > 0.'
    )

    parser.add_argument(
        '--valid_num_workers',
        type=int,
        default=2,
        help='Validation DataLoader workers. Formal validation keeps all metrics, but uses fewer workers to avoid RAM/pinned-memory spikes.'
    )

    parser.add_argument(
        '--valid_prefetch_factor',
        type=int,
        default=2,
        help='Validation DataLoader prefetch factor when valid_num_workers > 0.'
    )

    parser.add_argument(
        '--valid_persistent_workers',
        action='store_true',
        help='Keep validation DataLoader workers persistent. Default off to avoid train+valid worker memory spikes.'
    )
    parser.add_argument('--amp', action='store_true',
                        help='Enable CUDA automatic mixed precision training.')
    parser.add_argument('--amp_dtype', type=str, default='bf16',
                        choices=['bf16', 'fp16'],
                        help='AMP dtype. bf16 is preferred on Ampere/A6000 for stability.')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Write training scalar logs every N steps to reduce CUDA sync.')
    parser.add_argument('--tqdm_mininterval', type=float, default=5.0,
                        help='Minimum seconds between tqdm screen refreshes.')
    parser.add_argument('--allow_tf32', action='store_true',
                        help='Allow TF32 matmul/cudnn on Ampere GPUs.')
    
    parser.add_argument('--save_interval', type=int, default=1,
                    help='Save full training-state checkpoint every N completed epochs. Set <=0 to disable periodic last checkpoint saves.')
    parser.add_argument('--resume_checkpoint', type=str, default='',
                        help='Path to a full training-state checkpoint. Empty means train from scratch.')
    parser.add_argument('--use_ema', action='store_true',
                        help='Enable EMA for training and validation. Keep disabled for first clean S3/S3CG retrain.')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay. Only used when --use_ema is set.')
    

    # reproducibility recording
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional subdirectory name under save_dir. Useful for ablations such as e8_dtm_core.')
    parser.add_argument('--auto_run_name', action='store_true',
                        help='If set, create save_dir/<timestamp or run_name>. If unset and run_name is empty, use save_dir directly to preserve old behavior.')
    parser.add_argument('--snapshot_max_mb', type=float, default=20.0,
                        help='Max file size in MB for copying files into record/config_snapshot or record/code_snapshot. Larger files are only hashed.')
    parser.add_argument('--no_snapshot', action='store_true',
                        help='Disable copying snapshot files. Manifests and cfg_runtime.json are still written.')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using torch.distributed launch/torchrun.")
    
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['AbFlow', 'AbFlowStruct', 'AbFlowOpt'],
                        help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--iter_round', type=int, default=3, help='Number of iterations for generation')
    parser.add_argument('--num_verts', type=int, default=50, help='Number of surface verts per epitope residue')

    # isMEANOpt related
    parser.add_argument('--seq_warmup', type=int, default=0, help='Number of epochs before starting training sequence')

    # task setting
    parser.add_argument('--pep_seq', action='store_true', help='use pep sequence')
    parser.add_argument('--pep_struct', action='store_true', help='use pep structure')
    parser.add_argument('--struct_only', action='store_true', help='Predict complex structure given the sequence')
    parser.add_argument('--bind_dist_cutoff', type=float, default=6.6, help='distance cutoff to decide the binding interface')

    # ablation
    parser.add_argument('--no_pred_edge_dist', action='store_true', help='Turn off edge distance prediction at the interface')
    parser.add_argument('--backbone_only', action='store_true', help='Model backbone only')
    parser.add_argument('--fix_channel_weights', action='store_true', help='Fix channel weights, may also for special use (e.g. antigen with modified AAs)')
    parser.add_argument('--no_memory', action='store_true', help='No memory passing')

    return parser.parse_args()


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
        print_log(args)
        print_log(f'Run dir: {args.save_dir}')
        print_log(f'CDR type: {args.cdr}')
        print_log(f'Paratope: {args.paratope}')
        print_log('structure only' if args.struct_only else 'sequence & structure codesign')
        print_log('ABFLOW env: ' + json.dumps(_collect_env(prefixes=["ABFLOW_"]), ensure_ascii=False))

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
                   pep_seq=args.pep_seq,
                   pep_struct=args.pep_struct,
                   struct_only=args.struct_only,
                   iter_round=args.iter_round,
                   backbone_only=args.backbone_only,
                   fix_channel_weights=args.fix_channel_weights,
                   pred_edge_dist=not args.no_pred_edge_dist,
                   keep_memory=not args.no_memory,
                   cdr_type=args.cdr, paratope=args.paratope)
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

    # ------------------------------------------------------------
    # V140 batch contract: JSON batch_size is PER-GPU micro-batch.
    # This matches the existing ABX trainer and MFDesign/Lightning data-module
    # convention.  DDP combines one micro-batch from every rank, therefore:
    #
    #   effective_global_batch = local_batch * world_size
    #
    # No hidden division by world_size is allowed here.
    # ------------------------------------------------------------
    local_batch_size = max(1, int(args.batch_size))
    effective_global_batch_size = local_batch_size * max(1, int(world_size))
    step_per_epoch = (
        len(train_set) + effective_global_batch_size - 1
    ) // effective_global_batch_size

    config.batch_size = local_batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)
    config.add_parameter(batch_size_per_gpu=local_batch_size)
    config.add_parameter(global_batch_size=effective_global_batch_size)
    config.add_parameter(batch_semantics='per_gpu')

    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, shuffle=args.shuffle
        )
    else:
        train_sampler = None

    # DataLoaders below consume args.batch_size, so keep it equal to the
    # explicit per-rank micro-batch instead of mutating it after TrainConfig.
    args.batch_size = local_batch_size
    config.local_rank = args.local_rank

    if _is_main_rank(args.local_rank):
        print_log(
            '[BatchContract] semantics=per_gpu '
            f'local_batch={local_batch_size} world_size={world_size} '
            f'effective_global_batch={effective_global_batch_size} '
            f'step_per_epoch={step_per_epoch}'
        )
        print_log(f'Batch size on each GPU: {local_batch_size}')
        print_log(f'Effective global batch size: {effective_global_batch_size}')
        print_log(f'step per epoch: {step_per_epoch}')
        print_log(f'world_size: {world_size}, rank: {rank}, local_rank: {args.local_rank}')

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
            f"DataLoader workers: train={args.num_workers}, "
            f"valid={args.valid_num_workers}, "
            f"train_prefetch={args.prefetch_factor}, "
            f"valid_prefetch={args.valid_prefetch_factor}, "
            f"valid_persistent_workers={args.valid_persistent_workers}"
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
    if _is_main_rank(args.local_rank):
        print_log(f'Actual trainer run dir: {actual_run_dir}')
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
