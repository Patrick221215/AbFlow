#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact watcher/evaluator for AbFlow TopK checkpoints with TensorBoard test logging.

Scientific policy:
- checkpoint saving/TopK membership is still controlled ONLY by validation loss;
- test metrics are diagnostics only and are never fed back into training;
- every successful test checkpoint is appended to CSV and additionally written
  into the same version_N TensorBoard run under the `Test/` namespace;
- test curves use checkpoint `epoch` as TensorBoard global_step by default;
- one persistent SummaryWriter is reused for the lifetime of the evaluator
  process, avoiding one new events.out file per evaluated checkpoint;
- training and evaluator remain separate processes, so the evaluator does NOT
  append to the training process's already-open physical event file. TensorBoard
  merges event data found under the same run directory.

This file is intentionally compatible with the existing launcher CLI used by
run_foldflow_r3_v59.sh.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import queue
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CKPT_RE = re.compile(
    r"(?P<ckpt>(?:/[^\s'\"]+)?epoch(?P<epoch>\d+)_step(?P<step>\d+)\.ckpt)"
)
PREFIX_FLOAT_RE = re.compile(
    r"^\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*:"
)

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
    r"^(?P<name>"
    + "|".join(re.escape(x) for x in METRIC_NAMES)
    + r"):\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
DOCKQ_PROP_RE = re.compile(
    r"proportion of DockQ above 0\.23:\s*"
    r"(?P<p23>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
    r"0\.49:\s*(?P<p49>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
    r"0\.8:\s*(?P<p80>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

ABFLOW_PAPER = {
    "AAR_mean": 0.4234,
    "CAAR_mean": 0.2824,
    "RMSDCA_CDRH3_mean": 8.25,
    "TMscore_mean": 0.9736,
    "LDDT_mean": 0.8522,
    "DockQ_mean": 0.423,
}
LOWER_BETTER = {"RMSDCA_CDRH3_mean"}
CORE_LOCATOR_FIELDS = [
    "AAR_mean", "CAAR_mean", "RMSDCA_CDRH3_mean", "DockQ_mean"
]
CORE_FIELDS = [
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
]
REQUIRED_SUCCESS_FIELDS = set(CORE_FIELDS)

# TensorBoard names deliberately do NOT reuse Loss/*.
# They are test-set diagnostics and must remain visually separated.
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

INTERNAL_BASE_EXP_ID = "PCS_RC_LC_R1_BASE"


def _env_on(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def metric_key(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def to_finite_float(x):
    value = to_float(x)
    if value is None or not math.isfinite(value):
        return None
    return value


def _core_string(row: Dict[str, str]) -> str:
    vals = [to_float(row.get(k)) for k in CORE_LOCATOR_FIELDS]
    if any(v is None for v in vals):
        return ""
    return (
        f"AAR={vals[0]:.4f} | CAAR={vals[1]:.4f} | "
        f"H3raw={vals[2]:.3f}A | DockQ={vals[3]:.4f}"
    )


def add_paper_gaps(row: Dict[str, str]) -> None:
    for k, base in ABFLOW_PAPER.items():
        v = to_float(row.get(k))
        if v is None:
            continue
        gap = (base - v) if k in LOWER_BETTER else (v - base)
        row[f"gap_vs_abflow_paper_{k.replace('_mean','')}"] = f"{gap:.6g}"
    row["core_metrics"] = _core_string(row)


def _find_internal_base_best(run_dir: Optional[str]) -> Optional[Path]:
    if not run_dir:
        return None
    run_dir = Path(run_dir).resolve()
    root = run_dir.parent
    candidates = list(
        (root / INTERNAL_BASE_EXP_ID).glob(
            "version_*/checkpoint/topk_eval_metrics_compact_best.json"
        )
    )
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_internal_base(run_dir: Optional[str], explicit: Optional[str] = None):
    path = Path(explicit).resolve() if explicit else _find_internal_base_best(run_dir)
    if path is None or not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    vals = {}
    for k in CORE_LOCATOR_FIELDS:
        v = to_float(data.get(k))
        if v is None:
            return None, None
        vals[k] = v
    return vals, str(path)


def add_internal_base_delta(
    row: Dict[str, str],
    base_vals: Optional[Dict[str, float]],
    base_path: Optional[str],
) -> None:
    if not base_vals:
        row["internal_base_status"] = "not_ready"
        return
    row["internal_base_status"] = "matched_base_loaded"
    row["internal_base_reference"] = base_path or ""
    for k, base in base_vals.items():
        v = to_float(row.get(k))
        if v is None:
            continue
        delta = (base - v) if k in LOWER_BETTER else (v - base)
        row[f"delta_vs_internal_base_{k.replace('_mean','')}"] = f"{delta:.6g}"


def add_references(row: Dict[str, str], args) -> None:
    add_paper_gaps(row)
    if row.get("exp_id") == INTERNAL_BASE_EXP_ID:
        row["internal_base_status"] = "this_run_is_internal_base"
        return
    base_vals, base_path = _load_internal_base(
        args.run_dir, getattr(args, "internal_base_json", None)
    )
    add_internal_base_delta(row, base_vals, base_path)


@dataclass(frozen=True)
class TopKEntry:
    ckpt: str
    epoch: int
    step: int
    line_no: int
    rank: int
    valid_loss: str = ""
    raw_line: str = ""

    @property
    def ckpt_key(self) -> str:
        return os.path.realpath(self.ckpt)

    @property
    def short_name(self) -> str:
        return f"epoch{self.epoch}_step{self.step}"


def version_sort_key(path: Path) -> Tuple[int, float]:
    m = re.search(r"version_(\d+)", str(path))
    ver = int(m.group(1)) if m else -1
    try:
        mt = path.stat().st_mtime
    except OSError:
        mt = 0.0
    return ver, mt


def infer_topk_map(run_dir: Optional[str]) -> Optional[Path]:
    if not run_dir:
        return None
    root = Path(run_dir)
    candidates = list(root.glob("version_*/checkpoint/topk_map.txt"))
    candidates += list(root.glob("checkpoint/topk_map.txt"))
    candidates = [p for p in candidates if p.exists()]
    return (
        sorted(candidates, key=version_sort_key, reverse=True)[0]
        if candidates else None
    )


def resolve_ckpt(raw: str, topk_dir: Path) -> str:
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    for base in [topk_dir, topk_dir.parent, topk_dir.parent.parent]:
        cand = base / raw
        if cand.exists():
            return str(cand)
    return str(topk_dir / raw)


def parse_topk_map(topk_map: Path) -> List[TopKEntry]:
    topk_dir = topk_map.parent
    entries, seen = [], set()
    for line_no, line in enumerate(
        topk_map.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines(), 1
    ):
        m = CKPT_RE.search(line)
        if not m:
            continue
        ckpt = resolve_ckpt(m.group("ckpt"), topk_dir)
        key = os.path.realpath(ckpt)
        if key in seen:
            continue
        seen.add(key)
        pm = PREFIX_FLOAT_RE.search(line)
        entries.append(
            TopKEntry(
                ckpt=ckpt,
                epoch=int(m.group("epoch")),
                step=int(m.group("step")),
                line_no=line_no,
                rank=len(entries) + 1,
                valid_loss=(pm.group("value") if pm else ""),
                raw_line=line.strip(),
            )
        )
    return entries


def parse_metrics(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        s = line.strip()
        m = METRIC_RE.match(s)
        if m:
            out[f"{metric_key(m.group('name'))}_mean"] = m.group("value")
            continue
        dm = DOCKQ_PROP_RE.search(s)
        if dm:
            out["DockQ_above_0.23"] = dm.group("p23")
            out["DockQ_above_0.49"] = dm.group("p49")
            out["DockQ_above_0.8"] = dm.group("p80")
    return out


def read_done(
    csv_path: Path,
    state_path: Path,
    *,
    retry_failed: bool = False,
) -> set:
    done = set()
    terminal = {"ok"} if retry_failed else {"ok", "failed"}
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (
                        row.get("status") in terminal
                        and row.get("checkpoint_realpath")
                    ):
                        done.add(row["checkpoint_realpath"])
        except Exception:
            pass
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            done |= {
                k for k, v in data.get("done", {}).items()
                if v in terminal
            }
        except Exception:
            pass
    return done


def update_state(path: Path, key: str, status: str) -> None:
    data = {"done": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {"done": {}}
    data.setdefault("done", {})[key] = status
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def append_csv(path: Path, row: Dict[str, str], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "status", "exp_id", "epoch", "step", "topk_rank",
        "valid_loss", "checkpoint", "checkpoint_realpath", "result_dir",
        "log_file", "returncode", "elapsed_sec", "raw_topk_line",
    ] + CORE_FIELDS + [
        "core_metrics",
        "internal_base_status",
        "internal_base_reference",
        "gap_vs_abflow_paper_AAR",
        "gap_vs_abflow_paper_CAAR",
        "gap_vs_abflow_paper_RMSDCA_CDRH3",
        "gap_vs_abflow_paper_DockQ",
        "delta_vs_internal_base_AAR",
        "delta_vs_internal_base_CAAR",
        "delta_vs_internal_base_RMSDCA_CDRH3",
        "delta_vs_internal_base_DockQ",
    ]
    with lock:
        old_rows, old_fields = [], []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames or []
                old_rows = list(reader)
        all_fields = list(
            dict.fromkeys(fields + old_fields + list(row.keys()))
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields)
            w.writeheader()
            for r in old_rows:
                w.writerow({k: r.get(k, "") for k in all_fields})
            w.writerow({k: row.get(k, "") for k in all_fields})
        os.replace(tmp, path)


def write_ranked(csv_path: Path) -> None:
    """Diagnostic ranking only.  Never use this ranking to save/train ckpts."""
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("status") == "ok"]
        fields = reader.fieldnames or []

    def key(r):
        def value(name: str, default: float) -> float:
            parsed = to_finite_float(r.get(name))
            return default if parsed is None else parsed
        return (
            value("DockQ_mean", -1e9),
            -value("RMSDCA_CDRH3_mean", 1e9),
            value("CAAR_mean", -1e9),
            value("AAR_mean", -1e9),
        )

    rows = sorted(rows, key=key, reverse=True)
    ranked = csv_path.with_name("topk_eval_metrics_compact_ranked.csv")
    with ranked.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    best = rows[0] if rows else {}
    csv_path.with_name(
        "topk_eval_metrics_compact_best.json"
    ).write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# TensorBoard test-metric logging
# --------------------------------------------------------------------------

def _tb_state_path(topk_map: Path, axis: str = "epoch") -> Path:
    axis = str(axis).strip().lower()
    return (
        topk_map.resolve().parent
        / f"test_tensorboard_state_{axis}_single_writer_v2.json"
    )


def _load_tb_done(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("logged", []))
    except Exception:
        return set()


def _save_tb_done(path: Path, done: set) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logged": sorted(done),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "policy": "diagnostic_only; never used for checkpoint saving",
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _tb_log_dir(args) -> Path:
    if getattr(args, "tensorboard_dir", None):
        return Path(args.tensorboard_dir).resolve()
    # .../version_N/checkpoint/topk_map.txt -> .../version_N
    return Path(args.topk_map).resolve().parent.parent


_TB_WRITERS: Dict[str, object] = {}


def _tb_axis(args) -> str:
    cli = str(getattr(args, "tensorboard_x_axis", "") or "").strip().lower()
    env = os.environ.get("ABFLOW_TEST_TENSORBOARD_X_AXIS", "").strip().lower()
    axis = cli or env or "epoch"
    if axis not in {"epoch", "step"}:
        raise ValueError(
            "TensorBoard x-axis must be 'epoch' or 'step', "
            f"got {axis!r}."
        )
    return axis


def _tb_global_step(row: Dict[str, str], args) -> int:
    key = "epoch" if _tb_axis(args) == "epoch" else "step"
    return int(float(row.get(key) or 0))


def _get_test_summary_writer(args):
    """One persistent test SummaryWriter per evaluator process and log_dir."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(
            "[TopK evaluator] TensorBoard unavailable; "
            f"skip Test/* scalars without affecting evaluation: {exc}",
            flush=True,
        )
        return None

    log_dir = _tb_log_dir(args)
    log_dir.mkdir(parents=True, exist_ok=True)
    axis = _tb_axis(args)
    key = f"{log_dir}|{axis}"
    writer = _TB_WRITERS.get(key)
    if writer is None:
        writer = SummaryWriter(
            log_dir=str(log_dir),
            filename_suffix=f".test_{axis}_metrics",
        )
        _TB_WRITERS[key] = writer
    return writer


def _close_test_summary_writers():
    for writer in list(_TB_WRITERS.values()):
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
    _TB_WRITERS.clear()


def write_test_tensorboard(
    row: Dict[str, str],
    args,
    lock: Optional[threading.Lock] = None,
) -> bool:
    """Write one successful test row to TensorBoard.

    Default x-axis is checkpoint epoch.  One persistent evaluator writer is
    reused across checkpoints in the same process.
    """
    if not _env_on("ABFLOW_TEST_TENSORBOARD", True):
        return False
    if row.get("status") != "ok":
        return False

    ckpt_key = (
        row.get("checkpoint_realpath")
        or os.path.realpath(row.get("checkpoint", ""))
    )
    if not ckpt_key:
        return False

    axis = _tb_axis(args)
    tb_state = _tb_state_path(Path(args.topk_map), axis=axis)
    guard = lock if lock is not None else threading.Lock()

    with guard:
        done = _load_tb_done(tb_state)
        if ckpt_key in done:
            return False

        x = _tb_global_step(row, args)
        writer = _get_test_summary_writer(args)
        if writer is None:
            return False

        for field, tag in TB_METRICS.items():
            value = to_float(row.get(field))
            if value is not None:
                writer.add_scalar(tag, value, x)

        valid_loss = to_float(row.get("valid_loss"))
        if valid_loss is not None:
            writer.add_scalar("Test/valid_loss_at_checkpoint", valid_loss, x)

        epoch = to_float(row.get("epoch"))
        step = to_float(row.get("step"))
        if epoch is not None:
            writer.add_scalar("TestMeta/checkpoint_epoch", epoch, x)
        if step is not None:
            writer.add_scalar("TestMeta/checkpoint_train_step", step, x)

        writer.add_scalar("TestReference/AbFlowPaper_AAR", 0.4234, x)
        writer.add_scalar("TestReference/AbFlowPaper_CAAR", 0.2824, x)
        writer.add_scalar("TestReference/AbFlowPaper_H3_raw_RMSD", 8.25, x)
        writer.add_scalar("TestReference/AbFlowPaper_DockQ", 0.423, x)
        writer.flush()

        done.add(ckpt_key)
        _save_tb_done(tb_state, done)
    return True


def backfill_tensorboard_from_csv(args) -> int:
    """Backfill already-evaluated rows once, without re-running test inference."""
    if not _env_on("ABFLOW_TEST_TENSORBOARD", True):
        return 0

    csv_path = Path(args.csv) if args.csv else None
    if csv_path is None or not csv_path.exists():
        return 0

    count = 0
    lock = threading.Lock()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            if write_test_tensorboard(row, args, lock):
                count += 1
    if count:
        print(
            f"[TopK evaluator] TensorBoard backfill: wrote {count} "
            f"checkpoint(s) into {_tb_log_dir(args)}",
            flush=True,
        )
    return count


def evaluate_one(
    entry: TopKEntry,
    gpu: str,
    args,
    lock: threading.Lock,
) -> Dict[str, str]:
    start = time.time()
    ckpt_real = os.path.realpath(entry.ckpt)
    result_dir = Path(args.result_root) / args.exp_id / entry.short_name
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else Path(args.topk_map).parent
        / "eval_logs_compact"
        / args.exp_id
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{entry.short_name}.log"

    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "exp_id": args.exp_id,
        "epoch": str(entry.epoch),
        "step": str(entry.step),
        "topk_rank": str(entry.rank),
        "valid_loss": entry.valid_loss,
        "checkpoint": entry.ckpt,
        "checkpoint_realpath": ckpt_real,
        "result_dir": str(result_dir),
        "log_file": str(log_file),
        "raw_topk_line": entry.raw_line,
    }

    if not Path(entry.ckpt).exists():
        row.update(
            {"status": "missing_ckpt", "returncode": "", "elapsed_sec": "0"}
        )
        append_csv(Path(args.csv), row, lock)
        update_state(Path(args.state_json), ckpt_real, "missing_ckpt")
        return row

    if args.launcher:
        # Compatibility path for historical train launchers whose test action
        # accepts: test EXP GPU CKPT RESULT_DIR TEST_JSON.
        cmd = [
            "bash",
            args.launcher,
            "test",
            args.exp_id,
            str(gpu),
            entry.ckpt,
            str(result_dir),
            args.test_json,
        ]
    else:
        # Formal path: the same shared epoch-test engine used in Trainer.
        cmd = [
            "bash",
            args.test_script,
            entry.ckpt,
            args.test_json,
            str(result_dir),
            "rabd",
        ]
        if args.surf_file:
            cmd.append(args.surf_file)
    env = os.environ.copy()
    env["GPU"] = str(gpu)
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.n_steps is not None:
        env["N_STEPS"] = str(args.n_steps)
    if args.show_sample_progress is not None:
        env["SHOW_SAMPLE_PROGRESS"] = (
            "1" if args.show_sample_progress else "0"
        )
    if args.pep_file:
        env["ABFLOW_EPOCH_TEST_PEP"] = str(args.pep_file)
    if args.surf_file:
        env["ABFLOW_EPOCH_TEST_SURF"] = str(args.surf_file)
    if args.eval_cdr:
        env["ABFLOW_EPOCH_TEST_CDR"] = str(args.eval_cdr)
    if args.metric_workers is not None:
        env["ABFLOW_EPOCH_TEST_METRIC_WORKERS"] = str(args.metric_workers)

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write(
            "[TopK evaluator command]\n"
            + " ".join(shlex.quote(x) for x in cmd)
            + "\n\n"
        )
        lf.flush()
        proc = subprocess.run(
            cmd,
            cwd=args.project_root,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.time() - start
    row["returncode"] = str(proc.returncode)
    row["elapsed_sec"] = f"{elapsed:.1f}"
    row.update(
        parse_metrics(
            log_file.read_text(encoding="utf-8", errors="replace")
        )
    )
    metric_fields_ok = all(
        to_finite_float(row.get(name)) is not None
        for name in REQUIRED_SUCCESS_FIELDS
    )
    row["status"] = "ok" if proc.returncode == 0 and metric_fields_ok else "failed"
    if row["status"] != "ok":
        missing = sorted(
            name for name in REQUIRED_SUCCESS_FIELDS
            if to_finite_float(row.get(name)) is None
        )
        row["error"] = (
            f"returncode={proc.returncode}; missing_or_non_numeric="
            + ",".join(missing)
        )

    if row["status"] == "ok":
        add_references(row, args)

    append_csv(Path(args.csv), row, lock)
    update_state(Path(args.state_json), ckpt_real, row["status"])

    # Diagnostic only; does not modify training, optimizer or TopK map.
    if row["status"] == "ok":
        write_test_tensorboard(row, args, lock)

    return row


def worker(
    gpu: str,
    q: "queue.Queue[TopKEntry]",
    args,
    lock: threading.Lock,
) -> None:
    while True:
        try:
            entry = q.get_nowait()
        except queue.Empty:
            return
        try:
            print(
                f"[TopK evaluator] GPU {gpu}: {entry.short_name}",
                flush=True,
            )
            row = evaluate_one(entry, gpu, args, lock)
            print(
                f"[TopK evaluator] GPU {gpu}: {entry.short_name} "
                f"status={row.get('status')} "
                f"{row.get('core_metrics','')}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[TopK evaluator] ERROR GPU {gpu} "
                f"{entry.ckpt}: {exc}",
                flush=True,
            )
        finally:
            q.task_done()


def _resolve_output_paths(args, inferred: Path) -> None:
    args.topk_map = str(inferred)
    topk_dir = inferred.resolve().parent
    if args.result_root is None:
        args.result_root = str(topk_dir / "eval_results_compact")
    if args.csv is None:
        args.csv = str(topk_dir / "topk_eval_metrics_compact.csv")
    if args.state_json is None:
        args.state_json = str(topk_dir / "topk_eval_state_compact.json")


def run_once(args) -> int:
    inferred = (
        Path(args.topk_map)
        if args.topk_map
        else infer_topk_map(args.run_dir)
    )
    if inferred is None or not inferred.exists():
        print(
            f"[TopK evaluator] topk_map not ready under "
            f"{args.run_dir or ''}",
            flush=True,
        )
        return 0

    _resolve_output_paths(args, inferred)

    # Existing successful evaluations are automatically copied into TensorBoard
    # once.  No test generation is re-run.
    backfill_tensorboard_from_csv(args)

    if args.sync_tensorboard_only:
        return 0

    entries = parse_topk_map(inferred)
    if args.limit is not None and args.limit > 0:
        entries = entries[: args.limit]
    if args.latest_only and entries:
        entries = [max(entries, key=lambda e: (e.epoch, e.step))]

    done = (
        set()
        if args.force
        else read_done(
            Path(args.csv),
            Path(args.state_json),
            retry_failed=args.retry_failed,
        )
    )
    new_entries = [e for e in entries if e.ckpt_key not in done]
    if args.max_new and args.max_new > 0:
        new_entries = new_entries[: args.max_new]

    print(
        f"[TopK evaluator] topk={inferred} parsed={len(entries)} "
        f"new={len(new_entries)} csv={args.csv}",
        flush=True,
    )

    if not new_entries:
        write_ranked(Path(args.csv))
        return 0

    q: "queue.Queue[TopKEntry]" = queue.Queue()
    for e in new_entries:
        q.put(e)

    gpus = [
        g.strip()
        for g in str(args.gpu_ids).split(",")
        if g.strip()
    ] or ["0"]
    lock = threading.Lock()
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, q, args, lock),
            daemon=True,
        )
        for gpu in gpus
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_ranked(Path(args.csv))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Compact AbFlow topk_map evaluator with watch mode "
            "and TensorBoard test metrics."
        )
    )
    p.add_argument("--exp-id", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--run-dir", default=None)
    p.add_argument(
        "--internal-base-json",
        default=None,
        help=(
            "Optional matched BASE best.json; otherwise infer sibling "
            "PCS_RC_LC_R1_BASE."
        ),
    )
    p.add_argument("--topk-map", default=None)
    p.add_argument(
        "--project-root",
        default="/home/data3/cjm/project/AbFlow",
    )
    p.add_argument(
        "--launcher",
        default=None,
        help=(
            "Historical train launcher with a 'test' action. Omit to use the "
            "formal shared scripts/test/test_epoch_ddp.sh path."
        ),
    )
    p.add_argument(
        "--test-script",
        default="scripts/test/test_epoch_ddp.sh",
        help="Formal shared Test entry used when --launcher is omitted.",
    )
    p.add_argument("--gpu-ids", default="0")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most the first N entries from topk_map.",
    )
    p.add_argument("--result-root", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--state-json", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--metric-workers", type=int, default=None)
    p.add_argument("--pep-file", default=None)
    p.add_argument("--surf-file", default=None)
    p.add_argument("--eval-cdr", default=None)
    p.add_argument(
        "--show-sample-progress",
        type=int,
        choices=[0, 1],
        default=None,
    )
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-interval", type=float, default=300.0)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--latest-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry checkpoints recorded as failed; off by default in watch mode.",
    )
    p.add_argument(
        "--tensorboard-dir",
        default=None,
        help=(
            "Optional TensorBoard directory. Default: version_N directory "
            "containing checkpoint/."
        ),
    )
    p.add_argument(
        "--tensorboard-x-axis",
        choices=["epoch", "step"],
        default="epoch",
        help=(
            "TensorBoard x-axis for Test/* scalars. Default: epoch; "
            "use step only for legacy compatibility."
        ),
    )
    p.add_argument(
        "--sync-tensorboard-only",
        action="store_true",
        help=(
            "Write already-evaluated rows from topk_eval_metrics_compact.csv "
            "to TensorBoard and exit; do not run test inference."
        ),
    )
    args = p.parse_args(argv)

    while True:
        rc = run_once(args)
        if not args.watch or args.sync_tensorboard_only:
            return rc
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        _close_test_summary_writers()
