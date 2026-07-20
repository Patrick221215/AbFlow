#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Goal-focused top-k checkpoint evaluator for AbFlow.

What this script does:
  1. Reads the current checkpoint/topk_map.txt.
  2. Evaluates every checkpoint currently listed there, usually top-5.
  3. Writes a compact CSV with only mean metrics and gap-to-target columns.
  4. Does not record lowest/highest/std PDB details, because those make the
     table hard to read and do not answer the current model-selection question.

Important principle:
  - This script can evaluate all top-k checkpoints for internal diagnosis.
  - For formal reporting, do not choose the test-best checkpoint unless you have
    a separate validation metric/protocol. The validation-loss top-1 remains the
    cleanest predeclared selection rule.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CKPT_RE = re.compile(r"(?P<ckpt>(?:/[^\s'\"]+)?epoch(?P<epoch>\d+)_step(?P<step>\d+)\.ckpt)")
PREFIX_FLOAT_RE = re.compile(r"^\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*:")
NAMED_VALUE_RE = re.compile(
    r"(?P<name>valid[_-]?loss|val[_-]?loss|loss|metric|score)\s*[:=]\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

METRIC_NAMES = [
    "AAR H3", "CAAR H3", "AAR", "CAAR",
    "RMSD(CA) aligned", "RMSD(CA) CDRH3", "RMSD(CA) CDRH3 aligned",
    "TMscore", "LDDT", "DockQ",
]
METRIC_RE = re.compile(
    r"^(?P<name>" + "|".join(re.escape(x) for x in METRIC_NAMES) +
    r"):\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
DOCKQ_PROP_RE = re.compile(
    r"proportion of DockQ above 0\.23:\s*(?P<p23>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
    r"0\.49:\s*(?P<p49>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*"
    r"0\.8:\s*(?P<p80>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)

# Current strong baseline target from PCS_RC_LC_R1, used only for compact gap reporting.
# Positive gap means better than or equal to the target direction.
R1_TARGETS = {
    "AAR_mean": 0.39952416993051654,
    "CAAR_mean": 0.263315665812141,
    "RMSD_CA_aligned_mean": 1.1144296254662673,
    "RMSD_CA_CDRH3_mean": 8.080138237240584,
    "RMSD_CA_CDRH3_aligned_mean": 1.832687902128891,
    "TMscore_mean": 0.9701333333333332,
    "LDDT_mean": 0.8285816666666667,
    "DockQ_mean": 0.3944333333333333,
    "DockQ_above_0.23": 0.9333333333333333,
    "DockQ_above_0.49": 0.2,
}
LOWER_IS_BETTER = {
    "RMSD_CA_aligned_mean",
    "RMSD_CA_CDRH3_mean",
    "RMSD_CA_CDRH3_aligned_mean",
}

KEYMAP = {
    "AAR H3": "AAR_H3_mean",
    "CAAR H3": "CAAR_H3_mean",
    "AAR": "AAR_mean",
    "CAAR": "CAAR_mean",
    "RMSD(CA) aligned": "RMSD_CA_aligned_mean",
    "RMSD(CA) CDRH3": "RMSD_CA_CDRH3_mean",
    "RMSD(CA) CDRH3 aligned": "RMSD_CA_CDRH3_aligned_mean",
    "TMscore": "TMscore_mean",
    "LDDT": "LDDT_mean",
    "DockQ": "DockQ_mean",
}

@dataclass
class Entry:
    rank: int
    line_no: int
    raw_line: str
    ckpt: str
    epoch: int
    step: int
    valid_metric_name: str
    valid_metric_value: str

    @property
    def short_name(self) -> str:
        return f"epoch{self.epoch}_step{self.step}"


def infer_topk_map(run_dir: Optional[str], topk_map: Optional[str]) -> Path:
    if topk_map:
        p = Path(topk_map)
        if not p.exists():
            raise FileNotFoundError(f"topk_map not found: {p}")
        return p
    if not run_dir:
        raise ValueError("Provide --topk-map or --run-dir")
    base = Path(run_dir)
    candidates = sorted(base.glob("version_*/checkpoint/topk_map.txt"))
    if not candidates:
        direct = base / "checkpoint" / "topk_map.txt"
        if direct.exists():
            return direct
        raise FileNotFoundError(f"No topk_map.txt under {base}")
    # Use newest version directory by mtime.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_valid_metric(line: str) -> Tuple[str, str]:
    # The standard topk_map format is usually: "4.87924: /path/epoch...ckpt".
    m = PREFIX_FLOAT_RE.search(line)
    if m:
        return "valid_loss", m.group("value")
    m = NAMED_VALUE_RE.search(line)
    if m:
        return m.group("name"), m.group("value")
    return "", ""


def parse_entries(topk_map: Path, limit: Optional[int] = None) -> List[Entry]:
    lines = topk_map.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: List[Entry] = []
    seen = set()
    for line_no, line in enumerate(lines, start=1):
        m = CKPT_RE.search(line)
        if not m:
            continue
        ckpt = m.group("ckpt")
        if not ckpt.startswith("/"):
            ckpt = str((topk_map.parent / ckpt).resolve())
        ckpt_real = os.path.realpath(ckpt)
        if ckpt_real in seen:
            continue
        seen.add(ckpt_real)
        name, value = parse_valid_metric(line)
        entries.append(Entry(
            rank=len(entries) + 1,
            line_no=line_no,
            raw_line=line.strip(),
            ckpt=ckpt,
            epoch=int(m.group("epoch")),
            step=int(m.group("step")),
            valid_metric_name=name,
            valid_metric_value=value,
        ))
        if limit is not None and len(entries) >= limit:
            break
    return entries


def parse_metrics(text: str) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        m = METRIC_RE.match(s)
        if m:
            metrics[KEYMAP[m.group("name")]] = m.group("value")
            continue
        dm = DOCKQ_PROP_RE.search(s)
        if dm:
            metrics["DockQ_above_0.23"] = dm.group("p23")
            metrics["DockQ_above_0.49"] = dm.group("p49")
            metrics["DockQ_above_0.8"] = dm.group("p80")
    return metrics


def add_gaps(row: Dict[str, str]) -> None:
    gap_sum = 0.0
    gap_count = 0
    for key, target in R1_TARGETS.items():
        raw = row.get(key, "")
        try:
            value = float(raw)
        except Exception:
            row[f"gap_vs_R1_{key}"] = ""
            continue
        gap = (target - value) if key in LOWER_IS_BETTER else (value - target)
        row[f"gap_vs_R1_{key}"] = f"{gap:.6g}"
        gap_sum += gap
        gap_count += 1
    row["mean_gap_vs_R1"] = f"{gap_sum / gap_count:.6g}" if gap_count else ""


def run_eval(entry: Entry, gpu: str, args: argparse.Namespace) -> Dict[str, str]:
    start = time.time()
    launcher = args.launcher if os.path.isabs(args.launcher) else str(Path(args.project_root) / args.launcher)
    result_dir = Path(args.result_root) / args.exp_id / entry.short_name
    log_dir = Path(args.log_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{entry.short_name}.log"

    row: Dict[str, str] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "started",
        "exp_id": args.exp_id,
        "topk_rank_in_map": str(entry.rank),
        "epoch": str(entry.epoch),
        "step": str(entry.step),
        "valid_metric_name": entry.valid_metric_name,
        "valid_metric_value": entry.valid_metric_value,
        "checkpoint": entry.ckpt,
        "raw_topk_line": entry.raw_line,
    }

    if not Path(entry.ckpt).exists():
        row.update({"status": "missing_ckpt", "elapsed_sec": "0", "returncode": ""})
        add_gaps(row)
        return row

    cmd = ["bash", launcher, "test", args.exp_id, str(gpu), entry.ckpt, str(result_dir), args.test_json]
    env = os.environ.copy()
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.n_steps is not None:
        env["N_STEPS"] = str(args.n_steps)
    if args.show_sample_progress is not None:
        env["SHOW_SAMPLE_PROGRESS"] = "1" if args.show_sample_progress else "0"

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("Command: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=args.project_root, env=env, stdout=lf, stderr=subprocess.STDOUT)

    text = log_file.read_text(encoding="utf-8", errors="replace")
    row.update(parse_metrics(text))
    row.update({
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": str(proc.returncode),
        "elapsed_sec": f"{time.time() - start:.1f}",
        "result_dir": str(result_dir),
        "log_file": str(log_file),
    })
    add_gaps(row)
    return row


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "timestamp", "status", "exp_id", "topk_rank_in_map", "epoch", "step",
        "valid_metric_name", "valid_metric_value",
        "AAR_mean", "gap_vs_R1_AAR_mean",
        "CAAR_mean", "gap_vs_R1_CAAR_mean",
        "RMSD_CA_CDRH3_mean", "gap_vs_R1_RMSD_CA_CDRH3_mean",
        "RMSD_CA_CDRH3_aligned_mean", "gap_vs_R1_RMSD_CA_CDRH3_aligned_mean",
        "RMSD_CA_aligned_mean", "gap_vs_R1_RMSD_CA_aligned_mean",
        "TMscore_mean", "gap_vs_R1_TMscore_mean",
        "LDDT_mean", "gap_vs_R1_LDDT_mean",
        "DockQ_mean", "gap_vs_R1_DockQ_mean",
        "DockQ_above_0.23", "gap_vs_R1_DockQ_above_0.23",
        "DockQ_above_0.49", "gap_vs_R1_DockQ_above_0.49",
        "mean_gap_vs_R1",
        "checkpoint", "result_dir", "log_file", "returncode", "elapsed_sec", "raw_topk_line",
    ]
    fields = list(dict.fromkeys(preferred + [k for r in rows for k in r.keys()]))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def worker(gpu: str, q: "queue.Queue[Entry]", args: argparse.Namespace, rows: List[Dict[str, str]], lock: threading.Lock) -> None:
    while True:
        try:
            entry = q.get_nowait()
        except queue.Empty:
            return
        try:
            row = run_eval(entry, gpu, args)
        except Exception as e:
            row = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "status": "exception",
                "exp_id": args.exp_id,
                "topk_rank_in_map": str(entry.rank),
                "epoch": str(entry.epoch),
                "step": str(entry.step),
                "valid_metric_name": entry.valid_metric_name,
                "valid_metric_value": entry.valid_metric_value,
                "checkpoint": entry.ckpt,
                "raw_topk_line": entry.raw_line,
                "error": repr(e),
            }
            add_gaps(row)
        with lock:
            rows.append(row)
        q.task_done()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate current AbFlow top-k checkpoints and write compact goal-focused metrics.")
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--run-dir", default=None, help="Experiment run dir, e.g. results/<EXP_ID>")
    ap.add_argument("--topk-map", default=None)
    ap.add_argument("--project-root", default="/home/data3/cjm/project/AbFlow")
    ap.add_argument("--launcher", default="scripts/train/run_state_consistent_ablation.sh")
    ap.add_argument("--gpu-ids", default="0")
    ap.add_argument("--limit", type=int, default=5, help="How many current top-k entries to evaluate; default 5")
    ap.add_argument("--result-root", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--show-sample-progress", type=int, choices=[0, 1], default=0)
    args = ap.parse_args(argv)

    topk_map = infer_topk_map(args.run_dir, args.topk_map)
    entries = parse_entries(topk_map, args.limit)
    if not entries:
        raise SystemExit(f"No checkpoints found in {topk_map}")

    topk_dir = topk_map.parent
    if args.result_root is None:
        args.result_root = str(topk_dir / "eval_results_compact")
    if args.csv is None:
        args.csv = str(topk_dir / "topk_eval_metrics_compact.csv")
    if args.log_dir is None:
        args.log_dir = str(topk_dir / "eval_logs_compact" / args.exp_id)

    q: "queue.Queue[Entry]" = queue.Queue()
    for e in entries:
        q.put(e)
    rows: List[Dict[str, str]] = []
    lock = threading.Lock()
    gpus = [x.strip() for x in args.gpu_ids.split(",") if x.strip()] or ["0"]
    threads = [threading.Thread(target=worker, args=(gpu, q, args, rows, lock), daemon=True) for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Deterministic ordering: topk rank in map first.
    rows.sort(key=lambda r: int(r.get("topk_rank_in_map", "999999") or 999999))
    write_csv(Path(args.csv), rows)
    print(f"Wrote compact metrics: {args.csv}")
    print("Formal selection rule: use topk_rank_in_map=1 unless you have a predeclared validation metric beyond validation loss.")
    print("Internal diagnosis: compare the gap_vs_R1_* columns across the current top-k rows.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
