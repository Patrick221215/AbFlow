#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate AbFlow checkpoints listed in topk_map.txt and write metric CSV/leaderboard next to topk_map.txt.

This script is intentionally conservative:
  - It does not change model/training code.
  - It treats topk_map.txt as the checkpoint-selection signal from training.
  - It parallelizes evaluation across checkpoints when multiple GPU ids are provided.
  - Each checkpoint is still evaluated by the existing launcher/test.sh/generate.py path, so metrics stay identical to manual test.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CKPT_RE = re.compile(r"(?P<ckpt>(?:/[^\s'\"]+)?epoch(?P<epoch>\d+)_step(?P<step>\d+)\.ckpt)")
LOSS_RE = re.compile(
    r"(?P<name>train[_-]?loss|valid[_-]?loss|val[_-]?loss|loss|score|metric|map)\s*[:=]\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
ANY_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

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
METRIC_RE = re.compile(r"^(?P<name>" + "|".join(re.escape(x) for x in METRIC_NAMES) + r"):\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")
STD_RE = re.compile(r"^\s*Standard deviation:\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
LOW_RE = re.compile(r"^\s*lowest:\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*pdb:\s*(?P<pdb>\S+)")
HIGH_RE = re.compile(r"^\s*highest:\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*pdb:\s*(?P<pdb>\S+)")
DOCKQ_PROP_RE = re.compile(
    r"proportion of DockQ above 0\.23:\s*(?P<p23>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*0\.49:\s*(?P<p49>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*0\.8:\s*(?P<p80>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


ABFLOW_TARGETS = {
    "AAR_mean": 0.4234,
    "CAAR_mean": 0.2824,
    "RMSDCA_CDRH3_mean": 8.25,
    "TMscore_mean": 0.9736,
    "LDDT_mean": 0.8522,
    "DockQ_mean": 0.4230,
}


def _float_or_none(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def add_target_gaps(row: Dict[str, str]) -> None:
    """Add transparent gaps to AbFlow-paper target lines. Positive is good."""
    for key, target in ABFLOW_TARGETS.items():
        val = _float_or_none(row.get(key))
        if val is None:
            continue
        short = key.replace("_mean", "")
        if key == "RMSDCA_CDRH3_mean":
            # RMSD is lower-is-better, so target - value is positive when we beat target.
            gap = target - val
        else:
            gap = val - target
        row[f"gap_vs_abflow_{short}"] = f"{gap:.6g}"

    aar = _float_or_none(row.get("AAR_mean"))
    caar = _float_or_none(row.get("CAAR_mean"))
    h3 = _float_or_none(row.get("RMSDCA_CDRH3_mean"))
    lddt = _float_or_none(row.get("LDDT_mean"))
    dockq = _float_or_none(row.get("DockQ_mean"))
    dockq49 = _float_or_none(row.get("DockQ_above_0.49"))
    if None not in (aar, caar, h3, lddt, dockq):
        # A readable checkpoint-selection score, not a training objective.
        # Positive means the checkpoint is closer to or above the AbFlow-paper line.
        score = 100.0 * (
            0.35 * ((dockq - ABFLOW_TARGETS["DockQ_mean"]) / ABFLOW_TARGETS["DockQ_mean"])
            + 0.25 * ((ABFLOW_TARGETS["RMSDCA_CDRH3_mean"] - h3) / ABFLOW_TARGETS["RMSDCA_CDRH3_mean"])
            + 0.20 * ((caar - ABFLOW_TARGETS["CAAR_mean"]) / ABFLOW_TARGETS["CAAR_mean"])
            + 0.10 * ((aar - ABFLOW_TARGETS["AAR_mean"]) / ABFLOW_TARGETS["AAR_mean"])
            + 0.10 * ((lddt - ABFLOW_TARGETS["LDDT_mean"]) / ABFLOW_TARGETS["LDDT_mean"])
        )
        row["selection_score_vs_abflow"] = f"{score:.6g}"
    if dockq49 is not None:
        row["DockQ_above_0.49_meaning"] = "medium_quality_fraction"


def write_ranked_outputs(csv_path: Path) -> None:
    """Write ranked CSV and a best JSON next to the raw incremental CSV."""
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    for r in ok_rows:
        add_target_gaps(r)
    def score_key(r):
        s = _float_or_none(r.get("selection_score_vs_abflow"))
        dockq = _float_or_none(r.get("DockQ_mean"))
        h3 = _float_or_none(r.get("RMSDCA_CDRH3_mean"))
        caar = _float_or_none(r.get("CAAR_mean"))
        return (s if s is not None else -1e9, dockq if dockq is not None else -1e9, -(h3 if h3 is not None else 1e9), caar if caar is not None else -1e9)
    ranked = sorted(ok_rows, key=score_key, reverse=True)
    extra_fields = [
        "selection_score_vs_abflow",
        "gap_vs_abflow_AAR",
        "gap_vs_abflow_CAAR",
        "gap_vs_abflow_RMSDCA_CDRH3",
        "gap_vs_abflow_TMscore",
        "gap_vs_abflow_LDDT",
        "gap_vs_abflow_DockQ",
        "DockQ_above_0.49_meaning",
    ]
    out_fields = list(dict.fromkeys(fields + extra_fields))
    ranked_path = csv_path.with_name("topk_eval_ranked.csv")
    with ranked_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in ranked:
            writer.writerow({k: r.get(k, "") for k in out_fields})
    best_path = csv_path.with_name("topk_eval_best.json")
    best = ranked[0] if ranked else {}
    best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metric_key(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )


@dataclass(frozen=True)
class TopKEntry:
    ckpt: str
    epoch: int
    step: int
    line_no: int
    rank: int
    topk_metric_name: str = ""
    topk_metric_value: str = ""
    raw_line: str = ""

    @property
    def ckpt_key(self) -> str:
        return os.path.realpath(self.ckpt)

    @property
    def short_name(self) -> str:
        return f"epoch{self.epoch}_step{self.step}"


def _infer_ckpt_path(raw_ckpt: str, topk_dir: Path) -> str:
    p = Path(raw_ckpt)
    if p.is_absolute():
        return str(p)
    # topk_map.txt normally sits in the checkpoint directory.  A relative
    # filename such as epoch64_step3380.ckpt therefore resolves next to it.
    cand = topk_dir / raw_ckpt
    if cand.exists():
        return str(cand)
    # Some topk_map lines may include a relative path from the experiment root.
    # Try parent directories before giving up; the final non-existing path is
    # still returned so the caller can report a clear missing checkpoint error.
    for base in [topk_dir.parent, topk_dir.parent.parent]:
        cand = base / raw_ckpt
        if cand.exists():
            return str(cand)
    return str(topk_dir / raw_ckpt)




def _version_sort_key(path: Path) -> Tuple[int, float]:
    m = re.search(r"version_(\d+)", str(path))
    ver = int(m.group(1)) if m else -1
    try:
        mt = path.stat().st_mtime
    except OSError:
        mt = 0.0
    return ver, mt


def infer_topk_map_from_run_dir(run_dir: Path) -> Optional[Path]:
    """Infer newest version_*/checkpoint/topk_map.txt under one experiment run dir."""
    candidates = list(run_dir.glob("version_*/checkpoint/topk_map.txt"))
    candidates += list(run_dir.glob("checkpoint/topk_map.txt"))
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=_version_sort_key, reverse=True)[0]


def wait_or_infer_topk_map(args: argparse.Namespace) -> Optional[Path]:
    if args.topk_map:
        p = Path(args.topk_map)
        return p if p.exists() else None
    if not args.run_dir:
        return None
    p = infer_topk_map_from_run_dir(Path(args.run_dir))
    return p

def parse_topk_map(topk_map: Path) -> List[TopKEntry]:
    topk_dir = topk_map.parent
    entries: List[TopKEntry] = []
    seen = set()
    if not topk_map.exists():
        return entries
    lines = topk_map.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_no, line in enumerate(lines, start=1):
        m = CKPT_RE.search(line)
        if not m:
            continue
        ckpt = _infer_ckpt_path(m.group("ckpt"), topk_dir)
        epoch, step = int(m.group("epoch")), int(m.group("step"))
        key = os.path.realpath(ckpt)
        if key in seen:
            continue
        seen.add(key)
        lm = LOSS_RE.search(line)
        loss_name, loss_value = "", ""
        if lm:
            loss_name, loss_value = lm.group("name"), lm.group("value")
        else:
            # Keep a weak fallback for unknown topk_map formats.  We do not use
            # this number for ranking; it is only recorded for traceability.
            floats = ANY_FLOAT_RE.findall(line)
            # Remove epoch/step if they are the only numbers; otherwise keep the last.
            if len(floats) >= 3:
                loss_name, loss_value = "line_last_float", floats[-1]
        entries.append(
            TopKEntry(
                ckpt=ckpt,
                epoch=epoch,
                step=step,
                line_no=line_no,
                rank=len(entries) + 1,
                topk_metric_name=loss_name,
                topk_metric_value=loss_value,
                raw_line=line.strip(),
            )
        )
    return entries


def parse_metrics_from_log(text: str) -> Dict[str, str]:
    rows = text.splitlines()
    metrics: Dict[str, str] = {}
    i = 0
    while i < len(rows):
        line = rows[i].strip()
        m = METRIC_RE.match(line)
        if not m:
            dm = DOCKQ_PROP_RE.search(line)
            if dm:
                metrics["DockQ_above_0.23"] = dm.group("p23")
                metrics["DockQ_above_0.49"] = dm.group("p49")
                metrics["DockQ_above_0.8"] = dm.group("p80")
            i += 1
            continue

        base = metric_key(m.group("name"))
        metrics[f"{base}_mean"] = m.group("value")
        # Parse the following statistic lines until the next metric or unrelated block.
        j = i + 1
        while j < len(rows):
            nxt = rows[j].strip()
            if METRIC_RE.match(nxt):
                break
            sm = STD_RE.match(nxt)
            if sm:
                metrics[f"{base}_std"] = sm.group("value")
                j += 1
                continue
            lm = LOW_RE.match(nxt)
            if lm:
                metrics[f"{base}_lowest"] = lm.group("value")
                metrics[f"{base}_lowest_pdb"] = lm.group("pdb")
                j += 1
                continue
            hm = HIGH_RE.match(nxt)
            if hm:
                metrics[f"{base}_highest"] = hm.group("value")
                metrics[f"{base}_highest_pdb"] = hm.group("pdb")
                j += 1
                continue
            dm = DOCKQ_PROP_RE.search(nxt)
            if dm:
                metrics["DockQ_above_0.23"] = dm.group("p23")
                metrics["DockQ_above_0.49"] = dm.group("p49")
                metrics["DockQ_above_0.8"] = dm.group("p80")
                j += 1
                continue
            # The metric block may include blank lines; skip them.  Otherwise stop.
            if nxt == "":
                j += 1
                continue
            if nxt.startswith("Done ") or nxt.startswith("Script:"):
                break
            # Unknown line in this block; move on but do not get stuck.
            j += 1
        i = j
    return metrics


def read_done_keys(csv_path: Path, state_path: Path) -> set:
    done = set()
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("checkpoint_realpath") and row.get("status") == "ok":
                        done.add(row["checkpoint_realpath"])
        except Exception:
            pass
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            for k, v in data.get("done", {}).items():
                if v == "ok":
                    done.add(k)
        except Exception:
            pass
    return done


def update_state(state_path: Path, ckpt_key: str, status: str) -> None:
    state = {"done": {}}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {"done": {}}
    state.setdefault("done", {})[ckpt_key] = status
    state["updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, state_path)


def append_csv(csv_path: Path, row: Dict[str, str], lock: threading.Lock) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        existing_fields: List[str] = []
        old_rows: List[Dict[str, str]] = []
        if csv_path.exists():
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_fields = reader.fieldnames or []
                old_rows = list(reader)
        preferred = [
            "timestamp",
            "status",
            "exp_id",
            "epoch",
            "step",
            "topk_rank",
            "topk_line_no",
            "topk_metric_name",
            "topk_metric_value",
            "checkpoint",
            "checkpoint_realpath",
            "result_dir",
            "log_file",
            "returncode",
            "elapsed_sec",
            "raw_topk_line",
        ]
        fields = list(dict.fromkeys(preferred + existing_fields + list(row.keys())))
        tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for old in old_rows:
                writer.writerow({k: old.get(k, "") for k in fields})
            writer.writerow({k: row.get(k, "") for k in fields})
        os.replace(tmp, csv_path)


def run_one(entry: TopKEntry, gpu: str, args: argparse.Namespace, csv_lock: threading.Lock) -> Dict[str, str]:
    start = time.time()
    ckpt_real = os.path.realpath(entry.ckpt)
    result_dir = Path(args.result_root) / args.exp_id / entry.short_name
    log_dir = Path(args.log_dir) if args.log_dir else Path(args.topk_map).parent / "eval_logs" / args.exp_id
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{entry.short_name}.log"

    row: Dict[str, str] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "exp_id": args.exp_id,
        "epoch": str(entry.epoch),
        "step": str(entry.step),
        "topk_rank": str(entry.rank),
        "topk_line_no": str(entry.line_no),
        "topk_metric_name": entry.topk_metric_name,
        "topk_metric_value": entry.topk_metric_value,
        "checkpoint": entry.ckpt,
        "checkpoint_realpath": ckpt_real,
        "result_dir": str(result_dir),
        "log_file": str(log_file),
        "raw_topk_line": entry.raw_line,
    }

    if not Path(entry.ckpt).exists():
        row.update({"status": "missing_ckpt", "returncode": "", "elapsed_sec": "0"})
        append_csv(Path(args.csv), row, csv_lock)
        update_state(Path(args.state_json), ckpt_real, "missing_ckpt")
        return row

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
    env = os.environ.copy()
    # Let callers tune test.sh without editing this script.
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.n_steps is not None:
        env["N_STEPS"] = str(args.n_steps)
    if args.show_sample_progress is not None:
        env["SHOW_SAMPLE_PROGRESS"] = "1" if args.show_sample_progress else "0"

    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("[TopK evaluator command]\n")
        lf.write(" ".join(shlex.quote(x) for x in cmd) + "\n\n")
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
    log_text = log_file.read_text(encoding="utf-8", errors="replace")
    row.update(parse_metrics_from_log(log_text))
    row["status"] = "ok" if proc.returncode == 0 and "DockQ_mean" in row else "failed"
    if row["status"] == "ok":
        add_target_gaps(row)
    append_csv(Path(args.csv), row, csv_lock)
    update_state(Path(args.state_json), ckpt_real, row["status"])
    return row


def worker_loop(gpu: str, q: "queue.Queue[TopKEntry]", args: argparse.Namespace, csv_lock: threading.Lock) -> None:
    while True:
        try:
            entry = q.get_nowait()
        except queue.Empty:
            return
        try:
            print(f"[TopK evaluator] GPU {gpu}: evaluating {entry.short_name} -> {entry.ckpt}", flush=True)
            row = run_one(entry, gpu, args, csv_lock)
            print(
                f"[TopK evaluator] GPU {gpu}: {entry.short_name} status={row.get('status')} "
                f"DockQ={row.get('DockQ_mean','')} AAR={row.get('AAR_mean','')} CAAR={row.get('CAAR_mean','')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[TopK evaluator] GPU {gpu}: ERROR for {entry.ckpt}: {exc}", file=sys.stderr, flush=True)
        finally:
            q.task_done()


def run_once(args: argparse.Namespace) -> int:
    inferred = wait_or_infer_topk_map(args)
    if inferred is None:
        print(f"[TopK evaluator] waiting for topk_map.txt under run_dir={args.run_dir or ''}", flush=True)
        return 0
    if str(inferred) != str(args.topk_map):
        # A new version directory appeared.  Redirect outputs next to the actual topk_map.
        args.topk_map = str(inferred)
        topk_dir = inferred.resolve().parent
        args.result_root = str(topk_dir / "eval_results")
        args.csv = str(topk_dir / "topk_eval_metrics.csv")
        args.state_json = str(topk_dir / "topk_eval_state.json")
        if args.log_dir is None:
            # keep default tied to current topk directory
            pass
    topk_map = Path(args.topk_map)
    entries = parse_topk_map(topk_map)
    if args.latest_only and entries:
        # topk_map is usually ordered by ranking or current map; keep the newest by step.
        entries = [max(entries, key=lambda x: (x.epoch, x.step))]
    if args.max_new is not None and args.max_new > 0:
        # Preserve topk_map order, but cap newly launched jobs.
        pass

    done = read_done_keys(Path(args.csv), Path(args.state_json)) if not args.force else set()
    new_entries = [e for e in entries if e.ckpt_key not in done]
    if args.max_new is not None and args.max_new > 0:
        new_entries = new_entries[: args.max_new]

    print(f"[TopK evaluator] parsed={len(entries)} new={len(new_entries)} csv={args.csv}", flush=True)
    if not new_entries:
        write_ranked_outputs(Path(args.csv))
        return 0

    gpus = [g.strip() for g in str(args.gpu_ids).split(",") if g.strip()]
    if not gpus:
        gpus = ["0"]
    q: "queue.Queue[TopKEntry]" = queue.Queue()
    for e in new_entries:
        q.put(e)
    csv_lock = threading.Lock()
    threads = []
    for gpu in gpus:
        t = threading.Thread(target=worker_loop, args=(gpu, q, args, csv_lock), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    write_ranked_outputs(Path(args.csv))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints from AbFlow topk_map.txt and write a CSV.")
    parser.add_argument("--exp-id", required=True, help="Experiment id passed to run_state_consistent_ablation.sh test")
    parser.add_argument("--topk-map", default=None, help="Path to checkpoint/topk_map.txt. If omitted, infer from --run-dir/version_*/checkpoint/topk_map.txt")
    parser.add_argument("--run-dir", default=None, help="Experiment run dir such as results_dtm/<EXP_ID>; used to infer topk_map.txt")
    parser.add_argument("--test-json", required=True, help="Path to RAbD/DiffAb test json")
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated GPU ids. Checkpoints are distributed across these GPUs.")
    parser.add_argument("--project-root", default="/home/data3/cjm/project/AbFlow", help="AbFlow project root")
    parser.add_argument("--launcher", default="scripts/train/run_state_consistent_ablation.sh", help="Launcher script path relative to project root or absolute")
    parser.add_argument("--result-root", default=None, help="Directory for generated PDB/results; default: <topk_dir>/eval_results")
    parser.add_argument("--csv", default=None, help="Metric CSV path; default: <topk_dir>/topk_eval_metrics.csv")
    parser.add_argument("--state-json", default=None, help="State json path; default: <topk_dir>/topk_eval_state.json")
    parser.add_argument("--log-dir", default=None, help="Log directory; default: <topk_dir>/eval_logs/<exp_id>")
    parser.add_argument("--watch", action="store_true", help="Keep polling topk_map for new checkpoints")
    parser.add_argument("--poll-interval", type=float, default=120.0, help="Polling interval in seconds for --watch")
    parser.add_argument("--max-new", type=int, default=None, help="At most evaluate this many new checkpoints per scan")
    parser.add_argument("--latest-only", action="store_true", help="Only evaluate the newest epoch/step entry from topk_map")
    parser.add_argument("--force", action="store_true", help="Re-evaluate checkpoints even if present in CSV/state")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional BATCH_SIZE override for scripts/test/test.sh")
    parser.add_argument("--n-steps", type=int, default=None, help="Optional N_STEPS override for scripts/test/test.sh")
    parser.add_argument("--show-sample-progress", type=int, choices=[0, 1], default=None, help="Optional SHOW_SAMPLE_PROGRESS override")
    args = parser.parse_args(argv)

    inferred_topk = wait_or_infer_topk_map(args)
    if inferred_topk is not None:
        args.topk_map = str(inferred_topk)
        topk_dir = inferred_topk.resolve().parent
    else:
        # In watch mode the training process may not have created version_*/checkpoint/topk_map.txt yet.
        # Use run_dir/checkpoint as a temporary default for log/state paths; run_once will re-infer on every poll.
        if not args.watch:
            raise SystemExit("topk_map.txt was not found. Provide --topk-map or --run-dir with an existing version_*/checkpoint/topk_map.txt")
        base = Path(args.run_dir) if args.run_dir else Path(args.project_root)
        topk_dir = base / "checkpoint"
    if args.result_root is None:
        args.result_root = str(topk_dir / "eval_results")
    if args.csv is None:
        args.csv = str(topk_dir / "topk_eval_metrics.csv")
    if args.state_json is None:
        args.state_json = str(topk_dir / "topk_eval_state.json")

    # Normalize launcher for subprocess cwd=args.project_root.
    if not os.path.isabs(args.launcher):
        args.launcher = args.launcher

    while True:
        rc = run_once(args)
        if not args.watch:
            return rc
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())
