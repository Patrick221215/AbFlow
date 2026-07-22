#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact watcher/evaluator for AbFlow topk_map.txt.

Purpose:
  - accept --watch / --poll-interval / --max-new, so the training launcher can start AutoTopK safely;
  - evaluate checkpoints by reusing run_state_consistent_ablation.sh test;
  - write only mean metrics and gaps versus the validated PCS_RC_LC_R1 baseline;
  - never use test metrics to decide training checkpoint saving. This script is an internal diagnostic/evaluation utility.
"""
from __future__ import annotations

import argparse, csv, datetime as dt, json, os, queue, re, shlex, subprocess, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CKPT_RE = re.compile(r"(?P<ckpt>(?:/[^\s'\"]+)?epoch(?P<epoch>\d+)_step(?P<step>\d+)\.ckpt)")
PREFIX_FLOAT_RE = re.compile(r"^\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*:")
METRIC_NAMES = [
    "AAR H3", "CAAR H3", "AAR", "CAAR", "RMSD(CA) aligned",
    "RMSD(CA) CDRH3", "RMSD(CA) CDRH3 aligned", "TMscore", "LDDT", "DockQ",
]
METRIC_RE = re.compile(r"^(?P<name>" + "|".join(re.escape(x) for x in METRIC_NAMES) + r"):\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")
DOCKQ_PROP_RE = re.compile(r"proportion of DockQ above 0\.23:\s*(?P<p23>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*0\.49:\s*(?P<p49>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s*0\.8:\s*(?P<p80>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

# Validated PCS_RC_LC_R1 reference line from the current project notes.
R1 = {
    "AAR_mean": 0.4234,
    "CAAR_mean": 0.2824,
    "RMSDCA_aligned_mean": 1.1144296254662673,
    "RMSDCA_CDRH3_mean": 8.25,
    "RMSDCA_CDRH3_aligned_mean": 1.832687902128891,
    "TMscore_mean": 0.9736,
    "LDDT_mean": 0.8522,
    "DockQ_mean": 0.423,
    "DockQ_above_0.23": 0.9333333333333333,
    "DockQ_above_0.49": 0.2,
}
LOWER_BETTER = {"RMSDCA_aligned_mean", "RMSDCA_CDRH3_mean", "RMSDCA_CDRH3_aligned_mean"}
CORE_FIELDS = [
    "AAR_mean", "CAAR_mean", "RMSDCA_aligned_mean", "RMSDCA_CDRH3_mean",
    "RMSDCA_CDRH3_aligned_mean", "TMscore_mean", "LDDT_mean", "DockQ_mean",
    "DockQ_above_0.23", "DockQ_above_0.49", "DockQ_above_0.8",
]


def metric_key(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def add_gaps(row: Dict[str, str]) -> None:
    vals = {}
    for k, base in R1.items():
        v = to_float(row.get(k))
        if v is None:
            continue
        gap = (base - v) if k in LOWER_BETTER else (v - base)
        row[f"gap_vs_R1_{k.replace('_mean','')}"] = f"{gap:.6g}"
        vals[k] = v
    required = ["DockQ_mean", "RMSDCA_CDRH3_mean", "CAAR_mean", "AAR_mean", "LDDT_mean"]
    if all(k in vals for k in required):
        score = 100.0 * (
            0.35 * ((vals["DockQ_mean"] - R1["DockQ_mean"]) / R1["DockQ_mean"])
            + 0.25 * ((R1["RMSDCA_CDRH3_mean"] - vals["RMSDCA_CDRH3_mean"]) / R1["RMSDCA_CDRH3_mean"])
            + 0.20 * ((vals["CAAR_mean"] - R1["CAAR_mean"]) / R1["CAAR_mean"])
            + 0.10 * ((vals["AAR_mean"] - R1["AAR_mean"]) / R1["AAR_mean"])
            + 0.10 * ((vals["LDDT_mean"] - R1["LDDT_mean"]) / R1["LDDT_mean"])
        )
        row["diagnostic_score_vs_R1"] = f"{score:.6g}"


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
    candidates = list(root.glob("version_*/checkpoint/topk_map.txt")) + list(root.glob("checkpoint/topk_map.txt"))
    candidates = [p for p in candidates if p.exists()]
    return sorted(candidates, key=version_sort_key, reverse=True)[0] if candidates else None


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
    for line_no, line in enumerate(topk_map.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        m = CKPT_RE.search(line)
        if not m:
            continue
        ckpt = resolve_ckpt(m.group("ckpt"), topk_dir)
        key = os.path.realpath(ckpt)
        if key in seen:
            continue
        seen.add(key)
        pm = PREFIX_FLOAT_RE.search(line)
        entries.append(TopKEntry(
            ckpt=ckpt,
            epoch=int(m.group("epoch")),
            step=int(m.group("step")),
            line_no=line_no,
            rank=len(entries) + 1,
            valid_loss=(pm.group("value") if pm else ""),
            raw_line=line.strip(),
        ))
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


def read_done(csv_path: Path, state_path: Path) -> set:
    done = set()
    if csv_path.exists():
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("status") == "ok" and row.get("checkpoint_realpath"):
                        done.add(row["checkpoint_realpath"])
        except Exception:
            pass
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            done |= {k for k, v in data.get("done", {}).items() if v == "ok"}
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
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def append_csv(path: Path, row: Dict[str, str], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp", "status", "exp_id", "epoch", "step", "topk_rank", "valid_loss",
        "checkpoint", "checkpoint_realpath", "result_dir", "log_file", "returncode", "elapsed_sec",
        "raw_topk_line",
    ] + CORE_FIELDS + [
        "gap_vs_R1_AAR", "gap_vs_R1_CAAR", "gap_vs_R1_RMSDCA_aligned",
        "gap_vs_R1_RMSDCA_CDRH3", "gap_vs_R1_RMSDCA_CDRH3_aligned",
        "gap_vs_R1_TMscore", "gap_vs_R1_LDDT", "gap_vs_R1_DockQ",
        "gap_vs_R1_DockQ_above_0.23", "gap_vs_R1_DockQ_above_0.49",
        "diagnostic_score_vs_R1",
    ]
    with lock:
        old_rows = []
        old_fields = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                old_fields = reader.fieldnames or []
                old_rows = list(reader)
        all_fields = list(dict.fromkeys(fields + old_fields + list(row.keys())))
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields)
            w.writeheader()
            for r in old_rows:
                w.writerow({k: r.get(k, "") for k in all_fields})
            w.writerow({k: row.get(k, "") for k in all_fields})
        os.replace(tmp, path)


def write_ranked(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("status") == "ok"]
        fields = reader.fieldnames or []
    def key(r):
        return (
            to_float(r.get("diagnostic_score_vs_R1")) or -1e9,
            to_float(r.get("DockQ_mean")) or -1e9,
            -(to_float(r.get("RMSDCA_CDRH3_mean")) or 1e9),
            to_float(r.get("CAAR_mean")) or -1e9,
        )
    rows = sorted(rows, key=key, reverse=True)
    ranked = csv_path.with_name("topk_eval_metrics_compact_ranked.csv")
    with ranked.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    best = rows[0] if rows else {}
    csv_path.with_name("topk_eval_metrics_compact_best.json").write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_one(entry: TopKEntry, gpu: str, args, lock: threading.Lock) -> Dict[str, str]:
    start = time.time()
    ckpt_real = os.path.realpath(entry.ckpt)
    result_dir = Path(args.result_root) / args.exp_id / entry.short_name
    log_dir = Path(args.log_dir) if args.log_dir else Path(args.topk_map).parent / "eval_logs_compact" / args.exp_id
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
        row.update({"status": "missing_ckpt", "returncode": "", "elapsed_sec": "0"})
        append_csv(Path(args.csv), row, lock)
        update_state(Path(args.state_json), ckpt_real, "missing_ckpt")
        return row
    cmd = ["bash", args.launcher, "test", args.exp_id, str(gpu), entry.ckpt, str(result_dir), args.test_json]
    env = os.environ.copy()
    if args.batch_size is not None:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.n_steps is not None:
        env["N_STEPS"] = str(args.n_steps)
    if args.show_sample_progress is not None:
        env["SHOW_SAMPLE_PROGRESS"] = "1" if args.show_sample_progress else "0"
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("[TopK evaluator command]\n" + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=args.project_root, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    row["returncode"] = str(proc.returncode)
    row["elapsed_sec"] = f"{elapsed:.1f}"
    row.update(parse_metrics(log_file.read_text(encoding="utf-8", errors="replace")))
    row["status"] = "ok" if proc.returncode == 0 and row.get("DockQ_mean") else "failed"
    if row["status"] == "ok":
        add_gaps(row)
    append_csv(Path(args.csv), row, lock)
    update_state(Path(args.state_json), ckpt_real, row["status"])
    return row


def worker(gpu: str, q: "queue.Queue[TopKEntry]", args, lock: threading.Lock) -> None:
    while True:
        try:
            entry = q.get_nowait()
        except queue.Empty:
            return
        try:
            print(f"[TopK evaluator] GPU {gpu}: {entry.short_name}", flush=True)
            row = evaluate_one(entry, gpu, args, lock)
            print(f"[TopK evaluator] GPU {gpu}: {entry.short_name} status={row.get('status')} DockQ={row.get('DockQ_mean','')} H3={row.get('RMSDCA_CDRH3_mean','')} score={row.get('diagnostic_score_vs_R1','')}", flush=True)
        except Exception as exc:
            print(f"[TopK evaluator] ERROR GPU {gpu} {entry.ckpt}: {exc}", flush=True)
        finally:
            q.task_done()


def run_once(args) -> int:
    inferred = Path(args.topk_map) if args.topk_map else infer_topk_map(args.run_dir)
    if inferred is None or not inferred.exists():
        print(f"[TopK evaluator] topk_map not ready under {args.run_dir or ''}", flush=True)
        return 0
    args.topk_map = str(inferred)
    topk_dir = inferred.resolve().parent
    if args.result_root is None:
        args.result_root = str(topk_dir / "eval_results_compact")
    if args.csv is None:
        args.csv = str(topk_dir / "topk_eval_metrics_compact.csv")
    if args.state_json is None:
        args.state_json = str(topk_dir / "topk_eval_state_compact.json")
    entries = parse_topk_map(inferred)
    if args.latest_only and entries:
        entries = [max(entries, key=lambda e: (e.epoch, e.step))]
    done = set() if args.force else read_done(Path(args.csv), Path(args.state_json))
    new_entries = [e for e in entries if e.ckpt_key not in done]
    if args.max_new and args.max_new > 0:
        new_entries = new_entries[:args.max_new]
    print(f"[TopK evaluator] topk={inferred} parsed={len(entries)} new={len(new_entries)} csv={args.csv}", flush=True)
    if not new_entries:
        write_ranked(Path(args.csv))
        return 0
    q: "queue.Queue[TopKEntry]" = queue.Queue()
    for e in new_entries:
        q.put(e)
    gpus = [g.strip() for g in str(args.gpu_ids).split(',') if g.strip()] or ["0"]
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(gpu, q, args, lock), daemon=True) for gpu in gpus]
    for t in threads: t.start()
    for t in threads: t.join()
    write_ranked(Path(args.csv))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compact AbFlow topk_map evaluator with watch mode.")
    p.add_argument("--exp-id", required=True)
    p.add_argument("--test-json", required=True)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--topk-map", default=None)
    p.add_argument("--project-root", default="/home/data3/cjm/project/AbFlow")
    p.add_argument("--launcher", default="scripts/train/run_state_consistent_ablation.sh")
    p.add_argument("--gpu-ids", default="0")
    p.add_argument("--limit", type=int, default=None, help="Accepted for compatibility; currently unused.")
    p.add_argument("--result-root", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--state-json", default=None)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--show-sample-progress", type=int, choices=[0, 1], default=None)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-interval", type=float, default=300.0)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--latest-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    while True:
        rc = run_once(args)
        if not args.watch:
            return rc
        time.sleep(args.poll_interval)

if __name__ == "__main__":
    raise SystemExit(main())
