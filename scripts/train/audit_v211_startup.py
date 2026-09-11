#!/usr/bin/env python3
"""Cross-run startup gate for the V211 R28/R29/R30 family.

The selected JSON files remain the sole source of run paths. This audit reads
their current run_time.log symlinks and verifies the causal contracts that can
be decided after two optimizer steps, before an expensive full epoch finishes.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


FINGERPRINT = re.compile(
    r"\[V211SharedInitFingerprint\]\s+sha256=([0-9a-f]{64})"
)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def fail(message: str) -> None:
    raise SystemExit(f"[V211StartupAuditFAIL] {message}")


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / value.removeprefix("./")


def fields(line: str) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in re.findall(rf"([A-Za-z0-9_]+)=({NUMBER})", line)
    }


def required_line(text: str, marker: str, label: str) -> str:
    matches = [line for line in text.splitlines() if marker in line]
    if not matches:
        fail(f"{label} has not emitted {marker}; wait for step 1 or inspect the run")
    return matches[-1]


def load_run(root: Path, config_path: Path) -> tuple[str, str, Path]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    meta = cfg.get("_experiment", {})
    exp_id = meta.get("exp_id", "")
    if exp_id.startswith("R28_"):
        label = "R28"
    elif exp_id.startswith("R29_"):
        label = "R29"
    elif exp_id.startswith("R30_"):
        label = "R30"
    else:
        fail(f"unrecognized experiment id in {config_path}: {exp_id!r}")
    formal = meta.get("formal_runtime", {})
    log_name = formal.get("log_filename")
    save_dir = cfg.get("save_dir")
    if not isinstance(log_name, str) or not isinstance(save_dir, str):
        fail(f"{label} JSON lacks save_dir/formal_runtime.log_filename")
    log_path = resolve(root, save_dir) / log_name
    if not log_path.is_file():
        fail(f"{label} log not found: {log_path}")
    return label, log_path.read_text(encoding="utf-8", errors="replace"), log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs=3, required=True)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    args = parser.parse_args()
    root = args.project_root.resolve()

    runs = {}
    paths = {}
    for config in args.configs:
        config_path = resolve(root, str(config)).resolve()
        if not config_path.is_file():
            fail(f"config not found: {config_path}")
        label, run_text, log_path = load_run(root, config_path)
        if label in runs:
            fail(f"duplicate {label} config")
        runs[label] = run_text
        paths[label] = log_path
    if set(runs) != {"R28", "R29", "R30"}:
        fail(f"expected R28/R29/R30, found {sorted(runs)}")

    hashes = {}
    parent_components = {}
    for label, run_text in runs.items():
        match = FINGERPRINT.search(run_text)
        if match is None:
            fail(f"{label} shared-initialization fingerprint missing")
        hashes[label] = match.group(1)
        required_line(run_text, "[V211BridgeColdStartPASS]", label)
        required_line(run_text, "[V211BridgeLivePASS]", label)
        step0 = next(
            (line for line in run_text.splitlines()
             if "[R05AbXStep]" in line and "step=0 " in line),
            None,
        )
        if step0 is None:
            fail(f"{label} step-0 parent components missing")
        values = fields(step0)
        missing = {"seq", "struct", "interface", "edge"} - set(values)
        if missing:
            fail(f"{label} step-0 fields missing: {sorted(missing)}")
        parent_components[label] = {
            key: values[key] for key in ("seq", "struct", "interface", "edge")
        }

    if len(set(hashes.values())) != 1:
        fail(f"shared parameter hashes differ: {hashes}")
    baseline = parent_components["R28"]
    for label in ("R29", "R30"):
        for key, reference in baseline.items():
            observed = parent_components[label][key]
            if not math.isclose(
                observed, reference, rel_tol=args.rtol, abs_tol=args.atol
            ):
                fail(
                    f"step-0 parent mismatch {label}.{key}: "
                    f"{observed} vs R28 {reference}"
                )

    required_line(runs["R29"], "[DistogramColdStartContract]", "R29")
    required_line(runs["R29"], "[LivePairGradientContract]", "R29")
    disto = fields(required_line(runs["R29"], "[DistoAudit]", "R29"))
    for key in ("scope_design_anchored", "objective_pairs", "donor_pairs"):
        if key not in disto:
            fail(f"R29 DistoAudit missing {key}")
    if disto["scope_design_anchored"] != 1.0:
        fail("R29 is not optimizing the design-anchored mask")
    if not 0.0 < disto["objective_pairs"] < disto["donor_pairs"]:
        fail(f"R29 task pair support is invalid: {disto}")

    required_line(
        runs["R30"], "[SmoothLDDTBridgeColdStartContract]", "R30"
    )
    required_line(
        runs["R30"], "[LiveSmoothLDDTPairGradientContract]", "R30"
    )
    smooth = fields(required_line(runs["R30"], "[SmoothLDDTAudit]", "R30"))
    restored = smooth.get("fixed_context_restored_rate")
    if restored is None or not 0.0 < restored <= 1.0:
        fail(f"R30 fixed-context restoration is invalid: {restored}")

    print(
        "[V211StartupAuditPASS] "
        f"shared_sha256={next(iter(hashes.values()))} "
        f"parent_step0={baseline} bridge_cold_live=PASS "
        f"R29_task_pairs={int(disto['objective_pairs'])}/"
        f"{int(disto['donor_pairs'])} R29_dLdz=LIVE "
        f"R30_fixed_context_restored={restored:.4f} R30_dLdz=LIVE"
    )
    for label in ("R28", "R29", "R30"):
        print(f"[V211StartupAuditLog] {label}={paths[label]}")


if __name__ == "__main__":
    main()
