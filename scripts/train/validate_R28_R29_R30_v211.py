#!/usr/bin/env python3
"""Fail-fast preflight for the V211 R28/R29/R30 experiment family."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


R05_PARENT = "R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
R28_ID = "R28_R05_ABX_NATIVE_PAIR_EGNN_U02"
PROTOCOL = "V211_EVIDENCE_DRIVEN_R28_PARENT"

EXPERIMENTS = {
    R28_ID: dict(
        parent=R05_PARENT, disto="off", lddt="off",
        disto_scope="all_resolved",
    ),
    "R29_R28_ABX_NATIVE_PAIR_EGNN_DISTOGRAM_U02": dict(
        parent=R28_ID, disto="on", lddt="off",
        disto_scope="design_anchored",
    ),
    "R30_R28_ABX_NATIVE_PAIR_EGNN_MF_DONOR_SMOOTH_LDDT_U02": dict(
        parent=R28_ID, disto="off", lddt="on",
        disto_scope="all_resolved",
    ),
}

# These are frozen scientific identities, not run routing values. GPU, port,
# batch size, seeds and every loss coefficient are intentionally absent: their
# sole authority is the selected JSON and this validator checks only schema,
# safety and cross-field consistency for those values.
SCIENTIFIC_ENV = {
    "ABFLOW_SOURCE_MODE": "pcs_rc",
    "ABFLOW_RECURRENT_PROPOSAL_CONTEXT": "on",
    "ABFLOW_RECURRENT_PROPOSAL_SEQUENCE_CONTEXT": "on",
    "ABFLOW_FINAL_READOUT_MODE": "integrated_endpoint",
    "ABFLOW_SCOREFM_LOSS_MODE": "f01_r3_endpoint_canonical_hybrid",
    "ABFLOW_SCOREFM_SAMPLER_MODE": "f01_canonical_carrier",
    "ABFLOW_R3_G_MODE": "foldflow_fixed_scaled",
    "ABFLOW_R3_FIXED_G_SCALED": "0.1",
    "ABFLOW_R3_FLOW_COORDINATE_SCALING": "0.1",
    "ABFLOW_R3_NOISE_SCOPE": "residue",
    "ABFLOW_R3_PATH_MIN_SIGMA": "0.0",
    "ABFLOW_ABX_NATIVE_REPR": "on",
    "ABFLOW_ABX_BRIDGE_MODE": "zero_reparameterized",
    "ABFLOW_ABX_RECYCLING": "off",
    "ABFLOW_ABX_WIDTH_PROFILE": "localized",
    "ABFLOW_ABX_ACTIVATION_CHECKPOINT": "on",
    "ABFLOW_SMOOTH_LDDT_CONTEXT_MODE": "fixed_observed",
    "ABFLOW_CONDITION_DIAGNOSTICS": "on",
    "ABFLOW_TASK_MASK_CONTRACT": "on",
    "ABFLOW_EPOCH_TEST": "on",
    "ABFLOW_EPOCH_TEST_INTERVAL": "1",
    "ABFLOW_EPOCH_TEST_N_STEPS": "10",
    "ABFLOW_EPOCH_TEST_FAIL_FAST": "on",
}


def fail(message: str) -> None:
    raise SystemExit(f"[V211PreflightFAIL] {message}")


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / value.removeprefix("./")


def require_text(text: str, symbols: tuple[str, ...], label: str) -> None:
    for symbol in symbols:
        if symbol not in text:
            fail(f"{label} symbol missing: {symbol}")


def validate_config(root: Path, config_path: Path, skip_data_files: bool) -> None:
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"invalid JSON {config_path}: {exc}")

    meta = cfg.get("_experiment")
    if not isinstance(meta, dict):
        fail("_experiment must be an object")
    exp_id = meta.get("exp_id")
    if exp_id not in EXPERIMENTS:
        fail(f"unknown exp_id={exp_id!r}")
    expected = EXPERIMENTS[exp_id]
    if meta.get("protocol") != PROTOCOL:
        fail(f"wrong protocol={meta.get('protocol')!r}; expected {PROTOCOL}")
    if meta.get("parent") != expected["parent"]:
        fail(
            f"{exp_id} parent={meta.get('parent')!r}, "
            f"expected {expected['parent']!r}"
        )

    task_regions = {}
    for key in ("cdr", "paratope"):
        value = cfg.get(key)
        if (
            not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)
        ):
            fail(f"{key} must be a non-empty unique JSON string list; got {value!r}")
        task_regions[key] = value
    if task_regions["cdr"] != task_regions["paratope"]:
        fail("cdr and paratope lists must match for the formal co-design protocol")

    for key in ("batch_size", "max_epoch", "iter_round"):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            fail(f"{key} must be a positive JSON integer; got {value!r}")
    # Three rounds are part of the frozen R05 physical parent, not a launcher
    # default. All other run-sized choices remain freely JSON-controlled.
    if cfg["iter_round"] != 3:
        fail(f"iter_round must preserve the R05 parent value 3; got {cfg['iter_round']!r}")
    ema_decay = cfg.get("ema_decay")
    if isinstance(ema_decay, bool) or not isinstance(ema_decay, (int, float)) \
            or not 0.0 <= float(ema_decay) < 1.0:
        fail(f"ema_decay must be numeric in [0,1); got {ema_decay!r}")
    if not isinstance(cfg.get("amp_dtype"), str) or not cfg["amp_dtype"]:
        fail("amp_dtype must be a non-empty JSON string")
    if cfg.get("resume_checkpoint") != "":
        fail("formal scratch runs require resume_checkpoint='' ")
    if cfg.get("save_dir") != f"./results_module/{exp_id}":
        fail("save_dir must be derived from the exact experiment id")

    env = meta.get("runtime_env")
    if not isinstance(env, dict):
        fail("_experiment.runtime_env must be an object")
    expected_env = {
        **SCIENTIFIC_ENV,
        "ABFLOW_ABX_DISTOGRAM": expected["disto"],
        "ABFLOW_DISTOGRAM_PAIR_SCOPE": expected["disto_scope"],
        "ABFLOW_MF_SMOOTH_LDDT": expected["lddt"],
    }
    for key, value in expected_env.items():
        if str(env.get(key)) != value:
            fail(f"runtime_env.{key}={env.get(key)!r}, expected {value!r}")
    curriculum = [key for key in env if "CURRICULUM" in key.upper()]
    if curriculum:
        fail(f"staged-loss curriculum is excluded; found {curriculum}")

    def env_nonnegative_float(key: str) -> float:
        try:
            value = float(env[key])
        except (KeyError, TypeError, ValueError):
            fail(f"runtime_env.{key} must be a non-negative numeric string")
        if value < 0.0:
            fail(f"runtime_env.{key} must be non-negative; got {value}")
        return value

    weights = {
        "sequence": env_nonnegative_float("ABFLOW_LOSS_SEQUENCE_WEIGHT"),
        "structure": env_nonnegative_float("ABFLOW_LOSS_STRUCTURE_WEIGHT"),
        "interface": env_nonnegative_float("ABFLOW_LOSS_INTERFACE_WEIGHT"),
        "edge": env_nonnegative_float("ABFLOW_LOSS_EDGE_WEIGHT"),
        "distogram": env_nonnegative_float("ABFLOW_LOSS_DISTOGRAM_WEIGHT"),
        "smooth_lddt": env_nonnegative_float("ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT"),
    }
    if expected["disto"] == "on" and weights["distogram"] <= 0.0:
        fail("R29 requires a positive JSON Distogram weight")
    if expected["disto"] == "off" and weights["distogram"] != 0.0:
        fail("Distogram-disabled experiments require JSON weight 0")
    if expected["lddt"] == "on" and weights["smooth_lddt"] <= 0.0:
        fail("R30 requires a positive JSON smooth-lDDT weight")
    if expected["lddt"] == "off" and weights["smooth_lddt"] != 0.0:
        fail("smooth-lDDT-disabled experiments require JSON weight 0")
    for key in ("ABFLOW_ABX_INIT_SEED", "ABFLOW_ABX_FORWARD_SEED"):
        value = str(env.get(key, ""))
        if not value.isdigit():
            fail(f"runtime_env.{key} must be a non-negative integer string")
    for key in ("ABFLOW_ABX_TRIANGLE_CHUNK_SIZE", "ABFLOW_SCI_LOG_FIRST_STEPS"):
        value = str(env.get(key, ""))
        if not value.isdigit() or int(value) <= 0:
            fail(f"runtime_env.{key} must be a positive integer string")
    if int(env["ABFLOW_SCI_LOG_FIRST_STEPS"]) < 2:
        fail("ABFLOW_SCI_LOG_FIRST_STEPS must cover cold and live bridge steps")

    formal = meta.get("formal_runtime")
    if not isinstance(formal, dict):
        fail("_experiment.formal_runtime must be an object")
    required_formal = {
        "mode", "physical_gpus", "world_size", "master_addr", "master_port",
        "rdzv_backend", "nnodes", "omp_num_threads", "version_policy",
        "force_scratch", "log_filename",
    }
    missing_formal = sorted(required_formal - set(formal))
    if missing_formal:
        fail(f"formal_runtime missing keys: {missing_formal}")
    if formal["mode"] != "train":
        fail("formal_runtime.mode must be 'train'")
    gpus = formal["physical_gpus"]
    if (
        not isinstance(gpus, list) or not gpus
        or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in gpus)
        or len(gpus) != len(set(gpus))
    ):
        fail(f"physical_gpus must be a non-empty unique integer list; got {gpus!r}")
    if formal["world_size"] != len(gpus):
        fail("world_size must equal len(physical_gpus)")
    port = formal["master_port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        fail(f"master_port must be an integer in [1024,65535]; got {port!r}")
    for key in ("nnodes", "omp_num_threads"):
        value = formal[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            fail(f"formal_runtime.{key} must be a positive integer")
    if formal["version_policy"] != "next_integer":
        fail("version_policy must be 'next_integer'")
    if formal["force_scratch"] is not True:
        fail("formal experiments require force_scratch=true")
    if not isinstance(formal["master_addr"], str) or not formal["master_addr"]:
        fail("master_addr must be a non-empty JSON string")
    if not isinstance(formal["rdzv_backend"], str) or not formal["rdzv_backend"]:
        fail("rdzv_backend must be a non-empty JSON string")
    log_filename = formal["log_filename"]
    if not isinstance(log_filename, str) or not log_filename \
            or Path(log_filename).name != log_filename:
        fail("log_filename must be a non-empty basename")

    loss_contract = meta.get("loss_contract")
    if not isinstance(loss_contract, dict):
        fail("_experiment.loss_contract must be an object")
    for key, value in weights.items():
        declared = loss_contract.get(key)
        if isinstance(declared, bool) or not isinstance(declared, (int, float)):
            fail(f"loss_contract.{key} must be numeric")
        if float(declared) != value:
            fail(
                f"loss_contract.{key}={declared!r} disagrees with "
                f"runtime_env JSON value {value!r}"
            )

    if not skip_data_files:
        for key in (
            "train_set", "valid_set", "train_pep", "valid_pep",
            "train_surf", "valid_surf",
        ):
            path = project_path(root, str(cfg.get(key, "")))
            if not path.is_file():
                fail(f"missing {key}: {path}")
        for key in (
            "ABFLOW_EPOCH_TEST_JSON", "ABFLOW_EPOCH_TEST_PEP",
            "ABFLOW_EPOCH_TEST_SURF",
        ):
            path = project_path(root, str(env.get(key, "")))
            if not path.is_file():
                fail(f"missing {key}: {path}")

    files = {
        "model": root / "models/AbFlow/AbFlow_model.py",
        "r3_matcher": root / "models/AbFlow/abflow_r3_matcher.py",
        "am_enc": root / "models/modules/am_enc.py",
        "am_egnn": root / "models/modules/am_egnn.py",
        "trainer": root / "trainer/AbFlow_trainer.py",
        "abs_trainer": root / "trainer/abs_trainer.py",
        "train": root / "scripts/train/train.sh",
        "launcher": root / "scripts/train/run_R28_R29_R30_v211.sh",
        "startup_audit": root / "scripts/train/audit_v211_startup.py",
        "epoch_test": root / "utils/epoch_test.py",
    }
    for label, path in files.items():
        if not path.is_file():
            fail(f"missing {label}: {path}")
    for label in (
        "model", "r3_matcher", "am_enc", "am_egnn", "trainer",
        "abs_trainer", "startup_audit", "epoch_test",
    ):
        ast.parse(files[label].read_text(encoding="utf-8"))

    model = files["model"].read_text(encoding="utf-8")
    enc = files["am_enc"].read_text(encoding="utf-8")
    egnn = files["am_egnn"].read_text(encoding="utf-8")
    trainer = files["trainer"].read_text(encoding="utf-8")
    train = files["train"].read_text(encoding="utf-8")
    launcher = files["launcher"].read_text(encoding="utf-8")
    abs_trainer = files["abs_trainer"].read_text(encoding="utf-8")

    require_text(model, (
        "class R05AbXNativeTrunk",
        "pseudo_beta, pseudo_beta_mask = pseudo_beta_fn_v2",
        'self.distogram_pair_scope == "design_anchored"',
        "design[:, :, None] | design[:, None, :]",
        "design_pair = pair & (design_atom[:, None] | design_atom[None, :])",
        "valid_atom_mask = self.batch_constants['xloss_mask'].bool()",
        "aux_pred_X = true_X.clone()",
        "aux_pred_X[cmask] = pred_X[cmask]",
        "with torch.random.fork_rng(devices=[])",
        "ABFLOW_ABX_FORWARD_SEED",
        "donor_dropout_rng=isolate",
        "shared_initialization_fingerprint",
        "value.reshape(-1).view(torch.uint8)",
        "final_sequence_diag",
    ), "model")
    require_text(enc, (
        "in_single_nf=0", "self.single_linear", "nn.init.zeros_",
        "h = base_pre + single_delta", "single_attr=None",
        "capture_bridge_diagnostics=False",
    ), "AMEncoder")
    require_text(egnn, (
        "self.edge_attr_linear", "nn.init.zeros_",
        "out = base_pre + pair_delta", "capture_bridge_diagnostics=False",
    ), "EGNN")
    if "H_gnn = torch.cat" in model:
        fail("old direct Single concatenation remains in model")
    if "input_edge + radial_nf + edges_in_d" in egnn:
        fail("old direct Pair concatenation remains in EGNN")
    require_text(trainer, (
        "[V211SharedInitFingerprint]", "[DistoAudit]",
        "[SmoothLDDTAudit]", "[V211BridgeAuthorityAudit]",
        "[V211BridgeColdStartPASS]", "[V211BridgeLivePASS]",
        "DistogramColdStartContract",
        "SmoothLDDTBridgeColdStartContract",
        "LiveSmoothLDDTPairGradientContract",
    ), "trainer")
    require_text(train, (
        'formal = meta.get("formal_runtime")',
        'gpus = formal["physical_gpus"]',
        "args.extend(str(item) for item in value)",
        "ABFLOW_JSON_FORCE_SCRATCH", "[V211JSONAuthority]",
    ), "train.sh")
    if "ABFLOW_FIXED_VERSION" not in abs_trainer:
        fail("abs_trainer lacks common DDP run-version contract")
    for forbidden in ('case "$EXP_ID"', "DEFAULT_GPU", "DEFAULT_PORT", "GPU_PAIR=${"):
        if forbidden in launcher:
            fail(f"launcher contains hard-coded experiment routing: {forbidden}")

    print(
        f"[V211PreflightPASS] exp={exp_id} parent={expected['parent']} "
        f"pair=on bridge=zero_reparameterized distogram={expected['disto']} "
        f"disto_scope={expected['disto_scope']} smooth_lddt={expected['lddt']} "
        f"GPU={gpus} port={port} batch={cfg['batch_size']} "
        f"weights={weights} cdr={task_regions['cdr']} "
        "run_json=PASS weights_json=PASS donor_localization=PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-data-files", action="store_true")
    args = parser.parse_args()
    validate_config(
        args.project_root.resolve(), args.config.resolve(), args.skip_data_files
    )


if __name__ == "__main__":
    main()
