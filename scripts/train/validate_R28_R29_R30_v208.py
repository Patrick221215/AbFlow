#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


R05_PARENT = "R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW"
R28_ID = "R28_R05_ABX_NATIVE_PAIR_EGNN_U02"

EXPERIMENTS = {
    R28_ID: {
        "parent": R05_PARENT,
        "gpu": [2, 3],
        "port": 29728,
        "pair": "on",
        "disto": "off",
        "lddt": "off",
        "disto_weight": "0.0",
        "lddt_weight": "0.0",
    },
    "R29_R28_ABX_NATIVE_PAIR_EGNN_DISTOGRAM_U02": {
        "parent": R28_ID,
        "gpu": [4, 5],
        "port": 29729,
        "pair": "on",
        "disto": "on",
        "lddt": "off",
        "disto_weight": "0.1",
        "lddt_weight": "0.0",
    },
    "R30_R28_ABX_NATIVE_PAIR_EGNN_MF_DONOR_SMOOTH_LDDT_U02": {
        "parent": R28_ID,
        "gpu": [6, 7],
        "port": 29730,
        "pair": "on",
        "disto": "off",
        "lddt": "on",
        "disto_weight": "0.0",
        "lddt_weight": "0.1",
    },
}

COMMON_ENV = {
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
    "ABFLOW_ABX_RECYCLING": "off",
    "ABFLOW_ABX_WIDTH_PROFILE": "localized",
    "ABFLOW_ABX_TRIANGLE_CHUNK_SIZE": "64",
    "ABFLOW_ABX_ACTIVATION_CHECKPOINT": "on",
    "ABFLOW_TASK_MASK_CONTRACT": "on",
    "ABFLOW_EPOCH_TEST": "on",
    "ABFLOW_EPOCH_TEST_INTERVAL": "1",
    "ABFLOW_EPOCH_TEST_N_STEPS": "10",
    "ABFLOW_EPOCH_TEST_FAIL_FAST": "on",
    "ABFLOW_LOSS_SEQUENCE_WEIGHT": "1.0",
    "ABFLOW_LOSS_STRUCTURE_WEIGHT": "1.0",
    "ABFLOW_LOSS_INTERFACE_WEIGHT": "1.0",
    "ABFLOW_LOSS_EDGE_WEIGHT": "1.0",
}


def fail(message: str) -> None:
    raise SystemExit(f"[V208PreflightFAIL] {message}")


def resolve_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / value.removeprefix("./")


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

    if meta.get("protocol") != "V208_JSON_AUTHORITY_R28_PARENT":
        fail(f"wrong protocol={meta.get('protocol')!r}")
    if meta.get("parent") != expected["parent"]:
        fail(
            f"{exp_id} parent={meta.get('parent')!r}, "
            f"expected {expected['parent']!r}"
        )
    for key in ("cdr", "paratope"):
        if cfg.get(key) != ["H3"]:
            fail(f"{key} must be JSON list ['H3']; got {cfg.get(key)!r}")

    scalar = {
        "batch_size": 20,
        "max_epoch": 200,
        "iter_round": 3,
        "ema_decay": 0.999,
        "amp_dtype": "bf16",
        "resume_checkpoint": "",
    }
    for key, value in scalar.items():
        if cfg.get(key) != value:
            fail(f"{key}={cfg.get(key)!r}, expected {value!r}")

    expected_save = f"./results_module/{exp_id}"
    if cfg.get("save_dir") != expected_save:
        fail(f"save_dir={cfg.get('save_dir')!r}, expected {expected_save!r}")

    env = meta.get("runtime_env")
    if not isinstance(env, dict):
        fail("_experiment.runtime_env must be an object")
    expected_env = {
        **COMMON_ENV,
        "ABFLOW_ABX_NATIVE_REPR": expected["pair"],
        "ABFLOW_ABX_DISTOGRAM": expected["disto"],
        "ABFLOW_MF_SMOOTH_LDDT": expected["lddt"],
        "ABFLOW_LOSS_DISTOGRAM_WEIGHT": expected["disto_weight"],
        "ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT": expected["lddt_weight"],
    }
    for key, value in expected_env.items():
        if str(env.get(key)) != value:
            fail(f"runtime_env.{key}={env.get(key)!r}, expected {value!r}")
    curriculum_keys = [key for key in env if "CURRICULUM" in key.upper()]
    if curriculum_keys:
        fail(f"staged-loss curriculum is excluded; found {curriculum_keys}")

    formal = meta.get("formal_runtime")
    if not isinstance(formal, dict):
        fail("_experiment.formal_runtime must be an object")
    formal_expected = {
        "mode": "train",
        "physical_gpus": expected["gpu"],
        "world_size": 2,
        "master_addr": "localhost",
        "master_port": expected["port"],
        "rdzv_backend": "c10d",
        "nnodes": 1,
        "omp_num_threads": 2,
        "version_policy": "next_integer",
        "force_scratch": True,
        "log_filename": "run_time.log",
    }
    for key, value in formal_expected.items():
        if formal.get(key) != value:
            fail(f"formal_runtime.{key}={formal.get(key)!r}, expected {value!r}")

    loss_contract = meta.get("loss_contract")
    if not isinstance(loss_contract, dict):
        fail("_experiment.loss_contract must be an object")
    contract_expected = {
        "sequence": 1.0,
        "structure": 1.0,
        "interface": 1.0,
        "edge": 1.0,
        "distogram": float(expected["disto_weight"]),
        "smooth_lddt": float(expected["lddt_weight"]),
    }
    for key, value in contract_expected.items():
        if loss_contract.get(key) != value:
            fail(f"loss_contract.{key}={loss_contract.get(key)!r}, expected {value!r}")

    if not skip_data_files:
        for key in (
            "train_set", "valid_set", "train_pep", "valid_pep",
            "train_surf", "valid_surf",
        ):
            path = resolve_project_path(root, str(cfg[key]))
            if not path.is_file():
                fail(f"missing {key}: {path}")
        for key in (
            "ABFLOW_EPOCH_TEST_JSON",
            "ABFLOW_EPOCH_TEST_PEP",
            "ABFLOW_EPOCH_TEST_SURF",
        ):
            path = resolve_project_path(root, str(env[key]))
            if not path.is_file():
                fail(f"missing {key}: {path}")

    required_files = {
        "model": root / "models/AbFlow/AbFlow_model.py",
        "r3_matcher": root / "models/AbFlow/abflow_r3_matcher.py",
        "am_enc": root / "models/modules/am_enc.py",
        "am_egnn": root / "models/modules/am_egnn.py",
        "trainer": root / "trainer/AbFlow_trainer.py",
        "abs_trainer": root / "trainer/abs_trainer.py",
        "train": root / "scripts/train/train.sh",
        "launcher": root / "scripts/train/run_R28_R29_R30_v208.sh",
        "epoch_test": root / "utils/epoch_test.py",
    }
    for label, path in required_files.items():
        if not path.is_file():
            fail(f"missing {label}: {path}")
    for label in (
        "model", "r3_matcher", "am_enc", "am_egnn",
        "trainer", "abs_trainer", "epoch_test",
    ):
        ast.parse(required_files[label].read_text(encoding="utf-8"))

    model_text = required_files["model"].read_text(encoding="utf-8")
    trainer_text = required_files["trainer"].read_text(encoding="utf-8")
    am_enc_text = required_files["am_enc"].read_text(encoding="utf-8")
    am_egnn_text = required_files["am_egnn"].read_text(encoding="utf-8")
    abs_trainer_text = required_files["abs_trainer"].read_text(encoding="utf-8")
    train_text = required_files["train"].read_text(encoding="utf-8")
    launcher_text = required_files["launcher"].read_text(encoding="utf-8")
    required_symbols = (
        "class R05AbXNativeTrunk",
        "pseudo_beta, pseudo_beta_mask = pseudo_beta_fn_v2",
        "pair_mask = pseudo_beta_mask[:, :, None] & pseudo_beta_mask[:, None, :]",
        "design_pair = pair & (design_atom[:, None] | design_atom[None, :])",
        "valid_atom_mask = self.batch_constants['xloss_mask'].bool()",
        "self._diagnostic_pair_probe_tensor",
        '("smooth_lddt", "endpoint")',
        "[GeometryUnitContract]",
        "ABFLOW_LOSS_STRUCTURE_WEIGHT",
    )
    for symbol in required_symbols:
        if symbol not in model_text:
            fail(f"model symbol missing: {symbol}")
    trainer_symbols = (
        "[DistoAudit]",
        "[PairLossAuthority]",
        'grad_diag.get("grad_pair_norm_distogram")',
        'g("grad_pair_norm_smooth_lddt")',
        "DistogramColdStartContract",
        "_live_pair_gradient_cold_start_observed",
        "LiveSmoothLDDTPairGradientContract",
        "_live_smooth_lddt_pair_gradient_contract_verified",
    )
    for symbol in trainer_symbols:
        if symbol not in trainer_text:
            fail(f"trainer symbol missing: {symbol}")
    for symbol in ("ctx_edge_attr=None", "inter_edge_attr=None", "surf_edge_attr=None"):
        if symbol not in am_enc_text:
            fail(f"AMEncoder pair-interface symbol missing: {symbol}")
    for symbol in ("edges_in_d", "edge_attr=None"):
        if symbol not in am_egnn_text:
            fail(f"AMEGNN native-edge symbol missing: {symbol}")
    if "ABFLOW_FIXED_VERSION" not in abs_trainer_text:
        fail("abs_trainer lacks DDP-safe JSON-run version contract")
    train_symbols = (
        'formal = meta.get("formal_runtime")',
        'gpus = formal["physical_gpus"]',
        "args.extend(str(item) for item in value)",
        'ABFLOW_JSON_FORCE_SCRATCH',
    )
    for symbol in train_symbols:
        if symbol not in train_text:
            fail(f"train.sh JSON-authority symbol missing: {symbol}")
    forbidden_launcher_symbols = (
        'case "$EXP_ID"', "DEFAULT_GPU", "DEFAULT_PORT", "GPU_PAIR=${",
    )
    for symbol in forbidden_launcher_symbols:
        if symbol in launcher_text:
            fail(f"launcher contains hard-coded experiment routing: {symbol}")

    print(
        f"[V208PreflightPASS] exp={exp_id} parent={expected['parent']} "
        f"cdr={cfg['cdr']} pair={expected['pair']} distogram={expected['disto']} "
        f"smooth_lddt={expected['lddt']} GPU={expected['gpu']} "
        f"port={expected['port']} run_json=PASS weights_json=PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-data-files", action="store_true")
    args = parser.parse_args()
    validate_config(
        args.project_root.resolve(),
        args.config.resolve(),
        args.skip_data_files,
    )


if __name__ == "__main__":
    main()
