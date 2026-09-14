#!/usr/bin/env python
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "scripts" / "train" / "configs" / "R05_R53_R58"
EXPECTED = {
    "R53_R05_ABX_NATIVE_PAIR_EGNN_PRENORM_RAW_TVT_U02": (0.0, "all_resolved", 0.0, "pred_design_endpoint"),
    "R56_R05_ABX_NATIVE_PAIR_EGNN_PRENORM_RAW_DISTOGRAM_GANCHOR_TVT_U02": (0.03, "generation_anchored", 0.0, "pred_design_endpoint"),
    "R57_R05_ABX_NATIVE_PAIR_EGNN_PRENORM_RAW_LDDT_CARRIER_TVT_U02": (0.0, "generation_anchored", 0.1, "carrier_implied_endpoint"),
    "R58_R05_ABX_NATIVE_PAIR_EGNN_PRENORM_RAW_DISTOGRAM_GANCHOR_LDDT_CARRIER_TVT_U02": (0.03, "generation_anchored", 0.1, "carrier_implied_endpoint"),
}

for exp_id, (dw, scope, lw, lsrc) in EXPECTED.items():
    path = CFG_DIR / f"{exp_id}.json"
    if not path.is_file():
        raise SystemExit(f"missing config: {path}")
    cfg = json.loads(path.read_text())
    exp = cfg["experiment"]
    assert exp["id"] == exp_id
    assert exp["protocol"] == "formal_train_val_test"
    assert pathlib.Path(cfg["training"]["output_dir"]).name == exp_id
    assert pathlib.Path(cfg["generation"]["save_dir"]).name == exp_id
    assert pathlib.Path(cfg["data"]["test"]["set"]).name == "test.json"
    assert int(cfg["generation"]["n_steps"]) == 10
    assert int(cfg["generation"]["seed"]) == 2023
    assert "gpus" not in cfg.get("runtime", {})
    sp = cfg["model"]["representation"]["single_pair"]
    assert sp["pair_coordinate"]["mode"] == "direct_shared"
    assert sp["coordinate_controller"]["mode"] == "egnn_prenorm_raw"
    d = cfg["loss"]["distogram"]
    l = cfg["loss"]["smooth_lddt"]
    assert abs(float(d["weight"]) - dw) < 1e-12
    assert str(d["pair_scope"]) == scope
    assert abs(float(l["weight"]) - lw) < 1e-12
    assert str(l["prediction_source"]) == lsrc
    if dw > 0:
        assert scope == "generation_anchored"

model = (ROOT / "models" / "AbFlow" / "AbFlow_model.py").read_text()
components = (ROOT / "models" / "modules" / "abflow_components.py").read_text()
trainer = (ROOT / "trainer" / "AbFlow_trainer.py").read_text()
assert "design_mask=cmask, aux_task_mask=paratope_mask" in model
assert "generation_anchored" in components
assert "distogram_context_context_optimized_pairs" in components
assert "compute_pair_gradient_authority" in model
assert "[PairGradientAudit]" in trainer
assert "[PairGradientAudit][UNAVAILABLE]" in trainer

print("V222_CONFIG_AND_SOURCE_CONTRACT=PASS")
for exp_id, values in EXPECTED.items():
    print(exp_id, values)
