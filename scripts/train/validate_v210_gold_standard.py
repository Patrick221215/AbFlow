#!/usr/bin/env python3
import ast, json, pathlib, sys
root=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else ".").resolve()
model=root/"models/AbFlow/AbFlow_model.py"
ast.parse(model.read_text(encoding="utf-8"))
cfgdir=root/"scripts/train/configs/R05_ABX_NATIVE_V210_GOLD_STANDARD"
names=[
"R28_R05_ABX_NATIVE_PAIR_EGNN_U02.json",
"R29_R05_ABX_NATIVE_PAIR_EGNN_DISTOGRAM_U02.json",
"R30_R05_MF_DONOR_SMOOTH_LDDT_U02.json",
]
for n in names:
    c=json.loads((cfgdir/n).read_text(encoding="utf-8"))
    assert isinstance(c["cdr"],list) and isinstance(c["paratope"],list)
    assert isinstance(c.get("save_dir"), str) and c["save_dir"].strip(), "save_dir missing"
    assert c["cdr"]==["H3"] and c["paratope"]==["H3"]
    e=c["_experiment"]["runtime_env"]
    assert all(k in e for k in (
      "ABFLOW_LOSS_SEQUENCE_WEIGHT","ABFLOW_LOSS_STRUCTURE_WEIGHT",
      "ABFLOW_LOSS_INTERFACE_WEIGHT","ABFLOW_LOSS_EDGE_WEIGHT",
      "ABFLOW_LOSS_DISTOGRAM_WEIGHT","ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT"))
src=model.read_text(encoding="utf-8")
for token in (
 "mf_boltz_design_localized_smooth_lddt_loss",
 "disto_resolved_pseudo_beta_rate",
 "grad_pair_norm_",
 "pair_mask = pb_mask[:, :, None] & pb_mask[:, None, :]",
 "AbXDistogramHeadLocal(pair_dim=self.pair_dim)",
 "Distogram runtime-width contract violated",
 "JSON-driven multi-CDR localization",
 "design_atom[:, None] | design_atom[None, :]",
 "[GeometryUnitContract]",
 "[LossWeightContract]",
):
    assert token in src, token
print("[V210StaticValidatorPASS] donor formulas/localization/task/loss-weight contracts present")

launcher=(root/"scripts/train/run_R28_R29_R30_v210.sh").read_text(encoding="utf-8")
assert 'cfg["save_dir"] = run_dir' in launcher
assert 'V210.1GeneratedConfigPASS' in launcher
