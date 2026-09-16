#!/usr/bin/env python3
import ast
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/AbFlow/AbFlow_model.py'
COMP = ROOT / 'models/AbFlow/abflow_components.py'
NN = ROOT / 'utils/nn_utils.py'
TRAINER = ROOT / 'trainer/AbFlow_trainer.py'
CFG = ROOT / 'scripts/train/configs/R05_R67_R68/R79_R05_ABX_R77_ROUND_STATE_SINGLEPAIR_ANALYTIC3R_TVT_U02.json'
LAUNCH = ROOT / 'scripts/train/run_R79_round_state_singlepair_v246.sh'

for p in (MODEL, COMP, NN, TRAINER, CFG, LAUNCH):
    if not p.is_file():
        raise SystemExit(f'MISSING: {p}')

texts = {p: p.read_text(encoding='utf-8') for p in (MODEL, COMP, NN, TRAINER, LAUNCH)}
for p in (MODEL, COMP, NN, TRAINER):
    ast.parse(texts[p], filename=str(p))


# Cross-file call-contract check: every explicit keyword sent to
# self.native_trunk(...) must exist in NativeTrunk.forward.  This catches API drift
# before DDP starts.
model_tree = ast.parse(texts[MODEL], filename=str(MODEL))
comp_tree = ast.parse(texts[COMP], filename=str(COMP))
native_forward = None
for node in comp_tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'NativeTrunk':
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                native_forward = item
                break
        break
assert native_forward is not None, 'NativeTrunk.forward not found'
accepted_native_kwargs = {
    a.arg for a in list(native_forward.args.args) + list(native_forward.args.kwonlyargs)
}
native_calls = []
for node in ast.walk(model_tree):
    if not isinstance(node, ast.Call):
        continue
    f = node.func
    if (
        isinstance(f, ast.Attribute)
        and f.attr == 'native_trunk'
        and isinstance(f.value, ast.Name)
        and f.value.id == 'self'
    ):
        native_calls.append(node)
assert native_calls, 'no self.native_trunk(...) calls found'
for call in native_calls:
    unexpected = [
        kw.arg for kw in call.keywords
        if kw.arg is not None and kw.arg not in accepted_native_kwargs
    ]
    assert not unexpected, (
        f'NativeTrunk call/signature mismatch at AbFlow_model.py:{call.lineno}: '
        f'unexpected kwargs={unexpected}; accepted={sorted(accepted_native_kwargs)}'
    )

cfg = json.loads(CFG.read_text(encoding='utf-8'))
exp = cfg['experiment']
assert exp['id'] == CFG.stem
assert exp['parent'] == 'R77_R05_ABX_R72_AUTHORITY_CLOSED_FINALROUND_ANALYTIC3R_TVT_U02'
assert exp['initialization'] == 'scratch'
assert exp['diagnostic_role'] == 'r79_round_state_singlepair_only'
assert cfg['model']['architecture']['iter_round'] == 3
sp = cfg['model']['representation']['single_pair']
pa = sp['physical_authority']
rs = sp['round_state_conditioning']
assert sp['pair_coordinate']['mode'] == 'direct_shared'
assert sp['coordinate_controller']['mode'] == 'egnn_prenorm_raw'
assert sp['time_embed'] is True
assert pa['mode'] == 'carrier_primary_analytic' and pa['physical_dof'] == 1
assert pa['geometric_operator_mode'] == 'latent_native_workspace'
assert pa['structure_supervision_mask'] == 'paratope_only'
assert pa['fixed_context_writeback'] == 'paratope_only'
assert pa['refinement_supervision'] == 'final_round_only_inherited_from_AbFlow'
assert rs['enabled'] is True
assert rs['geometry_source'] == 'current_authoritative_state'
assert rs['round0_state'] == 'outer_Xt'
assert rs['later_round_state'] == 'previous_analytic_endpoint'
assert rs['condition_design_geometry'] is True
assert rs['geometry_scope'] == 'paratope_only'
assert rs['prev_seq'] is False and rs['prev_pair'] is False and rs['prev_pos'] is False
assert rs['detach_between_rounds'] is False
assert rs['pair_static_within_round'] is True
assert float(cfg['loss']['sequence']) == 1.0
assert float(cfg['loss']['structure']) == 1.0
assert float(cfg['loss']['interface']) == 1.0
assert float(cfg['loss']['edge']) == 1.0
assert float(cfg['loss']['distogram']['weight']) == 0.0
assert float(cfg['loss']['smooth_lddt']['weight']) == 0.0
assert cfg['generation']['n_steps'] == 10 and cfg['generation']['seed'] == 2023
assert cfg['generation']['single_physical_field'] is True
assert cfg['generation']['terminal_kabsch_fusion'] is False

m = texts[MODEL]
assert 'self.round_state_conditioning_enabled' in m
assert 'current_relational_h3_native = self._interface_to_native_model(' in m
assert 'condition_design_geometry=True' in m
assert 'current_relational_h3_native = authority_endpoint_native' in m
assert "trunk_state=trunk_state" in m
assert m.index('for round_idx in range(self.round):') < m.index('condition_design_geometry=True')
for forbidden in ('set_round_recycle_context(', 'prev_seq =', 'prev_pair =', 'prev_pos =', 'previous_recycle_pos'):
    assert forbidden not in m, f'forbidden prev-recycle implementation found: {forbidden}'
assert "X[paratope_mask] = authority_endpoint_native" in m
assert "self.physical_authority_mode == 'carrier_primary_analytic'" in m
assert 'sequential_local_then_transport' not in m
assert "elif self.physical_authority_mode == 'endpoint_primary_analytic'" not in m

c = texts[COMP]
assert 'condition_design_geometry=False' in c
assert 'aux_task_mask=None' in c
assert "'geometry_condition_mask': GeometryCondition" in c
assert "'_aux_task_mask': AuxTask" in c
assert 'GeometryCondition = Fixed | AuxTask' in c
assert 'Fixed = M & (~Design)' in c
assert 'outside_design = AuxTask & (~Design)' in c
assert 'relational_task_geometry_visible_fraction' in c
assert 'relational_non_task_design_geometry_visible_fraction' in c

n = texts[NN]
assert n.count("batch.get('geometry_condition_mask', batch['fixed_mask'])") >= 2

t = texts[TRAINER]
assert "'[RoundStateValidation] '" in t
assert "'[TimeFieldValidation] '" not in t
assert "'[StateExposureAudit] '" not in t
assert "'[TestFieldTrajectory] '" in t
assert 'prev_recycling=off' in t
assert 'h3_geom_visible=' in t
assert 'other_design_geom_visible=' in t

l = texts[LAUNCH]
assert 'ABFLOW_STATE_EXPOSURE_AUDIT=off' in l
assert 'prev_seq=0 prev_pair=0 prev_pos=0' in l
assert 'versus_R77=round_state_singlepair_only' in l
assert 'geometry_scope=paratope_only' in l
assert 'other_cmask_geometry=blocked' in l

print('R79 static contract: PASS')
print('delta=round_state_singlepair_only prev_recycle=OFF physical_dof=1 sampler=UNCHANGED')
