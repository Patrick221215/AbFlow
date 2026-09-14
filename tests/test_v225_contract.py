from pathlib import Path
import hashlib, json, py_compile, re

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/AbFlow/AbFlow_model.py'
TRAINER = ROOT / 'trainer/AbFlow_trainer.py'
LAUNCHER = ROOT / 'scripts/train/run_R67_R68_v225.sh'
CFGDIR = ROOT / 'scripts/train/configs/R05_R67_R68'

for p in [MODEL, TRAINER, ROOT/'models/modules/am_enc.py', ROOT/'models/modules/am_egnn.py']:
    py_compile.compile(str(p), doraise=True)

model = MODEL.read_text()
trainer = TRAINER.read_text()
launcher = LAUNCHER.read_text()

assert '"r05_recurrent_carrier"' in model
branch = model.split('elif self.coordinate_state_mode == "r05_recurrent_carrier":',1)[1].split('r_interface_X.append(interface_X.clone())',1)[0]
assert '_carrier_from_single_endpoint' not in branch
assert '_close_scoreflow_endpoint_state' not in branch
assert 'pass' in branch
assert 'r_interface_X.append(interface_X.clone())' in model
assert 'X[cmask] = pred_X[cmask]' in model
assert '{"scoreflow_endpoint_fused", "scoreflow_single_endpoint", "r05_recurrent_carrier"}' in model
assert '[SampleGeometryOutlier]' not in model
assert '[SampleDynamicsAlert]' in model
assert '[RecurrentRole]' in trainer
assert '[GeometryDriftAlert]' in trainer
assert '[CoarseAnchor]' not in trainer
assert 'r05_recurrent_carrier' in launcher
assert 'rounds=3 carrier_feedback=egnn_output_exact_cross_round' in launcher

cfgs=sorted(CFGDIR.glob('*.json'))
assert len(cfgs)==2, cfgs
for p in cfgs:
    c=json.loads(p.read_text())
    assert 'gpus' not in c.get('runtime',{})
    assert c['model']['architecture']['iter_round']==3
    st=c['model']['representation']['single_pair']['coordinate_state']
    assert st['mode']=='r05_recurrent_carrier'
    assert st['carrier_recurrence']=='egnn_output_exact_cross_round'
    assert st['structural_recurrence']=='pred_X_cross_round'
    assert st['no_projection'] is True and st['no_kabsch_fusion'] is True
    assert c['loss']['generated_region_only'] is True
    assert 'coarse_anchor_distance' not in c['loss']
    assert float(c['loss']['smooth_lddt']['weight'])==0.0
    d=c['loss']['distogram']
    assert d['pair_scope']=='generation_anchored'
    assert d['reduction']=='relation_balanced'
weights={json.loads(p.read_text())['experiment']['id']: json.loads(p.read_text())['loss']['distogram']['weight'] for p in cfgs}
assert sorted(float(v) for v in weights.values())==[0.0,0.03]

# Exact controller files must remain the already-tested V219/V221 parent files.
for rel,prefix in [('models/modules/am_enc.py','8d7478017c493a93'),('models/modules/am_egnn.py','f39fb75cc89aba89')]:
    h=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    assert h.startswith(prefix), (rel,h)

print('V225_CONTRACT_TEST=PASS')
print('RECURRENT_GEOMETRY_RESTORED=PASS')
print('DISTOGRAM_RELATION_BALANCED=PASS')
print('DIAGNOSTICS_COMPACT=PASS')
