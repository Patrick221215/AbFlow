#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = (ROOT/'models/AbFlow/AbFlow_model.py').read_text()
components = (ROOT/'models/modules/abflow_components.py').read_text()
trainer = (ROOT/'trainer/AbFlow_trainer.py').read_text()
launcher = (ROOT/'scripts/train/run_R64_R66_v224.sh').read_text()

assert 'scoreflow_single_endpoint' in model
assert 'def _carrier_from_single_endpoint' in model
assert 'shadow_carrier=interface_X' in model
assert 'canonical_carrier_target_gfree' in model
assert 'def _coarse_anchor_distance_loss' in model
assert "self.distogram_reduction == 'relation_balanced'" in components
assert "torch.stack([dd_mean, df_mean, da_mean]" in components
assert '[StateAuthority]' in trainer
assert '[CoarseAnchor]' in trainer
assert 'pair_bounded_absmax=' not in trainer
assert 'edge_z=(' not in trainer
assert "tag = 'GeometryAuthorityAlert'" not in trainer
assert "if allow_alert:" in trainer
assert "state_mode != 'scoreflow_single_endpoint'" in launcher

cfg_dir=ROOT/'scripts/train/configs/R05_R64_R66'
files=sorted(cfg_dir.glob('*.json'))
assert len(files)==3
for path in files:
    cfg=json.loads(path.read_text())
    rid=cfg['experiment']['id']
    assert rid==path.stem
    assert Path(cfg['training']['output_dir']).name==rid
    st=cfg['model']['representation']['single_pair']['coordinate_state']
    assert st['mode']=='scoreflow_single_endpoint'
    assert st['clean_endpoint_authority']=='pred_X_only'
    assert cfg['loss']['smooth_lddt']['weight']==0.0
    assert 'gpus' not in cfg.get('runtime', {})

r65=json.loads((cfg_dir/'R65_R05_ABX_NATIVE_PAIR_EGNN_PRENORM_RAW_SINGLE_ENDPOINT_DISTOGRAM_RELBAL_TVT_U02.json').read_text())
assert r65['loss']['distogram']['weight']==0.03
assert r65['loss']['distogram']['reduction']=='relation_balanced'
r66=json.loads((cfg_dir/'R66_R05_ABX_NATIVE_PAIR_EGNN_PRENORM_RAW_SINGLE_ENDPOINT_COARSE_ANCHOR_TVT_U02.json').read_text())
assert r66['loss']['coarse_anchor_distance']['weight']==1.0
assert r66['loss']['coarse_anchor_distance']['cutoff_A']==21.6875
print('V224 contract tests: PASS')
