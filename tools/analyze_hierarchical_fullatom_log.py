#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Summarize V217 hierarchical actuator audits without empirical cutoffs."""
import math, re, sys
from pathlib import Path
if len(sys.argv)!=2:
    raise SystemExit('Usage: python tools/analyze_hierarchical_fullatom_log.py <run_time.log>')
path=Path(sys.argv[1]); text=path.read_text(errors='replace')
lines=[x for x in text.splitlines() if '[HierarchicalFullAtomActuatorAudit]' in x]
pat=re.compile(
 r'r(?P<round>\d+):trans_p99_A=(?P<tp>[-+eE0-9.nan]+) trans_max_A=(?P<tm>[-+eE0-9.nan]+) '
 r'rot_p99_deg=(?P<rp>[-+eE0-9.nan]+) int_p99_A=(?P<ip>[-+eE0-9.nan]+) '
 r'bb_int_p99_A=(?P<bp>[-+eE0-9.nan]+) sc_int_p99_A=(?P<sp>[-+eE0-9.nan]+) '
 r'atom_p99_A=(?P<ap>[-+eE0-9.nan]+) atom_max_A=(?P<am>[-+eE0-9.nan]+) '
 r'frame_valid=(?P<fv>[-+eE0-9.nan]+) fixed_update_A=(?P<fx>[-+eE0-9.nan]+) '
 r'ca_internal_A=(?P<ca>[-+eE0-9.nan]+) coarse_rigid_err_A=(?P<re>[-+eE0-9.nan]+) '
 r'bb_internal_dchange_p99_A=(?P<bd>[-+eE0-9.nan]+) rigid_w=(?P<rw>[-+eE0-9.nan]+) '
 r'internal_w=(?P<iw>[-+eE0-9.nan]+)')
rows=[]
for line in lines:
    for m in pat.finditer(line):
        d={'round':int(m.group('round'))}
        for k in ['tp','tm','rp','ip','bp','sp','ap','am','fv','fx','ca','re','bd','rw','iw']:
            try:d[k]=float(m.group(k))
            except:d[k]=float('nan')
        rows.append(d)
def vals(k):return [r[k] for r in rows if math.isfinite(r[k])]
def vmax(k):
    a=vals(k); return max(a) if a else float('nan')
def vmin(k):
    a=vals(k); return min(a) if a else float('nan')
def fmt(x):return 'nan' if not math.isfinite(x) else f'{x:.6g}'
print(f'file={path}')
print(f'audit_lines={len(lines)} round_records={len(rows)}')
for label,key,op in [
 ('max_translation_p99_A','tp',vmax),('max_translation_A','tm',vmax),
 ('max_rotation_p99_deg','rp',vmax),('max_internal_residual_p99_A','ip',vmax),
 ('max_backbone_internal_residual_p99_A','bp',vmax),('max_sidechain_internal_residual_p99_A','sp',vmax),
 ('max_atom_update_p99_A','ap',vmax),('max_atom_update_A','am',vmax),
 ('min_movable_frame_valid_fraction','fv',vmin),('max_fixed_context_update_A','fx',vmax),
 ('max_ca_internal_update_A','ca',vmax),('max_coarse_rigid_distance_error_A','re',vmax),
 ('max_backbone_internal_distance_change_p99_A','bd',vmax),
 ('max_rigid_head_weight_rms','rw',vmax),('max_internal_head_weight_rms','iw',vmax)]:
    print(f'{label}={fmt(op(key))}')
print(f'train_loss_outliers={text.count("[TrainLossOutlier]")}')
print(f'sample_geometry_outliers={text.count("[SampleGeometryOutlier]")}')
print('NOTE: no displacement cutoff is encoded; compare R44 trajectory directly with matched healthy-parent regimes.')
