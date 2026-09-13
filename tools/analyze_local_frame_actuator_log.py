#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Summarize V216 LocalFrameActuatorAudit lines without imposing heuristic cutoffs."""
import math, re, sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit('Usage: python tools/analyze_local_frame_actuator_log.py <run_time.log>')
path=Path(sys.argv[1])
text=path.read_text(errors='replace')
lines=[x for x in text.splitlines() if '[LocalFrameActuatorAudit]' in x]
pat=re.compile(
    r'r(?P<round>\d+):trans_p99_A=(?P<tp>[-+eE0-9.nan]+) '
    r'trans_max_A=(?P<tm>[-+eE0-9.nan]+) rot_p99_deg=(?P<rp>[-+eE0-9.nan]+) '
    r'sc_p99_A=(?P<sp>[-+eE0-9.nan]+) atom_p99_A=(?P<ap>[-+eE0-9.nan]+) '
    r'atom_max_A=(?P<am>[-+eE0-9.nan]+) frame_valid=(?P<fv>[-+eE0-9.nan]+) '
    r'fixed_update_A=(?P<fx>[-+eE0-9.nan]+) bb_rigid_err_A=(?P<be>[-+eE0-9.nan]+) '
    r'rigid_w=(?P<rw>[-+eE0-9.nan]+) atom_w=(?P<aw>[-+eE0-9.nan]+)'
)
rows=[]
for line in lines:
    for m in pat.finditer(line):
        d={'round':int(m.group('round'))}
        for k in ['tp','tm','rp','sp','ap','am','fv','fx','be','rw','aw']:
            try: d[k]=float(m.group(k))
            except: d[k]=float('nan')
        rows.append(d)

def vals(k): return [r[k] for r in rows if math.isfinite(r[k])]
def vmax(k):
    a=vals(k); return max(a) if a else float('nan')
def vmin(k):
    a=vals(k); return min(a) if a else float('nan')
def fmt(x): return 'nan' if not math.isfinite(x) else f'{x:.6g}'

print(f'file={path}')
print(f'audit_lines={len(lines)} round_records={len(rows)}')
print(f'max_translation_p99_A={fmt(vmax("tp"))}')
print(f'max_translation_A={fmt(vmax("tm"))}')
print(f'max_rotation_p99_deg={fmt(vmax("rp"))}')
print(f'max_sidechain_residual_p99_A={fmt(vmax("sp"))}')
print(f'max_atom_update_p99_A={fmt(vmax("ap"))}')
print(f'max_atom_update_A={fmt(vmax("am"))}')
print(f'min_movable_frame_valid_fraction={fmt(vmin("fv"))}')
print(f'max_fixed_context_update_A={fmt(vmax("fx"))}')
print(f'max_backbone_rigid_distance_error_A={fmt(vmax("be"))}')
print(f'max_rigid_head_weight_rms={fmt(vmax("rw"))}')
print(f'max_atom_head_weight_rms={fmt(vmax("aw"))}')
print(f'train_loss_outliers={text.count("[TrainLossOutlier]")}')
print(f'sample_geometry_outliers={text.count("[SampleGeometryOutlier]")}')
print('NOTE: no empirical displacement threshold is encoded here; compare R41 to matched R31 trajectory.')
