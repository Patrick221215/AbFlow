#!/usr/bin/env python3
import math, re, sys
from pathlib import Path

PAT = re.compile(r"r(?P<round>\d+):raw_alpha=(?P<raw>\S+) eff_alpha=(?P<eff>\S+) raw_dnorm=(?P<dn>\S+) dir_norm=(?P<un>\S+) trans=(?P<tr>\S+) update=(?P<up>\S+) upd_in_rms=(?P<ur>\S+)")

def f(x):
    try: return float(x)
    except: return math.nan

def main(path):
    vals=[]
    for line in Path(path).read_text(errors='replace').splitlines():
        if '[CoordinateControllerAudit]' not in line: continue
        call_m=re.search(r'train_call=(\d+)',line)
        call=int(call_m.group(1)) if call_m else -1
        for m in PAT.finditer(line):
            d={k:f(v) for k,v in m.groupdict().items() if k!='round'}
            d['round']=int(m.group('round')); d['call']=call; vals.append(d)
    if not vals:
        raise SystemExit('No [CoordinateControllerAudit] records found.')
    def mx(k):
        z=[v[k] for v in vals if math.isfinite(v[k])]
        return max(z) if z else math.nan
    print(f'records={len(vals)} calls={len(set(v["call"] for v in vals))}')
    print(f'max_raw_alpha={mx("raw"):.6g}')
    print(f'max_effective_alpha={mx("eff"):.6g}   contract<=1')
    print(f'max_raw_distance_norm={mx("dn"):.6g}')
    print(f'max_direction_norm={mx("un"):.6g}    contract<=1')
    print(f'max_edge_translation_abs={mx("tr"):.6g} contract<=1 componentwise')
    print(f'max_aggregated_update_abs={mx("up"):.6g} contract<=1 for mean aggregation')
    print(f'max_update_input_rms_ratio={mx("ur"):.6g}')
    bad_eff=[v for v in vals if math.isfinite(v['eff']) and v['eff']>1.0001]
    bad_dir=[v for v in vals if math.isfinite(v['un']) and v['un']>1.0001]
    bad_up=[v for v in vals if math.isfinite(v['up']) and v['up']>1.0001]
    print('contract_status=' + ('PASS' if not (bad_eff or bad_dir or bad_up) else 'FAIL'))
    if bad_eff or bad_dir or bad_up:
        print(f'violations: effective_alpha={len(bad_eff)} direction_norm={len(bad_dir)} update={len(bad_up)}')

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('usage: python tools/analyze_coordinate_controller_log.py /path/to/run_time.log')
    main(sys.argv[1])
