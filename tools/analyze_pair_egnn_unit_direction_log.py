#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

EPOCH_RE = re.compile(
    r'\[EpochSummary\]\s+epoch=(\d+).*?train=([\d.eE+-]+)\s+val=([\d.eE+-]+)'
    r'.*?AAR=([\d.eE+-]+)\s+CAAR=([\d.eE+-]+)\s+H3raw=([\d.eE+-]+)A'
    r'\s+H3aligned=([\d.eE+-]+)A\s+TM=([\d.eE+-]+)\s+lDDT=([\d.eE+-]+)\s+DockQ=([\d.eE+-]+)'
)
AUDIT_RE = re.compile(r'\[CoordinateControllerAudit\]\s+train_call=(\d+).*?mode=(\S+)\s+.*?(r0:.*)')
FIELD_RE = re.compile(r'(raw_alpha|pair_delta|eff_alpha|raw_dnorm|dir_norm|trans|update|upd_in_rms)=([\d.eE+-]+|nan)')
ROUND_RE = re.compile(r'r(\d+):([^;]+)')


def f(x):
    try: return float(x)
    except: return float('nan')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('log')
    args=ap.parse_args()
    text=Path(args.log).read_text(errors='replace')

    epochs=[]
    for m in EPOCH_RE.finditer(text):
        epochs.append({
            'epoch':int(m.group(1)), 'train':f(m.group(2)), 'val':f(m.group(3)),
            'AAR':f(m.group(4)), 'CAAR':f(m.group(5)), 'H3raw':f(m.group(6)),
            'H3aligned':f(m.group(7)), 'TM':f(m.group(8)), 'lDDT':f(m.group(9)),
            'DockQ':f(m.group(10)),
        })

    audits=[]
    for m in AUDIT_RE.finditer(text):
        rec={'call':int(m.group(1)), 'mode':m.group(2), 'rounds':[]}
        for rm in ROUND_RE.finditer(m.group(3)):
            vals={k:f(v) for k,v in FIELD_RE.findall(rm.group(2))}
            vals['round']=int(rm.group(1))
            rec['rounds'].append(vals)
        audits.append(rec)

    print(f'log={args.log}')
    print(f'epoch_summaries={len(epochs)} controller_audits={len(audits)}')
    if epochs:
        print('\n[Epoch trajectory]')
        for e in epochs:
            print(
                f"e{e['epoch']:03d} train={e['train']:.5g} val={e['val']:.5g} "
                f"AAR={e['AAR']:.4f} CAAR={e['CAAR']:.4f} raw={e['H3raw']:.3f} "
                f"aligned={e['H3aligned']:.3f} TM={e['TM']:.4f} lDDT={e['lDDT']:.4f} DockQ={e['DockQ']:.4f}"
            )
        if len(epochs)>=2:
            a,b=epochs[0],epochs[-1]
            print('\n[First -> last]')
            print(f"val_delta={b['val']-a['val']:+.5g}")
            print(f"aligned_delta_A={b['H3aligned']-a['H3aligned']:+.4f}")
            print(f"TM_delta={b['TM']-a['TM']:+.5f}")
            print(f"lDDT_delta={b['lDDT']-a['lDDT']:+.5f}")

    if audits:
        print('\n[Controller envelope]')
        keys=['raw_alpha','pair_delta','eff_alpha','raw_dnorm','dir_norm','trans','update','upd_in_rms']
        for key in keys:
            vals=[r[key] for a in audits for r in a['rounds'] if key in r and math.isfinite(r[key])]
            if vals:
                print(f'{key}: max={max(vals):.6g} last={vals[-1]:.6g}')
        dn=[r.get('dir_norm',float('nan')) for a in audits for r in a['rounds']]
        finite=[x for x in dn if math.isfinite(x)]
        if finite and max(finite) > 1.0005:
            print('CONTRACT_FAIL: dir_norm exceeded unit-vector tolerance')
        else:
            print('direction_contract=PASS')

    # Historical catastrophe signatures are observational only; do not clip or skip.
    catastrophic = ('[TrainLossOutlier]' in text or 'coord_update_absmax=1e+07' in text)
    print(f'historical_catastrophe_marker={int(catastrophic)}')

if __name__=='__main__':
    main()
