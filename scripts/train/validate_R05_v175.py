#!/usr/bin/env python3
import argparse, ast, json, math, pathlib, subprocess, sys, torch
IDS=['R23_R05_ABX_PAIR_TIME_U02','R24_R05_ENDPOINT_RELATION_U02','R25_R05_GEOMETRY_PAIR_U02']
PARENT='R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW'

def load(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def static(root):
    files=[root/'models/AbFlow/AbFlow_model.py',root/'models/modules/am_enc.py',root/'models/modules/am_enc_pair_time.py',root/'trainer/AbFlow_trainer.py',root/'trainer/abs_trainer.py',root/'scripts/train/validate_R05_v175.py',root/'scripts/train/validate_R05_v173.py']
    for p in files: ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    subprocess.run(['bash','-n',str(root/'scripts/train/run_R04_R05_R06_foldflow_noise_v104.sh')],check=True)
    model=(root/'models/AbFlow/AbFlow_model.py').read_text(encoding='utf-8')
    pt=(root/'models/modules/am_enc_pair_time.py').read_text(encoding='utf-8')
    for mark in ['R05MF_CAUSAL_MODULES_V175','pair_time_diagnostics','class R05GeometryPair','_r05_decode_endpoint']:
        if mark not in model: raise SystemExit('missing model marker '+mark)
    for mark in ['_abx_timestep_embedding','Preserve same-layer coordinate authority exactly.','pair_time_diagnostics']:
        if mark not in pt: raise SystemExit('missing pair-time marker '+mark)
    # Hard user boundary: scientific/runtime files contain no validation-generation subsystem.
    scan = [
        root/'models/AbFlow/AbFlow_model.py',
        root/'trainer/AbFlow_trainer.py',
        root/'trainer/abs_trainer.py',
        root/'scripts/train/run_R04_R05_R06_foldflow_noise_v104.sh',
    ] + list((root/'scripts/train/configs/R05_CAUSAL_V175').glob('*.json'))
    forbidden = ('ABFLOW_EPOCH_' + 'VALGEN').lower()
    forbidden2 = ('epoch_' + 'valgen').lower()
    for p in scan:
        txt=p.read_text(encoding='utf-8',errors='ignore').lower()
        if forbidden in txt or forbidden2 in txt:
            raise SystemExit('forbidden validation-generation implementation marker in '+str(p))
    print('[V175StaticPASS] AST + shell + no-ValGen implementation + module markers')

def config_check(path):
    c=load(path); eid=c['_experiment']['exp_id']; e=c['_experiment']['runtime_env']
    if eid not in IDS: raise SystemExit('unknown v175 id '+eid)
    for k,v in {'batch_size':56,'max_epoch':200,'iter_round':3,'use_ema':True}.items():
        if c.get(k)!=v: raise SystemExit(f'{eid}: {k} mismatch')
    if abs(float(c['ema_decay'])-.999)>1e-12: raise SystemExit('EMA mismatch')
    if c['_experiment']['parent']!=PARENT: raise SystemExit('parent mismatch')
    if e['ABFLOW_SEQUENCE_CONTEXT_MODE']!='legacy': raise SystemExit('sequence contract changed')
    expected={
      IDS[0]:('residue','off','off','off','0.0'),
      IDS[1]:('off','on','off','on','0.1'),
      IDS[2]:('off','off','on','off','0.0'),
    }[eid]
    got=(e['ABFLOW_PAIR_TIME_SCOPE'],e['ABFLOW_R05_ENDPOINT_RELATION'],e['ABFLOW_R05_GEOM_PAIR'],e['ABFLOW_MF_SMOOTH_LDDT'],e['ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT'])
    if got!=expected: raise SystemExit(f'{eid}: scientific delta mismatch {got} != {expected}')
    if e['ABFLOW_MF_REPR_CORE']!='off' or e['ABFLOW_MF_PAIR_ATOM_REFINER']!='off' or e['ABFLOW_MF_DISTOGRAM']!='off':
        raise SystemExit(eid+': confounded MF module')
    print('[V175ConfigPASS]',eid,got)

def tensor_checks():
    # U02 inverse and Jacobian
    torch.manual_seed(7)
    n=16; X0=torch.randn(n,3); X1=torch.randn(n,3); t=torch.rand(n,1)*.78+.21
    mu=(1-t)*X0+t*X1; r=torch.randn(n,3)*.1; Xt=mu+r
    Y=X1+r/(2*t)
    inv=2*Y-(Xt-(1-t)*X0)/t
    if not torch.allclose(inv,X1,atol=2e-5,rtol=2e-5): raise SystemExit('U02 inverse failed')
    Yv=Y.clone().requires_grad_(True); invv=2*Yv-(Xt-(1-t)*X0)/t
    g=torch.autograd.grad(invv.sum(),Yv)[0]
    if not torch.allclose(g,torch.full_like(g,2.0)): raise SystemExit('U02 Jacobian failed')
    print('[TensorPASS] U02 carrier -> clean endpoint inverse, Jacobian=2I')
    # endpoint relation: rigid-translation invariant; robust gradient non-zero at large errors
    A=torch.randn(8,3); B=A+torch.tensor([3.,-2.,1.]);
    da=torch.cdist(A,A); db=torch.cdist(B,B)
    if not torch.allclose(da,db,atol=1e-5): raise SystemExit('relation invariance failed')
    e=torch.tensor([0.,1.,10.],requires_grad=True); rho=(torch.sqrt(1+e*e)-1).mean(); rho.backward()
    if not (e.grad[-1].abs()>0.2): raise SystemExit('pseudo-Huber large-error gradient failed')
    print('[TensorPASS] endpoint relation: translation invariant + non-saturating pseudo-Huber gradient')
    # Pair-time math: exact zero-start, then explicit time dependence when activated.
    def emb(t,dim=32,maxp=10000):
        t=t.reshape(-1).float()*maxp; half=dim//2; sc=math.log(maxp)/(half-1)
        f=torch.exp(torch.arange(half)*(-sc)); ph=t[:,None]*f[None]
        return torch.cat([torch.sin(ph),torch.cos(ph)],-1)
    base=torch.randn(5,12); tt=torch.tensor([0.,.2,.4,.7,1.]); tau=emb(tt)
    W=torch.zeros(12,32); res=tau@W.t()
    if not torch.equal(base+res,base): raise SystemExit('pair-time zero-start failed')
    W[0,0]=.1; y=tau@W.t()
    if torch.allclose(y[1],y[4]): raise SystemExit('pair-time t sensitivity failed')
    print('[TensorPASS] AbX pair-time: exact zero-start + activated time sensitivity')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--config'); ap.add_argument('--runtime',action='store_true'); ap.add_argument('--tensor',action='store_true'); ap.add_argument('--all-configs',action='store_true'); a=ap.parse_args(); root=pathlib.Path(a.root)
    static(root)
    if a.config: config_check(a.config)
    if a.all_configs:
        for eid in IDS: config_check(root/'scripts/train/configs/R05_CAUSAL_V175'/f'{eid}.json')
    if a.tensor: tensor_checks()
    if a.runtime: print('[V175RuntimePASS] config/runtime contract static checks complete; CUDA/DDP real-data execution not claimed')
if __name__=='__main__': main()
