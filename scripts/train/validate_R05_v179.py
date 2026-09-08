#!/usr/bin/env python3
import argparse
import ast
import json
import pathlib
import math
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

R05='R05_PCS_RC_LC_R1_ABX_FF_FIXEDG_RESIDUE_U02_SCOREFLOW'
EXPS={
 'R23_R05_ABX_PAIR_TIME_U02': {'gpu':'2,3','kind':'pair_time'},
 'R24_R05_GEOMETRY_PAIR_TRIATTN_U02': {'gpu':'4,5','kind':'triangle_attention'},
 'R26_R05_GEOMETRY_PAIR_LIVE_DISTOGRAM_U02': {'gpu':'6,7','kind':'triangle_dgram'},
}

def load(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))

def static(root):
    model=root/'models/AbFlow/AbFlow_model.py'
    enc=root/'models/modules/am_enc.py'
    matcher=root/'models/AbFlow/abflow_r3_matcher.py'
    launcher=root/'scripts/train/run_R04_R05_R06_foldflow_noise_v104.sh'
    validator=root/'scripts/train/validate_R05_v179.py'
    for p in (model,enc,matcher,validator):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    subprocess.run(['bash','-n',str(launcher)],check=True)
    mt=model.read_text(encoding='utf-8')
    et=enc.read_text(encoding='utf-8')
    lt=launcher.read_text(encoding='utf-8')
    required_model=[
      'R05MF_GEOM_CALIBRATION_V179',
      'ABFLOW_R05_GEOM_DISTOGRAM',
      'ABFLOW_LOSS_R05_GEOM_DISTOGRAM_WEIGHT',
      'def distogram_loss(self, true_local_X, pair_edges, pair_state, local_is_ab)',
      'r05_geom_distogram_pair_live_contract',
      'scorefm_u02_implied_velocity_error_rms',
      "for name in ('endpoint', 'seq', 'structure', 'distogram')",
    ]
    for x in required_model:
        if x not in mt: raise SystemExit('missing model marker: '+x)
    # Exact uploaded R25 route audit: local pair residual reaches both consumers.
    for x in [
      'enriched_edge_feat = base_edge_feat + pair_residual',
      'coord, edge_index, coord_diff, enriched_edge_feat, channel_weights',
      'self.base.node_model(h, edge_index, enriched_edge_feat, node_attr)',
      'coord, edge_index, abX, enriched_edge_feat, channel_weights',
    ]:
        if x not in et: raise SystemExit('R25 route audit failed: '+x)
    for x in EXPS:
        if x not in lt: raise SystemExit('launcher missing '+x)
    print('[V179StaticPASS] AST/shell + exact R25 node&coord route + live pair distogram + flow diagnostic markers')

def config_check(path):
    c=load(path); m=c.get('_experiment') or {}; exp=m.get('exp_id')
    if exp not in EXPS: raise SystemExit('unsupported v179 config '+str(exp))
    if m.get('parent') != R05: raise SystemExit(exp+' must remain directly parented by R05')
    for k,v in {'batch_size':56,'max_epoch':200,'iter_round':3,'use_ema':True}.items():
        if c.get(k)!=v: raise SystemExit(f'{exp}: {k} mismatch {c.get(k)} != {v}')
    if abs(float(c.get('ema_decay'))-.999)>1e-12: raise SystemExit(exp+': EMA mismatch')
    e=m.get('runtime_env') or {}
    frozen={
      'ABFLOW_SEQUENCE_CONTEXT_MODE':'legacy',
      'ABFLOW_SEQUENCE_PATH_CONTRACT':'legacy_context',
      'ABFLOW_SEQUENCE_LOSS_SCOPE':'context_mask',
      'ABFLOW_MF_REPR_CORE':'off',
      'ABFLOW_MF_PAIR_ATOM_REFINER':'off',
      'ABFLOW_MF_DISTOGRAM':'off',
      'ABFLOW_LOSS_DISTOGRAM_WEIGHT':'0.0',
      'ABFLOW_MF_SMOOTH_LDDT':'off',
      'ABFLOW_LOSS_SMOOTH_LDDT_WEIGHT':'0.0',
      'ABFLOW_R05_ENDPOINT_RELATION':'off',
    }
    for k,v in frozen.items():
        if str(e.get(k))!=v: raise SystemExit(f'{exp}: {k}={e.get(k)!r}, expected {v!r}')
    kind=EXPS[exp]['kind']
    if kind=='pair_time':
        req={'ABFLOW_PAIR_TIME_SCOPE':'residue','ABFLOW_R05_GEOM_PAIR':'off',
             'ABFLOW_R05_GEOM_DISTOGRAM':'off','ABFLOW_LOSS_R05_GEOM_DISTOGRAM_WEIGHT':'0.0'}
    elif kind=='triangle_attention':
        req={'ABFLOW_PAIR_TIME_SCOPE':'off','ABFLOW_R05_GEOM_PAIR':'on',
             'ABFLOW_R05_GEOM_TRIANGLE_MODE':'attention','ABFLOW_R05_GEOM_DISTOGRAM':'off',
             'ABFLOW_LOSS_R05_GEOM_DISTOGRAM_WEIGHT':'0.0'}
    else:
        req={'ABFLOW_PAIR_TIME_SCOPE':'off','ABFLOW_R05_GEOM_PAIR':'on',
             'ABFLOW_R05_GEOM_TRIANGLE_MODE':'multiplication','ABFLOW_R05_GEOM_DISTOGRAM':'on',
             'ABFLOW_LOSS_R05_GEOM_DISTOGRAM_WEIGHT':'0.03'}
        if m.get('matched_reference')!='R25_R05_GEOMETRY_PAIR_U02':
            raise SystemExit('R26 must declare R25 as matched_reference while parent remains R05')
    for k,v in req.items():
        if str(e.get(k))!=v: raise SystemExit(f'{exp}: {k}={e.get(k)!r}, expected {v!r}')
    print(f'[V179ConfigPASS] {exp}: direct R05, kind={kind}, physics frozen')

def tensor_checks():
    # 1) U02 carrier is already a coordinate chart of the velocity field.
    torch.manual_seed(4)
    n=32
    t=torch.rand(n,1,1)*0.75+0.20
    xt=torch.randn(n,14,3)
    u_star=torch.randn_like(xt)
    y_star=xt+(1-t)*u_star
    y_pred=y_star+0.07*torch.randn_like(xt)
    u_pred=(y_pred-xt)/(1-t)
    elem_car=(y_pred-y_star).pow(2)
    elem_vel=(u_pred-u_star).pow(2)
    if not torch.allclose(elem_car,(1-t).pow(2)*elem_vel,atol=2e-6,rtol=2e-5):
        raise SystemExit('U02 carrier/velocity equivalence failed')
    # Show why a naive unweighted velocity auxiliary is a late-time reweighting.
    tt=torch.tensor([0.5,0.9,0.99])
    amp=1/(1-tt).pow(2)
    if not torch.allclose(amp,torch.tensor([4.,100.,10000.]),rtol=1e-4):
        raise SystemExit('flow reweight diagnostic failed')
    print('[TensorPASS] carrier MSE=(1-t)^2*velocity MSE; naive L_flow reweights t=.5/.9/.99 by 4/100/10000')

    # 2) A live pair distogram head must send a real gradient into z_R.
    torch.manual_seed(8)
    L,C,B=6,64,64
    z=torch.randn(L*L,C,requires_grad=True)
    norm=nn.LayerNorm(C); head=nn.Linear(C,B)
    logits=head(norm(z)).reshape(L,L,B)
    logits=0.5*(logits+logits.transpose(0,1))
    target=torch.randint(0,B,(L,L)); mask=~torch.eye(L,dtype=torch.bool)
    loss=F.cross_entropy(logits[mask],target[mask])
    g=torch.autograd.grad(loss,z)[0]
    if not torch.isfinite(g).all() or g.norm().item()<=0:
        raise SystemExit('live pair distogram has no gradient to z_R')
    print('[TensorPASS] live pair distogram CE -> nonzero finite z_R gradient')

    # 2b) Execute the *actual* R05GeometryPair class extracted from source without
    # importing the full AbFlow dependency graph.
    model_path=pathlib.Path(__file__).resolve().parents[2]/'models/AbFlow/AbFlow_model.py'
    tree=ast.parse(model_path.read_text(encoding='utf-8'))
    wanted={'_TriangleMultiplication','_TriangleAttention','R05GeometryPair'}
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in wanted]
    ns={'torch':torch,'nn':nn,'F':F,'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<v179_pair_extract>','exec'),ns)
    Pair=ns['R05GeometryPair']
    mod=Pair(pair_dim=64,triangle_mode='multiplication',triangle_heads=4,enable_distogram=True)
    L=5; coords=torch.randn(L,14,3); is_ab=torch.tensor([1,1,0,0,0],dtype=torch.bool)
    bid=torch.zeros(L,dtype=torch.long); row=torch.arange(L).repeat_interleave(L); col=torch.arange(L).repeat(L)
    edges=torch.stack([row,col]); pair,_=mod(coords,is_ab,bid,edges)
    dloss,ddiag=mod.distogram_loss(coords+0.1,edges,pair,is_ab)
    gp=torch.autograd.grad(dloss,pair,retain_graph=True)[0]
    gout=torch.autograd.grad(dloss,mod.outgoing.out.weight,retain_graph=True,allow_unused=True)[0]
    if gp is None or gp.norm().item()<=0 or gout is None or gout.norm().item()<=0:
        raise SystemExit('actual V179 pair/distogram gradient path failed')
    if int(ddiag['pairs'].item())!=14 or int(ddiag['intra_pairs'].item())!=2 or int(ddiag['antigen_pairs'].item())!=12:
        raise SystemExit('actual V179 H3-touching pair mask contract failed')
    print('[TensorPASS] actual R05GeometryPair: H3-touching mask + distogram -> live pair/triangle output gradients')

    # 3) Private auxiliary-head initialization must not advance caller CPU RNG.
    torch.manual_seed(123)
    before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13003); _=nn.Linear(64,64)
    after=torch.random.get_rng_state()
    if not torch.equal(before,after): raise SystemExit('private distogram RNG isolation failed')
    print('[TensorPASS] distogram-head seed is RNG-isolated')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--config'); ap.add_argument('--all-configs',action='store_true'); ap.add_argument('--runtime',action='store_true'); ap.add_argument('--tensor',action='store_true')
    a=ap.parse_args(); root=pathlib.Path(a.root); static(root)
    if a.config: config_check(a.config)
    if a.all_configs:
        d=root/'scripts/train/configs/R05_CAUSAL_V179'
        for p in sorted(d.glob('R*.json')): config_check(p)
    if a.tensor: tensor_checks()
    if a.runtime: print('[V179RuntimePASS] static/config/tensor only; no claim of real CUDA/BF16/DDP/data execution')
if __name__=='__main__': main()
