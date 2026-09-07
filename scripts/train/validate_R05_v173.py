#!/usr/bin/env python3
"""V173 exact configuration + CPU tensor contracts; not an end-to-end run.

--tensor executes selected AST definitions from the delivered model, without
mocking or importing missing project dependencies. --project-import additionally
imports the real project when those dependencies are installed. Runtime validation
is fail-closed and occurs before expensive training.
"""
import argparse, ast, json, math, os, subprocess, sys, hashlib, copy
from pathlib import Path


def validate_config(root, path, runtime=False):
    contract = json.loads((root/'scripts/train/configs/R05_CAUSAL_V173/EXPERIMENT_CONTRACT.json').read_text())
    cfg = json.loads(path.read_text()); meta=cfg.get('_experiment', {})
    name=meta.get('exp_id'); assert name in contract['experiments'], 'Unknown experiment'
    assert meta.get('parent')==contract['parent'], 'Not a direct R05 child'
    assert meta.get('implementation_revision')=='v173_r05_causal_modules'
    assert {k:v for k,v in cfg.items() if k!='_experiment'}==contract['parent_training'], 'Parent training settings changed'
    env=meta['runtime_env']; expected=dict(contract['parent_env'])
    expected.update(contract['experiments'][name]['allowed_env_delta'])
    assert env==expected, 'Unexpected module combination or environment change'
    for rel, marker in [('models/AbFlow/AbFlow_model.py','R05MF_CAUSAL_MODULES_V173'),
                        ('models/modules/am_enc.py','presence = None'),
                        ('trainer/AbFlow_trainer.py','R05ModuleAudit:Validation')]:
        assert marker in (root/rel).read_text(), 'Incomplete overlay: '+rel
    if runtime:
        for key,value in env.items():
            assert os.environ.get(key)==value, 'Runtime/config mismatch: '+key
        locked={'ABFLOW_SOURCE_MODE':'pcs_rc','ABFLOW_RECURRENT_PROPOSAL_CONTEXT':'on',
                'ABFLOW_ABX_COMMON_CENTER':'on','ABFLOW_R3_NOISE_SCOPE':'residue',
                'ABFLOW_R3_FIXED_G_SCALED':'0.1','ABFLOW_R3_FLOW_COORDINATE_SCALING':'0.1',
                'ABFLOW_SCOREFM_LOSS_MODE':'f01_r3_endpoint_canonical_hybrid',
                'ABFLOW_SCOREFM_SAMPLER_MODE':'f01_canonical_carrier',
                'ABFLOW_FINAL_READOUT_MODE':'integrated_endpoint',
                'ABFLOW_DUAL_SEQUENCE_STATE':'off','ABFLOW_DDP_COST_BALANCED':'off',
                'ABFLOW_PAIR_TIME_SCOPE':'off','ABFLOW_R3_SCORE_DSM_WEIGHT':'0.0',
                'ABFLOW_R3_PATHFLOW_WEIGHT':'0.0','ABFLOW_AUTO_TOPK_EVAL':'off',
                'ABFLOW_EPOCH_TEST':'on','ABFLOW_EPOCH_TEST_INTERVAL':'1',
                'ABFLOW_EPOCH_TEST_N_STEPS':'10'}
        for k,v in locked.items(): assert os.environ.get(k)==v, 'R05 invariant changed: '+k
        assert os.environ.get('ABFLOW_MAX_EPOCH','') in ('','200'), 'Formal protocol is 200 epochs'
        assert os.environ.get('ABFLOW_BOUNDARY_PCGRAD','off') in ('off','0','false'), 'PCGrad must stay off'
    delta={k:[contract['parent_env'][k],v] for k,v in env.items() if contract['parent_env'][k]!=v}
    print('[V173ConfigPASS]', name, json.dumps(delta,ensure_ascii=False,sort_keys=True))


def load_selected(root):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    env={'torch':torch,'nn':nn,'F':F,'math':math}
    tree=ast.parse((root/'models/AbFlow/AbFlow_model.py').read_text())
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            if node.name in ('_TriangleMultiplication','R05GeometryPair'):
                selected.append(node)
            if node.name in ('MFRepresentationEnrichment','AbFlowModel'):
                names={'design_region_smooth_lddt_loss','_r05_decode_endpoint','_sample_categorical_path'}
                for child in node.body:
                    if isinstance(child,ast.FunctionDef) and child.name in names:
                        child=copy.deepcopy(child); child.decorator_list=[]
                        selected.append(child)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(root/'models/AbFlow/AbFlow_model.py'),'exec'),env)
    return env


def tensor_checks(root):
    import torch
    torch.set_num_threads(1)
    env=load_selected(root); decode=env['_r05_decode_endpoint']; relation=env['design_region_smooth_lddt_loss']; Pair=env['R05GeometryPair']
    torch.manual_seed(2023)
    # 1. Exact U02 chart inversion across both sides of t=.2, endpoints included.
    t=torch.tensor([0.,.05,.1999,.2,.4,.9,1.]).reshape(-1,1,1)
    x0=torch.randn(7,14,3);x1=torch.randn_like(x0)
    noise=torch.randn(7,1,3)*torch.sqrt(t*(1-t))
    xt=(1-t)*x0+t*x1+noise
    carrier=torch.where(t>=.2,x1+noise/(2*t.clamp_min(.2)),x1).requires_grad_()
    endpoint=decode(carrier,xt,x0,t)
    assert torch.allclose(endpoint,x1,atol=2e-6)
    grad=torch.autograd.grad(endpoint.sum(),carrier)[0]
    assert torch.equal(grad,torch.where(t>=.2,torch.full_like(carrier,2),torch.ones_like(carrier)))
    print('[TensorPASS] U02 endpoint inversion t=0/.05/.1999/.2/.4/.9/1 and Jacobian 1 or 2')
    # 2. No new grads into source or GT through target/context construction;
    # test the loss primitive with a design/context universe including missing slots.
    truth=torch.tensor([[[0.,0,0],[0,1,0]],[[3.,0,0],[3,1,0]],[[0.,4,0],[0,5,0]],[[0.,0,4],[0,1,4]]])
    design=torch.tensor([1,1,0,0],dtype=torch.bool);ag=torch.tensor([0,0,0,1],dtype=torch.bool)
    valid=torch.ones(4,2,dtype=torch.bool);bid=torch.zeros(4,dtype=torch.long)
    perfect,diag=relation(truth.clone(),truth,valid,design,bid,ag,metric='pseudo_huber')
    assert perfect.item()==0 and diag['perfect_floor'].item()==0
    moved=truth.clone();moved[design]+=torch.tensor([40.,0,0]);moved.requires_grad_()
    loss,_=relation(moved,truth,valid,design,bid,ag,metric='pseudo_huber')
    g=torch.autograd.grad(loss,moved)[0]
    assert g[design].norm()>0.01 and torch.equal(g[~design],torch.zeros_like(g[~design]))
    smooth,_=relation(moved,truth,valid,design,bid,ag)
    gs=torch.autograd.grad(smooth,moved)[0]
    assert gs[design].norm()<1e-4
    assert g[design,0,0].mean()>0, 'gradient descent should move the displaced design back'
    translation=torch.tensor([300.,-10.,20.])
    translated,_=relation(moved+translation,truth+translation,valid,design,bid,ag,metric='pseudo_huber')
    assert torch.allclose(loss,translated,atol=1e-5)
    q,_=torch.linalg.qr(torch.randn(3,3))
    rotated,_=relation(moved@q,truth@q,valid,design,bid,ag,metric='pseudo_huber')
    assert torch.allclose(loss,rotated,atol=2e-5)
    padded=torch.cat([moved.detach(),torch.ones(4,1,3)*999],1)
    truth_pad=torch.cat([truth,torch.ones(4,1,3)*-999],1)
    loss_pad,_=relation(padded,truth_pad,torch.cat([valid,torch.zeros(4,1,dtype=torch.bool)],1),design,bid,ag,metric='pseudo_huber')
    assert torch.allclose(loss,loss_pad,atol=1e-5)
    empty,_=relation(moved,truth,valid,torch.zeros_like(design),bid,ag,metric='pseudo_huber')
    assert empty.item()==0
    # End-to-end *loss path* connection to carrier, not a full model run.
    y=carrier.detach().clone().requires_grad_(); y_shift=y+1.
    ep=decode(y_shift,xt,x0,t)
    full=torch.cat([ep,torch.zeros(7,14,3)],0)
    native=torch.cat([x1,torch.zeros_like(x1)],0)
    ds=torch.cat([torch.ones(7),torch.zeros(7)]).bool();ids=torch.arange(7).repeat(2)
    l,_=relation(full,native,torch.ones(14,14,dtype=torch.bool),ds,ids,~ds,metric='pseudo_huber')
    assert torch.autograd.grad(l,y)[0].norm()>0
    print('[TensorPASS] relation oracle=0; 40A error gradient nonzero; fixed-context/padding masks; rigid invariance; carrier gradient')
    # 3. Geometry representation uses two disconnected complete graph universes.
    def edges(nodes):
        return torch.stack([nodes[:,None].expand(-1,len(nodes)).reshape(-1),nodes[None,:].expand(len(nodes),-1).reshape(-1)])
    coords=torch.randn(7,14,3);roles=torch.tensor([1,1,0,0,1,0,0],dtype=torch.bool)
    gids=torch.tensor([0,0,0,0,1,1,1]);edge=torch.cat([edges(torch.arange(4)),edges(torch.arange(4,7))],1)
    before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]): pair=Pair(64)
    assert torch.equal(before,torch.random.get_rng_state())
    z,diag=pair(coords,roles,gids,edge)
    z_transform,_=pair(coords@q+translation,roles,gids,edge)
    assert torch.allclose(z,z_transform,atol=2e-5,rtol=1e-4)
    other=coords.clone();other[gids==1]+=torch.randn_like(other[gids==1])*50
    z_other,_=pair(other,roles,gids,edge)
    assert torch.equal(z[:16],z_other[:16]), 'cross-complex mixing'
    coord_grad=coords.clone().requires_grad_();zx,_=pair(coord_grad,roles,gids,edge)
    assert torch.autograd.grad(zx.sum(),coord_grad,allow_unused=True)[0] is None
    query=torch.tensor([[0,0,4,1],[1,4,6,3]])
    attrs=pair.gather(edge,z,query,7)
    assert attrs.shape==(4,65) and torch.equal(attrs[:,-1],torch.tensor([1.,0.,1.,1.]))
    projection=torch.nn.Parameter(torch.zeros(8,64));norm=torch.nn.LayerNorm(64)
    base=torch.randn(4,8);res=torch.nn.functional.linear(norm(attrs[:,:-1]),projection)*attrs[:,-1:]
    assert torch.equal(base+res,base)
    loss=(base+res).square().mean();g1=torch.autograd.grad(loss,projection)[0]
    assert g1.norm()>0
    with torch.no_grad():projection-=0.1*g1;norm.bias.fill_(2.)
    zx,_=pair(coords,roles,gids,edge);attrs=pair.gather(edge,zx,query,7)
    res=torch.nn.functional.linear(norm(attrs[:,:-1]),projection)*attrs[:,-1:]
    assert torch.equal(res[1],torch.zeros_like(res[1])), 'missing pair must remain absent after learned LN bias'
    grads=torch.autograd.grad((base+res).square().mean(),[pair.input[0].weight,pair.outgoing.out.weight],allow_unused=True)
    assert all(g is not None and g.norm()>0 for g in grads)
    print('[TensorPASS] geometry invariance/graph isolation/RNG isolation; missing-pair mask; zero-start parity; subsequent pair/triangle gradients')
    # 4. Actual delivered wrapper receives the presence bit and preserves base edge.
    wrapper_tree=ast.parse((root/'models/modules/am_enc.py').read_text())
    chosen=[x for x in wrapper_tree.body if isinstance(x,ast.ClassDef) and x.name=='_PairResidualAMEGCL']
    def radial(e,c,a,w,r):return torch.ones(e.shape[1],1),torch.ones(e.shape[1],1,3)
    wenv=dict(env,coord2radial=radial)
    exec(compile(ast.Module(body=chosen,type_ignores=[]),'am_enc.py','exec'),wenv)
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__();self.edge_mlp=torch.nn.Sequential(torch.nn.Linear(3,8));self.radial_linear=None
        def edge_model(self,h1,h2,r,edge_attr=None):return self.edge_mlp(torch.cat([h1,h2,r],-1))
        def coord_model(self,c,e,d,m,w):self.last_message=m;return c+m.sum()*0.01
        def node_model(self,h,e,m,a):return h+m.sum()*0.01,None
    base_module=Base();wrapped=wenv['_PairResidualAMEGCL'](base_module,64)
    h=torch.randn(7,1);coord=torch.randn(7,1,3)
    parent_msg=base_module.edge_model(h[query[0]],h[query[1]],torch.ones(4,1))
    wrapped(h,query,coord,None,None,edge_attr=attrs)
    assert torch.equal(parent_msg,base_module.last_message)
    with torch.no_grad():wrapped.pair_weight.fill_(.1);wrapped.pair_norm.bias.fill_(2.)
    wrapped(h,query,coord,None,None,edge_attr=attrs)
    assert torch.equal(parent_msg[1],base_module.last_message[1])
    print('[TensorPASS] delivered edge wrapper presence semantics (synthetic base; not original AMEGNN parity)')
    # 5. Audit noisy branches without altering preexisting random draw or sampled St.
    from types import SimpleNamespace
    sampler=env['_sample_categorical_path'];obj=SimpleNamespace(deterministic_validation=False,training=True)
    native=torch.tensor([1,2,3,4]);source=torch.tensor([1,0,0,0]);mask=torch.ones(4,dtype=torch.bool)
    rng=torch.random.get_rng_state();a=sampler(obj,native,source,torch.tensor([.3]),torch.zeros(4,dtype=torch.long),mask)
    after=torch.random.get_rng_state();torch.random.set_rng_state(rng)
    b,noisy=sampler(obj,native,source,torch.tensor([.3]),torch.zeros(4,dtype=torch.long),mask,True)
    assert torch.equal(a,b) and torch.equal(after,torch.random.get_rng_state())
    _,branch=sampler(obj,native,source,torch.tensor([0.]),torch.zeros(4,dtype=torch.long),mask,True)
    assert branch[0] and native[0]==source[0]
    print('[TensorPASS] branch audit preserves original RNG/St, including coincident proposal/native tokens')
    print('[V173TensorPASS] torch='+torch.__version__+' device=CPU; no full model / dataset / CUDA-DDP test')


def main():
    if not __debug__:
        raise SystemExit("Validation requires assertions enabled; remove Python -O/PYTHONOPTIMIZE")
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--config',type=Path);ap.add_argument('--runtime',action='store_true');ap.add_argument('--tensor',action='store_true');ap.add_argument('--project-import',action='store_true');args=ap.parse_args();root=args.root.resolve()
    for rel in ('models/AbFlow/AbFlow_model.py','models/AbFlow/abflow_r3_matcher.py','models/modules/am_enc.py','trainer/AbFlow_trainer.py','trainer/abs_trainer.py'):
        tree=ast.parse((root/rel).read_text(),filename=rel)
        if rel.endswith('AbFlow_model.py'):
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AbFlowModel')
            assert any(isinstance(n,ast.FunctionDef) and n.name=='forward' for n in cls.body), 'forward is outside AbFlowModel'
    subprocess.run(['bash','-n',str(root/'scripts/train/run_R04_R05_R06_foldflow_noise_v104.sh')],check=True)
    configs=[args.config] if args.config else sorted((root/'scripts/train/configs/R05_CAUSAL_V173').glob('R*.json'))
    assert len(configs)>0
    for cfg in configs:validate_config(root,cfg,args.runtime)
    if args.tensor:tensor_checks(root)
    if args.project_import:
        sys.path.insert(0,str(root));__import__('models.AbFlow.AbFlow_model')
        print('[ProjectImportPASS] actual repository import only; forward/backward not executed')
    print('[V173StaticPASS] Python syntax, class structure, launcher syntax and allowed experiment deltas')

if __name__=='__main__':main()
