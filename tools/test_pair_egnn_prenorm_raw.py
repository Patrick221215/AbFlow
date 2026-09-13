#!/usr/bin/env python3
import importlib.util, sys, types
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
MOD=ROOT/'models/modules/am_egnn.py'
# Minimal import stubs for the standalone contract test.
ts=types.ModuleType('torch_scatter')
ts.scatter_softmax=lambda src,index,*a,**k: src
sys.modules['torch_scatter']=ts
u=types.ModuleType('utils'); us=types.ModuleType('utils.singleton')
def singleton(cls): return cls
us.singleton=singleton
sys.modules['utils']=u; sys.modules['utils.singleton']=us
spec=importlib.util.spec_from_file_location('am_egnn_v219',MOD)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

torch.manual_seed(7)
layer=m.AM_E_GCL(
    input_nf=16, output_nf=16, hidden_nf=16, n_channel=4,
    channel_nf=4, radial_nf=8, edges_in_d=6, residual=True,
    dropout=0.0, pair_coord_mode='direct_shared', tanh=False,
    normalize=False, coords_agg='mean', coord_prenorm=True,
).double().eval()

# 0) Zero-start Pair bridge is exact; live Pair changes the shared edge message.
source=torch.randn(19,16,dtype=torch.double)
target=torch.randn(19,16,dtype=torch.double)
radial=torch.randn(19,8,dtype=torch.double)
pair=torch.randn(19,6,dtype=torch.double)
state0,base0=layer.edge_model(source,target,radial,pair)
pair_zero_err=(state0-base0).abs().max().item()
assert pair_zero_err == 0.0
with torch.no_grad(): layer.edge_attr_linear.weight.fill_(0.01)
state1,base1=layer.edge_model(source,target,radial,pair)
pair_live_effect=(state1-base1).abs().max().item()
assert pair_live_effect > 0.0

# 1) PreNorm isolates action head from pure representation-amplitude scaling.
feat=torch.randn(31,16,dtype=torch.double)
a,ain,ahid=layer._coord_head_forward(feat)
b,bin,bhid=layer._coord_head_forward(feat*100.0)
prenorm_in_err=(ain-bin).abs().max().item()
prenorm_out_err=(a-b).abs().max().item()
assert prenorm_in_err < 1e-4, prenorm_in_err
assert prenorm_out_err < 1e-4, prenorm_out_err

# 2) Scalar is not tanh/clipped: force a large final readout and confirm >1.
with torch.no_grad():
    layer.coord_mlp[2].weight.fill_(2.0)
large,_,_=layer._coord_head_forward(feat)
scalar_absmax=large.abs().max().item()
assert scalar_absmax > 1.0, scalar_absmax

# 3) Raw R05 vector basis: scaling geometry scales the same fixed-scalar action.
# Use direct coord_model with a zeroed head then set deterministic constant-like hidden mapping.
with torch.no_grad():
    layer.coord_mlp[0].weight.zero_(); layer.coord_mlp[0].bias.fill_(1.0)
    layer.coord_mlp[2].weight.fill_(0.25)
edge_feat=torch.randn(2,16,dtype=torch.double)
coord=torch.tensor([[[0.,0.,0.]]*4, [[1.,0.,0.]]*4],dtype=torch.double)
edge_index=(torch.tensor([0,1]),torch.tensor([1,0]))
coord_diff=coord[edge_index[0]]-coord[edge_index[1]]
weights=torch.ones(2,4,dtype=torch.double)
out1=layer.coord_model(coord.clone(),edge_index,coord_diff,edge_feat,weights)
coord10=coord*10.0
coord_diff10=coord10[edge_index[0]]-coord10[edge_index[1]]
out10=layer.coord_model(coord10.clone(),edge_index,coord_diff10,edge_feat,weights)
d1=out1-coord; d10=out10-coord10
raw_scale_ratio=(d10.abs().max()/d1.abs().max()).item()
assert abs(raw_scale_ratio-10.0)<1e-9, raw_scale_ratio

# 4) E(3) equivariance of the raw-vector update.
Q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.double))
if torch.linalg.det(Q)<0: Q[:,0]*=-1
shift=torch.tensor([2.3,-1.1,0.7],dtype=torch.double)
coord_rt=coord@Q.T+shift
coord_diff_rt=coord_rt[edge_index[0]]-coord_rt[edge_index[1]]
out_rt=layer.coord_model(coord_rt.clone(),edge_index,coord_diff_rt,edge_feat,weights)
expected=out1@Q.T+shift
eq_err=(out_rt-expected).abs().max().item()
assert eq_err < 1e-9, eq_err

# 5) Gradients finite and non-zero through LayerNorm -> scalar head.
layer_g=m.AM_E_GCL(
    input_nf=16, output_nf=16, hidden_nf=16, n_channel=4,
    channel_nf=4, radial_nf=8, edges_in_d=6, residual=True,
    dropout=0.0, pair_coord_mode='direct_shared', tanh=False,
    normalize=False, coords_agg='mean', coord_prenorm=True,
).double().eval()
feat_g=torch.randn(23,16,dtype=torch.double,requires_grad=True)
out_g,_,_=layer_g._coord_head_forward(feat_g)
loss=out_g.square().mean(); loss.backward()
grad_rms=feat_g.grad.square().mean().sqrt().item()
assert torch.isfinite(feat_g.grad).all()
assert grad_rms > 0.0

print(f'pair_zero_start_err={pair_zero_err:.3e}')
print(f'pair_live_edge_effect={pair_live_effect:.3e}')
print(f'prenorm_input_scale100_err={prenorm_in_err:.3e}')
print(f'prenorm_scalar_scale100_err={prenorm_out_err:.3e}')
print(f'unbounded_scalar_absmax={scalar_absmax:.6f}')
print(f'raw_vector_scale10_ratio={raw_scale_ratio:.6f}')
print(f'equivariance_err={eq_err:.3e}')
print(f'grad_input_rms={grad_rms:.3e}')
print('PAIR_EGNN_PRENORM_RAW_TEST=PASS')
