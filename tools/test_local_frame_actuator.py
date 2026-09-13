#!/usr/bin/python
import importlib.util
from pathlib import Path
import torch

MODULE = Path(__file__).resolve().parents[1] / 'models/modules/local_frame_actuator.py'
spec = importlib.util.spec_from_file_location('local_frame_actuator', MODULE)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
Act = mod.LocalFrameFullAtomActuator

def rotation_matrix(dtype=torch.float64):
    A=torch.tensor([[0.2,-0.5,0.8],[0.9,0.3,-0.1],[-0.2,0.7,0.6]],dtype=dtype)
    q,_=torch.linalg.qr(A)
    if torch.linalg.det(q)<0: q[:,0]*=-1
    return q

def make_coords(N=6,C=14,dtype=torch.float64):
    x=torch.randn(N,C,3,dtype=dtype)
    # Guarantee a non-degenerate N-CA-C frame for every residue.
    ca=torch.randn(N,3,dtype=dtype)
    x[:,1]=ca
    x[:,2]=ca+torch.tensor([1.4,0.2,-0.1],dtype=dtype)
    x[:,0]=ca+torch.tensor([-0.4,1.3,0.3],dtype=dtype)
    x[:,3]=ca+torch.tensor([0.1,-0.7,1.1],dtype=dtype)
    return x

torch.manual_seed(7)
N,C,H,A=6,14,32,8
x=make_coords(N,C)
h=torch.randn(N,H,dtype=torch.float64)
attr=torch.randn(N,C,A,dtype=torch.float64)
w=torch.ones(N,C,dtype=torch.float64)
mov=torch.tensor([1,1,1,0,1,0],dtype=torch.bool)
act=Act(H,A,frame_eps=1e-8,coordinate_scale=0.1).double()

# 1) Zero-initialized identity must be exact to numerical precision.
y=act(h,x,attr,w,mov,capture_diagnostics=True)
identity_err=(y-x).abs().max().item()
assert identity_err < 1e-12, identity_err

# 2) Give the actuator non-zero learned weights, then fixed residues must remain exact.
with torch.no_grad():
    act.rigid_head.weight.normal_(0,0.03); act.rigid_head.bias.normal_(0,0.01)
    act.atom_hidden.weight.normal_(0,0.03); act.atom_hidden.bias.normal_(0,0.01)
    act.atom_out.weight.normal_(0,0.03); act.atom_out.bias.normal_(0,0.01)
y=act(h,x,attr,w,mov,capture_diagnostics=True)
fixed_err=(y[~mov]-x[~mov]).abs().max().item()
assert fixed_err < 1e-12, fixed_err

# 3) Backbone N/CA/C/O moves rigidly per residue: internal distances are preserved.
def pd4(z):
    ids=[]
    for a in range(4):
        for b in range(a+1,4): ids.append(torch.linalg.norm(z[:,a]-z[:,b],dim=-1))
    return torch.stack(ids,dim=-1)
bb_err=(pd4(y)-pd4(x)).abs()[mov].max().item()
assert bb_err < 1e-9, bb_err

# 4) Global E(3) equivariance under arbitrary rotation + translation.
Q=rotation_matrix(); t=torch.tensor([2.3,-1.4,0.7],dtype=torch.float64)
x2=torch.einsum('ij,naj->nai',Q,x)+t
z=act(h,x2,attr,w,mov,capture_diagnostics=False)
y_expected=torch.einsum('ij,naj->nai',Q,y)+t
eq_err=(z-y_expected).abs().max().item()
assert eq_err < 2e-9, eq_err

# 5) Invalid local frame: no action, never substitute a global-axis fallback.
x_bad=x.clone(); x_bad[2,0]=x_bad[2,1]; x_bad[2,2]=x_bad[2,1]
y_bad=act(h,x_bad,attr,w,mov,capture_diagnostics=True)
invalid_err=(y_bad[2]-x_bad[2]).abs().max().item()
assert invalid_err < 1e-12, invalid_err

print(f'identity_err={identity_err:.3e}')
print(f'fixed_err={fixed_err:.3e}')
print(f'backbone_rigid_distance_err={bb_err:.3e}')
print(f'equivariance_err={eq_err:.3e}')
print(f'invalid_frame_update_err={invalid_err:.3e}')

# 6) Gradients through learned action and Cartesian state remain finite.
xg=x.clone().requires_grad_(True); hg=h.clone().requires_grad_(True)
yg=act(hg,xg,attr,w,mov,capture_diagnostics=False)
loss=(yg[mov]**2).mean(); loss.backward()
assert torch.isfinite(xg.grad).all()
assert torch.isfinite(hg.grad).all()
print(f'grad_x_rms={xg.grad.square().mean().sqrt().item():.3e}')
print(f'grad_h_rms={hg.grad.square().mean().sqrt().item():.3e}')

print('LOCAL_FRAME_ACTUATOR_TEST=PASS')
