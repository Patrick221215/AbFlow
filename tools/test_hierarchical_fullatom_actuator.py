#!/usr/bin/python
import importlib.util
from pathlib import Path
import torch

MODULE = Path(__file__).resolve().parents[1] / 'models/modules/local_frame_actuator.py'
spec = importlib.util.spec_from_file_location('local_frame_actuator', MODULE)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
Act = mod.HierarchicalLocalFrameFullAtomActuator


def rotation_matrix(dtype=torch.float64):
    A=torch.tensor([[0.2,-0.5,0.8],[0.9,0.3,-0.1],[-0.2,0.7,0.6]],dtype=dtype)
    q,_=torch.linalg.qr(A)
    if torch.linalg.det(q)<0: q[:,0]*=-1
    return q


def make_coords(N=6,C=14,dtype=torch.float64):
    x=torch.randn(N,C,3,dtype=dtype)
    ca=torch.randn(N,3,dtype=dtype)
    x[:,1]=ca
    x[:,2]=ca+torch.tensor([1.4,0.2,-0.1],dtype=dtype)
    x[:,0]=ca+torch.tensor([-0.4,1.3,0.3],dtype=dtype)
    x[:,3]=ca+torch.tensor([0.1,-0.7,1.1],dtype=dtype)
    return x


def pd4(z):
    vals=[]
    for a in range(4):
        for b in range(a+1,4):
            vals.append(torch.linalg.norm(z[:,a]-z[:,b],dim=-1))
    return torch.stack(vals,dim=-1)


torch.manual_seed(7)
N,C,H,A=6,14,32,8
x=make_coords(N,C)
h=torch.randn(N,H,dtype=torch.float64)
attr=torch.randn(N,C,A,dtype=torch.float64)
w=torch.ones(N,C,dtype=torch.float64)
mov=torch.tensor([1,1,1,0,1,0],dtype=torch.bool)
act=Act(H,A,frame_eps=1e-8,coordinate_scale=0.1).double()

# 1) Both zero-initialized heads give exact identity.
y=act(h,x,attr,w,mov,capture_diagnostics=True)
identity_err=(y-x).abs().max().item()
assert identity_err < 1e-12, identity_err

# 2) Non-zero learned action must leave fixed residues exactly unchanged.
with torch.no_grad():
    act.rigid_head.weight.normal_(0,0.03); act.rigid_head.bias.normal_(0,0.01)
    act.atom_hidden.weight.normal_(0,0.03); act.atom_hidden.bias.normal_(0,0.01)
    act.atom_out.weight.normal_(0,0.03); act.atom_out.bias.normal_(0,0.01)
y=act(h,x,attr,w,mov,capture_diagnostics=True)
fixed_err=(y[~mov]-x[~mov]).abs().max().item()
assert fixed_err < 1e-12, fixed_err
ca_internal=float(act.last_diagnostics['ca_internal_update_absmax_A'])
assert ca_internal < 1e-12, ca_internal

# 3) The coarse component by itself is exactly residue-rigid.
atom_mask=w!=0
R,origin,valid=act._frame(x,atom_mask)
hn=act.state_norm(h)
rigid=act.rigid_head(hn)
drot=mod._quat_vec_to_rot(rigid[:,:3]); trans=rigid[:,3:]
zero_internal=torch.zeros_like(x)
rigid_only,_,_=act._apply_local_action(
    x,R,origin,drot,trans,zero_internal,mov & valid,atom_mask)
rigid_err=(pd4(rigid_only)-pd4(x)).abs()[mov & valid].max().item()
assert rigid_err < 1e-9, rigid_err

# 4) Full action is globally E(3)-equivariant.
Q=rotation_matrix(); t=torch.tensor([2.3,-1.4,0.7],dtype=torch.float64)
x2=torch.einsum('ij,naj->nai',Q,x)+t
z=act(h,x2,attr,w,mov,capture_diagnostics=False)
y_expected=torch.einsum('ij,naj->nai',Q,y)+t
eq_err=(z-y_expected).abs().max().item()
assert eq_err < 2e-9, eq_err

# 5) Invalid local frame never falls back to an arbitrary global axis.
x_bad=x.clone(); x_bad[2,0]=x_bad[2,1]; x_bad[2,2]=x_bad[2,1]
y_bad=act(h,x_bad,attr,w,mov,capture_diagnostics=True)
invalid_err=(y_bad[2]-x_bad[2]).abs().max().item()
assert invalid_err < 1e-12, invalid_err

# 6) Expressivity theorem in code: any atom14 target is reachable with a valid
# current frame.  Set DeltaR=I, choose translation to target CA, and use local
# internal corrections for every non-CA atom.
source=make_coords(N,C)
target=torch.randn_like(source)
# target itself need not have canonical geometry; this is a pure parameterization test.
R0,o0,v0=act._frame(source,torch.ones(N,C,dtype=torch.bool))
assert bool(v0.all())
I=torch.eye(3,dtype=source.dtype).expand(N,3,3).clone()
target_ca=target[:,1]
trans_local=torch.einsum('nj,njk->nk',target_ca-o0,R0)
source_local=torch.einsum('naj,njk->nak',source-o0[:,None,:],R0)
target_local_about_target_ca=torch.einsum(
    'naj,njk->nak', target-target_ca[:,None,:], R0)
internal=target_local_about_target_ca-source_local
internal[:,1]=0.0
reconstructed,_,_=act._apply_local_action(
    source,R0,o0,I,trans_local,internal,
    torch.ones(N,dtype=torch.bool),torch.ones(N,C,dtype=torch.bool))
expressivity_err=(reconstructed-target).abs().max().item()
assert expressivity_err < 2e-9, expressivity_err

# 7) Unlike V216 rigid-only backbone, the learned internal branch can change
# backbone internal geometry (this is capability, not a required magnitude).
bb_change=(pd4(y)-pd4(rigid_only)).abs()[mov & valid].max().item()
assert bb_change > 1e-10, bb_change

# 8) Gradients through coordinate state and learned heads remain finite.
xg=x.clone().requires_grad_(True); hg=h.clone().requires_grad_(True)
yg=act(hg,xg,attr,w,mov,capture_diagnostics=False)
loss=(yg[mov]**2).mean(); loss.backward()
assert torch.isfinite(xg.grad).all()
assert torch.isfinite(hg.grad).all()

print(f'identity_err={identity_err:.3e}')
print(f'fixed_err={fixed_err:.3e}')
print(f'ca_internal_update_A={ca_internal:.3e}')
print(f'coarse_rigid_distance_err={rigid_err:.3e}')
print(f'equivariance_err={eq_err:.3e}')
print(f'invalid_frame_update_err={invalid_err:.3e}')
print(f'full_atom14_expressivity_err={expressivity_err:.3e}')
print(f'backbone_internal_capability={bb_change:.3e}')
print(f'grad_x_rms={xg.grad.square().mean().sqrt().item():.3e}')
print(f'grad_h_rms={hg.grad.square().mean().sqrt().item():.3e}')
print('HIERARCHICAL_FULLATOM_ACTUATOR_TEST=PASS')
