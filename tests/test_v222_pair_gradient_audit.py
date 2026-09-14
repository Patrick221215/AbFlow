#!/usr/bin/env python
"""Execute the exact AbFlowModel.compute_pair_gradient_authority method in isolation."""
import ast
import pathlib
import types
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "AbFlow" / "AbFlow_model.py"
tree = ast.parse(MODEL.read_text())
method = None
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "AbFlowModel":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "compute_pair_gradient_authority":
                method = item
                break
assert method is not None
# Turn the class method into a standalone function with the same body/signature.
fn = ast.FunctionDef(
    name="compute_pair_gradient_authority",
    args=method.args,
    body=method.body,
    decorator_list=[],
    returns=method.returns,
    type_comment=None,
)
ast.fix_missing_locations(fn)
ns = {"torch": torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MODEL), "exec"), ns)

class Dummy:
    pass

obj = Dummy()
obj.last_pair_gradient_audit = {}
obj._pair_gradient_audit_error = ""
z = torch.ones((1, 2, 2, 3), requires_grad=True)
obj._pair_gradient_audit_state = {
    "pair_z": z,
    "loss_primary": (z.square()).sum(),
    "loss_endpoint": 0.5 * (z.square()).sum(),
    "loss_structure": 0.25 * (z.square()).sum(),
    "loss_distogram": 3.0 * z.sum(),
    "loss_smooth_lddt": -1.0 * z.sum(),
    "t_min": torch.tensor(0.1),
    "t_mean": torch.tensor(0.5),
    "t_max": torch.tensor(0.9),
}
out = ns["compute_pair_gradient_authority"](obj)
assert float(out["grad_pair_norm_primary"]) > 0
assert float(out["grad_pair_norm_distogram"]) > 0
assert float(out["grad_pair_norm_smooth_lddt"]) > 0
assert abs(float(out["grad_pair_cos_distogram_primary"]) - 1.0) < 1e-6
assert abs(float(out["grad_pair_cos_smooth_lddt_primary"]) + 1.0) < 1e-6
assert abs(float(out["grad_pair_cos_distogram_smooth_lddt"]) + 1.0) < 1e-6
assert z.grad is None  # autograd.grad must not write Parameter/input .grad
assert obj._pair_gradient_audit_state == {}

# Unavailable probe must be non-blocking.
obj._pair_gradient_audit_state = {}
out2 = ns["compute_pair_gradient_authority"](obj)
assert out2 == {}
assert obj._pair_gradient_audit_error
print("V222_PAIR_GRADIENT_AUDIT_TEST=PASS")
print("autograd_grad_no_dot_grad=1 unavailable_nonblocking=1")
