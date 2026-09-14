#!/usr/bin/env python
"""Self-contained V222 localization contract test.

The test stubs only unrelated project imports, then executes the real
NativeTrunk.distogram_loss_from_native implementation from this package.
"""
import importlib.util
import pathlib
import sys
import types
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "models" / "modules" / "abflow_components.py"

# Minimal import stubs: the contract test does not instantiate SinglePairEncoder.
configs = types.ModuleType("configs")
configs.CDR_TO_ENUM = {"H3": 3}
configs.UNKNOWN_AB_REGION_INDEX = 0
configs.imgt_region_index = lambda *args, **kwargs: 0
configs.normalize_regions = lambda x: [x] if isinstance(x, str) else list(x)
sys.modules["configs"] = configs
utils = types.ModuleType("utils")
nn_utils = types.ModuleType("utils.nn_utils")
class _Dummy: pass
nn_utils.DistogramHead = _Dummy
nn_utils.SinglePairEncoder = _Dummy
nn_utils._atom14_chemical_mask = lambda *a, **k: None
nn_utils._atom14_exists_from_seq = lambda *a, **k: None
nn_utils._torsions_from_atom14 = lambda *a, **k: None
# Real test pseudo-beta: use atom14 slot 1 and mark all packed tokens resolved.
def _pseudo_beta(seq, X, obs=None):
    pb = X[:, :, 1]
    mask = torch.ones(pb.shape[:2], dtype=torch.bool, device=pb.device)
    return pb, mask
nn_utils.pseudo_beta_fn_v2 = _pseudo_beta
sys.modules["utils"] = utils
sys.modules["utils.nn_utils"] = nn_utils

spec = importlib.util.spec_from_file_location("v222_abflow_components", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

obj = mod.NativeTrunk.__new__(mod.NativeTrunk)
nn.Module.__init__(obj)
obj.enable_distogram = True
obj.distogram_pair_scope = "generation_anchored"
obj.distogram_bins = 64
obj.distogram_min = 2.3125
obj.distogram_max = 21.6875
obj.distogram_head = nn.Module()
obj.distogram_head.proj = nn.Linear(1, 1, bias=False)

B, L = 1, 4
# token 0 = generated H3, tokens 1/2 = fixed antibody framework, token 3 = antigen
true_X = torch.zeros(B * L, 14, 3)
true_X[:, 1, 0] = torch.tensor([0.0, 4.0, 8.0, 6.0])
true_S = torch.zeros(B * L, dtype=torch.long)
logits = torch.randn(B, L, L, 64, requires_grad=True)
state = {
    "global_index": torch.arange(L).reshape(1, L),
    "mask": torch.ones(B, L, dtype=torch.bool),
    "atom_exists_padded": torch.ones(B, L, 14, dtype=torch.bool),
    "design_padded": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),  # deliberately broader cmask
    "aux_task_padded": torch.tensor([[1, 0, 0, 0]], dtype=torch.bool),
    "is_antigen_padded": torch.tensor([[0, 0, 0, 1]], dtype=torch.bool),
    "distogram_logits": logits,
}
loss, audit = obj.distogram_loss_from_native(state, true_X, true_S, collect_audit=True)
loss.backward()

# 4 tokens -> 12 resolved directed non-self donor pairs.
assert int(audit["distogram_donor_valid_pairs"].item()) == 12
# Only pairs touching token 0 survive: 6 directed pairs.
assert int(audit["distogram_valid_pairs"].item()) == 6
assert int(audit["distogram_DD_pairs"].item()) == 0
assert int(audit["distogram_DF_pairs"].item()) == 4
assert int(audit["distogram_DA_pairs"].item()) == 2
assert int(audit["distogram_context_context_optimized_pairs"].item()) == 0
assert abs(float(audit["distogram_task_pair_fraction"].item()) - 0.5) < 1e-7

# Gradient support must obey the same contract: context-context logits never
# receive Distogram gradient, while generated-context pairs do.
grad = logits.grad.detach()
assert float(grad[0, 1, 2].abs().sum()) == 0.0
assert float(grad[0, 2, 3].abs().sum()) == 0.0
assert float(grad[0, 0, 1].abs().sum()) > 0.0
assert float(grad[0, 1, 0].abs().sum()) > 0.0
assert float(grad[0, 0, 3].abs().sum()) > 0.0

# Smooth-lDDT localization: construct the same way as AbFlowModel does -- fixed
# context comes from native true_X, only the generated row is a live prediction.
valid_atom = torch.zeros(L, 14, dtype=torch.bool)
valid_atom[:, 1] = True
generation_mask = torch.tensor([1, 0, 0, 0], dtype=torch.bool)
antigen_mask = torch.tensor([0, 0, 0, 1], dtype=torch.bool)
batch_id = torch.zeros(L, dtype=torch.long)
design_leaf = true_X[0:1].clone().detach().requires_grad_(True)
design_pred = design_leaf + 0.75
aux_pred = true_X.clone()
aux_pred[generation_mask] = design_pred
lddt, lddt_audit = mod.design_region_smooth_lddt_loss(
    aux_pred, true_X, valid_atom, generation_mask, batch_id,
    is_antigen_mask=antigen_mask, cutoff=15.0, collect_audit=True,
)
lddt.backward()
assert float(design_leaf.grad.abs().sum()) > 0.0
assert abs(float(lddt_audit["support_weight"].item()) - 6.0) < 1e-7

print("V222_AUX_LOCALIZATION_TEST=PASS")
print("distogram: donor_pairs=12 task_pairs=6 DD=0 DF=4 DA=2 ctxctx=0")
print("smooth_lddt: generated_CA_to_3_context support_weight=6 gradient=live")
