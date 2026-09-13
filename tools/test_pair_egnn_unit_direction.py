#!/usr/bin/env python3
import importlib.util
import math
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / 'models' / 'modules' / 'am_egnn.py'

# Minimal stubs so this standalone contract test does not depend on the full
# AbFlow runtime environment.
torch_scatter = types.ModuleType('torch_scatter')
def scatter_softmax(src, index, dim=0):
    out = torch.empty_like(src)
    for idx in torch.unique(index):
        mask = index == idx
        out[mask] = torch.softmax(src[mask], dim=dim)
    return out
torch_scatter.scatter_softmax = scatter_softmax
sys.modules.setdefault('torch_scatter', torch_scatter)

utils = types.ModuleType('utils')
singleton_mod = types.ModuleType('utils.singleton')
def singleton(cls):
    return cls
singleton_mod.singleton = singleton
sys.modules.setdefault('utils', utils)
sys.modules.setdefault('utils.singleton', singleton_mod)

spec = importlib.util.spec_from_file_location('v218_am_egnn', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
AM_E_GCL = mod.AM_E_GCL
MS_E_GCL = mod.MS_E_GCL
coord2radial = mod.coord2radial

# Load AMEncoder against the same standalone module.
models_pkg = types.ModuleType('models'); models_pkg.__path__ = []
modules_pkg = types.ModuleType('models.modules'); modules_pkg.__path__ = []
sys.modules.setdefault('models', models_pkg)
sys.modules.setdefault('models.modules', modules_pkg)
sys.modules['models.modules.am_egnn'] = mod
enc_spec = importlib.util.spec_from_file_location(
    'models.modules.am_enc', ROOT / 'models' / 'modules' / 'am_enc.py'
)
enc_mod = importlib.util.module_from_spec(enc_spec)
enc_spec.loader.exec_module(enc_mod)
AMEncoder = enc_mod.AMEncoder


def rotation_matrix(dtype):
    axis = torch.tensor([0.3, -0.4, 0.5], dtype=dtype)
    axis = axis / axis.norm()
    theta = torch.tensor(0.73, dtype=dtype)
    x, y, z = axis
    K = torch.stack([
        torch.stack([x*0, -z, y]),
        torch.stack([z, y*0, -x]),
        torch.stack([-y, x, z*0]),
    ])
    I = torch.eye(3, dtype=dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def complete_edges(n):
    row, col = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                row.append(i); col.append(j)
    return torch.tensor(row), torch.tensor(col)


def maxerr(a, b):
    return float((a - b).abs().max().item())


def assert_finite(name, x):
    if not torch.isfinite(x).all():
        raise AssertionError(f'{name} contains non-finite values')


def main():
    torch.manual_seed(7)
    dtype = torch.float64
    n, c = 5, 4
    hidden, channel_nf, radial_nf, edge_nf = 12, 5, 10, 7
    edges = complete_edges(n)
    e = edges[0].numel()

    layer = AM_E_GCL(
        hidden, hidden, hidden, c, channel_nf, radial_nf,
        edges_in_d=edge_nf, dropout=0.0,
        pair_coord_mode='direct_shared', normalize=True, tanh=False,
    ).to(dtype=dtype)
    layer.eval()

    h = torch.randn(n, hidden, dtype=dtype)
    coord = torch.randn(n, c, 3, dtype=dtype)
    channel_attr = torch.randn(n, c, channel_nf, dtype=dtype)
    channel_weights = torch.ones(n, c, dtype=dtype)
    pair = torch.randn(e, edge_nf, dtype=dtype)
    pair_zero = torch.zeros_like(pair)

    # Contract 1: zero-start Pair adapter exactly preserves the pair-free parent.
    h_pair0, x_pair0 = layer(
        h, edges, coord, channel_attr, channel_weights,
        edge_attr=pair, capture_bridge_diagnostics=True,
    )
    h_zero, x_zero = layer(
        h, edges, coord, channel_attr, channel_weights,
        edge_attr=pair_zero, capture_bridge_diagnostics=True,
    )
    cold_h_err = maxerr(h_pair0, h_zero)
    cold_x_err = maxerr(x_pair0, x_zero)
    if cold_h_err != 0.0 or cold_x_err != 0.0:
        raise AssertionError(f'cold-start parent identity failed: h={cold_h_err} x={cold_x_err}')

    # Make Pair live without changing any coordinate-controller formula.
    with torch.no_grad():
        layer.edge_attr_linear.weight.normal_(mean=0.0, std=0.08)
    h_live, x_live = layer(
        h, edges, coord, channel_attr, channel_weights,
        edge_attr=pair, capture_bridge_diagnostics=True,
    )
    _, x_live_zero = layer(
        h, edges, coord, channel_attr, channel_weights,
        edge_attr=pair_zero, capture_bridge_diagnostics=True,
    )
    pair_h_effect = float((h_live - h_zero).abs().max().item())
    pair_x_effect = float((x_live - x_live_zero).abs().max().item())
    if pair_h_effect <= 1e-10 or pair_x_effect <= 1e-10:
        raise AssertionError(
            f'live Pair must affect state and Cartesian update; h={pair_h_effect} x={pair_x_effect}'
        )

    # Contract 2: exact E(3) equivariance of the coordinate update.
    Q = rotation_matrix(dtype)
    t = torch.tensor([1.7, -0.8, 2.3], dtype=dtype)
    coord_rt = torch.einsum('nca,ba->ncb', coord, Q) + t
    h_rt, x_rt = layer(
        h, edges, coord_rt, channel_attr, channel_weights,
        edge_attr=pair, capture_bridge_diagnostics=True,
    )
    x_expected = torch.einsum('nca,ba->ncb', x_live, Q) + t
    equiv_x_err = maxerr(x_rt, x_expected)
    equiv_h_err = maxerr(h_rt, h_live)
    if equiv_x_err > 5e-10 or equiv_h_err > 5e-10:
        raise AssertionError(f'E(3) equivariance failed: x={equiv_x_err} h={equiv_h_err}')

    # Contract 3: distance magnitude is removed from the final vector basis.
    # Hold the invariant edge feature fixed and scale only d_ij by 100x.
    radial, diff = coord2radial(edges, coord, channel_attr, channel_weights, layer.radial_linear)
    state_edge, _ = layer.edge_model(h[edges[0]], h[edges[1]], radial, pair)
    layer.capture_bridge_diagnostics = True
    out_a = layer.coord_model(coord, edges, diff, state_edge, channel_weights, base_edge_feat=None)
    scale = 100.0
    coord_b = coord * scale
    diff_b = diff * scale
    out_b = layer.coord_model(coord_b, edges, diff_b, state_edge, channel_weights, base_edge_feat=None)
    delta_a = out_a - coord
    delta_b = out_b - coord_b
    scale_invariance_err = maxerr(delta_a, delta_b)
    if scale_invariance_err > 5e-10:
        raise AssertionError(f'unit-direction basis still carries distance gain: {scale_invariance_err}')

    # Contract 4: formal V218 has no tanh / scalar hard cap.  A large invariant
    # scalar remains large while the vector basis stays unit norm.
    class ConstantCoeff(torch.nn.Module):
        def __init__(self, channels, value):
            super().__init__(); self.channels = channels; self.value = value
        def forward(self, x):
            return x.new_full((x.shape[0], self.channels), self.value)
    original = layer.coord_mlp
    layer.coord_mlp = ConstantCoeff(c, 5.0)
    layer.capture_bridge_diagnostics = True
    _ = layer.coord_model(coord, edges, diff, state_edge, channel_weights, base_edge_feat=None)
    diag = layer.last_coord_diagnostics
    scalar_absmax = float(diag['coord_coeff_absmax'])
    direction_norm_max = float(diag['coord_direction_norm_absmax'])
    if scalar_absmax < 4.999:
        raise AssertionError('formal scalar authority is unexpectedly bounded')
    if direction_norm_max > 1.0 + 1e-10:
        raise AssertionError(f'unit direction norm > 1: {direction_norm_max}')
    layer.coord_mlp = original

    # Contract 5: gradients stay finite through differentiable normalization.
    h_g = h.clone().requires_grad_(True)
    x_g = coord.clone().requires_grad_(True)
    out_h, out_x = layer(
        h_g, edges, x_g, channel_attr, channel_weights,
        edge_attr=pair, capture_bridge_diagnostics=False,
    )
    loss = out_h.square().mean() + out_x.square().mean()
    loss.backward()
    assert_finite('grad_h', h_g.grad)
    assert_finite('grad_x', x_g.grad)
    grad_h_rms = float(h_g.grad.square().mean().sqrt().item())
    grad_x_rms = float(x_g.grad.square().mean().sqrt().item())

    # Surface layer must expose the same formal controller mode; this catches
    # accidental drift between native/context and antigen-surface EGNN paths.
    surf = MS_E_GCL(
        hidden, hidden, hidden, c, channel_nf, radial_nf, surf_nf=6,
        edges_in_d=edge_nf, dropout=0.0,
        pair_coord_mode='direct_shared', normalize=True, tanh=False,
    ).to(dtype=dtype)
    if surf.pair_coord_mode != 'direct_shared' or not surf.normalize or surf.tanh:
        raise AssertionError('surface EGNN controller contract mismatch')

    # Contract 6: the full AMEncoder direct-shared path executes as a single
    # Pair-conditioned state/geometry stream (surface path may be empty).
    enc = AMEncoder(
        in_node_nf=hidden, hidden_nf=hidden, out_node_nf=hidden,
        n_channel=c, channel_nf=channel_nf, radial_nf=radial_nf,
        in_edge_nf=edge_nf, in_single_nf=6, num_verts=6,
        n_layers=1, dropout=0.0, pair_coord_mode='direct_shared',
        coord_tanh=False, coord_normalize=True,
    ).to(dtype=dtype)
    enc.eval()
    inter_mask = torch.tensor([True, True, False, False, False])
    inter_x = coord[inter_mask].clone()
    inter_edges = complete_edges(int(inter_mask.sum()))
    aligned_edges = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    epi_index = torch.empty(0, dtype=torch.long)
    surf_verts = torch.zeros(0, 6, 3, dtype=dtype)
    update_mask = torch.zeros(n, dtype=torch.bool)
    inter_update_mask = torch.zeros(int(inter_mask.sum()), dtype=torch.bool)
    ctx_pair = torch.randn(edges[0].numel(), edge_nf, dtype=dtype)
    inter_pair = torch.randn(inter_edges[0].numel(), edge_nf, dtype=dtype)
    surf_pair = torch.zeros(0, edge_nf, dtype=dtype)
    single_attr = torch.randn(n, 6, dtype=dtype)
    enc_h, enc_x, enc_inter_x = enc(
        h, coord, edges, inter_mask, inter_x, surf_verts,
        inter_edges, update_mask, inter_update_mask, aligned_edges, epi_index,
        channel_attr, channel_weights, ctx_edge_attr=ctx_pair,
        inter_edge_attr=inter_pair, surf_edge_attr=surf_pair,
        single_attr=single_attr, capture_bridge_diagnostics=True,
    )
    assert_finite('encoder_h', enc_h)
    assert_finite('encoder_x', enc_x)
    assert_finite('encoder_inter_x', enc_inter_x)
    if float(enc.last_coord_diagnostics['pair_direct_shared']) != 1.0:
        raise AssertionError('AMEncoder did not execute the direct-shared path')
    enc_direct_update = float(enc.last_coord_diagnostics['coord_update_absmax_max'])

    print(f'cold_start_h_err={cold_h_err:.3e}')
    print(f'cold_start_x_err={cold_x_err:.3e}')
    print(f'live_pair_h_effect={pair_h_effect:.3e}')
    print(f'live_pair_x_effect={pair_x_effect:.3e}')
    print(f'equivariance_x_err={equiv_x_err:.3e}')
    print(f'equivariance_h_err={equiv_h_err:.3e}')
    print(f'distance_gain_scale100_err={scale_invariance_err:.3e}')
    print(f'unbounded_scalar_absmax={scalar_absmax:.6f}')
    print(f'unit_direction_norm_max={direction_norm_max:.6f}')
    print(f'grad_h_rms={grad_h_rms:.3e}')
    print(f'grad_x_rms={grad_x_rms:.3e}')
    print(f'encoder_direct_update_absmax={enc_direct_update:.3e}')
    print('PAIR_EGNN_UNIT_DIRECTION_TEST=PASS')


if __name__ == '__main__':
    main()
