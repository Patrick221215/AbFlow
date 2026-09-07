#!/usr/bin/python
# -*- coding:utf-8 -*-
"""AbX-style zero-start pair-time representation conditioning for AbFlow R05.

Scientific contract
-------------------
AbX (ICML 2024) embeds continuous diffusion time and exposes it to both residue
and residue-pair representations.  R05 already exposes time at node level. This
module adds the missing *pair-level* access while preserving the validated R05
coordinate authority.

For each enabled residue edge e=(i,j):

    tau(t) = [sin(w_k * 10000 t), cos(w_k * 10000 t)]_k
    delta_m_e(t) = W_t tau(t)
    m_sem_e(t) = m_base_e + mask_e * delta_m_e(t)

W_t is initialized to exactly zero with torch.zeros, so step-0 is the exact
parent function and no RNG is consumed by the new parameters. Critically, the
same-layer coordinate MLP receives m_base, NOT m_sem. Time therefore enters as
representation context, not as a hand-written geometric force. Later geometry
can still react through updated node states, which is the intended learned path.

Scope 'residue' mirrors the project AbX adaptation: context residue edges and
true residue-residue interface edges see time; antigen-surface vertices do not,
because they are not residue-pair entities in AbX semantics.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .am_enc import AMEncoder
from .am_egnn import coord2radial, coord_SR


def _abx_timestep_embedding(t, dim=32, max_positions=10000):
    """Match the supplied AbX sinusoidal timestep convention."""
    t = t.reshape(-1).float() * float(max_positions)
    half = dim // 2
    if half <= 1:
        raise ValueError('pair-time embedding dim must be >= 4')
    scale = math.log(float(max_positions)) / float(half - 1)
    freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-scale))
    phase = t[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _ZeroStartPairTimeBias(nn.Module):
    def __init__(self, hidden_nf, time_dim=32):
        super().__init__()
        self.time_dim = int(time_dim)
        # IMPORTANT: no nn.Linear constructor -> no random draw -> parent RNG parity.
        self.weight = nn.Parameter(torch.zeros(int(hidden_nf), self.time_dim))
        self.bias = nn.Parameter(torch.zeros(int(hidden_nf)))
        self._last_diag = {}

    def forward(self, base_edge_feat, pair_time_attr):
        self._last_diag = {}
        if pair_time_attr is None:
            return base_edge_feat
        if pair_time_attr.dim() != 2 or pair_time_attr.shape[-1] != 2:
            raise ValueError('pair_time_attr must be [E,2]=[t,enabled_mask]')
        t = pair_time_attr[:, 0].to(base_edge_feat.device)
        enabled = pair_time_attr[:, 1:2].to(
            device=base_edge_feat.device, dtype=base_edge_feat.dtype)
        tau = _abx_timestep_embedding(t, self.time_dim).to(base_edge_feat.dtype)
        residual = F.linear(tau, self.weight.to(base_edge_feat.dtype), self.bias.to(base_edge_feat.dtype))
        residual = residual * enabled
        semantic = base_edge_feat + residual
        with torch.no_grad():
            base_rms = torch.sqrt(base_edge_feat.detach().float().square().mean() + 1e-8)
            res_rms = torch.sqrt(residual.detach().float().square().mean() + 1e-8)
            self._last_diag = {
                'enabled_fraction': enabled.detach().float().mean(),
                'time_mean': t.detach().float().mean() if t.numel() else base_rms.new_tensor(0.),
                'base_rms': base_rms,
                'residual_rms': res_rms,
                'residual_to_base': res_rms / base_rms.clamp_min(1e-8),
                'weight_rms': torch.sqrt(self.weight.detach().float().square().mean() + 1e-12),
            }
        return semantic


class _TimeSemanticAMEGCL(nn.Module):
    """Time changes the node/semantic message; same-layer coordinate force is parent R05."""
    def __init__(self, base_gcl, time_dim=32):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.time_bias = _ZeroStartPairTimeBias(hidden_nf, time_dim)

    def forward(self, h, edge_index, coord, channel_attr, channel_weights,
                edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, channel_attr, channel_weights, self.base.radial_linear)
        base_edge = self.base.edge_model(h[row], h[col], radial, edge_attr=None)
        # Preserve same-layer coordinate authority exactly.
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, base_edge, channel_weights)
        semantic_edge = self.time_bias(base_edge, edge_attr)
        h, _ = self.base.node_model(h, edge_index, semantic_edge, node_attr)
        return h, coord


class _TimeSemanticMSGCL(nn.Module):
    def __init__(self, base_gcl, time_dim=32):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.time_bias = _ZeroStartPairTimeBias(hidden_nf, time_dim)

    def forward(self, h, edge_index, epi_index, coord, surf_verts,
                channel_attr, channel_weights, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, abX = coord_SR(
            edge_index, epi_index, coord, surf_verts, channel_attr,
            self.base.scale_linear, self.base.radial_linear)
        base_edge = self.base.edge_model(h[row], h[col], radial, edge_attr=None)
        coord = self.base.coord_model(
            coord, edge_index, abX, base_edge, channel_weights)
        semantic_edge = self.time_bias(base_edge, edge_attr)
        h, _ = self.base.node_model(h, edge_index, semantic_edge, node_attr)
        return h, coord


class AMEncoderPairTime(AMEncoder):
    """Original AMEncoder plus AbX-style explicit residue-pair time context."""
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, num_verts=50, act_fn=nn.SiLU(), n_layers=4,
                 residual=True, dropout=0.1, dense=False, pair_time_scope='residue',
                 time_dim=32):
        # Construct complete parent first: all original random params retain order.
        super().__init__(in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                         radial_nf, in_edge_nf=in_edge_nf, num_verts=num_verts,
                         act_fn=act_fn, n_layers=n_layers, residual=residual,
                         dropout=dropout, dense=dense)
        scope = str(pair_time_scope).strip().lower()
        if scope not in {'interface','context','residue','all'}:
            raise ValueError('pair_time_scope must be interface/context/residue/all')
        self.pair_time_scope = scope
        self.pair_time_dim = int(time_dim)
        for i in range(self.n_layers):
            if scope in {'context','residue','all'}:
                self._modules[f'ctx_gcl_{i}'] = _TimeSemanticAMEGCL(
                    self._modules[f'ctx_gcl_{i}'], self.pair_time_dim)
            # local inter GCL contains local-context + true Ab-Ag edges; upstream
            # [t,mask] decides which subset is active.
            self._modules[f'inter_gcl_{i}'] = _TimeSemanticAMEGCL(
                self._modules[f'inter_gcl_{i}'], self.pair_time_dim)
            if scope in {'interface','all'}:
                self._modules[f'surf_gcl_{i}'] = _TimeSemanticMSGCL(
                    self._modules[f'surf_gcl_{i}'], self.pair_time_dim)
        if scope in {'context','residue','all'}:
            self.out_layer = _TimeSemanticAMEGCL(self.out_layer, self.pair_time_dim)

    def pair_time_diagnostics(self):
        vals = {}
        mods = []
        for i in range(self.n_layers):
            for prefix in ('ctx_gcl_','inter_gcl_','surf_gcl_'):
                m = self._modules.get(f'{prefix}{i}')
                if m is not None and hasattr(m, 'time_bias'):
                    mods.append(m.time_bias)
        if hasattr(self.out_layer, 'time_bias'):
            mods.append(self.out_layer.time_bias)
        for m in mods:
            for k,v in getattr(m, '_last_diag', {}).items():
                if torch.is_tensor(v) and v.numel()==1:
                    vals.setdefault(k, []).append(v.detach().float())
        return {k: torch.stack(v).mean() for k,v in vals.items() if v}

    def forward(self, h, x, ctx_edges, inter_mask, inter_x, surf_verts, inter_edges,
                update_mask, inter_update_mask, aligned_edges, epi_index,
                channel_attr, channel_weights, ctx_edge_attr=None,
                inter_edge_attr=None, surf_edge_attr=None):
        h = self.linear_in(h); h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]
        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(self.n_layers):
            ctx_attr = ctx_edge_attr if self.pair_time_scope in {'context','residue','all'} else None
            h, x = self._modules[f'ctx_gcl_{i}'](
                h, ctx_edges, x, channel_attr, channel_weights, edge_attr=ctx_attr)
            inter_h = inter_h.clone(); inter_h[inter_update_mask] = h[update_mask]
            inter_h, inter_x = self._modules[f'inter_gcl_{i}'](
                inter_h, inter_edges, inter_x, inter_channel_attr,
                inter_channel_weights, edge_attr=inter_edge_attr)
            surf_attr = surf_edge_attr if self.pair_time_scope in {'interface','all'} else None
            inter_h, inter_x = self._modules[f'surf_gcl_{i}'](
                inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                inter_channel_attr, inter_channel_weights, edge_attr=surf_attr)
            h = h.clone(); h[inter_mask] = inter_h
            ctx_states.append(h); ctx_coords.append(x); inter_coords.append(inter_x)
        ctx_attr = ctx_edge_attr if self.pair_time_scope in {'context','residue','all'} else None
        h, x = self.out_layer(h, ctx_edges, x, channel_attr, channel_weights, edge_attr=ctx_attr)
        ctx_states.append(h); ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, -1)
            x = torch.mean(torch.stack(ctx_coords), 0)
            inter_x = torch.mean(torch.stack(inter_coords), 0)
        h = self.dropout(h); h = self.linear_out(h)
        return h, x, inter_x
