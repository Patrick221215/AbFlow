#!/usr/bin/python
# -*- coding:utf-8 -*-
"""AbX-style explicit pair-time adapted to AbFlow's native EGNN edge MLP.

This file intentionally REUSES the existing ``am_enc_pair_time.py`` location.
No new pair-time module/head is introduced.

Why this differs from the older G00 prototype
---------------------------------------------
The original AbFlow AM_E_GCL builds an edge message as

    m_ij = edge_mlp([h_i, h_j, radial_ij, edge_attr_ij]).

AbX makes sinusoidal time an explicit pair feature *before* pair reasoning.
Therefore the closest sparse-EGNN adaptation is to extend the FIRST edge-MLP
preactivation with a learned projection of tau(t):

    a_ij = W_base [h_i,h_j,radial] + b + mask * W_t tau(t)
    m_ij = remaining_edge_mlp(a_ij).

``W_t`` is initialized to exact zeros, so step-0 is bitwise the parent function
(up to ordinary floating-point execution of the unchanged base branch) and no
RNG is consumed.  Unlike the previous post-edge additive residual, geometry and
time meet before the edge nonlinearity, allowing F(radial,t) interactions.

This is AbX-*style*, not an exact AbX Seqformer reproduction: AbFlow uses sparse
recomputed EGNN messages rather than a dense persistent pair tensor.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .am_enc import AMEncoder
from .am_egnn import coord2radial, coord_SR


def _pair_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    timesteps = timesteps.reshape(-1).float() * float(max_positions)
    half_dim = embedding_dim // 2
    if half_dim <= 1:
        emb = timesteps[:, None]
        return F.pad(emb, (0, max(0, embedding_dim - 1)))[:, :embedding_dim]
    freq = math.log(max_positions) / float(half_dim - 1)
    freq = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -freq
    )
    emb = timesteps[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


class _PreEdgeTimeFusion(nn.Module):
    """Zero-init extension of the first edge linear layer with tau(t)."""

    def __init__(self, hidden_nf):
        super().__init__()
        self.hidden_nf = int(hidden_nf)
        # Equivalent to appending hidden_nf time channels to the first edge
        # Linear and initializing only the new columns to zero.  torch.zeros is
        # RNG-neutral, preserving all historical parent parameter draws.
        self.weight = nn.Parameter(
            torch.zeros(self.hidden_nf, self.hidden_nf)
        )

    def forward(self, pair_time_attr, *, device, dtype):
        if pair_time_attr is None:
            # Keep parameter in the DDP graph even for disabled families.
            return None, 0.0 * self.weight.sum()
        if pair_time_attr.dim() != 2 or pair_time_attr.shape[-1] != 2:
            raise ValueError(
                "pair_time_attr must be [E,2]=[t,enabled_mask], got "
                f"{tuple(pair_time_attr.shape)}"
            )
        t = pair_time_attr[:, 0].to(device=device)
        enabled = pair_time_attr[:, 1:2].to(device=device, dtype=dtype)
        tau = _pair_timestep_embedding(t, self.hidden_nf).to(
            device=device, dtype=dtype
        )
        delta = F.linear(tau, self.weight.to(dtype=dtype), bias=None)
        return enabled * delta, None


def _edge_model_with_pre_time(base, source, target, radial, pair_time_attr,
                              time_fusion):
    """Run the original edge MLP with zero-init time added pre-nonlinearity."""
    radial = radial.reshape(radial.shape[0], -1)
    base_input = torch.cat([source, target, radial], dim=1)

    # Original AM_E_GCL/MS_E_GCL edge_mlp is:
    # Linear -> act -> Linear -> act.
    first = base.edge_mlp[0](base_input)
    time_delta, dummy = time_fusion(
        pair_time_attr, device=first.device, dtype=first.dtype
    )
    if time_delta is not None:
        first = first + time_delta
    else:
        first = first + dummy
    out = first
    for layer in list(base.edge_mlp.children())[1:]:
        out = layer(out)
    out = base.dropout(out)
    if base.attention:
        out = out * base.att_mlp(out)
    return out


class _TimeWrappedAMEGCL(nn.Module):
    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.time_fusion = _PreEdgeTimeFusion(hidden_nf)

    def forward(self, h, edge_index, coord, channel_attr, channel_weights,
                edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, channel_attr, channel_weights,
            self.base.radial_linear,
        )
        edge_feat = _edge_model_with_pre_time(
            self.base, h[row], h[col], radial, edge_attr, self.time_fusion
        )
        # Faithful to an actual edge_attr concatenation: the same conditioned
        # message drives both coordinate and node updates, exactly as original
        # AM_E_GCL.forward would use its edge_feat.
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, edge_feat, channel_weights
        )
        h, _ = self.base.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord


class _TimeWrappedMSGCL(nn.Module):
    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.time_fusion = _PreEdgeTimeFusion(hidden_nf)

    def forward(self, h, edge_index, epi_index, coord, surf_verts,
                channel_attr, channel_weights, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, abX = coord_SR(
            edge_index, epi_index, coord, surf_verts, channel_attr,
            self.base.scale_linear, self.base.radial_linear,
        )
        edge_feat = _edge_model_with_pre_time(
            self.base, h[row], h[col], radial, edge_attr, self.time_fusion
        )
        coord = self.base.coord_model(
            coord, edge_index, abX, edge_feat, channel_weights
        )
        h, _ = self.base.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord


class AMEncoderPairTime(AMEncoder):
    """Original AMEncoder + explicit pre-edge sinusoidal pair-time."""

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel,
                 channel_nf, radial_nf, in_edge_nf=0, num_verts=50,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, dropout=0.1,
                 dense=False, pair_time_scope="residue"):
        # Build the complete historical parent first to preserve RNG order.
        super().__init__(
            in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
            radial_nf, in_edge_nf=0, num_verts=num_verts, act_fn=act_fn,
            n_layers=n_layers, residual=residual, dropout=dropout, dense=dense,
        )
        scope = str(pair_time_scope).strip().lower()
        if scope not in {"interface", "context", "residue", "all"}:
            raise ValueError(
                "pair_time_scope must be interface, context, residue, or all, got "
                f"{pair_time_scope!r}"
            )
        self.pair_time_scope = scope

        for i in range(self.n_layers):
            if scope in {"context", "residue", "all"}:
                self._modules[f"ctx_gcl_{i}"] = _TimeWrappedAMEGCL(
                    self._modules[f"ctx_gcl_{i}"]
                )
            # Inter GCL contains both local-context and true Ab-Ag edges; the
            # [t,mask] attribute chooses which are enabled per scope.
            self._modules[f"inter_gcl_{i}"] = _TimeWrappedAMEGCL(
                self._modules[f"inter_gcl_{i}"]
            )
            if scope in {"interface", "all"}:
                self._modules[f"surf_gcl_{i}"] = _TimeWrappedMSGCL(
                    self._modules[f"surf_gcl_{i}"]
                )
        if scope in {"context", "residue", "all"}:
            self.out_layer = _TimeWrappedAMEGCL(self.out_layer)

    def forward(self, h, x, ctx_edges, inter_mask, inter_x, surf_verts,
                inter_edges, update_mask, inter_update_mask, aligned_edges,
                epi_index, channel_attr, channel_weights, ctx_edge_attr=None,
                inter_edge_attr=None, surf_edge_attr=None):
        h = self.linear_in(h)
        h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(self.n_layers):
            h, x = self._modules[f"ctx_gcl_{i}"](
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=(ctx_edge_attr if self.pair_time_scope in {"context", "residue", "all"} else None),
            )
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h, inter_x = self._modules[f"inter_gcl_{i}"](
                inter_h, inter_edges, inter_x, inter_channel_attr,
                inter_channel_weights, edge_attr=inter_edge_attr,
            )
            if self.pair_time_scope in {"interface", "all"}:
                inter_h, inter_x = self._modules[f"surf_gcl_{i}"](
                    inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                    inter_channel_attr, inter_channel_weights,
                    edge_attr=surf_edge_attr,
                )
            else:
                inter_h, inter_x = self._modules[f"surf_gcl_{i}"](
                    inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                    inter_channel_attr, inter_channel_weights,
                )
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=(ctx_edge_attr if self.pair_time_scope in {"context", "residue", "all"} else None),
        )
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x, inter_x
