#!/usr/bin/python
# -*- coding:utf-8 -*-
"""AbX-inspired zero-start semantic pair-time conditioning for AbFlow.

This file is the existing ``am_enc_pair_time.py`` implementation, revised in
place.  No new pair-time module is introduced.

AbX does not multiply a geometric force by raw time.  It sinusoidally embeds
the continuous timestep and appends that representation to sequence/pair
features before Seqformer reasoning.  AbFlow already has node-level sinusoidal
time conditioning, so this module tests the missing pair-level counterpart on
true antibody-antigen interface messages.

For each enabled edge:
    tau(t) = fixed sinusoidal embedding of the graph-level continuous time
    m_sem  = m_base + tau(t) * a_l
where a_l is a zero-initialized learnable channel scale.

Crucially:
    m_coord = m_base
    m_node  = m_sem

Thus pair-time does NOT directly enter the coordinate MLP of the same wrapped
GCL.  It can still influence later coordinates indirectly through the updated
node representation, which is the intended representation-conditioning
mechanism.

The new parameters are created with ``torch.zeros`` only, so initialization is
RNG-neutral and the step-0 function equals the original AMEncoder.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .am_enc import AMEncoder
from .am_egnn import coord2radial, coord_SR


def _pair_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """Fixed sinusoidal embedding matching the AbX/standard diffusion style."""
    timesteps = timesteps.reshape(-1).float() * float(max_positions)
    half_dim = embedding_dim // 2
    if half_dim <= 1:
        emb = timesteps[:, None]
        return F.pad(emb, (0, max(0, embedding_dim - 1)))[:, :embedding_dim]
    freq = math.log(max_positions) / float(half_dim - 1)
    freq = torch.exp(
        torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        ) * -freq
    )
    emb = timesteps[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode="constant")
    return emb


class _ChannelTimeScale(nn.Module):
    """Zero-start AbX-style semantic time residual."""

    def __init__(self, hidden_nf):
        super().__init__()
        self.hidden_nf = int(hidden_nf)
        # Same parameter budget as the historical pair-time prototype:
        # one learned scale per hidden channel, initialized without RNG.
        self.scale = nn.Parameter(torch.zeros(self.hidden_nf))

    def forward(self, edge_feat, pair_time_attr):
        if pair_time_attr is None:
            dummy = self.scale.sum()
            return edge_feat + 0.0 * dummy
        if pair_time_attr.dim() != 2 or pair_time_attr.shape[-1] != 2:
            raise ValueError(
                "pair_time_attr must have shape [E,2]=[t,enabled_mask], got "
                f"{tuple(pair_time_attr.shape)}"
            )
        t = pair_time_attr[:, 0].to(device=edge_feat.device)
        enabled = pair_time_attr[:, 1:2].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        time_feat = _pair_timestep_embedding(
            t, self.hidden_nf
        ).to(device=edge_feat.device, dtype=edge_feat.dtype)
        semantic_residual = (
            time_feat
            * self.scale.to(edge_feat.dtype).unsqueeze(0)
        )
        return edge_feat + enabled * semantic_residual


class _TimeWrappedAMEGCL(nn.Module):
    """Wrap an existing AM_E_GCL without changing its original parameters."""

    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.time_scale = _ChannelTimeScale(hidden_nf)

    def forward(
        self, h, edge_index, coord, channel_attr, channel_weights,
        edge_attr=None, node_attr=None
    ):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, channel_attr, channel_weights,
            self.base.radial_linear,
        )
        # Original edge model receives no extra edge attributes, so its
        # dimensions/weights remain exactly the same as the current AMEncoder.
        base_edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        semantic_edge_feat = self.time_scale(base_edge_feat, edge_attr)
        # Same wrapped GCL coordinate branch is exactly the parent message.
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, base_edge_feat, channel_weights
        )
        h, _ = self.base.node_model(
            h, edge_index, semantic_edge_feat, node_attr
        )
        return h, coord


class _TimeWrappedMSGCL(nn.Module):
    """Wrap an existing MS_E_GCL without changing its original parameters."""

    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.time_scale = _ChannelTimeScale(hidden_nf)

    def forward(
        self, h, edge_index, epi_index, coord, surf_verts,
        channel_attr, channel_weights, edge_attr=None, node_attr=None
    ):
        row, col = edge_index
        radial, abX = coord_SR(
            edge_index, epi_index, coord, surf_verts, channel_attr,
            self.base.scale_linear, self.base.radial_linear,
        )
        base_edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        semantic_edge_feat = self.time_scale(base_edge_feat, edge_attr)
        # Same wrapped GCL coordinate branch is exactly the parent message.
        coord = self.base.coord_model(
            coord, edge_index, abX, base_edge_feat, channel_weights
        )
        h, _ = self.base.node_model(
            h, edge_index, semantic_edge_feat, node_attr
        )
        return h, coord


class AMEncoderPairTime(AMEncoder):
    """Original AMEncoder + zero-start AbX-inspired semantic pair-time."""

    def __init__(
        self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
        radial_nf, in_edge_nf=0, num_verts=50, act_fn=nn.SiLU(),
        n_layers=4, residual=True, dropout=0.1, dense=False,
        pair_time_scope="interface",
    ):
        # IMPORTANT: build the complete original AMEncoder first.  Thus all
        # original random parameters are sampled in the same order as the base.
        super().__init__(
            in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
            radial_nf, in_edge_nf=0, num_verts=num_verts, act_fn=act_fn,
            n_layers=n_layers, residual=residual, dropout=dropout, dense=dense,
        )
        scope = str(pair_time_scope).strip().lower()
        if scope not in {"interface", "context", "all"}:
            raise ValueError(
                "pair_time_scope must be 'interface', 'context' or 'all', got "
                f"{pair_time_scope!r}"
            )
        self.pair_time_scope = scope

        # Added parameters are zeros only, so no RNG is consumed here.
        for i in range(self.n_layers):
            if scope in {"context", "all"}:
                self._modules[f"ctx_gcl_{i}"] = _TimeWrappedAMEGCL(
                    self._modules[f"ctx_gcl_{i}"]
                )
            self._modules[f"inter_gcl_{i}"] = _TimeWrappedAMEGCL(
                self._modules[f"inter_gcl_{i}"]
            )
            if scope in {"interface", "all"}:
                self._modules[f"surf_gcl_{i}"] = _TimeWrappedMSGCL(
                    self._modules[f"surf_gcl_{i}"]
                )

        if scope in {"context", "all"}:
            self.out_layer = _TimeWrappedAMEGCL(self.out_layer)

    def forward(
        self, h, x, ctx_edges, inter_mask, inter_x, surf_verts, inter_edges,
        update_mask, inter_update_mask, aligned_edges, epi_index,
        channel_attr, channel_weights, ctx_edge_attr=None,
        inter_edge_attr=None, surf_edge_attr=None,
    ):
        h = self.linear_in(h)
        h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(self.n_layers):
            h, x = self._modules[f"ctx_gcl_{i}"](
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=(ctx_edge_attr if self.pair_time_scope in {"context", "all"} else None),
            )

            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]

            inter_h, inter_x = self._modules[f"inter_gcl_{i}"](
                inter_h, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
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
            edge_attr=(ctx_edge_attr if self.pair_time_scope in {"context", "all"} else None),
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
