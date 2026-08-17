#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Minimal zero-start pair-level flow-time conditioning for AbFlow.

Design goal
-----------
Preserve the full-atom AMEncoder as the base architecture and add the smallest
possible explicit pair-time intervention.

For a base edge message m_ij, the module applies

    m_ij(t) = m_ij * (1 + mask_ij * t * w_l)

where w_l is a learnable hidden-channel scale initialized to exactly zero.
Therefore:
  * at initialization the forward function is exactly the original AMEncoder;
  * existing AM_E_GCL / MS_E_GCL parameter shapes are unchanged;
  * the new parameters consume no random numbers (torch.zeros only);
  * pair-time can be targeted only to true antibody-antigen edges.

Scopes
------
interface:
    - global/context GCL: unchanged
    - local inter GCL: time modulation only on true Ab-Ag edges; local-context
      edges are explicitly masked to zero
    - antigen-surface GCL: time modulation enabled

all:
    - everything in "interface"
    - global/context GCL and output context GCL are also time-conditioned
    - local-context edges inside the local inter GCL are also enabled

This is intentionally more conservative than concatenating a new learned time
embedding into every edge MLP.  The experiment should test the scientific role
of explicit pair-time, not a large capacity increase.
"""

import torch
import torch.nn as nn

from .am_enc import AMEncoder
from .am_egnn import coord2radial, coord_SR


class _ChannelTimeScale(nn.Module):
    """Zero-start channel-wise edge-message modulation."""

    def __init__(self, hidden_nf):
        super().__init__()
        # Zero initialization is deliberate: exact original forward at step 0
        # and no RNG consumption that could shift initialization of later layers.
        self.scale = nn.Parameter(torch.zeros(hidden_nf))

    def forward(self, edge_feat, pair_time_attr):
        if pair_time_attr is None:
            return edge_feat
        if pair_time_attr.dim() != 2 or pair_time_attr.shape[-1] != 2:
            raise ValueError(
                "pair_time_attr must have shape [E, 2] = [t, enabled_mask], "
                f"got {tuple(pair_time_attr.shape)}"
            )
        t = pair_time_attr[:, :1].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        enabled = pair_time_attr[:, 1:2].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        return edge_feat * (
            1.0 + enabled * t * self.scale.to(edge_feat.dtype).unsqueeze(0)
        )


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
        edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        edge_feat = self.time_scale(edge_feat, edge_attr)
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, edge_feat, channel_weights
        )
        h, _ = self.base.node_model(
            h, edge_index, edge_feat, node_attr
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
        edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        edge_feat = self.time_scale(edge_feat, edge_attr)
        coord = self.base.coord_model(
            coord, edge_index, abX, edge_feat, channel_weights
        )
        h, _ = self.base.node_model(
            h, edge_index, edge_feat, node_attr
        )
        return h, coord


class AMEncoderPairTime(AMEncoder):
    """Original AMEncoder + zero-start explicit pair-time modulation."""

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
