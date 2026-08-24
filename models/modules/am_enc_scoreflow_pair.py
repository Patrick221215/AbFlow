#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Parameter-matched Score--Flow pair-representation adapter for AbFlow.

The adapter never changes the same-layer coordinate message.  It adds a
zero-start semantic bias only to the node update:

    m_node = m_base + enabled * P(features)
    m_coord = m_base

All v83 configurations instantiate the same 5->hidden projector.  A
fixed feature mask, constructed by the model, determines which scientific
signal is active.  Consequently G00/G01/G02 have identical parameter counts
and initialization; they differ only in their enabled information.
"""

import torch
import torch.nn as nn

from .am_enc import AMEncoder
from .am_egnn import coord2radial, coord_SR


PAIR_FEATURE_DIM = 5


class _ZeroStartScoreFlowProjection(nn.Module):
    """RNG-free linear projection with an exact zero initial function."""

    def __init__(self, hidden_nf):
        super().__init__()
        self.hidden_nf = int(hidden_nf)
        # Construct Parameters directly from zeros.  Calling nn.Linear and then
        # zeroing it would still consume random numbers and shift initialization
        # of trainable parent modules created later in AbFlowModel.__init__.
        self.weight = nn.Parameter(torch.zeros(
            self.hidden_nf, PAIR_FEATURE_DIM
        ))
        self.bias = nn.Parameter(torch.zeros(self.hidden_nf))

    def forward(self, edge_feat, pair_attr):
        if pair_attr is None:
            # DDP safety: mark every adapter parameter as used while preserving
            # the exact base function for disabled/context messages.
            dummy = sum(parameter.sum() for parameter in self.parameters())
            return edge_feat + 0.0 * dummy
        if pair_attr.dim() != 2 or pair_attr.shape[-1] != PAIR_FEATURE_DIM + 1:
            raise ValueError(
                "pair_attr must be [E,6]=[five_features,enabled_mask], "
                f"got {tuple(pair_attr.shape)}"
            )
        features = pair_attr[:, :PAIR_FEATURE_DIM].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        enabled = pair_attr[:, PAIR_FEATURE_DIM:].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        projected = torch.nn.functional.linear(
            features, self.weight, self.bias
        )
        return edge_feat + enabled * projected


class _ScoreFlowAMEGCL(nn.Module):
    """Condition the node representation, never the same-step coordinates."""

    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.pair_adapter = _ZeroStartScoreFlowProjection(hidden_nf)

    def forward(
        self, h, edge_index, coord, channel_attr, channel_weights,
        edge_attr=None, node_attr=None,
    ):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, channel_attr, channel_weights,
            self.base.radial_linear,
        )
        base_edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        node_edge_feat = self.pair_adapter(base_edge_feat, edge_attr)
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, base_edge_feat, channel_weights
        )
        h, _ = self.base.node_model(h, edge_index, node_edge_feat, node_attr)
        return h, coord


class _ScoreFlowMSGCL(nn.Module):
    """Surface-message counterpart with the same branch separation."""

    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.pair_adapter = _ZeroStartScoreFlowProjection(hidden_nf)

    def forward(
        self, h, edge_index, epi_index, coord, surf_verts,
        channel_attr, channel_weights, edge_attr=None, node_attr=None,
    ):
        row, col = edge_index
        radial, ab_x = coord_SR(
            edge_index, epi_index, coord, surf_verts, channel_attr,
            self.base.scale_linear, self.base.radial_linear,
        )
        base_edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        node_edge_feat = self.pair_adapter(base_edge_feat, edge_attr)
        coord = self.base.coord_model(
            coord, edge_index, ab_x, base_edge_feat, channel_weights
        )
        h, _ = self.base.node_model(h, edge_index, node_edge_feat, node_attr)
        return h, coord


class AMEncoderScoreFlowPair(AMEncoder):
    """Original AMEncoder plus interface-only Score--Flow pair semantics."""

    def __init__(
        self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
        radial_nf, in_edge_nf=0, num_verts=50, act_fn=nn.SiLU(),
        n_layers=4, residual=True, dropout=0.1, dense=False,
    ):
        # Build the exact parent encoder first.  Added modules come afterwards,
        # so the parent's random-parameter initialization order is preserved.
        super().__init__(
            in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
            radial_nf, in_edge_nf=0, num_verts=num_verts, act_fn=act_fn,
            n_layers=n_layers, residual=residual, dropout=dropout, dense=dense,
        )
        for layer_idx in range(self.n_layers):
            inter_key = f"inter_gcl_{layer_idx}"
            surf_key = f"surf_gcl_{layer_idx}"
            self._modules[inter_key] = _ScoreFlowAMEGCL(
                self._modules[inter_key]
            )
            self._modules[surf_key] = _ScoreFlowMSGCL(
                self._modules[surf_key]
            )

    def forward(
        self, h, x, ctx_edges, inter_mask, inter_x, surf_verts, inter_edges,
        update_mask, inter_update_mask, aligned_edges, epi_index,
        channel_attr, channel_weights, inter_edge_attr=None,
        surf_edge_attr=None,
    ):
        h = self.linear_in(h)
        h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for layer_idx in range(self.n_layers):
            h, x = self._modules[f"ctx_gcl_{layer_idx}"](
                h, ctx_edges, x, channel_attr, channel_weights
            )
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h, inter_x = self._modules[f"inter_gcl_{layer_idx}"](
                inter_h, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
            )
            inter_h, inter_x = self._modules[f"surf_gcl_{layer_idx}"](
                inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                inter_channel_attr, inter_channel_weights,
                edge_attr=surf_edge_attr,
            )
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights
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
