#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Parameter-matched Flow/Score semantic pair conditioning for F01-v59 AbFlow.

This module deliberately leaves same-layer coordinate messages unchanged.
Only semantic/node messages on true local antibody-antigen interaction edges
receive a zero-start residual built from four invariant features:

    [log1p(||mean_flow||), cos(mean_flow, r_ij),
     log1p(||score_correction||), cos(score_correction, r_ij)]

The exact same 4->hidden adapter exists in ZERO, FLOW, SCORE, and SCOREFLOW modes.
The formal three-run stage uses ZERO / FLOW / SCOREFLOW. Inactive channels are
zeroed upstream, so all formal runs have identical parameter count/capacity.
"""

import torch
import torch.nn as nn

from .am_enc import AMEncoder
from .am_egnn import coord2radial


class _ZeroStartFlowScorePairBias(nn.Module):
    def __init__(self, hidden_nf):
        super().__init__()
        self.proj = nn.Linear(4, hidden_nf, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, edge_feat, pair_attr):
        if pair_attr is None:
            return edge_feat
        if pair_attr.dim() != 2 or pair_attr.shape[-1] != 5:
            raise ValueError(
                "pair_attr must have shape [E,5] = "
                "[flow_mag, flow_cos, score_mag, score_cos, enabled_mask], "
                f"got {tuple(pair_attr.shape)}"
            )
        values = pair_attr[:, :4].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        enabled = pair_attr[:, 4:5].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        return edge_feat + enabled * self.proj(values)


class _FlowScoreSemanticAMEGCL(nn.Module):
    """Wrap one AM_E_GCL without changing its coordinate update."""

    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.pair_bias = _ZeroStartFlowScorePairBias(hidden_nf)

    def forward(
        self, h, edge_index, coord, channel_attr, channel_weights,
        edge_attr=None, node_attr=None
    ):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, channel_attr, channel_weights,
            self.base.radial_linear,
        )
        base_edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )

        # Preserve the validated F01 same-layer coordinate force exactly.
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, base_edge_feat, channel_weights
        )

        # New field information conditions semantic reasoning only.
        semantic_edge_feat = self.pair_bias(base_edge_feat, edge_attr)
        h, _ = self.base.node_model(
            h, edge_index, semantic_edge_feat, node_attr
        )
        return h, coord


class AMEncoderFlowScorePair(AMEncoder):
    """F01 AMEncoder + zero-start Flow/Score semantic interaction conditioning."""

    def __init__(
        self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
        radial_nf, in_edge_nf=0, num_verts=50, act_fn=nn.SiLU(),
        n_layers=4, residual=True, dropout=0.1, dense=False,
    ):
        # Build the complete validated encoder first, preserving initialization
        # order for all original parameters.
        super().__init__(
            in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
            radial_nf, in_edge_nf=0, num_verts=num_verts, act_fn=act_fn,
            n_layers=n_layers, residual=residual, dropout=dropout, dense=dense,
        )

        # Only local residue-residue interaction GCLs are conditioned.
        # ctx_gcl and surf_gcl remain untouched.
        for i in range(self.n_layers):
            self._modules[f"inter_gcl_{i}"] = _FlowScoreSemanticAMEGCL(
                self._modules[f"inter_gcl_{i}"]
            )

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
                edge_attr=ctx_edge_attr,
            )

            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]

            inter_h, inter_x = self._modules[f"inter_gcl_{i}"](
                inter_h, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
            )

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
            edge_attr=ctx_edge_attr,
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
