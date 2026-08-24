#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Endpoint-induced analytic Score conditioning for AbFlow interface pair semantics.

Design principles
-----------------
1. Follow AbX's *representation conditioning* idea rather than the failed E02
   multiplicative coordinate-message scaling.
2. Do NOT alter the same-layer coordinate update.  The original edge feature is
   sent unchanged to coord_model; only the semantic/node message receives the
   zero-start Score-conditioned residual.
3. Do NOT learn a Score head.  The model receives two invariant pair features
   computed analytically from the previous refinement round:
       score_norm : ||q_t||, q_t = sigma_t * score_t
       score_proj : cos(q_t, r_ij)
   where r_ij is the current antibody<-antigen edge direction.
4. The projection is zero-initialized, so enabling this module starts from the
   exact base AMEncoder function without changing any original GCL weights.

This is intentionally restricted to true antigen -> antibody interaction edges.
Context edges and antigen-surface coordinate updates remain unchanged.
"""

import torch
import torch.nn as nn

from .am_enc import AMEncoder
from .am_egnn import coord2radial


class _ZeroStartScorePairBias(nn.Module):
    """Map [score_norm, score_proj] -> hidden semantic edge bias."""

    def __init__(self, hidden_nf):
        super().__init__()
        self.proj = nn.Linear(2, hidden_nf, bias=True)
        # Exact base function at initialization and no RNG dependence.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, edge_feat, pair_score_attr):
        if pair_score_attr is None:
            return edge_feat
        if pair_score_attr.dim() != 2 or pair_score_attr.shape[-1] != 3:
            raise ValueError(
                "pair_score_attr must have shape [E,3] = "
                "[score_norm, score_proj, enabled_mask], got "
                f"{tuple(pair_score_attr.shape)}"
            )
        values = pair_score_attr[:, :2].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        enabled = pair_score_attr[:, 2:3].to(
            device=edge_feat.device, dtype=edge_feat.dtype
        )
        return edge_feat + enabled * self.proj(values)


class _ScoreSemanticAMEGCL(nn.Module):
    """Wrap AM_E_GCL: coordinate message unchanged, node message Score-conditioned."""

    def __init__(self, base_gcl):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.score_pair_bias = _ZeroStartScorePairBias(hidden_nf)

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

        # IMPORTANT: same-layer geometry is exactly the base AMEncoder geometry.
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, base_edge_feat, channel_weights
        )

        # Only semantic/node reasoning sees the analytic Score condition.
        semantic_edge_feat = self.score_pair_bias(
            base_edge_feat, edge_attr
        )
        h, _ = self.base.node_model(
            h, edge_index, semantic_edge_feat, node_attr
        )
        return h, coord


class AMEncoderScorePair(AMEncoder):
    """Base AMEncoder + zero-start analytic-Score semantic pair conditioning."""

    def __init__(
        self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
        radial_nf, in_edge_nf=0, num_verts=50, act_fn=nn.SiLU(),
        n_layers=4, residual=True, dropout=0.1, dense=False,
    ):
        # Construct complete base encoder first so original parameters are
        # initialized in the same order as AMEncoder.
        super().__init__(
            in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
            radial_nf, in_edge_nf=0, num_verts=num_verts, act_fn=act_fn,
            n_layers=n_layers, residual=residual, dropout=dropout, dense=dense,
        )

        # Only local residue-residue interaction GCLs are conditioned.
        # ctx_gcl and surf_gcl remain exactly unchanged.
        for i in range(self.n_layers):
            self._modules[f"inter_gcl_{i}"] = _ScoreSemanticAMEGCL(
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
            # Base context path.
            h, x = self._modules[f"ctx_gcl_{i}"](
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
            )

            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]

            # Only this semantic interaction path receives Score pair features.
            inter_h, inter_x = self._modules[f"inter_gcl_{i}"](
                inter_h, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
            )

            # Surface geometry remains the original AMEncoder behavior.
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
