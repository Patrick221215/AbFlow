#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from torch_scatter import scatter_softmax
from .am_egnn import AM_E_GCL, MS_E_GCL, coord2radial, coord_SR


class _PairResidualAMEGCL(nn.Module):
    """Zero-start pair-state residual on top of an existing AM_E_GCL.

    The base edge MLP is evaluated exactly as in R05.  The persistent pair
    representation z_ij is projected into the already-computed hidden edge
    message through a zero-initialized matrix.  Therefore the complete forward
    function is exactly the R05 parent at initialization, while gradients can
    immediately learn how much pair information should enter coordinate/node
    updates.  The projection is represented by a zero Parameter directly (not
    nn.Linear) so enabling the module consumes no RNG and does not perturb the
    initialization of later legacy parameters.
    """

    def __init__(self, base_gcl, pair_dim):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.pair_norm = nn.LayerNorm(int(pair_dim))
        self.pair_weight = nn.Parameter(torch.zeros(hidden_nf, int(pair_dim)))

    def forward(self, h, edge_index, coord, channel_attr, channel_weights,
                edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = coord2radial(
            edge_index, coord, channel_attr, channel_weights,
            self.base.radial_linear,
        )
        base_edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        enriched_edge_feat = base_edge_feat
        if edge_attr is not None:
            pair = self.pair_norm(edge_attr.to(
                device=base_edge_feat.device, dtype=base_edge_feat.dtype
            ))
            enriched_edge_feat = base_edge_feat + torch.nn.functional.linear(
                pair, self.pair_weight.to(base_edge_feat.dtype)
            )
        # Semantic closure: AbX consumes its final pair state directly inside the
        # structure trunk.  In the R05 hybrid, the analogous consumer is the *same*
        # existing EGNN layer.  Therefore the enriched invariant edge message must
        # drive both node and coordinate updates.  This does NOT add a second
        # coordinate authority: coord_model is still the sole legacy R05 decoder,
        # and pair_weight=0 preserves the exact parent function at initialization.
        coord = self.base.coord_model(
            coord, edge_index, coord_diff, enriched_edge_feat, channel_weights
        )
        h, _ = self.base.node_model(h, edge_index, enriched_edge_feat, node_attr)
        return h, coord


class _PairResidualMSGCL(nn.Module):
    """Surface counterpart of _PairResidualAMEGCL."""

    def __init__(self, base_gcl, pair_dim):
        super().__init__()
        self.base = base_gcl
        hidden_nf = int(base_gcl.edge_mlp[0].out_features)
        self.pair_norm = nn.LayerNorm(int(pair_dim))
        self.pair_weight = nn.Parameter(torch.zeros(hidden_nf, int(pair_dim)))

    def forward(self, h, edge_index, epi_index, coord, surf_verts,
                channel_attr, channel_weights, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, abX = coord_SR(
            edge_index, epi_index, coord, surf_verts, channel_attr,
            self.base.scale_linear, self.base.radial_linear,
        )
        base_edge_feat = self.base.edge_model(
            h[row], h[col], radial, edge_attr=None
        )
        enriched_edge_feat = base_edge_feat
        if edge_attr is not None:
            pair = self.pair_norm(edge_attr.to(
                device=base_edge_feat.device, dtype=base_edge_feat.dtype
            ))
            enriched_edge_feat = base_edge_feat + torch.nn.functional.linear(
                pair, self.pair_weight.to(base_edge_feat.dtype)
            )
        coord = self.base.coord_model(
            coord, edge_index, abX, enriched_edge_feat, channel_weights
        )
        h, _ = self.base.node_model(h, edge_index, enriched_edge_feat, node_attr)
        return h, coord


class AMEncoder(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, num_verts=50, act_fn=nn.SiLU(), n_layers=4,
                 residual=True, dropout=0.1, dense=False):
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        :param dense: if dense, then context states will be concatenated for all layers,
                      coordination will be averaged
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f'ctx_gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
            self.add_module(f'inter_gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
            self.add_module(f'surf_gcl_{i}', MS_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                surf_nf=num_verts, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
        self.out_layer = AM_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
            radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual
        )

    def enable_pair_representation(self, pair_dim):
        """Inject persistent pair states into local interface messages.

        Only the local inter/surface pathways are wrapped.  Global/context R05
        message passing and graph topology stay untouched.  Calling this method
        twice is an error because nested wrappers would change semantics.
        """
        if getattr(self, "_pair_representation_enabled", False):
            raise RuntimeError("pair representation is already enabled")
        pair_dim = int(pair_dim)
        if pair_dim <= 0:
            raise ValueError("pair_dim must be positive")
        for i in range(self.n_layers):
            self._modules[f'inter_gcl_{i}'] = _PairResidualAMEGCL(
                self._modules[f'inter_gcl_{i}'], pair_dim
            )
            self._modules[f'surf_gcl_{i}'] = _PairResidualMSGCL(
                self._modules[f'surf_gcl_{i}'], pair_dim
            )
        self._pair_representation_enabled = True
        self._pair_representation_dim = pair_dim
    
    def forward(self, h, x, ctx_edges, inter_mask, inter_x, surf_verts, inter_edges, update_mask, inter_update_mask, aligned_edges, epi_index, channel_attr, channel_weights,
                ctx_edge_attr=None, inter_edge_attr=None, surf_edge_attr=None):
        h = self.linear_in(h)
        h = self.dropout(h)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(0, self.n_layers):
            h, x = self._modules[f'ctx_gcl_{i}'](
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr)
            # synchronization of the shadow paratope (native -> shadow)
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h, inter_x = self._modules[f'inter_gcl_{i}'](
                inter_h, inter_edges, inter_x, inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr
            )
            inter_h, inter_x = self._modules[f'surf_gcl_{i}'](
                inter_h, aligned_edges, epi_index, inter_x, surf_verts, inter_channel_attr, inter_channel_weights,
                edge_attr=surf_edge_attr
            )
            # synchronization of the shadow paratope (shadow -> native)
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x, inter_x