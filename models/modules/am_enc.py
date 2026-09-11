#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .am_egnn import AM_E_GCL, MS_E_GCL


class AMEncoder(nn.Module):
    """R05 multi-channel EGNN with parent-preserving single/pair conditioning."""

    def __init__(
        self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
        radial_nf, in_edge_nf=0, in_single_nf=0, num_verts=50,
        act_fn=nn.SiLU(), n_layers=4, residual=True, dropout=0.1, dense=False,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, hidden_nf)
        self.single_linear = None
        if int(in_single_nf) > 0:
            with torch.random.fork_rng(devices=[]):
                self.single_linear = nn.Linear(
                    int(in_single_nf), hidden_nf, bias=False
                )
            nn.init.zeros_(self.single_linear.weight)

        self.last_bridge_diagnostics = {}

        self.dense = dense
        self.linear_out = nn.Linear(
            hidden_nf * (n_layers + 1) if dense else hidden_nf,
            out_node_nf,
        )

        for i in range(n_layers):
            self.add_module(
                f'ctx_gcl_{i}',
                AM_E_GCL(
                    hidden_nf, hidden_nf, hidden_nf, n_channel,
                    channel_nf, radial_nf, edges_in_d=in_edge_nf,
                    act_fn=act_fn, residual=residual, dropout=dropout,
                ),
            )
            self.add_module(
                f'inter_gcl_{i}',
                AM_E_GCL(
                    hidden_nf, hidden_nf, hidden_nf, n_channel,
                    channel_nf, radial_nf, edges_in_d=in_edge_nf,
                    act_fn=act_fn, residual=residual, dropout=dropout,
                ),
            )
            self.add_module(
                f'surf_gcl_{i}',
                MS_E_GCL(
                    hidden_nf, hidden_nf, hidden_nf, n_channel,
                    channel_nf, radial_nf, surf_nf=num_verts,
                    edges_in_d=in_edge_nf, act_fn=act_fn,
                    residual=residual, dropout=dropout,
                ),
            )

        self.out_layer = AM_E_GCL(
            hidden_nf, hidden_nf, hidden_nf, n_channel,
            channel_nf, radial_nf, edges_in_d=in_edge_nf,
            act_fn=act_fn, residual=residual,
        )

    def forward(
        self, h, x, ctx_edges, inter_mask, inter_x, surf_verts,
        inter_edges, update_mask, inter_update_mask, aligned_edges,
        epi_index, channel_attr, channel_weights, ctx_edge_attr=None,
        inter_edge_attr=None, surf_edge_attr=None, single_attr=None,
        capture_bridge_diagnostics=False,
    ):
        # Parent-preserving single bridge:
        # h0 = W_parent h + W_single s, with W_single initialized to zero.
        base_pre = self.linear_in(h)
        if self.single_linear is None:
            single_delta = torch.zeros_like(base_pre)
        else:
            if single_attr is None:
                single_attr = base_pre.new_zeros(
                    (base_pre.shape[0], self.single_linear.in_features)
                )
            single_delta = self.single_linear(single_attr)
        h = self.dropout(base_pre + single_delta)

        capture_bridge_diagnostics = bool(capture_bridge_diagnostics)
        self.last_bridge_diagnostics = {}
        if capture_bridge_diagnostics:
            with torch.no_grad():
                base_rms = base_pre.float().square().mean().sqrt()
                delta_rms = single_delta.float().square().mean().sqrt()
                self.last_bridge_diagnostics.update({
                    'bridge_single_delta_to_base_ratio': (
                        delta_rms / base_rms.clamp_min(1.0e-8)
                    ).to(base_pre.dtype),
                    'bridge_single_adapter_weight_rms': (
                        base_rms.new_zeros(()) if self.single_linear is None
                        else self.single_linear.weight.detach().float().square().mean().sqrt()
                    ).to(base_pre.dtype),
                })
        pair_ratios = []
        pair_weights = []

        def collect_pair(module):
            if not capture_bridge_diagnostics:
                return
            diag = getattr(module, 'last_bridge_diagnostics', {}) or {}
            ratio = diag.get('pair_delta_to_base_ratio')
            weight = diag.get('pair_adapter_weight_rms')
            if ratio is not None:
                pair_ratios.append(ratio)
            if weight is not None:
                pair_weights.append(weight)

        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(self.n_layers):
            h, x = self._modules[f'ctx_gcl_{i}'](
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair(self._modules[f'ctx_gcl_{i}'])

            # Native R05 shadow interface: native -> shadow.
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]

            inter_h, inter_x = self._modules[f'inter_gcl_{i}'](
                inter_h, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair(self._modules[f'inter_gcl_{i}'])

            inter_h, inter_x = self._modules[f'surf_gcl_{i}'](
                inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                inter_channel_attr, inter_channel_weights,
                edge_attr=surf_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair(self._modules[f'surf_gcl_{i}'])

            # Shadow -> native.
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr,
            capture_bridge_diagnostics=capture_bridge_diagnostics,
        )
        collect_pair(self.out_layer)
        ctx_states.append(h)
        ctx_coords.append(x)

        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)

        h = self.dropout(h)
        h = self.linear_out(h)
        if capture_bridge_diagnostics:
            if pair_ratios:
                stacked = torch.stack(pair_ratios)
                self.last_bridge_diagnostics['bridge_pair_delta_to_base_ratio_mean'] = stacked.mean()
                self.last_bridge_diagnostics['bridge_pair_delta_to_base_ratio_max'] = stacked.max()
            if pair_weights:
                self.last_bridge_diagnostics['bridge_pair_adapter_weight_rms_mean'] = torch.stack(pair_weights).mean()
        return h, x, inter_x
