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
        pair_coord_mode='bounded_residual', pair_coord_delta_bound=1.0,
        coord_tanh=False, coord_normalize=False,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.pair_coord_mode = str(pair_coord_mode or 'bounded_residual').strip().lower()
        self.pair_coord_delta_bound = float(pair_coord_delta_bound)
        self.coord_tanh = bool(coord_tanh)
        self.coord_normalize = bool(coord_normalize)

        self.linear_in = nn.Linear(in_node_nf, hidden_nf)
        self.single_linear = None
        if int(in_single_nf) > 0:
            with torch.random.fork_rng(devices=[]):
                self.single_linear = nn.Linear(
                    int(in_single_nf), hidden_nf, bias=False
                )
            nn.init.zeros_(self.single_linear.weight)

        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}

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
                    pair_coord_mode=self.pair_coord_mode,
                    pair_coord_delta_bound=self.pair_coord_delta_bound,
                    tanh=self.coord_tanh, normalize=self.coord_normalize,
                ),
            )
            self.add_module(
                f'inter_gcl_{i}',
                AM_E_GCL(
                    hidden_nf, hidden_nf, hidden_nf, n_channel,
                    channel_nf, radial_nf, edges_in_d=in_edge_nf,
                    act_fn=act_fn, residual=residual, dropout=dropout,
                    pair_coord_mode=self.pair_coord_mode,
                    pair_coord_delta_bound=self.pair_coord_delta_bound,
                    tanh=self.coord_tanh, normalize=self.coord_normalize,
                ),
            )
            self.add_module(
                f'surf_gcl_{i}',
                MS_E_GCL(
                    hidden_nf, hidden_nf, hidden_nf, n_channel,
                    channel_nf, radial_nf, surf_nf=num_verts,
                    edges_in_d=in_edge_nf, act_fn=act_fn,
                    residual=residual, dropout=dropout,
                    pair_coord_mode=self.pair_coord_mode,
                    pair_coord_delta_bound=self.pair_coord_delta_bound,
                    tanh=self.coord_tanh, normalize=self.coord_normalize,
                ),
            )

        self.out_layer = AM_E_GCL(
            hidden_nf, hidden_nf, hidden_nf, n_channel,
            channel_nf, radial_nf, edges_in_d=in_edge_nf,
            act_fn=act_fn, residual=residual,
            pair_coord_mode=self.pair_coord_mode,
            pair_coord_delta_bound=self.pair_coord_delta_bound,
            tanh=self.coord_tanh, normalize=self.coord_normalize,
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
        state_pre = base_pre + single_delta
        if self.training and self.dropout.p > 0.0:
            shared_dropout_scale = self.dropout(torch.ones_like(state_pre))
            h = state_pre * shared_dropout_scale
            h_base = base_pre * shared_dropout_scale
        else:
            h = state_pre
            h_base = base_pre

        capture_bridge_diagnostics = bool(capture_bridge_diagnostics)
        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}
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
        coord_updates = []
        coord_coeffs = []
        coord_base_raw_coeffs = []
        coord_state_raw_coeffs = []
        coord_diff_norms = []
        coord_direction_norms = []
        coord_trans = []
        coord_update_input_ratios = []

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

        def collect_coord(module, stage):
            if not capture_bridge_diagnostics:
                return
            diag = getattr(module, 'last_coord_diagnostics', {}) or {}
            for key, value in diag.items():
                self.last_coord_diagnostics[f'{stage}.{key}'] = value
            update = diag.get('coord_update_absmax')
            coeff = diag.get('coord_coeff_absmax')
            base_raw = diag.get('coord_base_coeff_raw_absmax')
            state_raw = diag.get('coord_state_coeff_raw_absmax')
            diff_norm = diag.get('coord_diff_norm_absmax')
            direction_norm = diag.get('coord_direction_norm_absmax')
            trans_absmax = diag.get('coord_trans_absmax')
            update_input_ratio = diag.get('coord_update_to_input_rms_ratio')
            if update is not None:
                coord_updates.append(update)
            if coeff is not None:
                coord_coeffs.append(coeff)
            if base_raw is not None:
                coord_base_raw_coeffs.append(base_raw)
            if state_raw is not None:
                coord_state_raw_coeffs.append(state_raw)
            if diff_norm is not None:
                coord_diff_norms.append(diff_norm)
            if direction_norm is not None:
                coord_direction_norms.append(direction_norm)
            if trans_absmax is not None:
                coord_trans.append(trans_absmax)
            if update_input_ratio is not None:
                coord_update_input_ratios.append(update_input_ratio)

        inter_h = h[inter_mask]
        inter_h_base = h_base[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(self.n_layers):
            h, h_base, x = self._modules[f'ctx_gcl_{i}'].forward_dual(
                h, h_base, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair(self._modules[f'ctx_gcl_{i}'])
            collect_coord(self._modules[f'ctx_gcl_{i}'], f'ctx_{i}')

            # Native R05 shadow interface: native -> shadow, for both streams.
            inter_h = inter_h.clone()
            inter_h_base = inter_h_base.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h_base[inter_update_mask] = h_base[update_mask]

            inter_h, inter_h_base, inter_x = self._modules[f'inter_gcl_{i}'].forward_dual(
                inter_h, inter_h_base, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair(self._modules[f'inter_gcl_{i}'])
            collect_coord(self._modules[f'inter_gcl_{i}'], f'inter_{i}')

            inter_h, inter_h_base, inter_x = self._modules[f'surf_gcl_{i}'].forward_dual(
                inter_h, inter_h_base, aligned_edges, epi_index, inter_x, surf_verts,
                inter_channel_attr, inter_channel_weights,
                edge_attr=surf_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair(self._modules[f'surf_gcl_{i}'])
            collect_coord(self._modules[f'surf_gcl_{i}'], f'surf_{i}')

            # Shadow -> native.
            h = h.clone()
            h_base = h_base.clone()
            h[inter_mask] = inter_h
            h_base[inter_mask] = inter_h_base
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, h_base, x = self.out_layer.forward_dual(
            h, h_base, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr,
            capture_bridge_diagnostics=capture_bridge_diagnostics,
        )
        collect_pair(self.out_layer)
        collect_coord(self.out_layer, 'out')
        ctx_states.append(h)
        ctx_coords.append(x)

        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)

        h = self.dropout(h)
        h = self.linear_out(h)
        if capture_bridge_diagnostics:
            self.last_coord_diagnostics['pair_coord_mode'] = self.pair_coord_mode
            self.last_coord_diagnostics['pair_coord_delta_bound'] = x.new_tensor(self.pair_coord_delta_bound)
            self.last_coord_diagnostics['coord_tanh'] = x.new_tensor(1.0 if self.coord_tanh else 0.0)
            self.last_coord_diagnostics['coord_normalize'] = x.new_tensor(1.0 if self.coord_normalize else 0.0)
            if pair_ratios:
                stacked = torch.stack(pair_ratios)
                self.last_bridge_diagnostics['bridge_pair_delta_to_base_ratio_mean'] = stacked.mean()
                self.last_bridge_diagnostics['bridge_pair_delta_to_base_ratio_max'] = stacked.max()
            if pair_weights:
                self.last_bridge_diagnostics['bridge_pair_adapter_weight_rms_mean'] = torch.stack(pair_weights).mean()
            if coord_updates:
                self.last_coord_diagnostics['coord_update_absmax_max'] = torch.stack(coord_updates).max()
                self.last_coord_diagnostics['coord_update_absmax_mean'] = torch.stack(coord_updates).mean()
            if coord_coeffs:
                self.last_coord_diagnostics['coord_coeff_absmax_max'] = torch.stack(coord_coeffs).max()
                self.last_coord_diagnostics['coord_coeff_absmax_mean'] = torch.stack(coord_coeffs).mean()
            if coord_base_raw_coeffs:
                self.last_coord_diagnostics['coord_base_coeff_raw_absmax_max'] = torch.stack(coord_base_raw_coeffs).max()
            if coord_state_raw_coeffs:
                self.last_coord_diagnostics['coord_state_coeff_raw_absmax_max'] = torch.stack(coord_state_raw_coeffs).max()
            if coord_diff_norms:
                self.last_coord_diagnostics['coord_diff_norm_absmax_max'] = torch.stack(coord_diff_norms).max()
            if coord_direction_norms:
                self.last_coord_diagnostics['coord_direction_norm_absmax_max'] = torch.stack(coord_direction_norms).max()
            if coord_trans:
                self.last_coord_diagnostics['coord_trans_absmax_max'] = torch.stack(coord_trans).max()
            if coord_update_input_ratios:
                self.last_coord_diagnostics['coord_update_to_input_rms_ratio_max'] = torch.stack(coord_update_input_ratios).max()
        return h, x, inter_x
