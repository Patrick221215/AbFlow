#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .am_egnn import AM_E_GCL, MS_E_GCL
from .local_frame_actuator import LocalFrameFullAtomActuator


class AMEncoder(nn.Module):
    """R05 multi-channel relational encoder with selectable geometry actuator.

    ``local_frame_fullatom_affine`` is the formal V216 path. Edge geometry and
    Pair features update invariant node states; Cartesian coordinates are then
    updated by a residue-local frame actuator. Legacy EGNN coordinate paths stay
    available for historical reproducibility only.
    """

    def __init__(
        self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
        radial_nf, in_edge_nf=0, in_single_nf=0, num_verts=50,
        act_fn=nn.SiLU(), n_layers=4, residual=True, dropout=0.1, dense=False,
        pair_coord_mode='bounded_residual', pair_coord_delta_bound=1.0,
        coord_tanh=False, coord_normalize=False,
        coord_controller_mode='legacy_unbounded', frame_eps=1.0e-6,
        coordinate_scale=0.1,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.pair_coord_mode = str(pair_coord_mode or 'bounded_residual').strip().lower()
        self.pair_coord_delta_bound = float(pair_coord_delta_bound)
        self.coord_tanh = bool(coord_tanh)
        self.coord_normalize = bool(coord_normalize)
        self.coord_controller_mode = str(coord_controller_mode or 'legacy_unbounded').strip().lower()
        self.local_frame_mode = self.coord_controller_mode == 'local_frame_fullatom_affine'

        self.linear_in = nn.Linear(in_node_nf, hidden_nf)
        self.single_linear = None
        if int(in_single_nf) > 0:
            with torch.random.fork_rng(devices=[]):
                self.single_linear = nn.Linear(int(in_single_nf), hidden_nf, bias=False)
            nn.init.zeros_(self.single_linear.weight)

        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}
        self.dense = dense
        self.linear_out = nn.Linear(
            hidden_nf * (n_layers + 1) if dense else hidden_nf,
            out_node_nf,
        )

        for i in range(n_layers):
            common = dict(
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
                dropout=dropout, pair_coord_mode=self.pair_coord_mode,
                pair_coord_delta_bound=self.pair_coord_delta_bound,
                tanh=self.coord_tanh, normalize=self.coord_normalize,
                update_coords=not self.local_frame_mode,
            )
            self.add_module(
                f'ctx_gcl_{i}',
                AM_E_GCL(hidden_nf, hidden_nf, hidden_nf, n_channel,
                         channel_nf, radial_nf, **common),
            )
            self.add_module(
                f'inter_gcl_{i}',
                AM_E_GCL(hidden_nf, hidden_nf, hidden_nf, n_channel,
                         channel_nf, radial_nf, **common),
            )
            surf_common = dict(common)
            surf_common['surf_nf'] = num_verts
            self.add_module(
                f'surf_gcl_{i}',
                MS_E_GCL(hidden_nf, hidden_nf, hidden_nf, n_channel,
                         channel_nf, radial_nf, **surf_common),
            )

        self.out_layer = AM_E_GCL(
            hidden_nf, hidden_nf, hidden_nf, n_channel,
            channel_nf, radial_nf, edges_in_d=in_edge_nf,
            act_fn=act_fn, residual=residual,
            pair_coord_mode=self.pair_coord_mode,
            pair_coord_delta_bound=self.pair_coord_delta_bound,
            tanh=self.coord_tanh, normalize=self.coord_normalize,
            update_coords=not self.local_frame_mode,
        )

        self.geometry_actuator = None
        if self.local_frame_mode:
            self.geometry_actuator = LocalFrameFullAtomActuator(
                hidden_nf=hidden_nf, channel_nf=channel_nf,
                frame_eps=frame_eps, coordinate_scale=coordinate_scale,
                backbone_channels=min(4, n_channel),
            )

    def _initial_state(self, h, single_attr, capture_bridge_diagnostics):
        base_pre = self.linear_in(h)
        if self.single_linear is None:
            single_delta = torch.zeros_like(base_pre)
        else:
            if single_attr is None:
                single_attr = base_pre.new_zeros((base_pre.shape[0], self.single_linear.in_features))
            single_delta = self.single_linear(single_attr)
        state_pre = base_pre + single_delta
        if self.training and self.dropout.p > 0.0:
            shared_dropout_scale = self.dropout(torch.ones_like(state_pre))
            state = state_pre * shared_dropout_scale
            base = base_pre * shared_dropout_scale
        else:
            state, base = state_pre, base_pre

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
        return state, base

    def _collect_pair(self, module, pair_ratios, pair_weights, capture):
        if not capture:
            return
        diag = getattr(module, 'last_bridge_diagnostics', {}) or {}
        if diag.get('pair_delta_to_base_ratio') is not None:
            pair_ratios.append(diag['pair_delta_to_base_ratio'])
        if diag.get('pair_adapter_weight_rms') is not None:
            pair_weights.append(diag['pair_adapter_weight_rms'])

    def _finalize_pair_diag(self, pair_ratios, pair_weights):
        if pair_ratios:
            stacked = torch.stack(pair_ratios)
            self.last_bridge_diagnostics['bridge_pair_delta_to_base_ratio_mean'] = stacked.mean()
            self.last_bridge_diagnostics['bridge_pair_delta_to_base_ratio_max'] = stacked.max()
        if pair_weights:
            self.last_bridge_diagnostics['bridge_pair_adapter_weight_rms_mean'] = torch.stack(pair_weights).mean()

    @staticmethod
    def _max_diag(records, key, ref):
        vals = [d[key] for d in records if key in d and torch.is_tensor(d[key])]
        return torch.stack(vals).max() if vals else ref.new_zeros(())

    @staticmethod
    def _min_diag(records, key, ref):
        vals = [d[key] for d in records if key in d and torch.is_tensor(d[key])]
        return torch.stack(vals).min() if vals else ref.new_zeros(())

    def _forward_local_frame(
        self, h, x, ctx_edges, inter_mask, inter_x, surf_verts,
        inter_edges, update_mask, inter_update_mask, aligned_edges,
        epi_index, channel_attr, channel_weights, ctx_edge_attr,
        inter_edge_attr, surf_edge_attr, single_attr, capture,
    ):
        h, _ = self._initial_state(h, single_attr, capture)
        pair_ratios, pair_weights = [], []
        actuator_records = []

        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]
        ctx_states, ctx_coords, inter_coords = [], [], []

        def actuate(stage, state, coord, cattr, cweight, movable):
            coord = self.geometry_actuator(
                state, coord, cattr, cweight, movable,
                capture_diagnostics=capture,
            )
            if capture:
                d = {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in (self.geometry_actuator.last_diagnostics or {}).items()
                }
                actuator_records.append(d)
                for key, value in d.items():
                    self.last_coord_diagnostics[f'{stage}.{key}'] = value
            return coord

        for i in range(self.n_layers):
            mod = self._modules[f'ctx_gcl_{i}']
            h, x = mod(
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
                capture_bridge_diagnostics=capture,
            )
            self._collect_pair(mod, pair_ratios, pair_weights, capture)
            # Representation first, then local-frame action (AlphaFold/AbX order).
            x = actuate(f'ctx_{i}', h, x, channel_attr, channel_weights, update_mask)

            # Native -> shadow state only. Shadow coordinates deliberately remain
            # a separate epitope-attached physical state, as in the R05 parent.
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]

            mod = self._modules[f'inter_gcl_{i}']
            inter_h, inter_x = mod(
                inter_h, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
                capture_bridge_diagnostics=capture,
            )
            self._collect_pair(mod, pair_ratios, pair_weights, capture)
            inter_x = actuate(
                f'inter_{i}', inter_h, inter_x,
                inter_channel_attr, inter_channel_weights, inter_update_mask,
            )

            mod = self._modules[f'surf_gcl_{i}']
            inter_h, inter_x = mod(
                inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                inter_channel_attr, inter_channel_weights,
                edge_attr=surf_edge_attr,
                capture_bridge_diagnostics=capture,
            )
            self._collect_pair(mod, pair_ratios, pair_weights, capture)
            inter_x = actuate(
                f'surf_{i}', inter_h, inter_x,
                inter_channel_attr, inter_channel_weights, inter_update_mask,
            )

            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr,
            capture_bridge_diagnostics=capture,
        )
        self._collect_pair(self.out_layer, pair_ratios, pair_weights, capture)
        x = actuate('out', h, x, channel_attr, channel_weights, update_mask)
        ctx_states.append(h)
        ctx_coords.append(x)

        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)

        h = self.dropout(h)
        h = self.linear_out(h)
        if capture:
            self._finalize_pair_diag(pair_ratios, pair_weights)
            ref = x
            # Compatibility keys keep existing trainer forensics meaningful while
            # making clear that no edge coefficient has Cartesian authority.
            self.last_coord_diagnostics['coord_base_coeff_absmax_max'] = ref.new_zeros(())
            self.last_coord_diagnostics['coord_state_coeff_absmax_max'] = ref.new_zeros(())
            self.last_coord_diagnostics['coord_pair_delta_bounded_absmax_max'] = ref.new_zeros(())
            self.last_coord_diagnostics['coord_update_absmax_max'] = self._max_diag(
                actuator_records, 'coord_update_absmax', ref)
            for key in [
                'translation_norm_A_p50','translation_norm_A_p95','translation_norm_A_p99','translation_norm_A_max',
                'rotation_angle_deg_p50','rotation_angle_deg_p95','rotation_angle_deg_p99','rotation_angle_deg_max',
                'sidechain_residual_norm_A_p50','sidechain_residual_norm_A_p95','sidechain_residual_norm_A_p99','sidechain_residual_norm_A_max',
                'atom_update_norm_A_p50','atom_update_norm_A_p95','atom_update_norm_A_p99','atom_update_norm_A_max',
                'frame_orthogonality_error_max', 'fixed_atom_update_absmax_A','backbone_rigid_distance_error_max_A',
                'rigid_head_weight_rms','atom_head_weight_rms',
            ]:
                self.last_coord_diagnostics[f'actuator_{key}_max'] = self._max_diag(actuator_records, key, ref)
            self.last_coord_diagnostics['actuator_frame_valid_fraction_min'] = self._min_diag(
                actuator_records, 'frame_valid_fraction', ref)
            self.last_coord_diagnostics['actuator_movable_frame_valid_fraction_min'] = self._min_diag(
                actuator_records, 'movable_frame_valid_fraction', ref)
        return h, x, inter_x

    def _forward_legacy(
        self, h, x, ctx_edges, inter_mask, inter_x, surf_verts,
        inter_edges, update_mask, inter_update_mask, aligned_edges,
        epi_index, channel_attr, channel_weights, ctx_edge_attr,
        inter_edge_attr, surf_edge_attr, single_attr, capture,
    ):
        h, h_base = self._initial_state(h, single_attr, capture)
        pair_ratios, pair_weights = [], []
        coord_updates, coord_coeffs = [], []
        coord_base_raw_coeffs, coord_state_raw_coeffs = [], []
        coord_diff_norms, coord_direction_norms = [], []
        coord_trans, coord_update_input_ratios = [], []

        def collect_coord(module, stage):
            if not capture:
                return
            diag = getattr(module, 'last_coord_diagnostics', {}) or {}
            for key, value in diag.items():
                self.last_coord_diagnostics[f'{stage}.{key}'] = value
            pairs = [
                ('coord_update_absmax', coord_updates),
                ('coord_coeff_absmax', coord_coeffs),
                ('coord_base_coeff_raw_absmax', coord_base_raw_coeffs),
                ('coord_state_coeff_raw_absmax', coord_state_raw_coeffs),
                ('coord_diff_norm_absmax', coord_diff_norms),
                ('coord_direction_norm_absmax', coord_direction_norms),
                ('coord_trans_absmax', coord_trans),
                ('coord_update_to_input_rms_ratio', coord_update_input_ratios),
            ]
            for key, dst in pairs:
                if diag.get(key) is not None:
                    dst.append(diag[key])

        inter_h = h[inter_mask]
        inter_h_base = h_base[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]
        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(self.n_layers):
            mod = self._modules[f'ctx_gcl_{i}']
            h, h_base, x = mod.forward_dual(
                h, h_base, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr, capture_bridge_diagnostics=capture)
            self._collect_pair(mod, pair_ratios, pair_weights, capture); collect_coord(mod, f'ctx_{i}')

            inter_h = inter_h.clone(); inter_h_base = inter_h_base.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_h_base[inter_update_mask] = h_base[update_mask]

            mod = self._modules[f'inter_gcl_{i}']
            inter_h, inter_h_base, inter_x = mod.forward_dual(
                inter_h, inter_h_base, inter_edges, inter_x,
                inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr, capture_bridge_diagnostics=capture)
            self._collect_pair(mod, pair_ratios, pair_weights, capture); collect_coord(mod, f'inter_{i}')

            mod = self._modules[f'surf_gcl_{i}']
            inter_h, inter_h_base, inter_x = mod.forward_dual(
                inter_h, inter_h_base, aligned_edges, epi_index, inter_x, surf_verts,
                inter_channel_attr, inter_channel_weights,
                edge_attr=surf_edge_attr, capture_bridge_diagnostics=capture)
            self._collect_pair(mod, pair_ratios, pair_weights, capture); collect_coord(mod, f'surf_{i}')

            h = h.clone(); h_base = h_base.clone()
            h[inter_mask] = inter_h; h_base[inter_mask] = inter_h_base
            ctx_states.append(h); ctx_coords.append(x); inter_coords.append(inter_x)

        h, h_base, x = self.out_layer.forward_dual(
            h, h_base, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr, capture_bridge_diagnostics=capture)
        self._collect_pair(self.out_layer, pair_ratios, pair_weights, capture); collect_coord(self.out_layer, 'out')
        ctx_states.append(h); ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)
        h = self.dropout(h); h = self.linear_out(h)
        if capture:
            self._finalize_pair_diag(pair_ratios, pair_weights)
            self.last_coord_diagnostics['pair_coord_delta_bound'] = x.new_tensor(self.pair_coord_delta_bound)
            self.last_coord_diagnostics['coord_tanh'] = x.new_tensor(1.0 if self.coord_tanh else 0.0)
            self.last_coord_diagnostics['coord_normalize'] = x.new_tensor(1.0 if self.coord_normalize else 0.0)
            groups = [
                ('coord_update_absmax', coord_updates), ('coord_coeff_absmax', coord_coeffs),
                ('coord_base_coeff_raw_absmax', coord_base_raw_coeffs), ('coord_state_coeff_raw_absmax', coord_state_raw_coeffs),
                ('coord_diff_norm_absmax', coord_diff_norms), ('coord_direction_norm_absmax', coord_direction_norms),
                ('coord_trans_absmax', coord_trans), ('coord_update_to_input_rms_ratio', coord_update_input_ratios),
            ]
            for name, vals in groups:
                if vals:
                    self.last_coord_diagnostics[f'{name}_max'] = torch.stack(vals).max()
                    if name in {'coord_update_absmax','coord_coeff_absmax'}:
                        self.last_coord_diagnostics[f'{name}_mean'] = torch.stack(vals).mean()
        return h, x, inter_x

    def forward(
        self, h, x, ctx_edges, inter_mask, inter_x, surf_verts,
        inter_edges, update_mask, inter_update_mask, aligned_edges,
        epi_index, channel_attr, channel_weights, ctx_edge_attr=None,
        inter_edge_attr=None, surf_edge_attr=None, single_attr=None,
        capture_bridge_diagnostics=False,
    ):
        capture = bool(capture_bridge_diagnostics)
        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}
        args = (
            h, x, ctx_edges, inter_mask, inter_x, surf_verts,
            inter_edges, update_mask, inter_update_mask, aligned_edges,
            epi_index, channel_attr, channel_weights, ctx_edge_attr,
            inter_edge_attr, surf_edge_attr, single_attr, capture,
        )
        if self.local_frame_mode:
            return self._forward_local_frame(*args)
        return self._forward_legacy(*args)
