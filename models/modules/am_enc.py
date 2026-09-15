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
        coord_tanh=False, coord_normalize=False, coord_prenorm=False,
        pair_adapter_prenorm=False, pair_semantic_gate=False,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.pair_coord_mode = str(pair_coord_mode or 'bounded_residual').strip().lower()
        self.pair_coord_delta_bound = float(pair_coord_delta_bound)
        self.coord_tanh = bool(coord_tanh)
        self.coord_normalize = bool(coord_normalize)
        self.coord_prenorm = bool(coord_prenorm)
        self.pair_adapter_prenorm = bool(pair_adapter_prenorm)
        self.pair_semantic_gate = bool(pair_semantic_gate)

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
                    coord_prenorm=self.coord_prenorm,
                    pair_adapter_prenorm=self.pair_adapter_prenorm,
                    pair_semantic_gate=self.pair_semantic_gate,
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
                    coord_prenorm=self.coord_prenorm,
                    pair_adapter_prenorm=self.pair_adapter_prenorm,
                    pair_semantic_gate=self.pair_semantic_gate,
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
                    coord_prenorm=self.coord_prenorm,
                    pair_adapter_prenorm=self.pair_adapter_prenorm,
                    pair_semantic_gate=self.pair_semantic_gate,
                ),
            )

        self.out_layer = AM_E_GCL(
            hidden_nf, hidden_nf, hidden_nf, n_channel,
            channel_nf, radial_nf, edges_in_d=in_edge_nf,
            act_fn=act_fn, residual=residual,
            pair_coord_mode=self.pair_coord_mode,
            pair_coord_delta_bound=self.pair_coord_delta_bound,
            tanh=self.coord_tanh, normalize=self.coord_normalize,
            coord_prenorm=self.coord_prenorm,
            pair_adapter_prenorm=self.pair_adapter_prenorm,
            pair_semantic_gate=self.pair_semantic_gate,
        )

    def forward(
        self, h, x, ctx_edges, inter_mask, inter_x, surf_verts,
        inter_edges, update_mask, inter_update_mask, aligned_edges,
        epi_index, channel_attr, channel_weights, ctx_edge_attr=None,
        inter_edge_attr=None, surf_edge_attr=None, single_attr=None,
        capture_bridge_diagnostics=False,
        endpoint_to_carrier_sync_fn=None, carrier_to_endpoint_sync_fn=None,
    ):
        """Run the R05 physical recurrence with optional Single/Pair conditioning.

        Formal R67 modes keep Pair-state PreNorm and coordinate-head PreNorm.
        Semantic Pair context enters through a zero-start hidden adapter.  Direct
        Pair geometry can either use historical ``factorized_direct`` hidden-message
        routing or the preferred ``gated_action_residual``
        scalar-action residual, which preserves the mature R05 coordinate actuator.
        No clipping, tanh controller, trust radius, or manual physical step scale is
        introduced.

        V242 single-state sequential-operator mode supplies BOTH analytic chart
        callbacks.  The H3 state then follows one physical trajectory:

            carrier C_i
              -> endpoint E_i = Phi(C_i)
              -> R05 local operator: E_i^L = L(E_i)
              -> carrier chart C_i^L = Phi^{-1}(E_i^L)
              -> interface/transport operator: C_i^T = T(C_i^L)
              -> endpoint chart E_{i+1} = Phi(C_i^T).

        Therefore native ctx/out coordinates are no longer a discarded shadow state:
        their H3 update is immediately written back into the SAME carrier state before
        the transport operator runs.  Conversely, transport updates are immediately
        mapped back to the endpoint chart before the next local operator.  There are
        two sequential geometric operators but only one physical Cartesian state.
        """
        use_dual = self.pair_coord_mode == 'bounded_residual'
        sequential_single_state = (
            endpoint_to_carrier_sync_fn is not None
            or carrier_to_endpoint_sync_fn is not None
        )
        if sequential_single_state and (
            endpoint_to_carrier_sync_fn is None
            or carrier_to_endpoint_sync_fn is None
        ):
            raise RuntimeError(
                'sequential single-state geometry requires both endpoint->carrier '
                'and carrier->endpoint analytic sync callbacks'
            )

        # Parent-preserving Single bridge:
        # h0 = W_parent h + W_single s, W_single(0)=0.
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
            h_base = base_pre * shared_dropout_scale if use_dual else None
        else:
            h = state_pre
            h_base = base_pre if use_dual else None

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

        pair_input_raw_rms = []
        pair_input_prenorm_rms = []
        pair_ratios = []
        pair_coord_ratios = []
        pair_semantic_gate_rms = []
        coord_update_rms = []
        coord_state_raw_rms = []
        coord_diff_norm_rms = []
        coord_head_input_rms = []
        local_to_carrier_sync_gap_rms = []
        carrier_to_endpoint_sync_gap_rms = []
        sequential_local_design_update_rms = []
        sequential_transport_design_update_rms = []

        # Path-authority diagnostics are deliberately computed outside the EGNN
        # layer itself, where AMEncoder knows which coordinate stream is being
        # updated.  This separates the native/full pred_X path (ctx/out) from
        # the shadow recurrent-carrier path (inter/surf) without changing either
        # computation.
        def _masked_rms(delta, mask=None):
            if not capture_bridge_diagnostics:
                return None
            with torch.no_grad():
                part = delta if mask is None else delta[mask]
                if part.numel() == 0:
                    return delta.new_zeros(())
                return part.detach().float().square().mean().sqrt().to(delta.dtype)

        def _masked_absmax(delta, mask=None):
            if not capture_bridge_diagnostics:
                return None
            with torch.no_grad():
                part = delta if mask is None else delta[mask]
                if part.numel() == 0:
                    return delta.new_zeros(())
                return part.detach().float().abs().amax().to(delta.dtype)

        def record_path_delta(stage, stream, before, after, design_mask):
            if not capture_bridge_diagnostics:
                return
            delta = after - before
            self.last_coord_diagnostics[f'{stage}.{stream}_all_update_rms'] = _masked_rms(delta)
            self.last_coord_diagnostics[f'{stage}.{stream}_all_update_absmax'] = _masked_absmax(delta)
            self.last_coord_diagnostics[f'{stage}.{stream}_design_update_rms'] = _masked_rms(delta, design_mask)
            self.last_coord_diagnostics[f'{stage}.{stream}_design_update_absmax'] = _masked_absmax(delta, design_mask)


        def collect_pair(module):
            if not capture_bridge_diagnostics:
                return
            diag = getattr(module, 'last_bridge_diagnostics', {}) or {}
            raw_pair = diag.get('pair_input_raw_rms')
            norm_pair = diag.get('pair_input_prenorm_rms')
            ratio = diag.get('pair_semantic_delta_to_base_ratio', diag.get('pair_delta_to_base_ratio'))
            coord_ratio = diag.get('pair_coordinate_delta_to_base_ratio')
            sem_gate = diag.get('pair_semantic_gate_rms')
            if raw_pair is not None:
                pair_input_raw_rms.append(raw_pair)
            if norm_pair is not None:
                pair_input_prenorm_rms.append(norm_pair)
            if ratio is not None:
                pair_ratios.append(ratio)
            if coord_ratio is not None:
                pair_coord_ratios.append(coord_ratio)
            if sem_gate is not None:
                pair_semantic_gate_rms.append(sem_gate)

        def collect_coord(module, stage):
            if not capture_bridge_diagnostics:
                return
            diag = getattr(module, 'last_coord_diagnostics', {}) or {}
            for key, value in diag.items():
                self.last_coord_diagnostics[f'{stage}.{key}'] = value
            mapping = [
                ('coord_update_rms', coord_update_rms),
                ('coord_state_coeff_raw_rms', coord_state_raw_rms),
                ('coord_diff_norm_rms', coord_diff_norm_rms),
                ('coord_state_head_input_rms', coord_head_input_rms),
            ]
            for key, bucket in mapping:
                value = diag.get(key)
                if value is not None:
                    bucket.append(value)

        inter_h = h[inter_mask]
        inter_h_base = h_base[inter_mask] if use_dual else None
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        for i in range(self.n_layers):
            ctx_layer = self._modules[f'ctx_gcl_{i}']
            native_before = x
            if use_dual:
                h, h_base, native_candidate = ctx_layer.forward_dual(
                    h, h_base, ctx_edges, x, channel_attr, channel_weights,
                    edge_attr=ctx_edge_attr,
                    capture_bridge_diagnostics=capture_bridge_diagnostics,
                )
            else:
                h, native_candidate = ctx_layer(
                    h, ctx_edges, x, channel_attr, channel_weights,
                    edge_attr=ctx_edge_attr,
                    capture_bridge_diagnostics=capture_bridge_diagnostics,
                )
            # Operator 1: mature R05 local/intrinsic geometry update.
            # In sequential-single-state mode this H3 update is PHYSICAL: it is
            # immediately converted into the carrier chart before the transport
            # operator, rather than becoming a discarded second coordinate field.
            x = native_candidate
            collect_pair(ctx_layer)
            collect_coord(ctx_layer, f'ctx_{i}')
            record_path_delta(f'ctx_{i}', 'local', native_before, x, update_mask)
            if capture_bridge_diagnostics and sequential_single_state:
                dlocal = x[update_mask] - native_before[update_mask]
                sequential_local_design_update_rms.append(_masked_rms(dlocal))

            # Shared hidden state flows from local reasoning into interface reasoning.
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            if use_dual:
                inter_h_base = inter_h_base.clone()
                inter_h_base[inter_update_mask] = h_base[update_mask]

            if sequential_single_state:
                # Close the SAME physical state after the local operator:
                # endpoint view -> analytic carrier view.  Only H3 rows change;
                # antigen/context coordinates in inter_x remain fixed.
                synced_carrier = endpoint_to_carrier_sync_fn(x[update_mask])
                if tuple(synced_carrier.shape) != tuple(inter_x[inter_update_mask].shape):
                    raise RuntimeError(
                        'endpoint_to_carrier_sync_fn shape mismatch: '
                        f'synced={tuple(synced_carrier.shape)} '
                        f'expected={tuple(inter_x[inter_update_mask].shape)}'
                    )
                inter_x = inter_x.clone()
                inter_x[inter_update_mask] = synced_carrier
                if capture_bridge_diagnostics:
                    endpoint_rt = carrier_to_endpoint_sync_fn(
                        inter_x[inter_update_mask]
                    )
                    local_to_carrier_sync_gap_rms.append(
                        _masked_rms(endpoint_rt - x[update_mask])
                    )

            inter_layer = self._modules[f'inter_gcl_{i}']
            carrier_before_inter = inter_x
            if use_dual:
                inter_h, inter_h_base, inter_x = inter_layer.forward_dual(
                    inter_h, inter_h_base, inter_edges, inter_x,
                    inter_channel_attr, inter_channel_weights,
                    edge_attr=inter_edge_attr,
                    capture_bridge_diagnostics=capture_bridge_diagnostics,
                )
            else:
                inter_h, inter_x = inter_layer(
                    inter_h, inter_edges, inter_x,
                    inter_channel_attr, inter_channel_weights,
                    edge_attr=inter_edge_attr,
                    capture_bridge_diagnostics=capture_bridge_diagnostics,
                )
            collect_pair(inter_layer)
            collect_coord(inter_layer, f'inter_{i}')
            record_path_delta(f'inter_{i}', 'transport', carrier_before_inter, inter_x, inter_update_mask)

            surf_layer = self._modules[f'surf_gcl_{i}']
            carrier_before_surf = inter_x
            if use_dual:
                inter_h, inter_h_base, inter_x = surf_layer.forward_dual(
                    inter_h, inter_h_base, aligned_edges, epi_index,
                    inter_x, surf_verts, inter_channel_attr,
                    inter_channel_weights, edge_attr=surf_edge_attr,
                    capture_bridge_diagnostics=capture_bridge_diagnostics,
                )
            else:
                inter_h, inter_x = surf_layer(
                    inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                    inter_channel_attr, inter_channel_weights,
                    edge_attr=surf_edge_attr,
                    capture_bridge_diagnostics=capture_bridge_diagnostics,
                )
            collect_pair(surf_layer)
            collect_coord(surf_layer, f'surf_{i}')
            record_path_delta(f'surf_{i}', 'transport', carrier_before_surf, inter_x, inter_update_mask)

            if sequential_single_state:
                # Operator 2 has now moved the carrier state.  Convert it back to
                # the endpoint/native chart before the next local operator.
                transport_before = x[update_mask]
                synced_design = carrier_to_endpoint_sync_fn(
                    inter_x[inter_update_mask]
                )
                if tuple(synced_design.shape) != tuple(x[update_mask].shape):
                    raise RuntimeError(
                        'carrier_to_endpoint_sync_fn shape mismatch: '
                        f'synced={tuple(synced_design.shape)} '
                        f'expected={tuple(x[update_mask].shape)}'
                    )
                x = x.clone()
                x[update_mask] = synced_design
                if capture_bridge_diagnostics:
                    sequential_transport_design_update_rms.append(
                        _masked_rms(x[update_mask] - transport_before)
                    )
                    carrier_rt = endpoint_to_carrier_sync_fn(x[update_mask])
                    carrier_to_endpoint_sync_gap_rms.append(
                        _masked_rms(carrier_rt - inter_x[inter_update_mask])
                    )

            # Transport hidden state -> local hidden state.
            h = h.clone()
            h[inter_mask] = inter_h
            if use_dual:
                h_base = h_base.clone()
                h_base[inter_mask] = inter_h_base
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        native_before_out = x
        if use_dual:
            h, h_base, native_candidate = self.out_layer.forward_dual(
                h, h_base, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
        else:
            h, native_candidate = self.out_layer(
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
        # Final R05 local readout is also part of the SAME physical state.
        # Convert it into the carrier chart so the returned carrier/endpoint pair
        # remains exactly one state in two analytic charts.
        x = native_candidate
        collect_pair(self.out_layer)
        collect_coord(self.out_layer, 'out')
        record_path_delta('out', 'local', native_before_out, x, update_mask)
        if sequential_single_state:
            if capture_bridge_diagnostics:
                sequential_local_design_update_rms.append(
                    _masked_rms(x[update_mask] - native_before_out[update_mask])
                )
            synced_carrier = endpoint_to_carrier_sync_fn(x[update_mask])
            inter_x = inter_x.clone()
            inter_x[inter_update_mask] = synced_carrier
            if capture_bridge_diagnostics:
                endpoint_rt = carrier_to_endpoint_sync_fn(
                    inter_x[inter_update_mask]
                )
                local_to_carrier_sync_gap_rms.append(
                    _masked_rms(endpoint_rt - x[update_mask])
                )
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
            self.last_coord_diagnostics['pair_coord_delta_bound'] = x.new_tensor(
                self.pair_coord_delta_bound if use_dual else float('nan')
            )
            self.last_coord_diagnostics['pair_direct_shared'] = x.new_tensor(
                1.0 if self.pair_coord_mode == 'direct_shared' else 0.0
            )
            self.last_coord_diagnostics['pair_factorized_direct'] = x.new_tensor(
                1.0 if self.pair_coord_mode == 'factorized_direct' else 0.0
            )
            self.last_coord_diagnostics['pair_gated_action_residual'] = x.new_tensor(
                1.0 if self.pair_coord_mode == 'gated_action_residual' else 0.0
            )
            self.last_coord_diagnostics['pair_semantic_gate_enabled'] = x.new_tensor(
                1.0 if self.pair_semantic_gate else 0.0
            )
            self.last_coord_diagnostics['coord_tanh'] = x.new_tensor(
                1.0 if self.coord_tanh else 0.0
            )
            self.last_coord_diagnostics['coord_normalize'] = x.new_tensor(
                1.0 if self.coord_normalize else 0.0
            )
            self.last_coord_diagnostics['coord_prenorm'] = x.new_tensor(
                1.0 if self.coord_prenorm else 0.0
            )
            if pair_input_raw_rms:
                self.last_bridge_diagnostics['bridge_pair_input_raw_rms_mean'] = torch.stack(pair_input_raw_rms).mean()
            if pair_input_prenorm_rms:
                self.last_bridge_diagnostics['bridge_pair_input_prenorm_rms_mean'] = torch.stack(pair_input_prenorm_rms).mean()
            if pair_ratios:
                self.last_bridge_diagnostics['bridge_pair_delta_to_base_ratio_mean'] = torch.stack(pair_ratios).mean()
            if pair_coord_ratios:
                self.last_bridge_diagnostics['bridge_pair_coordinate_delta_to_base_ratio_mean'] = torch.stack(pair_coord_ratios).mean()
            if pair_semantic_gate_rms:
                self.last_bridge_diagnostics['bridge_pair_semantic_gate_rms_mean'] = torch.stack(pair_semantic_gate_rms).mean()
            if coord_update_rms:
                self.last_coord_diagnostics['coord_update_rms_max'] = torch.stack(coord_update_rms).max()
            if coord_state_raw_rms:
                self.last_coord_diagnostics['coord_state_coeff_raw_rms_max'] = torch.stack(coord_state_raw_rms).max()
            if coord_diff_norm_rms:
                self.last_coord_diagnostics['coord_diff_norm_rms_max'] = torch.stack(coord_diff_norm_rms).max()
            if coord_head_input_rms:
                self.last_coord_diagnostics['coord_state_head_input_rms_max'] = torch.stack(coord_head_input_rms).max()

            # Aggregate only the stream-specific quantities that answer the
            # current scientific question: which coordinate path first becomes
            # unstable?  Stage-level entries remain available for sparse
            # outlier forensics.
            def _stage_values(prefixes, suffix):
                vals = []
                for key, value in self.last_coord_diagnostics.items():
                    stage = key.split('.', 1)[0]
                    if stage.startswith(prefixes) and key.endswith(suffix) and torch.is_tensor(value):
                        vals.append(value)
                return vals

            native_dx = _stage_values(('ctx_', 'out'), '.local_design_update_rms')
            carrier_dx = _stage_values(('inter_', 'surf_'), '.transport_design_update_rms')
            native_dx_max = _stage_values(('ctx_', 'out'), '.local_design_update_absmax')
            carrier_dx_max = _stage_values(('inter_', 'surf_'), '.transport_design_update_absmax')
            native_alpha = _stage_values(('ctx_', 'out'), '.coord_state_coeff_raw_rms')
            carrier_alpha = _stage_values(('inter_', 'surf_'), '.coord_state_coeff_raw_rms')
            native_alpha_max = _stage_values(('ctx_', 'out'), '.coord_state_coeff_raw_absmax')
            carrier_alpha_max = _stage_values(('inter_', 'surf_'), '.coord_state_coeff_raw_absmax')
            native_lever = _stage_values(('ctx_', 'out'), '.coord_diff_norm_rms')
            carrier_lever = _stage_values(('inter_', 'surf_'), '.coord_diff_norm_rms')
            native_lever_max = _stage_values(('ctx_', 'out'), '.coord_diff_norm_absmax')
            carrier_lever_max = _stage_values(('inter_', 'surf_'), '.coord_diff_norm_absmax')
            native_basis = _stage_values(('ctx_', 'out'), '.coord_direction_norm_rms')
            native_mech = _stage_values(('ctx_', 'out'), '.coord_update_to_alpha_rms_ratio')
            if native_dx:
                self.last_coord_diagnostics['native_design_update_rms_max'] = torch.stack(native_dx).max()
            if carrier_dx:
                self.last_coord_diagnostics['carrier_design_update_rms_max'] = torch.stack(carrier_dx).max()
            if native_dx_max:
                self.last_coord_diagnostics['native_design_update_absmax_max'] = torch.stack(native_dx_max).max()
            if carrier_dx_max:
                self.last_coord_diagnostics['carrier_design_update_absmax_max'] = torch.stack(carrier_dx_max).max()
            if native_alpha:
                self.last_coord_diagnostics['native_alpha_rms_max'] = torch.stack(native_alpha).max()
            if carrier_alpha:
                self.last_coord_diagnostics['carrier_alpha_rms_max'] = torch.stack(carrier_alpha).max()
            if native_alpha_max:
                self.last_coord_diagnostics['native_alpha_absmax_max'] = torch.stack(native_alpha_max).max()
            if carrier_alpha_max:
                self.last_coord_diagnostics['carrier_alpha_absmax_max'] = torch.stack(carrier_alpha_max).max()
            if native_lever:
                self.last_coord_diagnostics['native_lever_rms_max'] = torch.stack(native_lever).max()
            if carrier_lever:
                self.last_coord_diagnostics['carrier_lever_rms_max'] = torch.stack(carrier_lever).max()
            if native_lever_max:
                self.last_coord_diagnostics['native_lever_absmax_max'] = torch.stack(native_lever_max).max()
            if carrier_lever_max:
                self.last_coord_diagnostics['carrier_lever_absmax_max'] = torch.stack(carrier_lever_max).max()
            if native_basis:
                self.last_coord_diagnostics['native_basis_norm_rms_max'] = torch.stack(native_basis).max()
            if native_mech:
                self.last_coord_diagnostics['native_update_to_alpha_rms_ratio_max'] = torch.stack(native_mech).max()

            self.last_coord_diagnostics['sequential_single_state'] = x.new_tensor(
                1.0 if sequential_single_state else 0.0
            )
            if sequential_local_design_update_rms:
                vals = torch.stack(sequential_local_design_update_rms)
                self.last_coord_diagnostics['sequential_local_update_rms_mean'] = vals.mean()
                self.last_coord_diagnostics['sequential_local_update_rms_max'] = vals.max()
            if sequential_transport_design_update_rms:
                vals = torch.stack(sequential_transport_design_update_rms)
                self.last_coord_diagnostics['sequential_transport_update_rms_mean'] = vals.mean()
                self.last_coord_diagnostics['sequential_transport_update_rms_max'] = vals.max()
            if local_to_carrier_sync_gap_rms:
                self.last_coord_diagnostics['sequential_local_to_carrier_sync_gap_rms_max'] = (
                    torch.stack(local_to_carrier_sync_gap_rms).max()
                )
            if carrier_to_endpoint_sync_gap_rms:
                self.last_coord_diagnostics['sequential_carrier_to_endpoint_sync_gap_rms_max'] = (
                    torch.stack(carrier_to_endpoint_sync_gap_rms).max()
                )

        return h, x, inter_x
