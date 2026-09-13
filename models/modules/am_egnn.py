#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_softmax

from utils.singleton import singleton


class AMEGNN(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4,
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
            self.add_module(f'gcl_{i}', AM_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf, radial_nf,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
        self.out_layer = AM_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel, channel_nf,
            radial_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual
        )
            
    def forward(self, h, x, edges, channel_attr, channel_weights, ctx_edge_attr=None):
        h = self.linear_in(h)
        h = self.dropout(h)

        ctx_states, ctx_coords = [], []
        for i in range(0, self.n_layers):
            h, x = self._modules[f'gcl_{i}'](
                h, edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr)

            ctx_states.append(h)
            ctx_coords.append(x)

        h, x = self.out_layer(
            h, edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x

'''
Below are the implementation of the adaptive multi-channel message passing mechanism
'''

@singleton
class RollerPooling(nn.Module):
    '''
    Adaptive average pooling for the adaptive scaler
    '''
    def __init__(self, n_channel) -> None:
        super().__init__()
        self.n_channel = n_channel
        with torch.no_grad():
            pool_matrix = []
            ones = torch.ones((n_channel, n_channel), dtype=torch.float)
            for i in range(n_channel):
                # i start from 0 instead of 1 !!! (less readable but higher implemetation efficiency)
                window_size = n_channel - i
                mat = torch.triu(ones) - torch.triu(ones, diagonal=window_size)
                pool_matrix.append(mat / window_size)
            self.pool_matrix = torch.stack(pool_matrix)
    
    def forward(self, hidden, target_size):
        '''
        :param hidden: [n_edges, n_channel]
        :param target_size: [n_edges]
        '''
        pool_mat = self.pool_matrix.to(hidden.device).type(hidden.dtype)
        pool_mat = pool_mat[target_size - 1]  # [n_edges, n_channel, n_channel]
        hidden = hidden.unsqueeze(-1)  # [n_edges, n_channel, 1]
        return torch.bmm(pool_mat, hidden)  # [n_edges, n_channel, 1]


class AM_E_GCL(nn.Module):
    '''
    Adaptive Multi-Channel E(n) Equivariant Convolutional Layer
    '''

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, channel_nf, radial_nf,
                 edges_in_d=0, node_attr_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False, dropout=0.1,
                 pair_coord_mode='bounded_residual', pair_coord_delta_bound=1.0):
        super(AM_E_GCL, self).__init__()

        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.pair_coord_mode = str(pair_coord_mode or 'bounded_residual').strip().lower()
        if self.pair_coord_mode not in {'bounded_residual', 'legacy_shared'}:
            raise ValueError(f'Unsupported pair_coord_mode={self.pair_coord_mode!r}')
        self.pair_coord_delta_bound = float(pair_coord_delta_bound)
        if self.pair_coord_delta_bound <= 0.0:
            raise ValueError('pair_coord_delta_bound must be > 0')

        self.dropout = nn.Dropout(dropout)

        # V211 parent-preserving Pair localization.
        # A concat([h_i, h_j, radial, z_ij]) layer is algebraically identical
        # to W_base*base+b + W_z*z.  Keeping the two terms separate lets W_z
        # start at exactly zero without changing any R05 parameter or equation.
        self.edges_in_d = int(edges_in_d)
        input_edge = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + radial_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.edge_attr_linear = None
        if self.edges_in_d > 0:
            # nn.Linear initializes before it can be zeroed.  Isolating that
            # draw is necessary for identical shared initialization across
            # R28/R29/R30 and against the R05 parent.
            with torch.random.fork_rng(devices=[]):
                self.edge_attr_linear = nn.Linear(
                    self.edges_in_d, hidden_nf, bias=False
                )
            nn.init.zeros_(self.edge_attr_linear.weight)
        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}
        self.capture_bridge_diagnostics = False
        self.radial_linear = nn.Linear(channel_nf ** 2, radial_nf)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + node_attr_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        # Keep the pre-activation explicit so diagnostics can distinguish an
        # upstream gain excursion from the bounded coordinate controller.
        # When ``self.tanh`` is enabled, the exact EGNN tanh transform is
        # applied in ``coord_model`` below; for tanh=False this is bitwise the
        # same computation as the previous implementation.
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr, base_source=None, base_target=None):
        """Return state and pair-free coordinate messages from one edge path.

        ``state_edge`` keeps the full learned pair conditioning.  ``base_edge``
        removes only the *direct* z_ij adapter before the shared nonlinear edge
        stack.  A single shared dropout mask is applied to both streams, so the
        state stream preserves the original stochastic contract while the
        coordinate stream can measure a clean direct-pair residual.
        """
        radial = radial.reshape(radial.shape[0], -1)
        if base_source is None:
            base_source = source
        if base_target is None:
            base_target = target
        state_input = torch.cat([source, target, radial], dim=1)
        base_input = torch.cat([base_source, base_target, radial], dim=1)
        base_pre = self.edge_mlp[0](base_input)
        state_parent_pre = self.edge_mlp[0](state_input)
        if self.edge_attr_linear is None:
            pair_delta = torch.zeros_like(base_pre)
        else:
            if edge_attr is None:
                edge_attr = base_pre.new_zeros((base_pre.shape[0], self.edges_in_d))
            pair_delta = self.edge_attr_linear(edge_attr)
        state_pre = state_parent_pre + pair_delta

        if self.capture_bridge_diagnostics:
            with torch.no_grad():
                base_f = state_parent_pre.detach().float()
                delta_f = pair_delta.detach().float()
                combined_f = state_pre.detach().float()
                base_rms = base_f.square().mean().sqrt()
                delta_rms = delta_f.square().mean().sqrt()
                self.last_bridge_diagnostics = {
                    'pair_base_preact_rms': base_rms.to(base_pre.dtype),
                    'pair_delta_rms': delta_rms.to(base_pre.dtype),
                    'pair_delta_absmax': delta_f.abs().amax().to(base_pre.dtype),
                    'pair_combined_preact_rms': combined_f.square().mean().sqrt().to(base_pre.dtype),
                    'pair_delta_to_base_ratio': (
                        delta_rms / base_rms.clamp_min(1.0e-8)
                    ).to(base_pre.dtype),
                    'pair_adapter_weight_rms': (
                        base_rms.new_zeros(()) if self.edge_attr_linear is None
                        else self.edge_attr_linear.weight.detach().float().square().mean().sqrt()
                    ).to(base_pre.dtype),
                }
        else:
            self.last_bridge_diagnostics = {}

        base_edge = base_pre
        state_edge = state_pre
        for layer in self.edge_mlp[1:]:
            base_edge = layer(base_edge)
            state_edge = layer(state_edge)

        # One Bernoulli mask, shared by both streams.  At zero pair adapter the
        # two values are exactly identical, preserving the R05 parent value.
        if self.training and self.dropout.p > 0.0:
            dropout_scale = self.dropout(torch.ones_like(state_edge))
            base_edge = base_edge * dropout_scale
            state_edge = state_edge * dropout_scale

        if self.attention:
            base_edge = base_edge * self.att_mlp(base_edge)
            state_edge = state_edge * self.att_mlp(state_edge)
        return state_edge, base_edge

    def node_model(self, x, edge_index, edge_attr, node_attr):
        '''
        :param x: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param edge_attr: [n_edge, hidden_size], refers to message from i to j
        :param node_attr: [bs * n_node, node_dim]
        '''
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # [bs * n_node, hidden_size]
        # print_log(f'agg1, {torch.isnan(agg).sum()}', level='DEBUG')
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # [bs * n_node, input_size + hidden_size]
        # print_log(f'agg, {torch.isnan(agg).sum()}', level='DEBUG')
        out = self.node_mlp(agg)  # [bs * n_node, output_size]
        # print_log(f'out, {torch.isnan(out).sum()}', level='DEBUG')
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def node_model_dual(self, state_x, base_x, edge_index, state_edge_attr,
                        base_edge_attr, node_attr):
        """Shared-parameter state/base node update with one dropout mask."""
        row, col = edge_index
        state_agg = unsorted_segment_sum(
            state_edge_attr, row, num_segments=state_x.size(0)
        )
        base_agg = unsorted_segment_sum(
            base_edge_attr, row, num_segments=base_x.size(0)
        )
        if node_attr is not None:
            state_in = torch.cat([state_x, state_agg, node_attr], dim=1)
            base_in = torch.cat([base_x, base_agg, node_attr], dim=1)
        else:
            state_in = torch.cat([state_x, state_agg], dim=1)
            base_in = torch.cat([base_x, base_agg], dim=1)
        state_out = self.node_mlp(state_in)
        base_out = self.node_mlp(base_in)
        if self.training and self.dropout.p > 0.0:
            dropout_scale = self.dropout(torch.ones_like(state_out))
            state_out = state_out * dropout_scale
            base_out = base_out * dropout_scale
        if self.residual:
            state_out = state_x + state_out
            base_out = base_x + base_out
        return state_out, base_out

    def coord_model(self, coord, edge_index, coord_diff, state_edge_feat,
                    channel_weights, base_edge_feat=None):
        '''Pair-aware Cartesian update with bounded direct-pair authority.

        The state/message stream may use the full pair-conditioned edge feature.
        Geometry uses the pair-free base coefficient plus a bounded residual from
        direct pair conditioning.  With the V215 EGNN controller enabled:

            alpha_base = tanh(alpha_base_raw)
            alpha_state = tanh(alpha_state_raw)
            alpha = alpha_base + B * tanh((alpha_state-alpha_base) / B).

        For a zero pair adapter, alpha_state == alpha_base exactly.  Around the
        origin tanh(z)=z+O(z^3), so the zero-start parent is first-order
        preserved; only high-gain coordinate authority is saturated.  This is
        scalar-field bounding plus canonical direction normalization, not coordinate clipping.
        '''
        row, col = edge_index
        n_channel = channel_weights.shape[-1]
        coord_before = coord

        # EGNN coordinate controller.  The official EGNN implementation
        # exposes ``tanh`` precisely to bound phi_x(m_ij).  We keep the raw
        # scalar for forensics, then apply tanh before any Cartesian authority
        # is granted.  This preserves E(n) equivariance because only an
        # invariant scalar is transformed; coordinate directions are unchanged.
        state_coeff_raw = self.coord_mlp(state_edge_feat)
        state_coeff = torch.tanh(state_coeff_raw) if self.tanh else state_coeff_raw
        if base_edge_feat is None or self.pair_coord_mode == 'legacy_shared':
            base_coeff_raw = state_coeff_raw
            base_coeff = state_coeff
            pair_coeff_raw = torch.zeros_like(state_coeff)
            pair_coeff_bounded = pair_coeff_raw
            coord_coeff = state_coeff
        else:
            base_coeff_raw = self.coord_mlp(base_edge_feat)
            base_coeff = torch.tanh(base_coeff_raw) if self.tanh else base_coeff_raw
            # Preserve V212 semantics: direct Pair authority is measured after
            # the same coordinate controller used by state/base.  Hence zero
            # Pair still gives exact equality, while the residual remains
            # independently bounded by ``pair_coord_delta_bound``.
            pair_coeff_raw = state_coeff - base_coeff
            bound = pair_coeff_raw.new_tensor(self.pair_coord_delta_bound)
            pair_coeff_bounded = bound * torch.tanh(pair_coeff_raw / bound)
            coord_coeff = base_coeff + pair_coeff_bounded

        # V215 canonical EGNN direction normalization.  Keep the original
        # distance-dependent radial/message features unchanged; normalize only
        # the equivariant vector that grants Cartesian authority.  This mirrors
        # the reference EGNN implementation: the norm is detached so the
        # denominator is not an auxiliary gradient-control path.
        coord_diff_raw = coord_diff
        coord_diff_norm = torch.norm(coord_diff_raw, dim=-1, keepdim=True)
        if self.normalize:
            coord_direction = coord_diff_raw / (coord_diff_norm.detach() + self.epsilon)
        else:
            coord_direction = coord_diff_raw

        channel_sum = (channel_weights != 0).long().sum(-1)
        pooled_edge_feat = RollerPooling(n_channel)(coord_coeff, channel_sum[row])
        trans = coord_direction * pooled_edge_feat

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord_before + agg

        if self.capture_bridge_diagnostics:
            with torch.no_grad():
                def _rms(v):
                    vf = v.detach().float()
                    return vf.square().mean().sqrt().to(coord_before.dtype) if vf.numel() else coord_before.new_zeros(())
                def _amax(v):
                    vf = v.detach().float()
                    return vf.abs().amax().to(coord_before.dtype) if vf.numel() else coord_before.new_zeros(())
                self.last_coord_diagnostics = {
                    'coord_input_rms': _rms(coord_before),
                    'coord_input_absmax': _amax(coord_before),
                    'coord_diff_rms': _rms(coord_diff_raw),
                    'coord_diff_absmax': _amax(coord_diff_raw),
                    'coord_diff_norm_rms': _rms(coord_diff_norm),
                    'coord_diff_norm_absmax': _amax(coord_diff_norm),
                    'coord_direction_rms': _rms(coord_direction),
                    'coord_direction_absmax': _amax(coord_direction),
                    'coord_direction_norm_absmax': _amax(torch.norm(coord_direction, dim=-1)),
                    'coord_normalize': coord_before.new_tensor(1.0 if self.normalize else 0.0),
                    'coord_base_coeff_raw_rms': _rms(base_coeff_raw),
                    'coord_base_coeff_raw_absmax': _amax(base_coeff_raw),
                    'coord_state_coeff_raw_rms': _rms(state_coeff_raw),
                    'coord_state_coeff_raw_absmax': _amax(state_coeff_raw),
                    'coord_base_coeff_rms': _rms(base_coeff),
                    'coord_base_coeff_absmax': _amax(base_coeff),
                    'coord_state_coeff_rms': _rms(state_coeff),
                    'coord_state_coeff_absmax': _amax(state_coeff),
                    'coord_pair_delta_raw_rms': _rms(pair_coeff_raw),
                    'coord_pair_delta_raw_absmax': _amax(pair_coeff_raw),
                    'coord_pair_delta_bounded_rms': _rms(pair_coeff_bounded),
                    'coord_pair_delta_bounded_absmax': _amax(pair_coeff_bounded),
                    'coord_coeff_rms': _rms(coord_coeff),
                    'coord_coeff_absmax': _amax(coord_coeff),
                    'coord_trans_rms': _rms(trans),
                    'coord_trans_absmax': _amax(trans),
                    'coord_update_rms': _rms(agg),
                    'coord_update_absmax': _amax(agg),
                    'coord_update_to_input_rms_ratio': (
                        _rms(agg).float() / _rms(coord_before).float().clamp_min(1.0e-8)
                    ).to(coord_before.dtype),
                    'coord_output_rms': _rms(coord),
                    'coord_output_absmax': _amax(coord),
                }
        else:
            self.last_coord_diagnostics = {}
        return coord

    def forward(self, h, edge_index, coord, channel_attr, channel_weights,
                edge_attr=None, node_attr=None, capture_bridge_diagnostics=False):
        '''
        h: [bs * n_node, hidden_size]
        edge_index: list of [n_row] and [n_col] where n_row == n_col (with no cutoff, n_row == bs * n_node * (n_node - 1))
        coord: [bs * n_node, n_channel, d]
        channel_attr: [bs * n_node, n_channel, channel_nf]
        channel_weights: [bs * n_node, n_channel]
        '''
        row, col = edge_index
        self.capture_bridge_diagnostics = bool(capture_bridge_diagnostics)
        # print('row, col : ', row, col)

        radial, coord_diff = coord2radial(edge_index, coord, channel_attr, channel_weights, self.radial_linear)
        edge_feat, base_edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(
            coord, edge_index, coord_diff, edge_feat, channel_weights,
            base_edge_feat=base_edge_feat,
        )
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord

    def forward_dual(self, h, h_base, edge_index, coord, channel_attr,
                     channel_weights, edge_attr=None, node_attr=None,
                     capture_bridge_diagnostics=False):
        """Relational state stream + R05-like geometry reference stream.

        ``h`` receives single/pair relational context. ``h_base`` never receives
        the direct single/pair adapters.  Coordinate authority is the base-stream
        coefficient plus a bounded residual from the relational stream.
        """
        row, col = edge_index
        self.capture_bridge_diagnostics = bool(capture_bridge_diagnostics)
        radial, coord_diff = coord2radial(
            edge_index, coord, channel_attr, channel_weights, self.radial_linear
        )
        edge_feat, base_edge_feat = self.edge_model(
            h[row], h[col], radial, edge_attr,
            base_source=h_base[row], base_target=h_base[col],
        )
        coord = self.coord_model(
            coord, edge_index, coord_diff, edge_feat, channel_weights,
            base_edge_feat=base_edge_feat,
        )
        h, h_base = self.node_model_dual(
            h, h_base, edge_index, edge_feat, base_edge_feat, node_attr
        )
        return h, h_base, coord


def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


CONSTANT = 1
NUM_SEG = 1  # if you do not have enough memory or you have large attr_size, increase this parameter

def coord2radial(edge_index, coord, attr, channel_weights, linear_map):
    '''
    :param edge_index: tuple([n_edge], [n_edge]) which is tuple of (row, col)
    :param coord: [N, n_channel, d]
    :param attr: [N, n_channel, attr_size], attribute embedding of each channel
    :param channel_weights: [N, n_channel], weights of different channels
    :param linear_map: nn.Linear, map features to d_out
    :param num_seg: split row/col into segments to reduce memory cost
    '''
    row, col = edge_index
    
    radials = []

    seg_size = (len(row) + NUM_SEG - 1) // NUM_SEG

    for i in range(NUM_SEG):
        start = i * seg_size
        end = min(start + seg_size, len(row))
        if end <= start:
            break
        seg_row, seg_col = row[start:end], col[start:end]

        coord_msg = torch.norm(
            coord[seg_row].unsqueeze(2) - coord[seg_col].unsqueeze(1),  # [n_edge, n_channel, n_channel, d]
            dim=-1, keepdim=False)  # [n_edge, n_channel, n_channel]
        
        coord_msg = coord_msg * torch.bmm(
            channel_weights[seg_row].unsqueeze(2),
            channel_weights[seg_col].unsqueeze(1)
            )  # [n_edge, n_channel, n_channel]
        
        radial = torch.bmm(
            attr[seg_row].transpose(-1, -2),  # [n_edge, attr_size, n_channel], [n_edge, 16, 14]
            coord_msg)  # [n_edge, n_channel, n_channel], [n_edge, 14, 14]
        radial = torch.bmm(radial, attr[seg_col])  # [n_edge, attr_size, attr_size]
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, attr_size * attr_size]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True) + CONSTANT  # post norm
        radial = linear_map(radial) / radial_norm # [n_edge, d_out]

        radials.append(radial)
    
    radials = torch.cat(radials, dim=0)  # [N_edge, d_out]

    # generate coord_diff by first mean src then minused by dst
    # message passed from col to row
    channel_mask = (channel_weights != 0).long()  # [N, n_channel]
    channel_sum = channel_mask.sum(-1)  # [N]
    pooled_col_coord = (coord[col] * channel_mask[col].unsqueeze(-1)).sum(1)  # [n_edge, d]
    pooled_col_coord = pooled_col_coord / channel_sum[col].unsqueeze(-1)  # [n_edge, d], denominator cannot be 0 since no pad node exists
    coord_diff = coord[row] - pooled_col_coord.unsqueeze(1)  # [n_edge, n_channel, d]

    return radials, coord_diff


class MS_E_GCL(nn.Module):
    '''
    Multi-Channel Surface E(n) Equivariant Convolutional Layer
    '''

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, channel_nf, radial_nf, surf_nf=50,
                 edges_in_d=0, node_attr_d=0, act_fn=nn.SiLU(), residual=True, attention=False,
                 normalize=False, coords_agg='mean', tanh=False, dropout=0.1,
                 pair_coord_mode='bounded_residual', pair_coord_delta_bound=1.0):
        super(MS_E_GCL, self).__init__()

        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.pair_coord_mode = str(pair_coord_mode or 'bounded_residual').strip().lower()
        if self.pair_coord_mode not in {'bounded_residual', 'legacy_shared'}:
            raise ValueError(f'Unsupported pair_coord_mode={self.pair_coord_mode!r}')
        self.pair_coord_delta_bound = float(pair_coord_delta_bound)
        if self.pair_coord_delta_bound <= 0.0:
            raise ValueError('pair_coord_delta_bound must be > 0')

        self.dropout = nn.Dropout(dropout)

        self.edges_in_d = int(edges_in_d)
        input_edge = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + radial_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.edge_attr_linear = None
        if self.edges_in_d > 0:
            with torch.random.fork_rng(devices=[]):
                self.edge_attr_linear = nn.Linear(
                    self.edges_in_d, hidden_nf, bias=False
                )
            nn.init.zeros_(self.edge_attr_linear.weight)
        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}
        self.capture_bridge_diagnostics = False
        self.radial_linear = nn.Linear(channel_nf ** 2, radial_nf)
        self.scale_linear = nn.Linear(surf_nf, channel_nf)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + node_attr_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        # Keep the pre-activation explicit so diagnostics can distinguish an
        # upstream gain excursion from the bounded coordinate controller.
        # When ``self.tanh`` is enabled, the exact EGNN tanh transform is
        # applied in ``coord_model`` below; for tanh=False this is bitwise the
        # same computation as the previous implementation.
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr, base_source=None, base_target=None):
        """Return surface state and pair-free coordinate messages from one edge path.

        ``state_edge`` keeps the full learned pair conditioning.  ``base_edge``
        removes only the *direct* z_ij adapter before the shared nonlinear edge
        stack.  A single shared dropout mask is applied to both streams, so the
        state stream preserves the original stochastic contract while the
        coordinate stream can measure a clean direct-pair residual.
        """
        radial = radial.reshape(radial.shape[0], -1)
        if base_source is None:
            base_source = source
        if base_target is None:
            base_target = target
        state_input = torch.cat([source, target, radial], dim=1)
        base_input = torch.cat([base_source, base_target, radial], dim=1)
        base_pre = self.edge_mlp[0](base_input)
        state_parent_pre = self.edge_mlp[0](state_input)
        if self.edge_attr_linear is None:
            pair_delta = torch.zeros_like(base_pre)
        else:
            if edge_attr is None:
                edge_attr = base_pre.new_zeros((base_pre.shape[0], self.edges_in_d))
            pair_delta = self.edge_attr_linear(edge_attr)
        state_pre = state_parent_pre + pair_delta

        if self.capture_bridge_diagnostics:
            with torch.no_grad():
                base_f = state_parent_pre.detach().float()
                delta_f = pair_delta.detach().float()
                combined_f = state_pre.detach().float()
                base_rms = base_f.square().mean().sqrt()
                delta_rms = delta_f.square().mean().sqrt()
                self.last_bridge_diagnostics = {
                    'pair_base_preact_rms': base_rms.to(base_pre.dtype),
                    'pair_delta_rms': delta_rms.to(base_pre.dtype),
                    'pair_delta_absmax': delta_f.abs().amax().to(base_pre.dtype),
                    'pair_combined_preact_rms': combined_f.square().mean().sqrt().to(base_pre.dtype),
                    'pair_delta_to_base_ratio': (
                        delta_rms / base_rms.clamp_min(1.0e-8)
                    ).to(base_pre.dtype),
                    'pair_adapter_weight_rms': (
                        base_rms.new_zeros(()) if self.edge_attr_linear is None
                        else self.edge_attr_linear.weight.detach().float().square().mean().sqrt()
                    ).to(base_pre.dtype),
                }
        else:
            self.last_bridge_diagnostics = {}

        base_edge = base_pre
        state_edge = state_pre
        for layer in self.edge_mlp[1:]:
            base_edge = layer(base_edge)
            state_edge = layer(state_edge)

        # One Bernoulli mask, shared by both streams.  At zero pair adapter the
        # two values are exactly identical, preserving the R05 parent value.
        if self.training and self.dropout.p > 0.0:
            dropout_scale = self.dropout(torch.ones_like(state_edge))
            base_edge = base_edge * dropout_scale
            state_edge = state_edge * dropout_scale

        if self.attention:
            base_edge = base_edge * self.att_mlp(base_edge)
            state_edge = state_edge * self.att_mlp(state_edge)
        return state_edge, base_edge

    def node_model(self, x, edge_index, edge_attr, node_attr):
        '''
        :param x: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param edge_attr: [n_edge, hidden_size], refers to message from i to j
        :param node_attr: [bs * n_node, node_dim]
        '''
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # [bs * n_node, hidden_size]
        # print_log(f'agg1, {torch.isnan(agg).sum()}', level='DEBUG')
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # [bs * n_node, input_size + hidden_size]
        # print_log(f'agg, {torch.isnan(agg).sum()}', level='DEBUG')
        out = self.node_mlp(agg)  # [bs * n_node, output_size]
        # print_log(f'out, {torch.isnan(out).sum()}', level='DEBUG')
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def node_model_dual(self, state_x, base_x, edge_index, state_edge_attr,
                        base_edge_attr, node_attr):
        """Shared-parameter state/base node update with one dropout mask."""
        row, col = edge_index
        state_agg = unsorted_segment_sum(
            state_edge_attr, row, num_segments=state_x.size(0)
        )
        base_agg = unsorted_segment_sum(
            base_edge_attr, row, num_segments=base_x.size(0)
        )
        if node_attr is not None:
            state_in = torch.cat([state_x, state_agg, node_attr], dim=1)
            base_in = torch.cat([base_x, base_agg, node_attr], dim=1)
        else:
            state_in = torch.cat([state_x, state_agg], dim=1)
            base_in = torch.cat([base_x, base_agg], dim=1)
        state_out = self.node_mlp(state_in)
        base_out = self.node_mlp(base_in)
        if self.training and self.dropout.p > 0.0:
            dropout_scale = self.dropout(torch.ones_like(state_out))
            state_out = state_out * dropout_scale
            base_out = base_out * dropout_scale
        if self.residual:
            state_out = state_x + state_out
            base_out = base_x + base_out
        return state_out, base_out

    def coord_model(self, coord, edge_index, coord_diff, state_edge_feat,
                    channel_weights, base_edge_feat=None):
        '''Pair-aware Cartesian update with bounded direct-pair authority.

        The state/message stream may use the full pair-conditioned edge feature.
        Geometry uses the pair-free base coefficient plus a bounded residual from
        direct pair conditioning.  With the V215 EGNN controller enabled:

            alpha_base = tanh(alpha_base_raw)
            alpha_state = tanh(alpha_state_raw)
            alpha = alpha_base + B * tanh((alpha_state-alpha_base) / B).

        For a zero pair adapter, alpha_state == alpha_base exactly.  Around the
        origin tanh(z)=z+O(z^3), so the zero-start parent is first-order
        preserved; only high-gain coordinate authority is saturated.  This is
        scalar-field bounding plus canonical direction normalization, not coordinate clipping.
        '''
        row, col = edge_index
        n_channel = channel_weights.shape[-1]
        coord_before = coord

        # EGNN coordinate controller.  The official EGNN implementation
        # exposes ``tanh`` precisely to bound phi_x(m_ij).  We keep the raw
        # scalar for forensics, then apply tanh before any Cartesian authority
        # is granted.  This preserves E(n) equivariance because only an
        # invariant scalar is transformed; coordinate directions are unchanged.
        state_coeff_raw = self.coord_mlp(state_edge_feat)
        state_coeff = torch.tanh(state_coeff_raw) if self.tanh else state_coeff_raw
        if base_edge_feat is None or self.pair_coord_mode == 'legacy_shared':
            base_coeff_raw = state_coeff_raw
            base_coeff = state_coeff
            pair_coeff_raw = torch.zeros_like(state_coeff)
            pair_coeff_bounded = pair_coeff_raw
            coord_coeff = state_coeff
        else:
            base_coeff_raw = self.coord_mlp(base_edge_feat)
            base_coeff = torch.tanh(base_coeff_raw) if self.tanh else base_coeff_raw
            # Preserve V212 semantics: direct Pair authority is measured after
            # the same coordinate controller used by state/base.  Hence zero
            # Pair still gives exact equality, while the residual remains
            # independently bounded by ``pair_coord_delta_bound``.
            pair_coeff_raw = state_coeff - base_coeff
            bound = pair_coeff_raw.new_tensor(self.pair_coord_delta_bound)
            pair_coeff_bounded = bound * torch.tanh(pair_coeff_raw / bound)
            coord_coeff = base_coeff + pair_coeff_bounded

        # V215 canonical EGNN direction normalization.  Keep the original
        # distance-dependent radial/message features unchanged; normalize only
        # the equivariant vector that grants Cartesian authority.  This mirrors
        # the reference EGNN implementation: the norm is detached so the
        # denominator is not an auxiliary gradient-control path.
        coord_diff_raw = coord_diff
        coord_diff_norm = torch.norm(coord_diff_raw, dim=-1, keepdim=True)
        if self.normalize:
            coord_direction = coord_diff_raw / (coord_diff_norm.detach() + self.epsilon)
        else:
            coord_direction = coord_diff_raw

        channel_sum = (channel_weights != 0).long().sum(-1)
        pooled_edge_feat = RollerPooling(n_channel)(coord_coeff, channel_sum[row])
        trans = coord_direction * pooled_edge_feat

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord_before + agg

        if self.capture_bridge_diagnostics:
            with torch.no_grad():
                def _rms(v):
                    vf = v.detach().float()
                    return vf.square().mean().sqrt().to(coord_before.dtype) if vf.numel() else coord_before.new_zeros(())
                def _amax(v):
                    vf = v.detach().float()
                    return vf.abs().amax().to(coord_before.dtype) if vf.numel() else coord_before.new_zeros(())
                self.last_coord_diagnostics = {
                    'coord_input_rms': _rms(coord_before),
                    'coord_input_absmax': _amax(coord_before),
                    'coord_diff_rms': _rms(coord_diff_raw),
                    'coord_diff_absmax': _amax(coord_diff_raw),
                    'coord_diff_norm_rms': _rms(coord_diff_norm),
                    'coord_diff_norm_absmax': _amax(coord_diff_norm),
                    'coord_direction_rms': _rms(coord_direction),
                    'coord_direction_absmax': _amax(coord_direction),
                    'coord_direction_norm_absmax': _amax(torch.norm(coord_direction, dim=-1)),
                    'coord_normalize': coord_before.new_tensor(1.0 if self.normalize else 0.0),
                    'coord_base_coeff_raw_rms': _rms(base_coeff_raw),
                    'coord_base_coeff_raw_absmax': _amax(base_coeff_raw),
                    'coord_state_coeff_raw_rms': _rms(state_coeff_raw),
                    'coord_state_coeff_raw_absmax': _amax(state_coeff_raw),
                    'coord_base_coeff_rms': _rms(base_coeff),
                    'coord_base_coeff_absmax': _amax(base_coeff),
                    'coord_state_coeff_rms': _rms(state_coeff),
                    'coord_state_coeff_absmax': _amax(state_coeff),
                    'coord_pair_delta_raw_rms': _rms(pair_coeff_raw),
                    'coord_pair_delta_raw_absmax': _amax(pair_coeff_raw),
                    'coord_pair_delta_bounded_rms': _rms(pair_coeff_bounded),
                    'coord_pair_delta_bounded_absmax': _amax(pair_coeff_bounded),
                    'coord_coeff_rms': _rms(coord_coeff),
                    'coord_coeff_absmax': _amax(coord_coeff),
                    'coord_trans_rms': _rms(trans),
                    'coord_trans_absmax': _amax(trans),
                    'coord_update_rms': _rms(agg),
                    'coord_update_absmax': _amax(agg),
                    'coord_update_to_input_rms_ratio': (
                        _rms(agg).float() / _rms(coord_before).float().clamp_min(1.0e-8)
                    ).to(coord_before.dtype),
                    'coord_output_rms': _rms(coord),
                    'coord_output_absmax': _amax(coord),
                }
        else:
            self.last_coord_diagnostics = {}
        return coord

    def forward(self, h, edge_index, epi_index, coord, surf_verts, channel_attr, channel_weights,
                edge_attr=None, node_attr=None, capture_bridge_diagnostics=False):
        '''
        h: [bs * n_node, hidden_size]
        edge_index: list of [n_row] and [n_col] where n_row == n_col (with no cutoff, n_row == bs * n_node * (n_node - 1))
        coord: [bs * n_node, n_channel, d]
        channel_attr: [bs * n_node, n_channel, channel_nf]
        channel_weights: [bs * n_node, n_channel]
        '''
        row, col = edge_index
        self.capture_bridge_diagnostics = bool(capture_bridge_diagnostics)
        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}
        # Empty aligned surface edges are a valid no-message case (especially
        # with local batch=1).  Do not fabricate geometry and do not bypass this
        # module in AMEncoder: return an exact identity value while attaching a
        # zero-valued dependency to every trainable parameter.  Consequently
        # each parameter receives a zero tensor gradient rather than grad=None,
        # which is compatible with DDP find_unused_parameters=False and the
        # repeated-checkpoint static graph used by the AbX/R05 integration.
        if row.numel() == 0 or epi_index is None or epi_index.numel() == 0:
            zero_anchor = None
            for param in self.parameters():
                if param.requires_grad:
                    term = param.reshape(-1)[0] * 0.0
                    zero_anchor = term if zero_anchor is None else zero_anchor + term
            if zero_anchor is None:
                return h, coord
            return (
                h + zero_anchor.to(device=h.device, dtype=h.dtype),
                coord + zero_anchor.to(device=coord.device, dtype=coord.dtype),
            )

        radial, abX = coord_SR(
            edge_index, epi_index, coord, surf_verts, channel_attr,
            self.scale_linear, self.radial_linear
        )
        edge_feat, base_edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(
            coord, edge_index, abX, edge_feat, channel_weights,
            base_edge_feat=base_edge_feat,
        )
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord

    def forward_dual(self, h, h_base, edge_index, epi_index, coord, surf_verts,
                     channel_attr, channel_weights, edge_attr=None, node_attr=None,
                     capture_bridge_diagnostics=False):
        row, col = edge_index
        self.capture_bridge_diagnostics = bool(capture_bridge_diagnostics)
        self.last_bridge_diagnostics = {}
        self.last_coord_diagnostics = {}
        if row.numel() == 0 or epi_index is None or epi_index.numel() == 0:
            zero_anchor = None
            for param in self.parameters():
                if param.requires_grad:
                    term = param.reshape(-1)[0] * 0.0
                    zero_anchor = term if zero_anchor is None else zero_anchor + term
            if zero_anchor is None:
                return h, h_base, coord
            z_h = zero_anchor.to(device=h.device, dtype=h.dtype)
            z_x = zero_anchor.to(device=coord.device, dtype=coord.dtype)
            return h + z_h, h_base + z_h, coord + z_x

        radial, abX = coord_SR(
            edge_index, epi_index, coord, surf_verts, channel_attr,
            self.scale_linear, self.radial_linear
        )
        edge_feat, base_edge_feat = self.edge_model(
            h[row], h[col], radial, edge_attr,
            base_source=h_base[row], base_target=h_base[col],
        )
        coord = self.coord_model(
            coord, edge_index, abX, edge_feat, channel_weights,
            base_edge_feat=base_edge_feat,
        )
        h, h_base = self.node_model_dual(
            h, h_base, edge_index, edge_feat, base_edge_feat, node_attr
        )
        return h, h_base, coord
        
        
def coord_SR(aligned_edge_index, epi_index, local_coord, surf_verts, attr, scale_map, linear_map):
    '''
    :param edge_index: tuple([n_edge], [n_edge]) which is tuple of (row, col)
    :param coord: [N, n_channel, d]
    :param attr: [N, n_channel, attr_size], attribute embedding of each channel
    :param channel_weights: [N, n_channel], weights of different channels
    :param linear_map: nn.Linear, map features to d_out
    :param num_seg: split row/col into segments to reduce memory cost
    '''
    row, col = aligned_edge_index
    
    radials = []

    seg_size = (len(row) + NUM_SEG - 1) // NUM_SEG

    for i in range(NUM_SEG):
        start = i * seg_size
        end = min(start + seg_size, len(row))
        if end <= start:
            break
        s_row, seg_col = row[start:end], col[start:end]
        seg_row = torch.zeros_like(s_row)
        for j in range(len(seg_row)) : 
            seg_row[j] = torch.where(epi_index == s_row[j])[0]

        coord_msg = torch.norm(
            local_coord[seg_col].unsqueeze(2) - surf_verts[seg_row].unsqueeze(1),  # [n_edge, n_channel, n_channel, d]
            dim=-1, keepdim=False)  # [n_edge, n_channel, n_channel]
        
        # coord_msg = coord_msg * torch.bmm(
        #     channel_weights[seg_row].unsqueeze(2),
        #     channel_weights[seg_col].unsqueeze(1)
        #     )  # [n_edge, n_channel, n_channel]
        
        radial = torch.bmm(
            attr[seg_row].transpose(-1, -2),  # [n_edge, attr_size, n_channel], [n_edge, 16, 14]
            coord_msg)  # [n_edge, n_channel, n_channel], [n_edge, 14, 14]
        radial = scale_map(radial)  # [n_edge, n_channel, n_channel]
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, attr_size * attr_size]
        radial_norm = torch.norm(radial, dim=-1, keepdim=True) + CONSTANT  # post norm
        radial = linear_map(radial) / radial_norm # [n_edge, d_out]

        radials.append(radial)

    radials = torch.cat(radials, dim=0)  # [N_edge, d_out]
    
    surf_coord = surf_verts[seg_row].unsqueeze(2)
    abX = local_coord[seg_col].clone()
    for i in range(surf_coord.shape[1]) : 
        abX -= surf_coord[:, i, :, :]
    abX /= surf_coord.shape[1]

    return radials, abX
