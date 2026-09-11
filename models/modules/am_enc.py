#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn

from torch_scatter import scatter_softmax
from .am_egnn import AM_E_GCL, MS_E_GCL


class AMEncoder(nn.Module):

    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, channel_nf,
                 radial_nf, in_edge_nf=0, in_single_nf=0, num_verts=50,
                 act_fn=nn.SiLU(), n_layers=4,
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
        self.in_single_nf = int(in_single_nf)
        self.single_linear = None
        if self.in_single_nf > 0:
            # Zero-start residual is exactly equivalent in capacity to a
            # concatenated input projection, but preserves the complete R05
            # input map at initialization.  RNG isolation prevents this extra
            # module from changing later parent parameters.
            with torch.random.fork_rng(devices=[]):
                self.single_linear = nn.Linear(
                    self.in_single_nf, self.hidden_nf, bias=False
                )
            nn.init.zeros_(self.single_linear.weight)
        self.last_bridge_diagnostics = {}

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
    
    def forward(self, h, x, ctx_edges, inter_mask, inter_x, surf_verts,
                inter_edges, update_mask, inter_update_mask, aligned_edges,
                epi_index, channel_attr, channel_weights, ctx_edge_attr=None,
                inter_edge_attr=None, surf_edge_attr=None, single_attr=None,
                capture_bridge_diagnostics=False):
        """Run the original R05 multi-channel EGNN with native edge attributes.

        ``ctx_edge_attr`` is aligned to ``ctx_edges`` (global residue graph),
        ``inter_edge_attr`` is aligned to ``inter_edges`` (local residue graph),
        and ``surf_edge_attr`` is aligned to ``aligned_edges``.  Importantly,
        ``aligned_edges`` still connects antigen *residues* to antibody
        residues; ``MS_E_GCL`` merely replaces their radial geometry with an
        antigen-surface-aware radial.  Therefore a residue-pair state z_ij has
        the same endpoint semantics on this path and can use AM_E_GCL/MS_E_GCL's
        pre-existing ``edge_attr`` interface without a second coordinate head.
        """
        base_pre = self.linear_in(h)
        if self.single_linear is None:
            single_delta = torch.zeros_like(base_pre)
        else:
            if single_attr is None:
                single_attr = base_pre.new_zeros(
                    (base_pre.shape[0], self.in_single_nf)
                )
            if single_attr.ndim != 2 or single_attr.shape[0] != base_pre.shape[0] \
                    or single_attr.shape[1] != self.in_single_nf:
                raise ValueError(
                    "native AbX single shape mismatch: expected "
                    f"({base_pre.shape[0]}, {self.in_single_nf}), "
                    f"got {tuple(single_attr.shape)}"
                )
            single_delta = self.single_linear(single_attr)
        h = base_pre + single_delta
        h = self.dropout(h)

        capture_bridge_diagnostics = bool(capture_bridge_diagnostics)
        if capture_bridge_diagnostics:
            with torch.no_grad():
                base_rms = torch.sqrt(base_pre.detach().float().square().mean())
                delta_rms = torch.sqrt(single_delta.detach().float().square().mean())
                weight_rms = (
                    base_rms.new_zeros(())
                    if self.single_linear is None
                    else torch.sqrt(
                        self.single_linear.weight.detach().float().square().mean()
                    )
                )
                self.last_bridge_diagnostics = {
                    "bridge_single_base_rms": base_rms.to(base_pre.dtype),
                    "bridge_single_delta_rms": delta_rms.to(base_pre.dtype),
                    "bridge_single_delta_to_base_ratio": (
                        delta_rms / base_rms.clamp_min(1.0e-8)
                    ).to(base_pre.dtype),
                    "bridge_single_adapter_weight_rms": weight_rms.to(base_pre.dtype),
                }
        else:
            self.last_bridge_diagnostics = {}

        pair_ratios = []

        def collect_pair_diagnostics(prefix, module):
            if not capture_bridge_diagnostics:
                return
            diag = getattr(module, "last_bridge_diagnostics", {}) or {}
            for key, value in diag.items():
                full_key = f"bridge_{prefix}_{key}"
                self.last_bridge_diagnostics[full_key] = value
                if key == "pair_delta_to_base_ratio":
                    pair_ratios.append(value)
        inter_h = h[inter_mask]
        inter_channel_attr = channel_attr[inter_mask]
        inter_channel_weights = channel_weights[inter_mask]

        ctx_states, ctx_coords, inter_coords = [], [], []
        has_surface_edges = (
            aligned_edges is not None
            and aligned_edges.numel() > 0
            and epi_index is not None
            and epi_index.numel() > 0
        )
        if (
            not has_surface_edges
            and os.environ.get("ABFLOW_ABX_MEMORY_DIAGNOSTICS", "off").lower()
            in {"1", "true", "yes", "y", "on"}
        ):
            rank = (
                int(torch.distributed.get_rank())
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            print(
                f"[R05SurfEdgeEmpty] rank={rank} "
                f"aligned_edges={0 if aligned_edges is None else aligned_edges.shape[-1]} "
                f"epi_index={0 if epi_index is None else epi_index.numel()} "
                f"action=MS_E_GCL_zero_message_identity",
                flush=True,
            )
        for i in range(0, self.n_layers):
            ctx_module = self._modules[f'ctx_gcl_{i}']
            h, x = ctx_module(
                h, ctx_edges, x, channel_attr, channel_weights,
                edge_attr=ctx_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics)
            collect_pair_diagnostics(f"ctx_l{i}", ctx_module)
            # synchronization of the shadow paratope (native -> shadow)
            inter_h = inter_h.clone()
            inter_h[inter_update_mask] = h[update_mask]
            inter_module = self._modules[f'inter_gcl_{i}']
            inter_h, inter_x = inter_module(
                inter_h, inter_edges, inter_x, inter_channel_attr, inter_channel_weights,
                edge_attr=inter_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair_diagnostics(f"inter_l{i}", inter_module)
            # MS_E_GCL itself owns the empty-edge zero-message semantics.
            # Always call it so the module participates in the autograd/DDP graph
            # on every iteration; no fake edge and no conditional parameter use.
            surf_module = self._modules[f'surf_gcl_{i}']
            inter_h, inter_x = surf_module(
                inter_h, aligned_edges, epi_index, inter_x, surf_verts,
                inter_channel_attr, inter_channel_weights,
                edge_attr=surf_edge_attr,
                capture_bridge_diagnostics=capture_bridge_diagnostics,
            )
            collect_pair_diagnostics(f"surf_l{i}", surf_module)
            # synchronization of the shadow paratope (shadow -> native)
            h = h.clone()
            h[inter_mask] = inter_h
            ctx_states.append(h)
            ctx_coords.append(x)
            inter_coords.append(inter_x)

        h, x = self.out_layer(
            h, ctx_edges, x, channel_attr, channel_weights,
            edge_attr=ctx_edge_attr,
            capture_bridge_diagnostics=capture_bridge_diagnostics)
        collect_pair_diagnostics("out", self.out_layer)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
            inter_x = torch.mean(torch.stack(inter_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        if capture_bridge_diagnostics and pair_ratios:
            stacked = torch.stack(pair_ratios)
            self.last_bridge_diagnostics[
                "bridge_pair_delta_to_base_ratio_mean"
            ] = stacked.mean()
            self.last_bridge_diagnostics[
                "bridge_pair_delta_to_base_ratio_max"
            ] = stacked.max()
        return h, x, inter_x
