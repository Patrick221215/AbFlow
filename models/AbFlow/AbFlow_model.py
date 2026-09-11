#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from utils.nn_utils import (
    SeparatedAminoAcidFeature, ProteinFeature, GMEdgeConstructor,
    SeperatedCoordNormalizer, _knn_edges, get_timestep_embedding,
    _abflow_ca_fill_observed_mask,
)
from evaluation.rmsd import kabsch_torch

from ..modules.am_enc import AMEncoder
from ..modules.am_egnn import AMEGNN
from .abflow_conditional_matcher import AbFlowConditionalMatcher
from .abflow_components import (
    AbFlowR3Matcher, NativeTrunk, design_region_smooth_lddt_loss,
)
from configs import normalize_regions

class AbFlowModel(nn.Module):


    def __init__(self, embed_size, hidden_size, n_channel, num_classes, num_verts,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1,
                 pep_seq=True, pep_struct=True, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False,
                 model_config=None, loss_config=None):
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.pep_seq = pep_seq
        self.pep_struct = pep_struct
        self.struct_only = struct_only
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        self.n_channel = 4 if backbone_only else n_channel
        self.cdr_type = list(normalize_regions(cdr_type)) or cdr_type
        self.paratope = list(normalize_regions(paratope)) or paratope
        self.eps = 1e-8
        model_config = model_config or {}
        loss_config = loss_config or {}
        r05_config = model_config.get("r05", {})
        source_config = r05_config.get("source", {})
        flow_config = r05_config.get("flow", {})
        r3_config = r05_config.get("r3", {})
        representation_config = model_config.get("representation", {}).get("single_pair", {})

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size, relative_position=relative_position,
            edge_constructor=GMEdgeConstructor, fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only)
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)

        if keep_memory:
            self.memory_ffn = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, embed_size))

        if pred_edge_dist:
            if keep_memory:
                self.edge_H_ffn = nn.Sequential(
                    nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(), nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, 1))
            self.init_gnn = AMEGNN(
                embed_size, hidden_size, hidden_size, self.n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False)

        if struct_only:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, 1))
        else:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, num_classes))

        distogram_config = loss_config.get("distogram", {})
        smooth_lddt_config = loss_config.get("smooth_lddt", {})
        self.loss_distogram_weight = float(distogram_config.get("weight", 0.0))
        self.loss_smooth_lddt_weight = float(
            smooth_lddt_config.get("weight", 0.0)
        )
        self.smooth_lddt_cutoff = float(
            smooth_lddt_config.get("cutoff", 15.0)
        )
        self.distogram_enabled = self.loss_distogram_weight > 0.0
        self.smooth_lddt_enabled = self.loss_smooth_lddt_weight > 0.0
        self.relational_trunk_enabled = True

        # Representation initialization is isolated from the R05 RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(
                int(representation_config["init_seed"])
            )
            self.aa_feature.configure_single_pair(representation_config)
            self.native_trunk = NativeTrunk(
                representation_config=representation_config,
                distogram_config=distogram_config,
                forward_seed=int(representation_config["forward_seed"]),
            )

        self.gnn = AMEncoder(
            embed_size, hidden_size, hidden_size, self.n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=self.native_trunk.pair_dim,
            in_single_nf=self.native_trunk.single_dim,
            num_verts=num_verts, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)

        self.normalizer = SeperatedCoordNormalizer()
        self.batch_constants = {}

        self.flow_matcher = AbFlowConditionalMatcher(
            min_sigma=float(flow_config["min_sigma"]),
            eps=self.eps,
        )
        self.r3_matcher = AbFlowR3Matcher(
            transport_fraction=float(r3_config["transport_fraction"]),
            path_min_sigma=float(r3_config["path_min_sigma"]),
            eps=self.eps,
        )

        self.flow_coordinate_scaling = float(r3_config["coordinate_scaling"])
        self.r3_fixed_g_scaled = float(r3_config["fixed_g_scaled"])
        self.f01_hybrid_t_min = float(flow_config["hybrid_t_min"])
        self.proposal_adapter_start_round = int(
            source_config["proposal_adapter_start_round"]
        )

        self.loss_sequence_weight = float(loss_config["sequence"])
        self.loss_structure_weight = float(loss_config["structure"])
        self.loss_interface_weight = float(loss_config["interface"])
        self.loss_edge_weight = float(loss_config["edge"])

        self.flow_time_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size), nn.SiLU(),
            nn.Linear(embed_size, embed_size))

        self.coord_pep_condition_dim = 6
        self.coord_pep_condition_adapter = nn.Sequential(
            nn.Linear(embed_size + self.coord_pep_condition_dim, embed_size),
            nn.SiLU(), nn.Linear(embed_size, embed_size))
        nn.init.zeros_(self.coord_pep_condition_adapter[-1].weight)
        nn.init.zeros_(self.coord_pep_condition_adapter[-1].bias)

        self.seq_pep_condition_embedding = nn.Embedding(num_classes, embed_size)
        self.seq_pep_condition_adapter = nn.Sequential(
            nn.Linear(2 * embed_size, embed_size), nn.SiLU(),
            nn.Linear(embed_size, embed_size))
        nn.init.zeros_(self.seq_pep_condition_adapter[-1].weight)
        nn.init.zeros_(self.seq_pep_condition_adapter[-1].bias)

        # Lightweight trainer compatibility; no diagnostic branch is executed.
        self.last_scorefm_losses = {}
        self.last_abflow_diagnostics = {}
        self._last_trunk_state = {}
        self._last_message_diagnostics = {}
        self.last_gradient_diagnostics = {}
        self.grad_conflict_diagnostics = False
        self._diagnostic_capture = False
        self._diagnostic_validation_mode = False

    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return (X, S)

    def _valid_proposal_backbone(self, coords):
        """Return a per-residue validity mask for proposal N/CA/C coordinates.

        The project runtime is PyTorch 1.11, whose ``Tensor.all`` accepts only
        one reduction dimension.  Flattening the N/CA/C xyz block keeps the
        original predicate exactly while remaining runtime-compatible:

            all coordinates finite AND total backbone magnitude > eps.

        This helper is the single authority for PCS-RC proposal-coordinate
        validity across initialization and recurrent proposal replacement.
        """
        n_bb = min(3, int(coords.shape[1]))
        bb_flat = coords[:, :n_bb].reshape(coords.shape[0], -1)
        return (
            torch.isfinite(bb_flat).all(dim=-1)
            & (bb_flat.abs().sum(dim=-1) > self.eps)
        )


    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep,
                    replace_seq=True, replace_struct=True):
        """Build the recurrent PCS-RC proposal context."""
        if replace_seq and self.pep_seq and S_pep is not None and S_pep.numel() == int(paratope_mask.sum()):
            pep_S = S_pep.to(device=S.device, dtype=torch.long)
            valid = (pep_S >= 0) & (pep_S < self.num_classes)
            local = S[paratope_mask].clone()
            S[paratope_mask] = torch.where(valid, pep_S, local)
        if replace_struct and self.pep_struct and X_pep is not None and X_pep.shape == X[paratope_mask].shape:
            pep_X = X_pep.to(device=X.device, dtype=X.dtype)
            valid = self._valid_proposal_backbone(pep_X)
            local = X[paratope_mask].clone()
            X[paratope_mask] = torch.where(valid[:, None, None], pep_X, local)
        return X, S


    def _condition_initial_interface(self, interface_X, interface_S, X_pep, S_pep):
        """Use valid proposal residues as the R05 source state."""
        if self.pep_struct and X_pep is not None and X_pep.shape == interface_X.shape:
            pep_X = X_pep.to(device=interface_X.device, dtype=interface_X.dtype)
            valid = self._valid_proposal_backbone(pep_X)
            interface_X = torch.where(valid[:, None, None], pep_X, interface_X)
        if not self.struct_only and self.pep_seq and S_pep is not None and S_pep.shape == interface_S.shape:
            pep_S = S_pep.to(device=interface_S.device, dtype=torch.long)
            valid = (pep_S >= 0) & (pep_S < self.num_classes)
            interface_S = torch.where(valid, pep_S, interface_S)
        return interface_X, interface_S


    def _sample_categorical_path(self, clean_S, base_S, t_graph,
                                 interface_batch_id, corrupt_mask=None):
        """Sample the linear categorical source-to-native path."""
        t = torch.as_tensor(t_graph, device=clean_S.device, dtype=torch.float32)
        prob = t.reshape(1).expand_as(clean_S) if t.numel() == 1 else t[interface_batch_id]
        if self.training:
            u = torch.rand(clean_S.shape, device=clean_S.device)
        else:
            idx = torch.arange(clean_S.numel(), device=clean_S.device, dtype=torch.float32)
            u = torch.frac(torch.sin((idx + 1.0) * 12.9898) * 43758.5453).abs()
        sampled = torch.where(u < prob.clamp(0.0, 1.0), clean_S, base_S).long()
        if corrupt_mask is None:
            return sampled
        return torch.where(corrupt_mask.to(clean_S.device).bool(), sampled, clean_S).long()

    def align_epi_ab(self, local_inter_edges, local_is_ab):
        row, col = local_inter_edges
        row_is_ab, col_is_ab = local_is_ab[row], local_is_ab[col]
        swap = row_is_ab & ~col_is_ab
        aligned = local_inter_edges.clone()
        aligned[0, swap], aligned[1, swap] = col[swap], row[swap]
        epi_index = torch.nonzero(~local_is_ab, as_tuple=False).reshape(-1)
        return aligned, epi_index


    def _sample_flow_times(self, batch_size, device, dtype=torch.float32):
        """R05 uses per-complex uniform time; validation is deterministic."""
        n = int(batch_size)
        if self.training:
            return torch.rand(n, device=device, dtype=dtype)
        if n == 1:
            return torch.full((1,), 0.5, device=device, dtype=dtype)
        return (torch.arange(n, device=device, dtype=dtype) + 0.5) / float(n)

    def _time_for_interface(self, t_graph, interface_batch_id, ref_tensor):
        if t_graph is None:
            return None
        t = torch.as_tensor(t_graph, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if t.numel() == 1:
            return t.reshape(1, 1, 1)
        return t[interface_batch_id].reshape(-1, 1, 1)

    def _flow_time_embedding_for_residues(self, flow_t, batch_id, H_0):
        if flow_t is None:
            return None
        t = torch.as_tensor(flow_t, device=H_0.device, dtype=H_0.dtype)
        if t.numel() == 1:
            n_graph = int(batch_id.max()) + 1 if batch_id.numel() else 1
            t = t.reshape(1).expand(n_graph)
        emb = get_timestep_embedding(t, H_0.shape[-1]).to(H_0)
        return self.flow_time_mlp(emb)[batch_id]

    def _build_coord_pep_condition_for_residues(
            self, pep_X_model, interface_X, paratope_mask, pep_coord_valid=None):
        if pep_X_model is None or pep_X_model.shape != interface_X.shape:
            return None, None
        if interface_X.shape[1] < 3 or int(paratope_mask.sum()) != interface_X.shape[0]:
            return None, None

        n_int = interface_X.shape[0]
        if pep_coord_valid is None:
            valid = torch.ones(n_int, device=interface_X.device, dtype=torch.bool)
        else:
            valid = torch.as_tensor(pep_coord_valid, device=interface_X.device, dtype=torch.bool).reshape(-1)
            if valid.numel() != n_int:
                return None, None

        delta = pep_X_model - interface_X
        ca_delta = delta[:, 1]
        ca_dist = torch.norm(ca_delta, dim=-1, keepdim=True)

        n_vec = pep_X_model[:, 0] - pep_X_model[:, 1]
        c_vec = pep_X_model[:, 2] - pep_X_model[:, 1]
        c_norm = torch.norm(c_vec, dim=-1, keepdim=True)
        e1 = F.normalize(c_vec, dim=-1, eps=self.eps)
        n_orth = n_vec - (n_vec * e1).sum(-1, keepdim=True) * e1
        n_orth_norm = torch.norm(n_orth, dim=-1, keepdim=True)
        e2 = F.normalize(n_orth, dim=-1, eps=self.eps)
        e3 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1, eps=self.eps)

        local_delta = torch.stack([
            (ca_delta * e1).sum(-1),
            (ca_delta * e2).sum(-1),
            (ca_delta * e3).sum(-1),
        ], dim=-1)
        local_delta = local_delta * (torch.log1p(ca_dist) / ca_dist.clamp_min(self.eps))
        frame_valid = (
            torch.isfinite(c_norm.squeeze(-1))
            & torch.isfinite(n_orth_norm.squeeze(-1))
            & (c_norm.squeeze(-1) > 1e-4)
            & (n_orth_norm.squeeze(-1) > 1e-4)
        )
        local_delta = torch.where(
            (valid & frame_valid)[:, None], local_delta, torch.zeros_like(local_delta))

        n_bb = min(4, interface_X.shape[1])
        bb_dist = torch.norm(delta[:, :n_bb], dim=-1)
        dist_feat = torch.cat([
            ca_dist,
            bb_dist.mean(-1, keepdim=True),
            torch.sqrt((bb_dist ** 2).mean(-1, keepdim=True) + self.eps),
        ], dim=-1)
        feat = torch.cat([local_delta, torch.log1p(dist_feat.clamp_min(0.0))], dim=-1)
        feat = torch.nan_to_num(feat) * valid[:, None].to(feat.dtype)

        full = interface_X.new_zeros((paratope_mask.shape[0], self.coord_pep_condition_dim))
        mask = torch.zeros(paratope_mask.shape[0], device=interface_X.device, dtype=torch.bool)
        full[paratope_mask], mask[paratope_mask] = feat, valid
        return full, mask


    def _build_seq_pep_condition_for_residues(self, S_pep, paratope_mask, ref_tensor):
        if S_pep is None or S_pep.numel() != int(paratope_mask.sum()):
            return None, None
        pep = S_pep.to(device=ref_tensor.device, dtype=torch.long).reshape(-1)
        valid = (pep >= 0) & (pep < self.num_classes)
        if not bool(valid.any()):
            return None, None
        idx = paratope_mask.nonzero(as_tuple=False).reshape(-1)[valid]
        tokens = torch.zeros(paratope_mask.shape[0], device=ref_tensor.device, dtype=torch.long)
        tokens = tokens.index_copy(0, idx, pep[valid])
        mask = torch.zeros_like(paratope_mask, dtype=torch.bool).index_fill(0, idx, True)
        emb = self.seq_pep_condition_embedding(tokens).to(ref_tensor.dtype)
        return emb * mask[:, None].to(emb.dtype), mask


    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask,
                        batch_id, memory_H=None, smooth_prob=None, smooth_mask=None,
                        flow_t=None, coord_pep_condition=None,
                        coord_pep_condition_mask=None, seq_pep_condition=None,
                        seq_pep_condition_mask=None, trunk_state=None):
        H_0, (ctx_edges, _), (atom_embeddings, atom_weights) = self.aa_feature(
            X, S, batch_id, self.k_neighbors, residue_pos,
            smooth_prob=smooth_prob, smooth_mask=smooth_mask)

        time_emb = self._flow_time_embedding_for_residues(flow_t, batch_id, H_0)
        if time_emb is not None:
            H_0 = H_0 + time_emb

        if coord_pep_condition is not None and coord_pep_condition_mask is not None:
            feat = coord_pep_condition.to(H_0)
            mask = coord_pep_condition_mask.to(H_0.device).bool()
            residual = self.coord_pep_condition_adapter(torch.cat([H_0, feat], dim=-1))
            H_0 = H_0 + residual * mask[:, None].to(H_0.dtype)
        else:
            H_0 = H_0 + 0.0 * sum(p.sum() for p in self.coord_pep_condition_adapter.parameters())

        if seq_pep_condition is not None and seq_pep_condition_mask is not None:
            feat = seq_pep_condition.to(H_0)
            mask = seq_pep_condition_mask.to(H_0.device).bool()
            residual = self.seq_pep_condition_adapter(torch.cat([H_0, feat], dim=-1))
            H_0 = H_0 + residual * mask[:, None].to(H_0.dtype)
        else:
            dummy = sum(p.sum() for p in self.seq_pep_condition_adapter.parameters())
            dummy = dummy + sum(p.sum() for p in self.seq_pep_condition_embedding.parameters())
            H_0 = H_0 + 0.0 * dummy

        if not self.keep_memory:
            memory_H = None
        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        if self.pred_edge_dist:
            if memory_H is None:
                edge_H, dummy_X = self.init_gnn(
                    H_0, X, ctx_edges, channel_attr=atom_embeddings,
                    channel_weights=atom_weights)
                X = X + 0.0 * dummy_X
            else:
                edge_H = self.edge_H_ffn(memory_H)

        X = self.aa_feature.update_global_coordinates(X, S)
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        local_X[local_is_ab] = interface_X

        local_ctx_edges = self.batch_constants['local_ctx_edges']
        local_inter_edges = self.batch_constants['local_inter_edges']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        edge_info = (offsets, local_batch_id, max_n, gni2lni)
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors, edge_info)

        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src, dst = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = (
                self.edge_dist_ffn(torch.cat([src, dst], dim=-1))
                + self.edge_dist_ffn(torch.cat([dst, src], dim=-1))).squeeze(-1)
        else:
            p_edge_dist = None

        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors, edge_info,
            given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)
        surf_edges, epi_index = self.align_epi_ab(local_inter_edges, local_is_ab)

        trunk_single = trunk_state['single_global'].to(H_0)
        ctx_pair_attr = self.native_trunk.gather_pair(
            trunk_state['pair_dense'], ctx_edges,
            trunk_state['node_graph'], trunk_state['node_local']).to(H_0)
        local_global = torch.nonzero(local_mask, as_tuple=False).flatten()
        inter_global = local_global[local_edges]
        surf_global = local_global[surf_edges]
        inter_pair_attr = self.native_trunk.gather_pair(
            trunk_state['pair_dense'], inter_global,
            trunk_state['node_graph'], trunk_state['node_local']).to(H_0)
        surf_pair_attr = self.native_trunk.gather_pair(
            trunk_state['pair_dense'], surf_global,
            trunk_state['node_graph'], trunk_state['node_local']).to(H_0)

        capture_diag = bool(getattr(self, '_diagnostic_capture', False))
        H, pred_X, pred_local_X = self.gnn(
            H_0, X, ctx_edges, local_mask, local_X, surf, local_edges,
            paratope_mask, local_is_ab, surf_edges, epi_index,
            channel_attr=atom_embeddings, channel_weights=atom_weights,
            ctx_edge_attr=ctx_pair_attr, inter_edge_attr=inter_pair_attr,
            surf_edge_attr=surf_pair_attr, single_attr=trunk_single,
            capture_bridge_diagnostics=capture_diag)
        if capture_diag:
            def _rms(value):
                if value is None or value.numel() == 0:
                    return H_0.new_zeros(())
                return value.detach().float().square().mean().sqrt().to(H_0.dtype)
            self._last_message_diagnostics = {
                'abx_ctx_edge_attr_rms': _rms(ctx_pair_attr),
                'abx_inter_edge_attr_rms': _rms(inter_pair_attr),
                'abx_surf_edge_attr_rms': _rms(surf_pair_attr),
            }
            self._last_message_diagnostics.update(
                getattr(self.gnn, 'last_bridge_diagnostics', {}) or {}
            )
        pred_logits = None if self.struct_only else self.ffn_residue(H)
        return pred_logits, pred_X, pred_local_X[local_is_ab], H, p_edge_dist

    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10 + ca_noise.unsqueeze(1)
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise
        init_local_S = torch.randint(0, self.num_classes, (paratope_mask.sum(),), device=X.device, dtype=torch.long)
        return (init_local_X, init_local_S)

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        batch_id = torch.zeros_like(S)
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1
        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(paratope_mask, torch.logical_and(is_ag, not_ag_global))
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = (local_segment_ids[row], local_segment_ids[col])
        row_is_ag = row_segment_ids == self.aa_feature.ag_seg_id
        col_is_ag = col_segment_ids == self.aa_feature.ag_seg_id
        is_inter = torch.logical_xor(row_is_ag, col_is_ag)
        is_ctx = torch.logical_not(is_inter)
        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)
        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id

    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        """Native minimum atom distance for interface edges."""
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)
        dist = torch.norm(dist, dim=-1)
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))
        dist = dist + pos_pad * 10000000000.0
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]
        return dist

    def _raw_interface_to_model_frame(self, interface_X, paratope_mask, batch_id):
        """Map raw paratope coordinates to the internal frame."""
        interface_batch_id = batch_id[paratope_mask]
        ag_centers = self.normalizer.ag_centers[interface_batch_id]
        return self.normalizer.normalize(interface_X - ag_centers.unsqueeze(1))

    def _interface_valid_graph_mask(self, interface_batch_id, n_graph, device):
        valid = torch.zeros(n_graph, device=device, dtype=torch.bool)
        if interface_batch_id.numel() > 0:
            valid[torch.unique(interface_batch_id)] = True
        return valid

    def _masked_residue_smooth_l1_per_graph(self, pred, target, atom_mask, interface_batch_id):
        """Per-complex masked coordinate SmoothL1."""
        if interface_batch_id.numel() == 0:
            return (pred.new_zeros(1), torch.zeros(1, device=pred.device, dtype=torch.bool))
        n_graph = int(interface_batch_id.max().item()) + 1
        atom_mask_f = atom_mask.to(pred.dtype)
        err = F.smooth_l1_loss(pred, target, reduction='none').sum(dim=-1)
        err = err * atom_mask_f
        per_res = err.sum(dim=-1) / (3.0 * atom_mask_f.sum(dim=-1).clamp_min(1.0))
        per_graph = scatter_mean(per_res, interface_batch_id, dim=0, dim_size=n_graph)
        valid_graph = self._interface_valid_graph_mask(interface_batch_id, n_graph, pred.device)
        return (per_graph, valid_graph)


    def _deterministic_standard_normal(self, shape, device, dtype):
        n = math.prod(int(d) for d in shape)
        idx = torch.arange(n, device=device, dtype=torch.float32) + 1.0
        u = torch.frac(torch.sin(idx * 12.9898 + 78.233) * 43758.5453).abs().clamp(1e-4, 1.0 - 1e-4)
        return (math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)).reshape(*shape).to(dtype)

    def _r05_primary_path(self, source_X0, target_X1, t_graph, t_int,
                          interface_batch_id):
        """Residue-level fixed-g FoldFlow R3 path used by R05."""
        mu = self.r3_matcher.linear_mean(source_X0, target_X1, t_int)
        if interface_batch_id.numel() == 0:
            return mu, {'r3_sigma_mean': target_X1.new_zeros(())}
        n_graph = int(interface_batch_id.max()) + 1
        t = torch.as_tensor(t_graph, device=target_X1.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(n_graph)
        g_raw = self.r3_matcher.foldflow_scaled_g_to_raw(
            g_scaled=self.r3_fixed_g_scaled,
            coordinate_scaling=self.flow_coordinate_scaling)
        g = torch.full_like(t, g_raw)
        sigma = self.r3_matcher.sigma_t(t, g)
        n_res = int(interface_batch_id.numel())
        if self.training:
            eps = torch.randn((n_res, 3), device=target_X1.device, dtype=torch.float32)
        else:
            eps = self._deterministic_standard_normal(
                (n_res, 3), target_X1.device, torch.float32)
        shift = (sigma[interface_batch_id, None] * eps).to(mu.dtype)
        Xt = mu + shift[:, None, :]
        return Xt, {'r3_sigma_mean': sigma.mean().to(target_X1.dtype)}

    @torch.no_grad()
    def _r05_coordinate_target(self, Xt, source_X0, target_X1, t_int):
        """U02 target: endpoint below t=0.2, canonical carrier above it."""
        canonical = self.r3_matcher.canonical_carrier_target_gfree(
            Xt, source_X0, target_X1, t_int,
            boundary_eps=self.f01_hybrid_t_min)
        t = torch.as_tensor(t_int, device=Xt.device, dtype=Xt.dtype)
        while t.dim() < Xt.dim():
            t = t.unsqueeze(-1)
        active = t >= self.f01_hybrid_t_min
        target = torch.where(active, canonical, target_X1)
        return target, {'r3_canonical_active_rate': active.reshape(active.shape[0], -1).any(-1).float().mean()}

    def _coordinate_training_objective(self, pred, target, atom_mask,
                                       interface_batch_id):
        per_graph, valid = self._masked_residue_smooth_l1_per_graph(
            pred, target, atom_mask, interface_batch_id)
        loss = per_graph[valid].mean() if bool(valid.any()) else pred.new_zeros(())
        return loss


    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                 surface, residue_pos, template, lengths, init_noise=None,
                 interface_init=None, sequence_init=None, flow_t=None):
        """R05 predictor at one transport state."""
        batch_id = self.batch_constants['batch_id']
        X, S, surface = X.clone(), S.clone(), surface.clone()
        X, S = self.init_mask(X, S, cmask, smask, template)
        X, S = self.replace_pep(X, S, paratope_mask, X_pep, S_pep)

        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)
        surface = self.normalizer.normalize(surface)
        X = self.aa_feature.update_global_coordinates(X, S)

        if interface_init is None:
            interface_X, interface_S = self.init_interface(
                X, S, paratope_mask, batch_id, init_noise)
            interface_X, interface_S = self._condition_initial_interface(
                interface_X, interface_S, X_pep, S_pep)
        else:
            interface_X = self._raw_interface_to_model_frame(
                interface_init, paratope_mask, batch_id)
            interface_S = (
                sequence_init.to(device=S.device, dtype=torch.long).clone()
                if sequence_init is not None else S[paratope_mask].clone())

        pep_X_model, pep_coord_valid = None, None
        if X_pep is not None and X_pep.shape == interface_X.shape:
            pep_X_raw = X_pep.to(device=X.device, dtype=X.dtype)
            pep_coord_valid = self._valid_proposal_backbone(pep_X_raw)
            if bool(pep_coord_valid.any()):
                pep_X_model = self._raw_interface_to_model_frame(
                    pep_X_raw, paratope_mask, batch_id)

        seq_ref = interface_X.new_zeros(
            (paratope_mask.shape[0], self.seq_pep_condition_embedding.embedding_dim))
        seq_cond, seq_cond_mask = self._build_seq_pep_condition_for_residues(
            S_pep, paratope_mask, seq_ref)

        trunk_S = S
        biological = paratope_mask.bool() | ((trunk_S >= 0) & (trunk_S < self.num_classes))
        trunk_X = X.clone()
        trunk_X[paratope_mask] = interface_X.to(trunk_X.dtype)
        # R05 coordinates are normalized by 10 for physical refinement.  The
        # compact relational trunk keeps AbX/DiffAb geometry conventions in
        # Angstrom, so restore scale without undoing the harmless centering.
        relational_X = self.normalizer.unnormalize(trunk_X)
        trunk_state = self.native_trunk(
            X=relational_X, S=trunk_S,
            segment_ids=self.batch_constants['segment_ids'],
            residue_pos=residue_pos, batch_id=batch_id,
            valid_mask=biological, is_antigen=self.batch_constants['is_ag'],
            design_mask=cmask, flow_t=flow_t,
            cdr_type=self.cdr_type, residue_feature=self.aa_feature,
            round_idx=-1,
            atom_observed_mask=self.batch_constants.get('xloss_mask'))
        self._last_trunk_state = trunk_state

        r_logits, r_interface_X, r_edge_dist = [], [interface_X.clone()], []
        pred_S_dist, memory_H = None, None
        for round_idx in range(self.round):
            if round_idx >= self.proposal_adapter_start_round:
                coord_cond, coord_mask = self._build_coord_pep_condition_for_residues(
                    pep_X_model, interface_X, paratope_mask,
                    pep_coord_valid=pep_coord_valid)
                seq_this, seq_mask_this = seq_cond, seq_cond_mask
            else:
                coord_cond = coord_mask = seq_this = seq_mask_this = None

            pred_logits, pred_X, interface_X, H, edge_dist = self.message_passing(
                X, S, residue_pos, interface_X, surface, paratope_mask,
                batch_id, memory_H=memory_H, smooth_prob=pred_S_dist,
                smooth_mask=smask, flow_t=flow_t,
                coord_pep_condition=coord_cond,
                coord_pep_condition_mask=coord_mask,
                seq_pep_condition=seq_this,
                seq_pep_condition_mask=seq_mask_this,
                trunk_state=trunk_state)

            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_logits.append((pred_logits, smask))
            r_edge_dist.append(edge_dist)
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_global_coordinates(X, S)

            if not self.struct_only:
                S = S.clone()
                if round_idx == self.round - 1:
                    S[smask] = torch.argmax(pred_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_logits[smask], dim=-1)

        interface_batch_id = self.batch_constants['interface_batch_id']
        prmsd = self.prmsd_ffn(H[cmask]).squeeze() if self.struct_only else None
        pred_X = self.normalizer.uncentering(
            self.normalizer.unnormalize(pred_X), batch_id)
        for i, value in enumerate(r_interface_X):
            value = self.normalizer.unnormalize(value)
            r_interface_X[i] = self.normalizer.uncentering(
                value, interface_batch_id, _type=4)
        self.normalizer.clear_cache()
        return H, S, r_logits, pred_X, r_interface_X, r_edge_dist, prmsd


    def forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths, xloss_mask,
                context_ratio=0):
        """Train the fixed R05/U02 state path."""
        cmask, smask = cmask.bool(), smask.bool()
        if self.backbone_only:
            X, template, xloss_mask = X[:, :4], template[:, :4], xloss_mask[:, :4]
            if X_pep is not None:
                X_pep = X_pep[:, :4]

        true_X, true_S = X.clone(), S.clone()
        self._prepare_batch_constants(S, paratope_mask, lengths)
        self.batch_constants['xloss_mask'] = xloss_mask.bool()
        batch_id = self.batch_constants['batch_id']
        interface_batch_id = self.batch_constants['interface_batch_id']
        batch_size_raw = self.batch_constants['batch_size']
        batch_size = int(batch_size_raw.item()) if torch.is_tensor(batch_size_raw) else int(batch_size_raw)

        # Historical R05 sequence-context curriculum.
        sequence_loss_mask = smask.clone()
        if context_ratio > 0:
            keep = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = smask & keep
            sequence_loss_mask = smask
        sequence_path_mask = smask

        gt_interface_X = true_X[paratope_mask]
        interface_X, interface_S = self.init_interface(
            X, S, paratope_mask, batch_id)
        interface_X, interface_S = self._condition_initial_interface(
            interface_X, interface_S, X_pep, S_pep)

        t_graph = self._sample_flow_times(batch_size, X.device, X.dtype)
        t_int = self._time_for_interface(
            t_graph, interface_batch_id, interface_X)
        Xt, path_info = self._r05_primary_path(
            interface_X, gt_interface_X, t_graph, t_int,
            interface_batch_id)
        coord_target, target_info = self._r05_coordinate_target(
            Xt, interface_X, gt_interface_X, t_int)

        if self.struct_only:
            St, sequence_state = interface_S, None
        else:
            St = self._sample_categorical_path(
                true_S[paratope_mask], interface_S, t_graph,
                interface_batch_id,
                corrupt_mask=sequence_path_mask[paratope_mask])
            sequence_state = St

        self._last_message_diagnostics = {}
        H, pred_S, r_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            interface_init=Xt, sequence_init=sequence_state,
            flow_t=t_graph)

        snll = X.new_zeros(())
        count = X.new_zeros(())
        if not self.struct_only:
            for logits, _ in r_logits:
                if bool(sequence_loss_mask.any()):
                    snll = snll + F.cross_entropy(
                        logits[sequence_loss_mask], true_S[sequence_loss_mask],
                        reduction='sum')
                    count = count + sequence_loss_mask.sum()
            snll = snll / count.clamp_min(1.0)

        struct_loss, struct_details, bb_rmsd, _ = self.protein_feature.structure_loss(
            pred_X, true_X, true_S, cmask, batch_id, xloss_mask,
            self.aa_feature)

        atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        atom_mask = atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = self._coordinate_training_objective(
            r_interface_X[-1], coord_target, atom_mask, interface_batch_id)

        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(
                self.normalizer.normalize(true_X), true_S)
            r_ed_losses = [F.smooth_l1_loss(v, gt_edge_dist) for v in r_edge_dist]
            ed_loss = sum(r_ed_losses, X.new_zeros(()))
        else:
            r_ed_losses = [X.new_zeros(()) for _ in range(self.round)]
            ed_loss = X.new_zeros(())
        dock_loss = interface_loss + ed_loss

        distogram_loss = X.new_zeros(())
        distogram_audit = {}
        if self.loss_distogram_weight > 0.0:
            distogram_loss, distogram_audit = self.native_trunk.distogram_loss_from_native(
                self._last_trunk_state, true_X, true_S,
                collect_audit=bool(getattr(self, "_diagnostic_capture", False)),
            )

        smooth_lddt_loss = X.new_zeros(())
        smooth_lddt_audit = {}
        if self.loss_smooth_lddt_weight > 0.0:
            valid_atom_mask = self.batch_constants['xloss_mask'].bool()
            # Only cmask coordinates are generated by the formal H3 task.  Restore
            # all fixed context exactly before the auxiliary distance objective so
            # fixed--fixed pairs cannot contribute a trivial signal.
            aux_pred_X = true_X.clone()
            aux_pred_X[cmask] = pred_X[cmask]
            smooth_lddt_loss, smooth_lddt_audit = design_region_smooth_lddt_loss(
                pred_X=aux_pred_X,
                true_X=true_X,
                valid_atom_mask=valid_atom_mask,
                design_residue_mask=cmask,
                batch_id=batch_id,
                is_antigen_mask=self.batch_constants['is_ag'],
                cutoff=self.smooth_lddt_cutoff,
                collect_audit=bool(getattr(self, "_diagnostic_capture", False)),
            )

        if self.struct_only:
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss = prmsd_loss = None

        loss = (
            self.loss_sequence_weight * snll
            + self.loss_structure_weight * struct_loss
            + self.loss_interface_weight * interface_loss
            + self.loss_edge_weight * ed_loss
            + self.loss_distogram_weight * distogram_loss
            + self.loss_smooth_lddt_weight * smooth_lddt_loss
        )
        if pdev_loss is not None:
            loss = loss + pdev_loss

        with torch.no_grad():
            aar = ((pred_S[sequence_loss_mask] == true_S[sequence_loss_mask]).float().mean()
                   if bool(sequence_loss_mask.any()) else X.new_zeros(()))
            trunk_diag = (self._last_trunk_state.get('diag', {})
                          if isinstance(self._last_trunk_state, dict) else {})
            smooth_intra = smooth_lddt_audit.get('intra', X.new_zeros(()))
            smooth_scaffold = smooth_lddt_audit.get('scaffold', X.new_zeros(()))
            smooth_antigen = smooth_lddt_audit.get('antigen', X.new_zeros(()))

            self.last_scorefm_losses = {
                'scorefm_total': interface_loss.detach(),
                'scorefm_endpoint': interface_loss.detach(),
                'distogram_loss': distogram_loss.detach(),
                'distogram_weighted_loss': (
                    self.loss_distogram_weight * distogram_loss
                ).detach(),
                'smooth_lddt_loss': smooth_lddt_loss.detach(),
                'smooth_lddt_weighted_loss': (
                    self.loss_smooth_lddt_weight * smooth_lddt_loss
                ).detach(),
                'smooth_lddt_DD': smooth_intra.detach(),
                'smooth_lddt_DF': smooth_scaffold.detach(),
                'smooth_lddt_DA': smooth_antigen.detach(),
                'smooth_lddt_support_weight': smooth_lddt_audit.get(
                    'support_weight', X.new_zeros(())
                ).detach(),
                'relational_single_rms': trunk_diag.get(
                    'relational_single_rms', X.new_zeros(())
                ).detach(),
                'relational_pair_rms': trunk_diag.get(
                    'relational_pair_rms', X.new_zeros(())
                ).detach(),
                'relational_antigen_keep_fraction': trunk_diag.get(
                    'relational_antigen_keep_fraction', X.new_ones(())
                ).detach(),
                **{k: v.detach() for k, v in distogram_audit.items()},
                **{k: v.detach() for k, v in path_info.items()},
                **{k: v.detach() for k, v in target_info.items()},
            }

            self.last_abflow_diagnostics = {
                't_min': t_graph.min().detach(),
                't_mean': t_graph.mean().detach(),
                't_max': t_graph.max().detach(),
                **{k: v.detach() if torch.is_tensor(v) else v
                   for k, v in self._last_message_diagnostics.items()},
            }

        self._clean_batch_constants()
        return (loss, (snll, aar), (struct_loss, *struct_details),
                (dock_loss, interface_loss, ed_loss, r_ed_losses),
                (pdev_loss, prmsd_loss))

    def compute_gradient_conflict_diagnostics(self):
        self.last_gradient_diagnostics = {}
        return {}


    def _sampling_time_grid(self, n_steps, device, dtype):
        return torch.linspace(0.0, 1.0, max(1, int(n_steps)) + 1,
                              device=device, dtype=dtype)

    @torch.no_grad()
    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
               surface, residue_pos, template, lengths, n_steps=10,
               init_noise=None, return_hidden=False, show_progress=False,
               progress_desc=None, xloss_mask=None):
        """Generate with the matched R05/U02 canonical-carrier sampler."""
        n_steps = max(1, int(n_steps))
        cmask, smask = cmask.bool(), smask.bool()
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]
            if X_pep is not None:
                X_pep = X_pep[:, :4]
            if xloss_mask is not None:
                xloss_mask = xloss_mask[:, :4]

        gen_X, gen_S = X.clone(), S.clone()
        self._prepare_batch_constants(S, paratope_mask, lengths)
        self.batch_constants['xloss_mask'] = (
            xloss_mask.bool() if xloss_mask is not None
            else _abflow_ca_fill_observed_mask(
                S, X, tol2=self.native_trunk.ca_fill_tol2
            ).bool())

        batch_id = self.batch_constants['batch_id']
        batch_size_raw = self.batch_constants['batch_size']
        batch_size = int(batch_size_raw.item()) if torch.is_tensor(batch_size_raw) else int(batch_size_raw)
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]
        interface_cmask = paratope_mask[cmask]

        Xt, St = self.init_interface(
            X, S, paratope_mask, batch_id, init_noise=init_noise)
        Xt, St = self._condition_initial_interface(Xt, St, X_pep, S_pep)
        source_X0 = Xt.clone()
        time_grid = self._sampling_time_grid(n_steps, X.device, X.dtype)

        steps = range(n_steps)
        if show_progress:
            steps = tqdm(steps, total=n_steps,
                         desc=progress_desc or 'Sampling ODE',
                         leave=False, dynamic_ncols=True)

        for i in steps:
            t, t_next = time_grid[i], time_grid[i + 1]
            dt = t_next - t
            flow_t = t.reshape(1).expand(batch_size)
            if show_progress and hasattr(steps, 'set_postfix'):
                steps.set_postfix(t=f'{float(t):.2f}')

            H, pred_S, r_logits, pred_X, r_interface_X, _, prmsd = self._forward(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                interface_init=Xt,
                sequence_init=None if self.struct_only else St,
                flow_t=flow_t)
            carrier = r_interface_X[-1]
            Xt, _ = self.r3_matcher.exact_carrier_scoreflow_step_gfree(
                x_t=Xt, x0=source_X0, carrier=carrier,
                t=t, t_next=t_next,
                canonical_t_min=self.f01_hybrid_t_min)

            if not self.struct_only:
                logits = r_logits[-1][0][paratope_mask]
                probs = F.softmax(logits - logits.max(-1, keepdim=True).values, dim=-1)
                proposed = torch.multinomial(probs.clamp_min(1e-8), 1).squeeze(-1)
                refresh_prob = self.flow_matcher.categorical_refresh_probability(t, dt)
                refresh = (torch.rand(St.shape, device=St.device) < refresh_prob)
                refresh = refresh & smask[paratope_mask]
                St = torch.where(refresh, proposed, St)

        H_final, pred_X_final, prmsd_final = H, pred_X, prmsd
        interface_X_final = Xt
        if not self.struct_only:
            final_logits = r_logits[-1][0]
            pred_S_final = pred_S.clone()
            if bool(smask.any()):
                pred_S_final[smask] = torch.argmax(final_logits[smask], dim=-1)
            logits = final_logits[smask]
            if logits.shape[0]:
                probs = torch.softmax(logits, dim=-1).max(-1).values
                metric = scatter_mean(
                    -torch.log(probs.clamp_min(1e-8)), s_batch_id,
                    dim=0, dim_size=batch_size)
            else:
                metric = X.new_zeros(batch_size)
        else:
            metric = scatter_mean(
                prmsd_final[interface_cmask], interface_batch_id,
                dim=0, dim_size=batch_size)

        gen_X[cmask] = pred_X_final[cmask]
        if not self.struct_only:
            gen_S[smask] = pred_S_final[smask]

        for b in range(batch_size):
            graph = batch_id == b
            design = graph & paratope_mask
            ori = gen_X[design][:, :4]
            pred = interface_X_final[interface_batch_id == b][:, :4]
            _, R, trans = kabsch_torch(ori.reshape(-1, 3), pred.reshape(-1, 3))
            ab = graph & is_ab
            gen_X[ab] = torch.matmul(gen_X[ab], R.T) + trans

        self._clean_batch_constants()
        if return_hidden:
            return gen_X, gen_S, metric, H_final
        return gen_X, gen_S, metric

    @torch.no_grad()
    def sample_many(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                    surface, residue_pos, template, lengths, n_samples=5,
                    n_steps=20, return_hidden=False, show_progress=False):
        outputs = []
        for i in range(int(n_samples)):
            n_atom = 4 if self.backbone_only else X.shape[1]
            noise = torch.randn(int(paratope_mask.sum()), n_atom, 3, device=X.device)
            outputs.append(self.sample(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface,
                residue_pos, template, lengths, n_steps=n_steps,
                init_noise=noise, return_hidden=return_hidden,
                show_progress=show_progress,
                progress_desc=f'Sample {i + 1}/{n_samples} ODE'))
        if return_hidden:
            xs, ss, ms, hs = zip(*outputs)
            return list(xs), list(ss), list(ms), list(hs)
        xs, ss, ms = zip(*outputs)
        return list(xs), list(ss), list(ms)
