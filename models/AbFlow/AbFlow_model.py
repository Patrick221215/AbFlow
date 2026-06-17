#!/usr/bin/python
# -*- coding:utf-8 -*-
import math, time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import GMEdgeConstructor, SeperatedCoordNormalizer
from utils.nn_utils import _knn_edges
from evaluation.rmsd import kabsch_torch

from ..modules.am_enc import AMEncoder
from ..modules.am_egnn import AMEGNN


class AbFlowModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes, num_verts, 
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, 
                 pep_seq=True, pep_struct=True, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        
        self.pep_seq = pep_seq
        self.pep_struct = pep_struct
        self.struct_only = struct_only

        # options
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        if self.backbone_only:
            n_channel = 4
        # Effective coordinate channel number after backbone_only.
        # Score-FM is defined directly on AbFlow full-atom Cartesian coordinates
        # X ∈ R^{N x n_channel x 3}, so all coordinate losses must use this value.
        self.n_channel = n_channel
        self.cdr_type = cdr_type
        self.paratope = paratope

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            relative_position=relative_position,
            edge_constructor=GMEdgeConstructor,
            fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only
        )
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)
        if keep_memory:
            self.memory_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, embed_size)
            )
        if self.pred_edge_dist:  # use predicted dist for KNN-graph at the interface
            if self.keep_memory:  # this ffn acts on the memory
                self.edge_H_ffn = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
            # this GNN encodes the initial hidden states for initial edge distance prediction
            self.init_gnn = AMEGNN(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False)
        if not struct_only:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.gnn = AMEncoder(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, num_verts=num_verts, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)
        
        self.normalizer = SeperatedCoordNormalizer()

        # training related cache
        self.batch_constants = {}

        # =========================================================
        # Coordinate Score-Factorized Flow Matching losses
        # =========================================================
        # We follow the AbX score-based principle at the level of AbFlow's
        # coordinate path: the network predicts a clean endpoint X_1^theta,
        # and the score is analytically induced from p_t(X_t | X_1^theta).
        # No arbitrary score head is introduced here.
        #
        # Path:               X_t = (1 - t) X_0 + t X_1
        # sigma_t:            sigma_t = 1 - t
        # conditional score:  s_t(X_t | X_1) = -(X_t - t X_1) / sigma_t^2
        self.scorefm_min_sigma = 5e-2
        self.scorefm_eps = 1e-8
        self.scorefm_t_threshold = 0.50

        # Overall weight of the new objective. The component weights below are
        # active by default; they are not placeholders. Keep the global weight
        # conservative because AbFlow already has sequence, structure and docking losses.
        self.scorefm_loss_weight = 5e-2
        self.scorefm_velocity_weight = 1.0
        self.scorefm_dsm_weight = 1.0
        self.scorefm_x1_weight = 0.25
        self.scorefm_local_dist_weight = 0.05
        self.scorefm_interface_contact_weight = 0.05
        self.scorefm_inter_clash_weight = 0.01
        self.scorefm_intra_clash_weight = 0.005

        # Geometry constants for auxiliary coordinate losses.
        self.scorefm_contact_cutoff = 8.0
        self.scorefm_contact_temperature = 1.0
        self.scorefm_inter_clash_cutoff = 2.0
        self.scorefm_intra_clash_cutoff = 1.5
        self.last_scorefm_losses = {}

        # self.timing_stats = {
        #     'surface_processing': 0.0,
        #     'sme_encoding': 0.0,
        #     'count': 0
        # }


    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S
    
    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep):
        if X_pep.abs().sum() == 0:
            return X, S
        pep_seq = getattr(self, 'pep_seq', True)
        pep_struct = getattr(self, 'pep_struct', True)

        if self.pep_seq:
            S[paratope_mask] = S_pep
        if self.pep_struct:
            X[paratope_mask] = X_pep
        return X, S
    
    def align_epi_ab(self, local_inter_edges, local_is_ab) : 
        aligned = torch.zeros_like(local_inter_edges)
        try : 
            for i in range(local_inter_edges.shape[1]) : 
                if local_is_ab[local_inter_edges[0][i]] == False and local_is_ab[local_inter_edges[1][i]] == True : 
                    aligned[:, i] = local_inter_edges[:, i]
                elif local_is_ab[local_inter_edges[0][i]] == True and local_is_ab[local_inter_edges[1][i]] == False : 
                    aligned[0, i] = local_inter_edges[1][i]
                    aligned[1, i] = local_inter_edges[0][i]
        except Exception as e : 
            print(e)
        
        epi_index = torch.nonzero(~local_is_ab).squeeze()
        return aligned, epi_index
    
    def optimal_alignment(self, X0, target_X):
        """
        计算X0到target_X的最优旋转和排序
        Args:
            X0: [N, n_channel, 3] 初始构象
            target_X: [N, n_channel, 3] 目标构象
        Returns:
            R: [3, 3] 最优旋转矩阵
            perm: [N] 最优排序
            X0_aligned: [N, n_channel, 3] 经过旋转和排序后的X0
        """
        from scipy.optimize import linear_sum_assignment
        # 1. 先计算最优旋转
        X0_flat = X0.reshape(-1, 3)
        target_X_flat = target_X.reshape(-1, 3)
        _, R, t = kabsch_torch(X0_flat, target_X_flat)
        X0_rotated = torch.matmul(X0, R.T) + t
        
        # 2. 计算最优排序 (使用匈牙利算法)
        cost_matrix = torch.cdist(X0_rotated.reshape(-1, 3), target_X.reshape(-1, 3))
        cost_matrix = cost_matrix.reshape(X0.shape[0], X0.shape[1], -1)  # [N, n_channel, N*n_channel]
        cost_matrix = cost_matrix.reshape(X0.shape[0]*X0.shape[1], -1)  # [N*n_channel, N*n_channel]
        
        # 使用匈牙利算法找最优匹配
        perm = linear_sum_assignment(cost_matrix.cpu().numpy())[1]
        perm = torch.from_numpy(perm).to(X0.device)
        
        # 应用旋转和排序
        X0_aligned = X0_rotated.reshape(-1, 3)[perm].reshape(X0.shape)
        
        return R, perm, X0_aligned
        

    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None):
        # embeddings, hidden state, (internal edges, external edges), (A : c * d, w : c * 1)
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)

        if not self.keep_memory:
            memory_H = None

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        if self.pred_edge_dist:
            if memory_H is not None:
                edge_H = self.edge_H_ffn(memory_H)
            else:
                # replace the MLP with gnn for initial edge distance prediction
                edge_H, dumb_X = self.init_gnn(H_0, X, ctx_edges,
                                       channel_attr=atom_embeddings,
                                       channel_weights=atom_weights)
                X = X + dumb_X * 0  # to cheat the autograd check

        # update coordination of the global node
        X = self.aa_feature.update_global_coordinates(X, S)

        # prepare local complex
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        # prepare local complex edges
        local_ctx_edges = self.batch_constants['local_ctx_edges']  # [2, Ec]
        local_inter_edges = self.batch_constants['local_inter_edges']  # [2, Ei]
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        # for context edges, use edges in the native paratope
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni))
        # for interative edges, use edges derived from the predicted distance
        local_X[local_is_ab] = interface_X
        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src_H, dst_H = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) +\
                          self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))  # perm-invariant
            p_edge_dist = p_edge_dist.squeeze()
        else:
            p_edge_dist = None
        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni), given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)
        
        #prepare surface
        # surf_start = time.time()
        aligned_local_inter_edges, epi_index = self.align_epi_ab(local_inter_edges, local_is_ab)
        # self.timing_stats['surface_processing'] += time.time() - surf_start

        # message passing
        # sme_start = time.time()
        H, pred_X, pred_local_X = self.gnn(H_0, X, ctx_edges,
                                           local_mask, local_X, surf, local_edges,
                                           paratope_mask, local_is_ab,
                                           aligned_local_inter_edges, epi_index,
                                           channel_attr=atom_embeddings,
                                           channel_weights=atom_weights)
        # self.timing_stats['sme_encoding'] += time.time() - sme_start

        interface_X = pred_local_X[local_is_ab]
        pred_logits = None if self.struct_only else self.ffn_residue(H)

        return pred_logits, pred_X, interface_X, H, p_edge_dist  # [N, num_classes], [N, n_channel, 3], [Ncdr, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]  # [bs, 3]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10  + ca_noise.unsqueeze(1) # scale other atoms
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise

        init_local_S = torch.randint(0, self.num_classes, 
                                   (paratope_mask.sum(),), 
                                   device=X.device,
                                   dtype=torch.long)
        return init_local_X, init_local_S

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        # generate batch id
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1

        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids

        # interface relatd
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(
            paratope_mask, torch.logical_and(is_ag, not_ag_global)
        )
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        # interface local edges
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = local_segment_ids[row], local_segment_ids[col]
        is_ctx = row_segment_ids == col_segment_ids
        is_inter = torch.logical_not(is_ctx)

        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])  # [2, Ec]
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])  # [2, Ei]
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)

        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id
    
    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        return dist
        is_binding = dist <= self.bind_dist_cutoff
        return is_binding
    
    def _raw_interface_to_model_frame(self, interface_X, paratope_mask, batch_id):
        """Convert raw paratope coordinates into AbFlow's internal shadow frame.

        `_forward` centers the antigen/antibody and normalizes coordinates before
        message passing. Shadow paratope coordinates are later uncentered with
        `_type=4`, i.e. by adding the antigen center. Therefore an externally
        supplied raw X_t must be represented internally as:

            X_t_model = (X_t_raw - antigen_center) / std.
        """
        interface_batch_id = batch_id[paratope_mask]
        ag_centers = self.normalizer.ag_centers[interface_batch_id]
        return self.normalizer.normalize(interface_X - ag_centers.unsqueeze(1))

    def _coord_score_from_clean(self, Xt, clean_X, t, sigma_t):
        """Analytic coordinate score for AbFlow's conditional path.

        Path:
            X_t = (1 - t) X_0 + t X_1, sigma_t = 1 - t
        Conditional density:
            p_t(X | X_1) = N(t X_1, sigma_t^2 I)
        Score:
            s_t(X_t | X_1) = -(X_t - t X_1) / sigma_t^2
        """
        return -(Xt - t * clean_X) / (sigma_t ** 2 + self.scorefm_eps)

    def _masked_residue_mse(self, diff, atom_mask, interface_batch_id):
        """ABX-style normalized vector MSE for [N_int, C, 3] tensors.

        We first sum over xyz, average valid atom channels in each residue,
        average residues inside each complex, then average complexes. This avoids
        biasing the loss toward residues or complexes with more valid atoms.
        """
        atom_mask_f = atom_mask.to(diff.dtype)
        atom_sq = (diff ** 2).sum(dim=-1) * atom_mask_f  # [N_int, C]
        per_res = atom_sq.sum(dim=-1) / atom_mask_f.sum(dim=-1).clamp_min(1.0)
        per_graph = scatter_mean(per_res, interface_batch_id, dim=0)
        return per_graph.mean()

    def _masked_residue_smooth_l1(self, pred, target, atom_mask, interface_batch_id):
        """ABX-style normalized SmoothL1 for coordinate tensors."""
        atom_mask_f = atom_mask.to(pred.dtype)
        err = F.smooth_l1_loss(pred, target, reduction='none').sum(dim=-1)  # [N_int, C]
        err = err * atom_mask_f
        per_res = err.sum(dim=-1) / atom_mask_f.sum(dim=-1).clamp_min(1.0)
        per_graph = scatter_mean(per_res, interface_batch_id, dim=0)
        return per_graph.mean()

    def _local_ca_distance_loss(self, pred_X, true_X, interface_batch_id):
        """Paratope internal C-alpha distance preservation.

        This is a coordinate-level analogue of a distogram/FAPE stabilizer for
        AbFlow, but it does not require adding a distogram head. It preserves the
        local loop geometry of the generated paratope endpoint.
        """
        if pred_X.shape[0] <= 1:
            return pred_X.new_tensor(0.0)
        ca_idx = 1 if pred_X.shape[1] > 1 else 0
        losses = []
        for b in torch.unique(interface_batch_id):
            mask = interface_batch_id == b
            if mask.sum() <= 1:
                continue
            pred_ca = pred_X[mask, ca_idx]
            true_ca = true_X[mask, ca_idx]
            pred_d = torch.cdist(pred_ca, pred_ca)
            true_d = torch.cdist(true_ca, true_ca)
            tri = torch.triu(torch.ones_like(pred_d, dtype=torch.bool), diagonal=1)
            if tri.any():
                losses.append(F.smooth_l1_loss(pred_d[tri], true_d[tri]))
        if len(losses) == 0:
            return pred_X.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _residue_min_dist(self, A, B, A_mask, B_mask):
        """Minimum valid atom distance for every residue pair.

        Args:
            A: [Na, Ca, 3], B: [Nb, Cb, 3]
            A_mask: [Na, Ca], B_mask: [Nb, Cb]
        Returns:
            min_d: [Na, Nb], valid_pair: [Na, Nb]
        """
        if A.shape[0] == 0 or B.shape[0] == 0:
            return None, None
        d = torch.norm(A[:, None, :, None, :] - B[None, :, None, :, :], dim=-1)  # [Na,Nb,Ca,Cb]
        valid = A_mask[:, None, :, None] & B_mask[None, :, None, :]
        d = d.masked_fill(~valid, 1e6)
        min_d = d.flatten(2).min(dim=-1)[0]
        valid_pair = valid.flatten(2).any(dim=-1)
        return min_d, valid_pair

    def _interface_contact_bce_loss(self, pred_X, true_interface_X, true_X, true_S,
                                    paratope_mask, batch_id, segment_ids,
                                    interface_batch_id, interface_atom_mask):
        """Differentiable antigen-paratope contact recovery loss.

        Ground-truth contacts are defined from native minimum atom distances,
        and predictions use a smooth logit (cutoff - predicted_distance) / tau.
        This directly targets interface contact quality without adding a new head.
        """
        atom_pos_full = self.aa_feature._construct_atom_pos(true_S)
        atom_mask_full = atom_pos_full != self.aa_feature.atom_pos_pad_idx
        ag_mask_full = segment_ids == self.aa_feature.ag_seg_id

        losses = []
        for b in torch.unique(interface_batch_id):
            p_mask = interface_batch_id == b
            a_mask = (batch_id == b) & ag_mask_full
            if p_mask.sum() == 0 or a_mask.sum() == 0:
                continue
            pred_par = pred_X[p_mask]
            true_par = true_interface_X[p_mask]
            par_atom_mask = interface_atom_mask[p_mask]
            ag_X = true_X[a_mask]
            ag_atom_mask = atom_mask_full[a_mask]

            pred_d, valid_pair = self._residue_min_dist(pred_par, ag_X, par_atom_mask, ag_atom_mask)
            true_d, _ = self._residue_min_dist(true_par, ag_X, par_atom_mask, ag_atom_mask)
            if pred_d is None or not valid_pair.any():
                continue
            label = (true_d < self.scorefm_contact_cutoff).to(pred_X.dtype)
            logits = (self.scorefm_contact_cutoff - pred_d) / self.scorefm_contact_temperature
            losses.append(F.binary_cross_entropy_with_logits(logits[valid_pair], label[valid_pair]))
        if len(losses) == 0:
            return pred_X.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _interface_clash_loss(self, pred_X, true_X, true_S, batch_id, segment_ids,
                              interface_batch_id, interface_atom_mask):
        """Repel predicted paratope atoms from antigen atoms if they clash."""
        atom_pos_full = self.aa_feature._construct_atom_pos(true_S)
        atom_mask_full = atom_pos_full != self.aa_feature.atom_pos_pad_idx
        ag_mask_full = segment_ids == self.aa_feature.ag_seg_id

        losses = []
        for b in torch.unique(interface_batch_id):
            p_mask = interface_batch_id == b
            a_mask = (batch_id == b) & ag_mask_full
            if p_mask.sum() == 0 or a_mask.sum() == 0:
                continue
            p_atoms = pred_X[p_mask].reshape(-1, 3)
            p_valid = interface_atom_mask[p_mask].reshape(-1)
            a_atoms = true_X[a_mask].reshape(-1, 3)
            a_valid = atom_mask_full[a_mask].reshape(-1)
            p_atoms = p_atoms[p_valid]
            a_atoms = a_atoms[a_valid]
            if p_atoms.shape[0] == 0 or a_atoms.shape[0] == 0:
                continue
            d = torch.cdist(p_atoms, a_atoms)
            losses.append(F.relu(self.scorefm_inter_clash_cutoff - d).pow(2).mean())
        if len(losses) == 0:
            return pred_X.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _intra_paratope_clash_loss(self, pred_X, interface_atom_mask, interface_batch_id):
        """Repel atoms from different paratope residues if they clash."""
        losses = []
        for b in torch.unique(interface_batch_id):
            mask = interface_batch_id == b
            if mask.sum() <= 1:
                continue
            Xb = pred_X[mask]
            Mb = interface_atom_mask[mask]
            n_res, n_ch = Mb.shape
            atoms = Xb.reshape(-1, 3)
            valid = Mb.reshape(-1)
            res_ids = torch.arange(n_res, device=pred_X.device).repeat_interleave(n_ch)
            atoms = atoms[valid]
            res_ids = res_ids[valid]
            if atoms.shape[0] <= 1:
                continue
            d = torch.cdist(atoms, atoms)
            same_res = res_ids[:, None] == res_ids[None, :]
            eye = torch.eye(d.shape[0], device=d.device, dtype=torch.bool)
            valid_pair = ~(same_res | eye)
            if valid_pair.any():
                losses.append(F.relu(self.scorefm_intra_clash_cutoff - d[valid_pair]).pow(2).mean())
        if len(losses) == 0:
            return pred_X.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _scorefm_loss(self, *, Xt, X0, X1, pred_clean_X, atom_mask,
                      true_X, true_S, paratope_mask, batch_id, segment_ids,
                      interface_batch_id, t, sigma_t):
        """Complete coordinate Score-FM objective for AbFlow.

        Components:
          1) ABX-style scaled DSM on analytically induced coordinate scores.
          2) Clean endpoint reconstruction for stability at high-noise states.
          3) Velocity identity loss tying score matching back to flow matching.
          4) Local C-alpha distance loss, a head-free distogram analogue.
          5) Interface contact recovery loss.
          6) Antigen-paratope and intra-paratope clash penalties.
        """
        gt_score = self._coord_score_from_clean(Xt, X1, t, sigma_t).detach()
        pred_score = self._coord_score_from_clean(Xt, pred_clean_X, t, sigma_t)

        # ABX-style score scaling: score_scaling = 1 / sigma_t.
        score_scaling = 1.0 / sigma_t.clamp_min(self.scorefm_min_sigma)
        dsm_diff = (pred_score - gt_score) / score_scaling
        dsm_loss = self._masked_residue_mse(dsm_diff, atom_mask, interface_batch_id)

        x1_loss = self._masked_residue_smooth_l1(pred_clean_X, X1, atom_mask, interface_batch_id)

        # Hybrid DSM/x1 loss, analogous to AbX's thresholded trans score/x0 objective.
        score_or_x1 = torch.where(
            t.reshape(()) > pred_clean_X.new_tensor(self.scorefm_t_threshold),
            dsm_loss,
            x1_loss,
        )

        # Velocity identity for the same path: u_t = X_1 - X_0 = X_1 + sigma_t s_t.
        pred_v = pred_clean_X + sigma_t * pred_score
        true_v = X1 - X0
        velocity_loss = self._masked_residue_smooth_l1(pred_v, true_v, atom_mask, interface_batch_id)

        local_dist_loss = self._local_ca_distance_loss(pred_clean_X, X1, interface_batch_id)
        interface_contact_loss = self._interface_contact_bce_loss(
            pred_clean_X, X1, true_X, true_S, paratope_mask, batch_id,
            segment_ids, interface_batch_id, atom_mask)
        inter_clash_loss = self._interface_clash_loss(
            pred_clean_X, true_X, true_S, batch_id, segment_ids,
            interface_batch_id, atom_mask)
        intra_clash_loss = self._intra_paratope_clash_loss(
            pred_clean_X, atom_mask, interface_batch_id)

        total = self.scorefm_loss_weight * (
            self.scorefm_velocity_weight * velocity_loss
            + self.scorefm_dsm_weight * score_or_x1
            + self.scorefm_x1_weight * x1_loss
            + self.scorefm_local_dist_weight * local_dist_loss
            + self.scorefm_interface_contact_weight * interface_contact_loss
            + self.scorefm_inter_clash_weight * inter_clash_loss
            + self.scorefm_intra_clash_weight * intra_clash_loss
        )

        details = {
            'scorefm_total': total.detach(),
            'scorefm_dsm': dsm_loss.detach(),
            'scorefm_x1': x1_loss.detach(),
            'scorefm_hybrid': score_or_x1.detach(),
            'scorefm_velocity': velocity_loss.detach(),
            'scorefm_local_dist': local_dist_loss.detach(),
            'scorefm_interface_contact': interface_contact_loss.detach(),
            'scorefm_inter_clash': inter_clash_loss.detach(),
            'scorefm_intra_clash': intra_clash_loss.detach(),
        }
        return total, details

    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise=None, interface_init=None):
        
        batch_id = self.batch_constants['batch_id']
        # print(batch_id[paratope_mask].shape, residue_pos[paratope_mask].shape)
        
        # print(X.shape, S.shape, cmask.shape, smask.shape, paratope_mask.shape, X_pep.shape, S_pep.shape, template.shape)

        # mask sequence and initialize coordinates with template
        X, S = self.init_mask(X, S, cmask, smask, template)
        
        # replace paratope with peptide
        X, S = self.replace_pep(X, S, paratope_mask, X_pep, S_pep)

        # normalize
        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)
        surface = self.normalizer.normalize(surface)

        # update center
        X = self.aa_feature.update_global_coordinates(X, S)

        # prepare initial interface
        # For coordinate Score-FM, the network must see the exact current state X_t
        # that defines the analytic score target. If interface_init is supplied, it
        # is in raw coordinates and must be converted into the internal normalized
        # antigen-centered shadow frame.
        if interface_init is None:
            interface_X, interface_S = self.init_interface(X, S, paratope_mask, batch_id, init_noise)
        else:
            interface_X = self._raw_interface_to_model_frame(interface_init, paratope_mask, batch_id)
            interface_S = S[paratope_mask].clone()
        # initial interface is replaced by peptide
        # interface_X = X[paratope_mask]

        # sequence and structure loss
        r_pred_S_logits, pred_S_dist, = [], None
        r_interface_X = [interface_X.clone()]  # init
        r_edge_dist = []
        memory_H = None
        # message passing
        for t in range(self.round):
            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(X, S, residue_pos, interface_X, surface, paratope_mask, batch_id, t, memory_H, pred_S_dist, smask)
            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_edge_dist.append(edge_dist)
            # 1. update X
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_global_coordinates(X, S)

            if not self.struct_only:
                # 2. update S
                S = S.clone()
                if t == self.round - 1:
                    S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask], dim=-1)

        interface_batch_id = self.batch_constants['interface_batch_id']

        if self.struct_only:
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        # uncentering and unnormalize
        pred_X = self.normalizer.unnormalize(pred_X)
        pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X in enumerate(r_interface_X):
            interface_X = self.normalizer.unnormalize(interface_X)
            interface_X = self.normalizer.uncentering(interface_X, interface_batch_id, _type=4)
            r_interface_X[i] = interface_X
        self.normalizer.clear_cache()


        return H, S, r_pred_S_logits, pred_X, r_interface_X,  r_edge_dist, prmsd

    def forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, xloss_mask, context_ratio=0):
        '''
        :param X: [N, n_channel, 3], Cartesian coordinates
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        '''
        # import ipdb; ipdb.set_trace()
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            xloss_mask = xloss_mask[:, :4]
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']

        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)
        
        gt_interface_X, gt_interface_S = true_X[paratope_mask], true_S[paratope_mask]
        interface_X, interface_S = self.init_interface(X, S, paratope_mask, batch_id)
        R, perm, interface_X_aligned = self.optimal_alignment(interface_X, gt_interface_X)
        interface_S_aligned = interface_S[torch.unique(perm // interface_X.shape[1])]
        
        # Continuous flow time. Avoid sigma_t = 1 - t being too small because
        # the analytic score contains 1 / sigma_t^2.
        t = torch.rand(1, device=X.device) * (1.0 - self.scorefm_min_sigma)
        sigma_t = (1.0 - t).clamp_min(self.scorefm_min_sigma)

        # AbFlow coordinate path: X_t = (1 - t) X_0 + t X_1.
        Xt = sigma_t * interface_X_aligned + t * gt_interface_X
        St = (1-t) * interface_S_aligned + t * gt_interface_S
        X[paratope_mask] = Xt.to(X.dtype)
        S[paratope_mask] = St.to(S.dtype)

        # get results
        # IMPORTANT: pass interface_init=Xt. Otherwise _forward reinitializes the
        # shadow interface and the score target no longer matches the model input.
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths, interface_init=Xt
        )

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss
        
        # 1. interface loss (shadow paratope)
        interface_atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        interface_atom_mask = interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = F.smooth_l1_loss(
            r_interface_X[-1][interface_atom_mask],
            gt_interface_X[interface_atom_mask])

        # complete coordinate Score-FM loss
        # This replaces the old raw flow loss. It follows the score-based principle:
        # predict a clean endpoint, analytically induce the score from the AbFlow
        # path, then use ABX-style scaled DSM plus geometry/interface terms.
        flow_loss, scorefm_details = self._scorefm_loss(
            Xt=Xt,
            X0=interface_X_aligned,
            X1=gt_interface_X,
            pred_clean_X=r_interface_X[-1],
            atom_mask=interface_atom_mask,
            true_X=true_X,
            true_S=true_S,
            paratope_mask=paratope_mask,
            batch_id=batch_id,
            segment_ids=self.batch_constants['segment_ids'],
            interface_batch_id=self.batch_constants['interface_batch_id'],
            t=t,
            sigma_t=sigma_t,
        )
        self.last_scorefm_losses = scorefm_details


        # 2. edge dist loss
        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(self.normalizer.normalize(true_X), true_S)
            ed_loss, r_ed_losses = 0, []
            for edge_dist in r_edge_dist:
                r_ed_loss = F.smooth_l1_loss(edge_dist, gt_edge_dist)
                ed_loss = ed_loss + r_ed_loss
                r_ed_losses.append(r_ed_loss)
        else:
            r_ed_losses = [0 for _ in range(self.round)]
            ed_loss = 0
        dock_loss = interface_loss + ed_loss

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # comprehensive loss
        loss = snll + struct_loss + dock_loss + flow_loss + (0 if pdev_loss is None else pdev_loss)
        # loss = snll + struct_loss + dock_loss + flow_loss + (0 if pdev_loss is None else pdev_loss)

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss)

    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths,n_steps=10, init_noise=None, return_hidden=False, show_progress=True, progress_desc=None):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        
        # self.timing_stats = {
        #     'surface_processing': 0.0,
        #     'sme_encoding': 0.0,
        #     'count': 0
        # }
        
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        interface_X, interface_S = self.init_interface(X, S, paratope_mask, batch_id)
        dt = 1.0 / n_steps
        Xt = interface_X.clone()
        St = interface_S.clone()
        
        step_iter = range(n_steps)
        if show_progress:
            step_iter = tqdm(
                step_iter,
                total=n_steps,
                desc=progress_desc or 'Sampling ODE',
                leave=False,
                dynamic_ncols=True
            )

        for i in step_iter:
            t = torch.tensor(i * dt, device=X.device)

            if show_progress and hasattr(step_iter, 'set_postfix'):
                step_iter.set_postfix(t=f'{float(t):.2f}')
            
            # 更新当前状态
            X_cur = X.clone()
            S_cur = S.clone()
            X_cur[paratope_mask] = Xt
            S_cur[paratope_mask] = St
            
            # 使用message passing获取速度场
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
                X_cur, S_cur, cmask, smask, paratope_mask, 
                X_pep, S_pep, surface, residue_pos, template, lengths,
                interface_init=Xt
            )

            # Score-factorized velocity. Given predicted clean endpoint Xhat_1,
            # induced score s_theta = -(X_t - t Xhat_1) / sigma_t^2 and
            # v_theta = Xhat_1 + sigma_t * s_theta = (Xhat_1 - X_t) / sigma_t.
            sigma_t = (1.0 - t).clamp_min(self.scorefm_min_sigma)
            pred_clean_X = r_interface_X[-1]
            pred_score = self._coord_score_from_clean(Xt, pred_clean_X, t, sigma_t)
            dX = pred_clean_X + sigma_t * pred_score
            if not self.struct_only:
                cur_logits = r_pred_S_logits[-1][0][paratope_mask]
                # 1. 数值稳定性处理
                cur_logits = cur_logits - cur_logits.max(dim=-1, keepdim=True)[0]
                # 2. 计算当前概率
                cur_probs = F.softmax(cur_logits, dim=-1)
                # 3. 计算序列的速度场
                dS = cur_probs - F.one_hot(St, num_classes=self.num_classes).float()
                
            # Euler步进
            Xt = Xt + dX * dt
            if not self.struct_only:
                next_probs = F.one_hot(St, num_classes=self.num_classes).float() + dS * dt
                # 确保数值稳定性
                next_probs = next_probs.clamp(min=1e-6)
                next_probs = next_probs / next_probs.sum(dim=-1, keepdim=True)
                # 从更新后的概率分布中采样
                St = torch.multinomial(next_probs, num_samples=1).squeeze(-1)
        
        X[paratope_mask] = Xt
        S[paratope_mask] = St
            
        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            # Use the final ODE state Xt as shadow-interface input instead of
            # reinitializing from noise.
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths, interface_init=Xt
            )

            # PPL or PRMSD
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                metric = scatter_mean(nlls, s_batch_id)  # [batch_size]
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id)  # [batch_size]

            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            # update metric history
            best_metric[update] = metric[update]

            # 1. set generated part
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
        
            interface_X = r_interface_X[-1]
            # 2. align by cdr
            for i in range(batch_size):
                if not update[i]:
                    continue
                # 1. align CDRH3
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]  # backbone
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))

                # 2. tranform antibody
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X

        self._clean_batch_constants()

        # self.timing_stats['count'] += 1

        if return_hidden:
            return gen_X, gen_S, metric, H
        return gen_X, gen_S, metric
    
    def struct_sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise=None, return_hidden=False):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise)

            # PPL or PRMSD
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                metric = scatter_mean(nlls, s_batch_id)  # [batch_size]
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id)  # [batch_size]

            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            # update metric history
            best_metric[update] = metric[update]

            # 1. set generated part
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
        
            interface_X = r_interface_X[-1]
            # 2. align by cdr
            for i in range(batch_size):
                if not update[i]:
                    continue
                # 1. align CDRH3
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]  # backbone
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))

                # 2. tranform antibody
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X

        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, metric, H
        return gen_X, gen_S, metric

    def sample_many(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths,n_samples=5, n_steps=20, return_hidden=False, show_progress=False):
        """
        Generate multiple samples in a single call
        
        Args:
            X, S, cmask, smask, paratope_mask, residue_pos, template, lengths: 
                Same parameters as in sample() method
            n_samples: Number of samples to generate
            n_steps: Number of flow steps for each sample
            return_hidden: Whether to return hidden states
            
        Returns:
            list_gen_X: List of n_samples generated coordinates
            list_gen_S: List of n_samples generated sequences
            list_metrics: List of n_samples metrics
            list_H: (Optional) List of n_samples hidden states if return_hidden=True
        """
        list_gen_X = []
        list_gen_S = []
        list_metrics = []
        list_H = [] if return_hidden else None
        
        # Generate multiple samples with different random noise
        for i in range(n_samples):
            # Generate different noise for each sample
            if self.backbone_only:
                init_noise = torch.randn(paratope_mask.sum(), 4, 3, device=X.device)
            else:
                init_noise = torch.randn(paratope_mask.sum(), X.shape[1], 3, device=X.device)
                
            # Generate a sample
            if return_hidden:
                gen_X, gen_S, metric, H = self.sample(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos,
                    template, lengths,
                    n_steps=n_steps,
                    init_noise=init_noise,
                    return_hidden=True,
                    show_progress=show_progress,
                    progress_desc=f'Sample {i + 1}/{n_samples} ODE'
                )
                list_H.append(H)
            else:
                gen_X, gen_S, metric = self.sample(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos,
                    template, lengths,
                    n_steps=n_steps,
                    init_noise=init_noise,
                    show_progress=show_progress,
                    progress_desc=f'Sample {i + 1}/{n_samples} ODE'
                )
            
            # Store results
            list_gen_X.append(gen_X)
            list_gen_S.append(gen_S)
            list_metrics.append(metric)
            
        if return_hidden:
            return list_gen_X, list_gen_S, list_metrics, list_H
        else:
            return list_gen_X, list_gen_S, list_metrics

isMEANModel = AbFlowModel
dyMEANModel = AbFlowModel
