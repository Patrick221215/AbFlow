#!/usr/bin/python
# -*- coding:utf-8 -*-
# V194_KABSCH_STOPGRAD_FP32: rigid alignment is a nuisance-frame target transform; do not backprop through SVD.
import functools as fn
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_mean, scatter_sum

from data.pdb_utils import VOCAB
from evaluation.rmsd import kabsch_torch
from configs import (
    RESTYPES, RESTYPE_1TO3, RESTYPE_NUM, ATOM14_ORDER, ATOM14_INDEX,
    ATOM14_MASK, CHI_ANGLES_ATOMS, NUM_AB_REGIONS,
)


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def normalize_vector(v, dim=-1, eps=1e-6):
    """Normalize vectors with the same stable convention used by DiffAb."""
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim=-1):
    """Project vector ``v`` onto unit vector ``e``."""
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_residue_basis(center, carbon, nitrogen, eps=1e-6):
    """Construct a right-handed residue frame from CA, C and N.

    This is the DiffAb backbone-frame construction:
      e1 = normalize(C - CA)
      e2 = normalize((N - CA) - proj_e1(N - CA))
      e3 = e1 x e2

    Args:
        center:   [..., 3], CA coordinates.
        carbon:   [..., 3], C coordinates.
        nitrogen: [..., 3], N coordinates.
    Returns:
        [..., 3, 3] matrix whose columns are (e1, e2, e3).
    """
    e1 = normalize_vector(carbon - center, dim=-1, eps=eps)
    v2 = nitrogen - center
    e2 = normalize_vector(v2 - project_v2v(v2, e1, dim=-1), dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)


def global_to_local(R, t, q):
    """Convert global coordinates to residue-local coordinates: R^T (q - t)."""
    q_size = q.shape
    B, L = q_size[:2]
    q_flat = q.reshape(B, L, -1, 3).transpose(-1, -2)
    local = torch.matmul(R.transpose(-1, -2), q_flat - t.unsqueeze(-1))
    return local.transpose(-1, -2).reshape(q_size)


def atom14_to_residue_local(
    atom14_positions, atom14_exists, fixed_mask,
    coordinate_scale=0.1, eps=1e-6,
):
    """Encode fixed-residue atom14 geometry in a DiffAb-style local frame.

    Frame construction is evaluated in FP32 even under BF16 autocast. Only the
    resulting invariant local coordinates are cast back before the learned MLP.
    """
    n_idx = ATOM14_ORDER['N']
    ca_idx = ATOM14_ORDER['CA']
    c_idx = ATOM14_ORDER['C']
    out_dtype = atom14_positions.dtype
    exists = atom14_exists.bool()
    fixed = fixed_mask.bool()
    with torch.cuda.amp.autocast(enabled=False):
        pos = atom14_positions.float()
        ca = pos[..., ca_idx, :]
        carbon = pos[..., c_idx, :]
        nitrogen = pos[..., n_idx, :]
        v1 = carbon - ca
        e1 = normalize_vector(v1, dim=-1, eps=eps)
        v2 = nitrogen - ca
        u2 = v2 - project_v2v(v2, e1, dim=-1)
        geometry_valid = (torch.linalg.norm(v1, dim=-1) > eps) & (torch.linalg.norm(u2, dim=-1) > eps)
        frame_valid = fixed & exists[..., n_idx] & exists[..., ca_idx] & exists[..., c_idx] & geometry_valid
        R = construct_residue_basis(ca, carbon, nitrogen, eps=eps)
        local = global_to_local(R, ca, pos)
        atom_valid = exists & frame_valid[..., None]
        local = torch.where(atom_valid[..., None], local, torch.zeros_like(local))
        local = local * float(coordinate_scale)
    return local.to(dtype=out_dtype), frame_valid


def graph_to_batch(tensor, batch_id, padding_value=0, mask_is_pad=True):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    batch = torch.ones((bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) * padding_value
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask


def _knn_edges(X, AP, src_dst, atom_pos_pad_idx, k_neighbors, batch_info, given_dist=None):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param AP: [N, n_channel], atom position with pad type need to be ignored
    :param src_dst: [Ef, 2], full possible edges represented in (src, dst)
    :param given_dist: [Ef], given distance of edges
    '''
    offsets, batch_id, max_n, gni2lni = batch_info

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = X.shape[0]
    if given_dist is None:
        dist = X[src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = AP[src_dst] == atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * BIGINT  # [Ef, n_channel, n_channel]
        del pos_pad  # release memory
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
    else:
        dist = given_dist
    src_dst = src_dst.transpose(0, 1)  # [2, Ef]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(src_dst[0], gni2lni[src_dst[1]])] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k_neighbors, dim=-1, largest=False)  # [N, topk]

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k_neighbors)
    src, dst = src.flatten(), dst.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)

    dst = dst + offsets[batch_id[src]]  # mapping from local to global node index

    edges = torch.stack([src, dst])  # message passed from dst to src
    return edges  # [2, E]


class EdgeConstructor:
    def __init__(self, boa_idx, boh_idx, bol_idx, atom_pos_pad_idx, ag_seg_id) -> None:
        self.boa_idx, self.boh_idx, self.bol_idx = boa_idx, boh_idx, bol_idx
        self.atom_pos_pad_idx = atom_pos_pad_idx
        self.ag_seg_id = ag_seg_id

        # buffer
        self._reset_buffer()

    def _reset_buffer(self):
        self.row = None
        self.col = None
        self.row_global = None
        self.col_global = None
        self.row_seg = None
        self.col_seg = None
        self.offsets = None
        self.max_n = None
        self.gni2lni = None
        self.not_global_edges = None

    def get_batch_edges(self, batch_id):
        # construct tensors to map between global / local node index
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)  # [bs]
        # global node index to local index. lni2gni can be implemented as lni + offsets[batch_id]
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]  # [N]

        # all possible edges (within the same graph)
        # same bid (get rid of self-loop and none edges)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        # shift right and pad 1 to the left
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0  # delete self loop
        row, col = torch.nonzero(same_bid).T  # [2, n_edge_all]
        col = col + offsets[batch_id[row]]  # mapping from local to global node index
        return (row, col), (offsets, max_n, gni2lni)

    def _prepare(self, S, batch_id, segment_ids) -> None:
        (row, col), (offsets, max_n, gni2lni) = self.get_batch_edges(batch_id)

        # not global edges
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx) # [N]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        
        # segment ids
        row_seg, col_seg = segment_ids[row], segment_ids[col]

        # add to buffer
        self.row, self.col = row, col
        self.offsets, self.max_n, self.gni2lni = offsets, max_n, gni2lni
        self.row_global, self.col_global = row_global, col_global
        self.not_global_edges = not_global_edges
        self.row_seg, self.col_seg = row_seg, col_seg

    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: same seg, not global
        select_edges = torch.logical_and(self.row_seg == self.col_seg, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: not same seg, not global
        select_edges = torch.logical_and(self.row_seg != self.col_seg, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        outer_edges = _knn_edges(
            X, atom_pos, torch.stack([inter_all_row, inter_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return outer_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        select_edges = torch.logical_and(self.row_global, self.col_global) # self-loop has been deleted
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_seq_edges(self):
        row, col = self.row, self.col
        # add additional edge to neighbors in 1D sequence (except epitope)
        select_edges = sequential_and(
            torch.logical_or((row - col) == 1, (row - col) == -1),  # adjacent in the graph
            self.not_global_edges,  # not global edges (also ensure the edges are in the same segment)
            self.row_seg != self.ag_seg_id  # not epitope
        )
        seq_adj = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return seq_adj

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, k_neighbors, atom_pos, segment_ids):
        '''
        Memory efficient with complexity of O(Nn) where n is the largest number of nodes in the batch
        '''
        # prepare inputs
        self._prepare(S, batch_id, segment_ids)

        ctx_edges, inter_edges = [], []

        # edges within chains
        inner_edges = self._construct_inner_edges(X, batch_id, k_neighbors, atom_pos)
        # edges between global nodes and normal/global nodes
        global_normal, global_global = self._construct_global_edges()
        # edges on the 1D sequence
        seq_edges = self._construct_seq_edges()

        # construct context edges
        ctx_edges = torch.cat([inner_edges, global_normal, global_global, seq_edges], dim=1)  # [2, E]

        # construct interaction edges
        inter_edges = self._construct_outer_edges(X, batch_id, k_neighbors, atom_pos)

        self._reset_buffer()
        return ctx_edges, inter_edges


class GMEdgeConstructor(EdgeConstructor):
    '''
    Edge constructor for graph matching (kNN internel edges and all bipartite edges)
    '''
    def _construct_inner_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible ctx edges: both in ag or ab, not global
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = torch.logical_and(row_is_ag == col_is_ag, self.not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        # ctx edges
        inner_edges = _knn_edges(
            X, atom_pos, torch.stack([ctx_all_row, ctx_all_col]).T,
            self.atom_pos_pad_idx, k_neighbors,
            (self.offsets, batch_id, self.max_n, self.gni2lni))
        return inner_edges

    def _construct_global_edges(self):
        row, col = self.row, self.col
        # edges between global and normal nodes
        select_edges = torch.logical_and(self.row_seg == self.col_seg, torch.logical_not(self.not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        # edges between global and global nodes
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = sequential_and(
            self.row_global, self.col_global, # self-loop has been deleted
            row_is_ag == col_is_ag)  # only inter-ag or inter-ab globals
        global_global = torch.stack([row[select_edges], col[select_edges]])  # [2, nE]
        return global_normal, global_global

    def _construct_outer_edges(self, X, batch_id, k_neighbors, atom_pos):
        row, col = self.row, self.col
        # all possible inter edges: one in ag and one in ab, not global
        row_is_ag = self.row_seg == self.ag_seg_id
        col_is_ag = self.col_seg == self.ag_seg_id
        select_edges = torch.logical_and(row_is_ag != col_is_ag, self.not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        return torch.stack([inter_all_row, inter_all_col])  # [2, E]


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings

# embedding of amino acids. (default: concat residue embedding and atom embedding to one vector)
class AminoAcidEmbedding(nn.Module):
    '''
    [residue embedding + position embedding, mean(atom embeddings + atom position embeddings)]
    '''
    def __init__(self, num_res_type, num_atom_type, num_atom_pos, res_embed_size, atom_embed_size,
                 atom_pad_id=VOCAB.get_atom_pad_idx(), relative_position=True, max_position=192):  # max position (with IMGT numbering)
        super().__init__()
        self.residue_embedding = nn.Embedding(num_res_type, res_embed_size)
        if relative_position:
            self.res_pos_embedding = SinusoidalPositionEmbedding(res_embed_size)  # relative positional encoding
        else:
            self.res_pos_embedding = nn.Embedding(max_position, res_embed_size)  # absolute position encoding
        self.atom_embedding = nn.Embedding(num_atom_type, atom_embed_size)
        self.atom_pos_embedding = nn.Embedding(num_atom_pos, atom_embed_size)
        self.atom_pad_id = atom_pad_id
        self.eps = 1e-10  # for mean of atom embedding (some residues have no atom at all)
    
    def forward(self, S, RP, A, AP):
        '''
        :param S: [N], residue types
        :param RP: [N], residue positions
        :param A: [N, n_channel], atom types
        :param AP: [N, n_channel], atom positions
        '''
        res_embed = self.residue_embedding(S) + self.res_pos_embedding(RP)  # [N, res_embed_size]
        atom_embed = self.atom_embedding(A) + self.atom_pos_embedding(AP)   # [N, n_channel, atom_embed_size]
        atom_not_pad = (AP != self.atom_pad_id)  # [N, n_channel]
        denom = torch.sum(atom_not_pad, dim=-1, keepdim=True) + self.eps
        atom_embed = torch.sum(atom_embed * atom_not_pad.unsqueeze(-1), dim=1) / denom  # [N, atom_embed_size]
        return torch.cat([res_embed, atom_embed], dim=-1)  # [N, res_embed_size + atom_embed_size]


class AminoAcidFeature(nn.Module):
    def __init__(self, embed_size, relative_position=True, edge_constructor=EdgeConstructor, backbone_only=False) -> None:
        super().__init__()

        self.backbone_only = backbone_only

        # number of classes
        self.num_aa_type = len(VOCAB)
        self.num_atom_type = VOCAB.get_num_atom_type()
        self.num_atom_pos = VOCAB.get_num_atom_pos()

        # atom-level special tokens
        self.atom_mask_idx = VOCAB.get_atom_mask_idx()
        self.atom_pad_idx = VOCAB.get_atom_pad_idx()
        self.atom_pos_mask_idx = VOCAB.get_atom_pos_mask_idx()
        self.atom_pos_pad_idx = VOCAB.get_atom_pos_pad_idx()
        
        # embedding
        self.aa_embedding = AminoAcidEmbedding(
            self.num_aa_type, self.num_atom_type, self.num_atom_pos,
            embed_size, embed_size, self.atom_pad_idx, relative_position)

        # global nodes and mask nodes
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)
        self.mask_idx = VOCAB.get_mask_idx()

        # segment ids
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 1, 2, 3

        # atoms encoding
        residue_atom_type, residue_atom_pos = [], []
        backbone = [VOCAB.atom_to_idx(atom[0]) for atom in VOCAB.backbone_atoms]
        n_channel = VOCAB.MAX_ATOM_NUMBER if not backbone_only else 4
        special_mask = VOCAB.get_special_mask()
        for i in range(len(VOCAB)):
            if i == self.boa_idx or i == self.boh_idx or i == self.bol_idx or i == self.mask_idx:
                # global nodes
                residue_atom_type.append([self.atom_mask_idx for _ in range(n_channel)])
                residue_atom_pos.append([self.atom_pos_mask_idx for _ in range(n_channel)])
            elif special_mask[i] == 1:
                # other special token (pad)
                residue_atom_type.append([self.atom_pad_idx for _ in range(n_channel)])
                residue_atom_pos.append([self.atom_pos_pad_idx for _ in range(n_channel)])
            else:
                # normal amino acids
                sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                atom_type = backbone
                atom_pos = [VOCAB.atom_pos_to_idx(VOCAB.atom_pos_bb) for _ in backbone]
                if not backbone_only:
                    sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                    atom_type = atom_type + [VOCAB.atom_to_idx(atom[0]) for atom in sidechain_atoms]
                    atom_pos = atom_pos + [VOCAB.atom_pos_to_idx(atom[1]) for atom in sidechain_atoms]
                num_pad = n_channel - len(atom_type)
                residue_atom_type.append(atom_type + [self.atom_pad_idx for _ in range(num_pad)])
                residue_atom_pos.append(atom_pos + [self.atom_pos_pad_idx for _ in range(num_pad)])
        
        # mapping from residue to atom types and positions
        self.residue_atom_type = nn.parameter.Parameter(
            torch.tensor(residue_atom_type, dtype=torch.long),
            requires_grad=False)
        self.residue_atom_pos = nn.parameter.Parameter(
            torch.tensor(residue_atom_pos, dtype=torch.long),
            requires_grad=False)

        # sidechain geometry
        if not backbone_only:
            sc_bonds, sc_bonds_mask = [], []
            sc_chi_atoms, sc_chi_atoms_mask = [], []
            for i in range(len(VOCAB)):
                if special_mask[i] == 1:
                    sc_bonds.append([])
                    sc_chi_atoms.append([])
                else:
                    symbol = VOCAB.idx_to_symbol(i)
                    atom_type = VOCAB.backbone_atoms + VOCAB.get_sidechain_info(symbol)
                    atom2channel = { atom: i for i, atom in enumerate(atom_type) }
                    chi_atoms, bond_atoms = VOCAB.get_sidechain_geometry(symbol)
                    sc_chi_atoms.append(
                        [[atom2channel[atom] for atom in atoms] for atoms in chi_atoms]
                    )
                    bonds = []
                    for src_atom in bond_atoms:
                        for dst_atom in bond_atoms[src_atom]:
                            bonds.append((atom2channel[src_atom], atom2channel[dst_atom]))
                    sc_bonds.append(bonds)
            max_num_chis = max([len(chis) for chis in sc_chi_atoms])
            max_num_bonds = max([len(bonds) for bonds in sc_bonds])
            for i in range(len(VOCAB)):
                num_chis, num_bonds = len(sc_chi_atoms[i]), len(sc_bonds[i])
                num_pad_chis, num_pad_bonds = max_num_chis - num_chis, max_num_bonds - num_bonds
                sc_chi_atoms_mask.append(
                    [1 for _ in range(num_chis)] + [0 for _ in range(num_pad_chis)]
                )
                sc_bonds_mask.append(
                    [1 for _ in range(num_bonds)] + [0 for _ in range(num_pad_bonds)]
                )
                sc_chi_atoms[i].extend([[-1, -1, -1, -1] for _ in range(num_pad_chis)])
                sc_bonds[i].extend([(-1, -1) for _ in range(num_pad_bonds)])

            # mapping residues to their sidechain chi angle atoms and bonds
            self.sidechain_chi_angle_atoms = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms, dtype=torch.long),
                requires_grad=False)
            self.sidechain_chi_mask = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms_mask, dtype=torch.bool),
                requires_grad=False
            )
            self.sidechain_bonds = nn.parameter.Parameter(
                torch.tensor(sc_bonds, dtype=torch.long),
                requires_grad=False
            )
            self.sidechain_bonds_mask = nn.parameter.Parameter(
                torch.tensor(sc_bonds_mask, dtype=torch.bool),
                requires_grad=False
            )

        # edge constructor
        self.edge_constructor = edge_constructor(self.boa_idx, self.boh_idx, self.bol_idx, self.atom_pos_pad_idx, self.ag_seg_id)

    def _is_global(self, S):
        return sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)  # [N]

    def _construct_residue_pos(self, S):
        # construct residue position. global node is 1, the first residue is 2, ... (0 for padding)
        glbl_node_mask = self._is_global(S)
        glbl_node_idx = torch.nonzero(glbl_node_mask).flatten()  # [batch_size * 3] (boa, boh, bol)
        shift = F.pad(glbl_node_idx[:-1] - glbl_node_idx[1:] + 1, (1, 0), value=1) # [batch_size * 3]
        residue_pos = torch.ones_like(S)
        residue_pos[glbl_node_mask] = shift
        residue_pos = torch.cumsum(residue_pos, dim=0)
        return residue_pos

    def _construct_segment_ids(self, S):
        # construct segment ids. 1/2/3 for antigen/heavy chain/light chain
        glbl_node_mask = self._is_global(S)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        return segment_ids

    def _construct_atom_type(self, S):
        # construct atom types
        return self.residue_atom_type[S]
    
    def _construct_atom_pos(self, S):
        # construct atom positions
        return self.residue_atom_pos[S]

    @torch.no_grad()
    def get_sidechain_chi_angles_atoms(self, S):
        chi_angles_atoms = self.sidechain_chi_angle_atoms[S]  # [N, max_num_chis, 4]
        chi_mask = self.sidechain_chi_mask[S]  # [N, max_num_chis]
        return chi_angles_atoms, chi_mask

    @torch.no_grad()
    def get_sidechain_bonds(self, S):
        bonds = self.sidechain_bonds[S]  # [N, max_num_bond, 2]
        bond_mask = self.sidechain_bonds_mask[S]
        return bonds, bond_mask

    def update_global_coordinates(self, X, S, atom_pos=None):
        X = X.clone()

        if atom_pos is None:  # [N, n_channel]
            atom_pos = self._construct_atom_pos(S)

        glbl_node_mask = self._is_global(S)
        chain_id = glbl_node_mask.long()
        chain_id = torch.cumsum(chain_id, dim=0)  # [N]
        chain_id[glbl_node_mask] = 0    # set global nodes to 0
        chain_id = chain_id.unsqueeze(-1).repeat(1, atom_pos.shape[-1])  # [N, n_channel]
        
        not_global = torch.logical_not(glbl_node_mask)
        not_pad = (atom_pos != self.atom_pos_pad_idx)[not_global]
        flatten_coord = X[not_global][not_pad]  # [N_atom, 3]
        flatten_chain_id = chain_id[not_global][not_pad]

        global_x = scatter_mean(
            src=flatten_coord, index=flatten_chain_id,
            dim=0, dim_size=glbl_node_mask.sum() + 1)  # because index start from 1
        X[glbl_node_mask] = global_x[1:].unsqueeze(1)

        return X

    def embedding(self, S, residue_pos=None, atom_type=None, atom_pos=None):
        '''
        :param S: [N], residue types
        '''
        if residue_pos is None:  # Residue positions in the chain
            residue_pos = self._construct_residue_pos(S)  # [N]

        if atom_type is None:  # Atom types in each residue
            atom_type = self.residue_atom_type[S]  # [N, n_channel]

        if atom_pos is None:   # Atom position in each residue
            atom_pos = self.residue_atom_pos[S]     # [N, n_channel]

        H = self.aa_embedding(S, residue_pos, atom_type, atom_pos)
        return H, (residue_pos, atom_type, atom_pos)

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id, k_neighbors, atom_pos=None, segment_ids=None):

        # prepare inputs
        if atom_pos is None:  # Atom position in each residue (pad need to be ignored)
            atom_pos = self.residue_atom_pos[S]
        
        if segment_ids is None:
            segment_ids = self._construct_segment_ids(S)

        ctx_edges, inter_edges = self.edge_constructor.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos, segment_ids)

        return ctx_edges, inter_edges

    def forward(self, X, S, batch_id, k_neighbors):
        H, (_, _, atom_pos) = self.embedding(S)
        ctx_edges, inter_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos)
        return H, (ctx_edges, inter_edges)


class SeparatedAminoAcidFeature(AminoAcidFeature):
    """Single residue/atom feature authority for R05 and the single/pair trunk."""

    def __init__(
        self, embed_size, atom_embed_size, relative_position=True,
        edge_constructor=EdgeConstructor, fix_atom_weights=False,
        backbone_only=False, representation_config=None,
    ) -> None:
        super().__init__(
            embed_size, relative_position=relative_position,
            edge_constructor=edge_constructor, backbone_only=backbone_only,
        )
        atom_weights_mask = self.residue_atom_type == self.atom_pad_idx
        self.register_buffer('atom_weights_mask', atom_weights_mask)
        self.fix_atom_weights = fix_atom_weights

        if fix_atom_weights:
            atom_weights = torch.ones_like(self.residue_atom_type, dtype=torch.float)
        else:
            atom_weights = torch.randn_like(self.residue_atom_type, dtype=torch.float)
        atom_weights[atom_weights_mask] = 0
        self.atom_weight = nn.Parameter(
            atom_weights, requires_grad=not fix_atom_weights
        )
        self.register_buffer(
            'zero_atom_weight', torch.zeros_like(atom_weights), persistent=False
        )

        # R05 residue/atom view.
        self.aa_embedding = AminoAcidEmbedding(
            self.num_aa_type, self.num_atom_type, self.num_atom_pos,
            embed_size, atom_embed_size, self.atom_pad_idx, relative_position,
        )

        self.representation_config = None
        self.single_pair_enabled = False
        if representation_config is not None:
            self.configure_single_pair(representation_config)

    def configure_single_pair(self, representation_config):
        """Attach the compact AbX-style single/pair view to the R05 residue authority."""
        self.representation_config = representation_config
        self.single_pair_enabled = True
        c = seqformer_config(representation_config)
        seq_channel = int(c.seq_channel)
        geometry_cfg = representation_config.get("geometry", {})
        self.single_pair_coordinate_scale = float(
            geometry_cfg.get("local_coordinate_scale", 0.1)
        )
        self.single_pair_frame_eps = float(
            geometry_cfg.get("frame_eps", 1e-6)
        )

        # These are independent learned views of the same canonical residue state.
        # They are intentionally not tied to the parent R05 residue table: tying
        # them would let R29/R30 auxiliary gradients rewrite the parent embedding
        # directly and would destroy the parent-preserving ablation boundary.
        self.single_pair_base_aa = nn.Embedding(
            RESTYPE_NUM + 3, seq_channel, padding_idx=20
        )
        self.single_pair_context_aa = nn.Embedding(
            RESTYPE_NUM + 3, seq_channel
        )
        self.single_pair_cdr = nn.Embedding(
            NUM_AB_REGIONS + 1, seq_channel
        )
        self.single_pair_coordinate = nn.Sequential(
            Linear(14 * 3 + 7 * 2, seq_channel, init='linear'),
            nn.ReLU(),
            Linear(seq_channel, seq_channel, init='linear'),
        )
        self.single_pair_antigen_proj = nn.Sequential(
            LayerNorm(seq_channel),
            Linear(seq_channel, seq_channel, init='linear'),
            nn.ReLU(),
            Linear(seq_channel, seq_channel, init='linear'),
        )
        self.single_pair_residue_mlp = nn.Sequential(
            Linear(seq_channel * 3 + 2, seq_channel * 2, init='linear'),
            nn.ReLU(),
            Linear(seq_channel * 2, seq_channel, init='linear'),
            nn.ReLU(),
            Linear(seq_channel, seq_channel, init='linear'),
            nn.ReLU(),
            Linear(seq_channel, seq_channel, init='linear'),
        )


    def get_atom_weights(self, residue_types):
        weights = torch.where(
            self.atom_weights_mask, self.zero_atom_weight, self.atom_weight
        )
        if not self.fix_atom_weights:
            weights = F.normalize(weights, dim=-1)
        return weights[residue_types]

    def encode_single_pair_base(self, seq, antigen_mask):
        raw = self.single_pair_base_aa(seq.long())
        antigen = self.single_pair_antigen_proj(raw).to(dtype=raw.dtype)
        return torch.where(antigen_mask[..., None], antigen, raw)

    def encode_single_pair_residue(
        self, batch, seq, atom14_positions, atom14_exists, angles_sin_cos
    ):
        """Build the contextual single residue view from one canonical residue state.

        Fixed/observed residues contribute local atom14 geometry.  Design residues
        keep sequence/topology context through the base single path, but their
        clean/native structural embedding is blocked.
        """
        geometry_mask = batch.get('geometry_condition_mask', batch['fixed_mask'])
        mask = batch['mask'].bool() & geometry_mask.bool()
        B, L = mask.shape

        aa = self.single_pair_context_aa(seq.long()) * mask[..., None]
        cdr = self.single_pair_cdr(batch['cdr_def'].long())

        local_atom14, frame_valid = atom14_to_residue_local(
            atom14_positions,
            atom14_exists,
            fixed_mask=mask,
            coordinate_scale=self.single_pair_coordinate_scale,
            eps=self.single_pair_frame_eps,
        )
        torsion = angles_sin_cos * frame_valid[..., None, None].to(
            angles_sin_cos.dtype
        )
        coord = self.single_pair_coordinate(torch.cat(
            [
                local_atom14.reshape(B, L, -1),
                torsion.reshape(B, L, -1),
            ],
            dim=-1,
        ))

        out = self.single_pair_residue_mlp(torch.cat(
            [
                aa,
                batch['chain_id'][..., None].to(aa.dtype),
                batch['residx'][..., None].to(aa.dtype),
                cdr,
                coord,
            ],
            dim=-1,
        ))
        return out * mask[..., None]


    def forward(
        self, X, S, batch_id, k_neighbors, residue_pos=None,
        smooth_prob=None, smooth_mask=None,
    ):
        if residue_pos is None:
            residue_pos = self._construct_residue_pos(S)
        atom_type = self.residue_atom_type[S]
        atom_pos = self.residue_atom_pos[S]

        pos_embedding = self.aa_embedding.res_pos_embedding(residue_pos)
        H = self.aa_embedding.residue_embedding(S)
        if smooth_prob is not None:
            res_embeddings = self.aa_embedding.residue_embedding(
                torch.arange(
                    smooth_prob.shape[-1], device=S.device, dtype=S.dtype
                )
            )
            with torch.cuda.amp.autocast(enabled=False):
                smooth_H = smooth_prob.float().mm(res_embeddings.float())
            H[smooth_mask] = smooth_H.to(dtype=H.dtype)
        H = H + pos_embedding

        atom_embedding = (
            self.aa_embedding.atom_embedding(atom_type)
            + self.aa_embedding.atom_pos_embedding(atom_pos)
        )
        atom_weights = self.get_atom_weights(S)

        ctx_edges, inter_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos
        )
        return H, (ctx_edges, inter_edges), (atom_embedding, atom_weights)


class ProteinFeature:
    def __init__(self, backbone_only=False):
        self.backbone_only = backbone_only

    def _cal_sidechain_bond_lengths(self, S, X, aa_feature: AminoAcidFeature):
        bonds, bonds_mask = aa_feature.get_sidechain_bonds(S)
        n = torch.nonzero(bonds_mask)[:, 0]  # [Nbonds]
        src, dst = bonds[bonds_mask].T
        src_X, dst_X = X[(n, src)], X[(n, dst)]  # [Nbonds, 3]
        bond_lengths = torch.norm(dst_X - src_X, dim=-1)
        return bond_lengths

    def _cal_sidechain_chis(self, S, X, aa_feature: AminoAcidFeature):
        chi_atoms, chi_mask = aa_feature.get_sidechain_chi_angles_atoms(S)
        n = torch.nonzero(chi_mask)[:, 0]  # [Nchis]
        a0, a1, a2, a3 = chi_atoms[chi_mask].T  # [Nchis]
        x0, x1, x2, x3 = X[(n, a0)], X[(n, a1)], X[(n, a2)], X[(n, a3)]  # [Nchis, 3]
        u_0, u_1, u_2 = (x1 - x0), (x2 - x1), (x3 - x2)  # [Nchis, 3]
        # normals of the two planes
        n_1 = F.normalize(torch.cross(u_0, u_1), dim=-1)  # [Nchis, 3]
        n_2 = F.normalize(torch.cross(u_1, u_2), dim=-1)  # [Nchis, 3]
        cosChi = (n_1 * n_2).sum(-1)  # [Nchis]
        eps = 1e-7
        cosChi = torch.clamp(cosChi, -1 + eps, 1 - eps)
        return cosChi

    def _cal_backbone_bond_lengths(self, X, seg_id):
        # loss of backbone (...N-CA-C(O)-N...) bond length
        # N-CA, CA-C, C=O
        bl1 = torch.norm(X[:, 1:4] - X[:, :3], dim=-1)  # [N, 3], (N-CA), (CA-C), (C=O)
        # C-N
        bl2 = torch.norm(X[1:, 0] - X[:-1, 2], dim=-1)  # [N-1]
        same_chain_mask = seg_id[1:] == seg_id[:-1]
        bl2 = bl2[same_chain_mask]
        bl = torch.cat([bl1.flatten(), bl2], dim=0)
        return bl

    def _cal_angles(self, X, seg_id):
        ori_X = X
        X = X[:, :3].reshape(-1, 3)  # [N * 3, 3], N, CA, C
        U = F.normalize(X[1:] - X[:-1], dim=-1)  # [N * 3 - 1, 3]

        # 1. dihedral angles
        u_2, u_1, u_0 = U[:-2], U[1:-1], U[2:]   # [N * 3 - 3, 3]
        # backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        # angle between normals
        eps = 1e-7
        cosD = (n_2 * n_1).sum(-1)  # [(N-1) * 3]
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        # D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        seg_id_atom = seg_id.repeat(1, 3).flatten()  # [N * 3]
        same_chain_mask = sequential_and(
            seg_id_atom[:-3] == seg_id_atom[1:-2],
            seg_id_atom[1:-2] == seg_id_atom[2:-1],
            seg_id_atom[2:-1] == seg_id_atom[3:]
        )  # [N * 3 - 3]
        # D = D[same_chain_mask]
        cosD = cosD[same_chain_mask]

        # 2. bond angles (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C=O), (CA-C, C-N_{n+1}), (O=C, C-Nn)
        u_0, u_1 = U[:-1], U[1:]  # [N*3 - 2, 3]
        cosA1 = ((-u_0) * u_1).sum(-1)  # [N*3 - 2], (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C-N_{n+1})
        same_chain_mask = sequential_and(
            seg_id_atom[:-2] == seg_id_atom[1:-1],
            seg_id_atom[1:-1] == seg_id_atom[2:]
        )
        cosA1 = cosA1[same_chain_mask]  # [N*3 - 2 * num_chain]
        u_co = F.normalize(ori_X[:, 3] - ori_X[:, 2], dim=-1)  # [N, 3], C=O
        u_cca = -U[1::3]  # [N, 3], C-CA
        u_cn = U[2::3] # [N-1, 3], C-N_{n+1}
        cosA2 = (u_co * u_cca).sum(-1)  # [N], (C=O, C-CA)
        cosA3 = (u_co[:-1] * u_cn).sum(-1)  # [N-1], (C=O, C-N_{n+1})
        same_chain_mask = (seg_id[:-1] == seg_id[1:]) # [N-1]
        cosA3 = cosA3[same_chain_mask]
        cosA = torch.cat([cosA1, cosA2, cosA3], dim=-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)

        return cosD, cosA

    def coord_loss(self, pred_X, true_X, batch_id, atom_mask, reference=None):
        """Alignment-invariant coordinate loss with a stable gradient path.

        Kabsch R/t are nuisance-frame variables used only to construct the aligned
        target. Backpropagating through SVD is ill-conditioned near repeated or
        close singular values. Compute the same optimal rigid alignment in FP32
        under no_grad, then differentiate SmoothL1 only through pred_X.
        """
        pred_X_f = pred_X.float()
        true_X_f = true_X.float().clone()
        pred_bb_f, true_bb_f = pred_X_f[:, :4], true_X_f[:, :4]
        bb_mask = atom_mask[:, :4]
        ops = []

        align_obj_f = pred_bb_f if reference is None else reference[:, :4].float()
        n_graph = int(torch.max(batch_id).detach().cpu().item()) + 1

        for i in range(n_graph):
            is_cur_graph = batch_id == i
            cur_bb_mask = bb_mask[is_cur_graph]
            if not bool(cur_bb_mask.any().detach().cpu().item()):
                eye = torch.eye(3, device=pred_X.device, dtype=torch.float32)
                zero = torch.zeros(3, device=pred_X.device, dtype=torch.float32)
                ops.append((eye, zero))
                continue

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    _, R, t = kabsch_torch(
                        true_bb_f[is_cur_graph][cur_bb_mask],
                        align_obj_f[is_cur_graph][cur_bb_mask],
                        requires_grad=False,
                    )
                    aligned_true = torch.matmul(
                        true_X_f[is_cur_graph], R.T
                    ) + t
            true_X_f[is_cur_graph] = aligned_true
            ops.append((R.detach(), t.detach()))

        atom_mask_sum = atom_mask.sum().clamp_min(1)
        xloss = F.smooth_l1_loss(
            pred_X_f[atom_mask], true_X_f[atom_mask], reduction='sum'
        ) / atom_mask_sum
        bb_sq = ((pred_X_f[:, :4] - true_X_f[:, :4]) ** 2).sum(-1).mean(-1)
        bb_rmsd = torch.sqrt(bb_sq.clamp_min(0.0))
        return xloss, bb_rmsd, ops

    def structure_loss(self, pred_X, true_X, S, cmask, batch_id, xloss_mask, aa_feature, full_profile=False, reference=None):
        atom_pos = aa_feature._construct_atom_pos(S)[cmask]
        seg_id = aa_feature._construct_segment_ids(S)[cmask]
        atom_mask = atom_pos != aa_feature.atom_pos_pad_idx
        atom_mask = torch.logical_and(atom_mask, xloss_mask[cmask])

        pred_X, true_X, batch_id = pred_X[cmask], true_X[cmask], batch_id[cmask]

        # Geometry losses stay in FP32 under BF16 autocast. The float() cast is
        # differentiable; only the Kabsch nuisance transform is detached.
        pred_X_f, true_X_f = pred_X.float(), true_X.float()

        # loss of absolute coordinates
        xloss, bb_rmsd, ops = self.coord_loss(
            pred_X_f, true_X_f, batch_id, atom_mask, reference
        )

        # loss of backbone (...N-CA-C(O)-N...) bond length
        true_bl = self._cal_backbone_bond_lengths(true_X_f, seg_id)
        pred_bl = self._cal_backbone_bond_lengths(pred_X_f, seg_id)
        bond_loss = F.smooth_l1_loss(pred_bl, true_bl)

        # loss of backbone dihedral angles
        if full_profile:
            true_cosD, true_cosA = self._cal_angles(true_X, seg_id)
            pred_cosD, pred_cosA = self._cal_angles(pred_X, seg_id)
            angle_loss = F.smooth_l1_loss(pred_cosD, true_cosD)
            bond_angle_loss = F.smooth_l1_loss(pred_cosA, true_cosA)

        S = S[cmask]
        if self.backbone_only:
            sc_bond_loss, sc_chi_loss = 0, 0
        else:
            # loss of sidechain bonds
            true_sc_bl = self._cal_sidechain_bond_lengths(S, true_X_f, aa_feature)
            pred_sc_bl = self._cal_sidechain_bond_lengths(S, pred_X_f, aa_feature)
            sc_bond_loss = F.smooth_l1_loss(pred_sc_bl, true_sc_bl)

            # loss of sidechain chis
            if full_profile:
                true_sc_chi = self._cal_sidechain_chis(S, true_X, aa_feature)
                pred_sc_chi = self._cal_sidechain_chis(S, pred_X, aa_feature)
                sc_chi_loss = F.smooth_l1_loss(pred_sc_chi, true_sc_chi)

        # exerting constraints on bond lengths only is sufficient
        violation_loss = bond_loss + sc_bond_loss
        loss = xloss + violation_loss

        if full_profile:
            details = (xloss, bond_loss, bond_angle_loss, angle_loss, sc_bond_loss, sc_chi_loss)
        else:
            details = (xloss, bond_loss, sc_bond_loss)

        return loss, details, bb_rmsd, ops


class SeperatedCoordNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.tensor(0)
        self.std = torch.tensor(10)
        self.mean = nn.parameter.Parameter(self.mean, requires_grad=False)
        self.std = nn.parameter.Parameter(self.std, requires_grad=False)
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)

    def normalize(self, X):
        X = (X - self.mean) / self.std
        return X

    def unnormalize(self, X):
        X = X * self.std + self.mean
        return X

    def centering(self, X, S, batch_id, aa_feature: AminoAcidFeature):
        # centering antigen and antibody separatedly
        segment_ids = aa_feature._construct_segment_ids(S)
        not_bol = S != aa_feature.bol_idx
        tmp_S = S[not_bol]
        tmp_X = aa_feature.update_global_coordinates(X[not_bol], tmp_S)
        self.ag_centers = tmp_X[tmp_S == aa_feature.boa_idx][:, 0]
        self.ab_centers = tmp_X[tmp_S == aa_feature.boh_idx][:, 0]

        is_ag = segment_ids == aa_feature.ag_seg_id
        is_ab = torch.logical_not(is_ag)

        # compose centers
        centers = torch.zeros(X.shape[0], X.shape[-1], dtype=X.dtype, device=X.device)
        centers[is_ag] = self.ag_centers[batch_id[is_ag]]
        centers[is_ab] = self.ab_centers[batch_id[is_ab]]
        X = X - centers.unsqueeze(1)
        self.is_ag, self.is_ab = is_ag, is_ab
        return X

    def uncentering(self, X, batch_id, _type=1):
        if _type == 0:
            # type 0: [N, 3]
            X = X.unsqueeze(1) # then it is type 1
        
        if _type == 0 or _type == 1:
            # type 1: [N, n_channel, 3]
            centers = torch.zeros(X.shape[0], X.shape[-1], dtype=X.dtype, device=X.device)
            centers[self.is_ag] = self.ag_centers[batch_id[self.is_ag]]
            centers[self.is_ab] = self.ab_centers[batch_id[self.is_ab]]
            X = X + centers.unsqueeze(1)
        elif _type == 2:
            # type 2: [2, bs, K, 3], X[0] for antigen, X[1] for antibody
            centers = torch.stack([self.ag_centers, self.ab_centers], dim=0)  # [2, bs, 3]
            X = X + centers.unsqueeze(-2)
        elif _type == 3:
            # type 3: [2, Ef, 3], X[0] for antigen, X[1] for antibody
            centers = torch.stack([self.ag_centers[batch_id], self.ab_centers[batch_id]], dim=0)
            X = X + centers
        elif _type == 4:
            # type 4: [N, n_channel, 3], but all uncentering to the center of antigen
            centers = self.ag_centers[batch_id]
            X = X + centers.unsqueeze(1)
        else:
            raise NotImplementedError(f'uncentering for type {_type} not implemented')

        if _type == 0:
            X = X.squeeze(1)
        return X

    def clear_cache(self):
        self.ag_centers, self.ab_centers, self.is_ag, self.is_ab = None, None, None, None

# ============================================================
# Dense single/pair representation and Seqformer operators
# ============================================================

class Linear_common(nn.Linear):
    def __init__(self, input_dim, output_dim, init, bias=True):
        super().__init__(input_dim, output_dim, bias=bias)
        assert init in ['gate', 'final', 'attn', 'relu', 'linear']
        if init in ['gate', 'final']:
            nn.init.constant_(self.weight, 0.)
        elif init == 'attn':
            torch.nn.init.xavier_uniform_(self.weight)
        elif init in ['relu', 'linear']:
            distribution_stddev = 0.87962566103423978
            scale = 2. if init == 'relu' else 1.
            stddev = np.sqrt(scale / input_dim) / distribution_stddev
            nn.init.trunc_normal_(self.weight, mean=0., std=stddev)
        else:
            raise NotImplementedError(f'{init} not Implemented')
        if bias:
            if init == 'gate':
                nn.init.constant_(self.bias, 1.)
            else:
                nn.init.constant_(self.bias, 0.)

def Linear(input_dim, output_dim, init, bias=True, config=None):
    assert init in ['gate', 'final', 'attn', 'relu', 'linear']
    return Linear_common(input_dim, output_dim, init, bias)

def apply_dropout(tensor, rate, is_training, broadcast_dim=None):
    if is_training and rate > 0.0:
        if broadcast_dim is not None:
            shape = list(tensor.shape)
            shape[broadcast_dim] = 1
            with torch.no_grad():
                scale = 1. / (1. - rate)
                keep_rate = torch.full(shape, 1. - rate, dtype=tensor.dtype, device=tensor.device)
                keep = torch.bernoulli(keep_rate)
            return scale * keep * tensor
        else:
            return F.dropout(tensor, rate)
    else:
        return tensor

def pseudo_beta_fn_v2(aatype, all_atom_positions, all_atom_masks=None):
    n_idx = ATOM14_ORDER['N']
    ca_idx = ATOM14_ORDER['CA']
    c_idx = ATOM14_ORDER['C']
    N = all_atom_positions[..., n_idx, :]
    CA = all_atom_positions[..., ca_idx, :]
    C = all_atom_positions[..., c_idx, :]
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    if all_atom_masks is not None:
        CB_mask = torch.all(torch.stack([
            all_atom_masks[...,n_idx], all_atom_masks[...,ca_idx], all_atom_masks[...,c_idx]
        ], dim=-1), dim=-1)
        return CB, CB_mask
    return CB

def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    breaks = torch.linspace(min_bin, max_bin, steps=num_bins-1, device=positions.device)
    sq_breaks = torch.square(breaks)
    dist2 = torch.sum(torch.square(
        rearrange(positions, 'b l c -> b l () c') -
        rearrange(positions, 'b l c -> b () l c')), dim=-1, keepdims=True)
    true_bins = torch.sum(dist2 > sq_breaks, axis=-1).long()
    return true_bins



# SEQFORMER_BATCH_RUNTIME
# SEQFORMER_WIDTH_CONTRACT
# - localized/donor width closure retained from V201
# - triangle chunk is supplied by the selected modular JSON
# - batch size remains a launcher/config concern; model code never owns batch size
# Fixes the V200 auxiliary-head width leak: every AbFlow consumer now derives its
# input width from the active trunk config (localized or donor), rather than
# silently retaining donor defaults. Scientific state / losses are unchanged.
# SEQFORMER_RUNTIME
# ATOM14_OBSERVED_MASK
# AbFlow's own l2_normalize uses sqrt(sum(square(v)) + epsilon).  Our online
# atom14->torsion adapter is differentiable w.r.t. the recurrent R05 state, so
# the epsilon must be inside sqrt; sqrt(r2).clamp_min(eps) still executes the
# singular SqrtBackward0 at r2==0 before clamp can protect the gradient.

def _dihedral_sin_cos(p0,p1,p2,p3,epsilon=1e-12,diag_tag=None):
    epsilon = float(epsilon)

    # Keep the geometric construction in FP32 even under BF16 autocast.  This
    # mirrors the fact that donor AbFlow torsions are precomputed numeric features
    # while preserving the gradient from current-state geometry in our adapter.
    with torch.cuda.amp.autocast(enabled=False):
        p0f,p1f,p2f,p3f = p0.float(),p1.float(),p2.float(),p3.float()
        b0 = p0f-p1f; b1=p2f-p1f; b2=p3f-p2f
        b1_sq = (b1*b1).sum(-1,keepdim=True)
        b1n = b1 / torch.sqrt(b1_sq + epsilon)
        v = b0 - (b0*b1n).sum(-1,keepdim=True)*b1n
        w = b2 - (b2*b1n).sum(-1,keepdim=True)*b1n
        x = (v*w).sum(-1)
        y = (torch.cross(b1n,v,dim=-1)*w).sum(-1)
        xy_sq = x*x+y*y
        n = torch.sqrt(xy_sq + epsilon)
        out = torch.stack([y/n,x/n],dim=-1)

    return out.to(dtype=p0.dtype) if p0.dtype in (torch.float16, torch.bfloat16) else out

def _atom14_chemical_mask(seq, coords):
    safe = seq.long().clamp(min=0, max=20)
    table = ATOM14_MASK.to(device=coords.device, dtype=coords.dtype)
    chem = table[safe]
    finite = torch.isfinite(coords).all(dim=-1).to(coords.dtype)
    return chem * finite

def _abflow_ca_fill_observed_mask(seq, coords, tol2=1e-12):
    """Infer AbFlow's resolved-atom mask from its CA-fill sentinel.

    AbFlow preprocessing initializes every atom slot to the residue CA and only
    overwrites slots that are actually resolved.  This fallback is used at
    inference by legacy generate.py paths that historically removed xloss_mask.
    Training/validation use the explicit xloss_mask and audit this fallback.
    """
    chem = _atom14_chemical_mask(seq, coords).bool()
    if coords.shape[-2] < 2:
        return chem.to(dtype=coords.dtype)
    ca = coords[..., 1:2, :]
    d2 = ((coords - ca) ** 2).sum(dim=-1)
    # A real non-CA atom cannot physically coincide with CA.  Use a tiny FP32
    # tolerance only to survive centering/scaling roundoff. CA itself is valid
    # whenever chemically present and finite.
    observed = d2 > float(tol2)
    observed[..., 1] = True
    return (chem & observed).to(dtype=coords.dtype)

def _atom14_exists_from_seq(seq, coords, observed_mask=None, tol2=1e-12):
    chem = _atom14_chemical_mask(seq, coords)
    if observed_mask is None:
        observed = _abflow_ca_fill_observed_mask(seq, coords, tol2=tol2)
    else:
        if tuple(observed_mask.shape) != tuple(coords.shape[:-1]):
            raise ValueError(
                f"atom observed-mask shape mismatch: {tuple(observed_mask.shape)} "
                f"vs coords {tuple(coords.shape)}"
            )
        observed = observed_mask.to(device=coords.device, dtype=coords.dtype)
    return chem * observed

# VECTOR_TORSION_TABLES: exact atom-index lookup, no Python/GPU sync loop.
_CHI_INDEX_ROWS = []
_CHI_VALID_ROWS = []
for _aa1 in RESTYPES:
    _res3 = RESTYPE_1TO3[_aa1]
    _idxmap = ATOM14_INDEX[_res3]
    _idx_row, _valid_row = [], []
    for _chi_i in range(4):
        if _chi_i < len(CHI_ANGLES_ATOMS[_res3]):
            _names = CHI_ANGLES_ATOMS[_res3][_chi_i]
            _ok = all(_n in _idxmap for _n in _names)
            _idx_row.append([_idxmap[_n] if _ok else 0 for _n in _names])
            _valid_row.append(_ok)
        else:
            _idx_row.append([0, 0, 0, 0])
            _valid_row.append(False)
    _CHI_INDEX_ROWS.append(_idx_row)
    _CHI_VALID_ROWS.append(_valid_row)
# unknown/MASK row
_CHI_INDEX_ROWS.append([[0, 0, 0, 0] for _ in range(4)])
_CHI_VALID_ROWS.append([False] * 4)
CHI_ATOM_INDEX_TABLE = torch.tensor(_CHI_INDEX_ROWS, dtype=torch.long)
CHI_VALID_TABLE = torch.tensor(_CHI_VALID_ROWS, dtype=torch.bool)


def _torsions_from_atom14(seq, coords, chain_id, mask, atom_exists=None, epsilon=1e-12):
    """Vectorized exact AbFlow 7-torsion construction for padded atom14 tensors.

    This is algebraically the same pre-omega/phi/psi/chi1..4 construction as
    V199.  It removes per-residue ``.item()/bool`` CUDA synchronizations only;
    invalid torsions retain the donor default [sin,cos]=[0,1].
    """
    B, L = seq.shape
    out = coords.new_zeros((B, L, 7, 2))
    out[..., 1] = 1.0
    exists = (
        _atom14_exists_from_seq(seq, coords).bool()
        if atom_exists is None
        else atom_exists.to(device=coords.device).bool()
    )
    valid_res = mask.bool()

    # Neighbor validity without wrap-around leakage.
    prev_mask = torch.roll(valid_res, shifts=1, dims=1)
    next_mask = torch.roll(valid_res, shifts=-1, dims=1)
    prev_chain = torch.roll(chain_id, shifts=1, dims=1)
    next_chain = torch.roll(chain_id, shifts=-1, dims=1)
    same_prev = valid_res & prev_mask & (chain_id == prev_chain)
    same_next = valid_res & next_mask & (chain_id == next_chain)
    if L:
        same_prev[:, 0] = False
        same_next[:, -1] = False

    prev_coords = torch.roll(coords, shifts=1, dims=1)
    next_coords = torch.roll(coords, shifts=-1, dims=1)
    prev_exists = torch.roll(exists, shifts=1, dims=1)
    next_exists = torch.roll(exists, shifts=-1, dims=1)

    pre_ok = same_prev & prev_exists[..., 1] & prev_exists[..., 2] & exists[..., 0] & exists[..., 1]
    phi_ok = same_prev & prev_exists[..., 2] & exists[..., 0] & exists[..., 1] & exists[..., 2]
    psi_ok = same_next & exists[..., 0] & exists[..., 1] & exists[..., 2] & next_exists[..., 0]

    pre = _dihedral_sin_cos(prev_coords[..., 1, :], prev_coords[..., 2, :], coords[..., 0, :], coords[..., 1, :], epsilon=epsilon, diag_tag="vector:pre_omega")
    phi = _dihedral_sin_cos(prev_coords[..., 2, :], coords[..., 0, :], coords[..., 1, :], coords[..., 2, :], epsilon=epsilon, diag_tag="vector:phi")
    psi = _dihedral_sin_cos(coords[..., 0, :], coords[..., 1, :], coords[..., 2, :], next_coords[..., 0, :], epsilon=epsilon, diag_tag="vector:psi")
    out[..., 0, :] = torch.where(pre_ok[..., None], pre, out[..., 0, :])
    out[..., 1, :] = torch.where(phi_ok[..., None], phi, out[..., 1, :])
    out[..., 2, :] = torch.where(psi_ok[..., None], psi, out[..., 2, :])

    # Four chi torsions from a residue-type lookup table.
    safe_seq = seq.long().clamp(min=0, max=20)
    chi_idx_table = CHI_ATOM_INDEX_TABLE.to(device=coords.device)
    chi_valid_table = CHI_VALID_TABLE.to(device=coords.device)
    chi_idx = chi_idx_table[safe_seq]                # [B,L,4,4]
    chi_type_ok = chi_valid_table[safe_seq] & valid_res[..., None]

    coords4 = coords[:, :, None, :, :].expand(B, L, 4, 14, 3)
    gather_idx = chi_idx[..., None].expand(B, L, 4, 4, 3)
    chi_pts = torch.gather(coords4, dim=3, index=gather_idx)  # [B,L,4,4,3]

    exists4 = exists[:, :, None, :].expand(B, L, 4, 14)
    chi_exists = torch.gather(exists4, dim=3, index=chi_idx)
    chi_ok = chi_type_ok & chi_exists.all(dim=-1)
    chi = _dihedral_sin_cos(
        chi_pts[..., 0, :], chi_pts[..., 1, :],
        chi_pts[..., 2, :], chi_pts[..., 3, :],
        epsilon=epsilon, diag_tag="vector:chi",
    )
    out[..., 3:7, :] = torch.where(
        chi_ok[..., None], chi, out[..., 3:7, :]
    )
    return out

class _Cfg(dict):
    """OmegaConf-like minimal attribute/dict compatibility used only by the local donor port."""
    def __getattr__(self, key):
        try: return self[key]
        except KeyError as exc: raise AttributeError(key) from exc
    def __setattr__(self, key, value): self[key] = value
    @classmethod
    def from_dict(cls,d):
        return cls({k:(cls.from_dict(v) if isinstance(v,dict) else v) for k,v in d.items()})

def seqformer_config(config, recycle_features=False, recycle_pos=False):
    """Build the single/pair trunk config from the selected experiment JSON."""
    channels = config['channels']
    attention = config['attention']
    execution = config.get('execution', {})
    dropout = config.get('dropout', {})

    seq_channel = int(channels['single'])
    pair_channel = int(channels['pair'])
    index_embed = int(channels['time_index'])
    seq_heads = int(attention['single_heads'])
    tri_heads = int(attention['triangle_heads'])
    outer_channel = int(attention['opm_channel'])
    tri_hidden = int(attention['triangle_hidden'])
    chunk = int(execution.get('triangle_chunk_size', 64))
    eval_chunk = int(execution.get('triangle_chunk_size_eval', chunk))

    return _Cfg.from_dict({
        'seqformer_num_block': int(config.get('blocks', 1)),
        'seq_channel': seq_channel,
        'pair_channel': pair_channel,
        'max_relative_feature': int(config.get('max_relative_feature', 32)),
        'index_embed_size': index_embed,
        'recycle_features': bool(recycle_features),
        'recycle_pos': bool(recycle_pos),
        'activation_checkpoint': bool(
            execution.get('activation_checkpoint', True)
        ),
        'pair_distance_chunk_size': int(
            execution.get('pair_distance_chunk_size', 8)
        ),
        'pair_distance_chunk_size_eval': int(
            execution.get(
                'pair_distance_chunk_size_eval',
                execution.get('pair_distance_chunk_size', 8),
            )
        ),
        'time_embed': bool(config.get('time_embed', True)),
        'prev_pos': {
            'min_bin': 3.375, 'num_bins': 15, 'max_bin': 21.375
        },
        'seqformer': {
            'seq_attention_with_pair_bias': {
                'orientation': 'per_row', 'num_head': seq_heads,
                'inp_kernels': [],
                'dropout_rate': float(dropout.get('single_attention', 0.1)),
                'shared_dropout': True,
            },
            'seq_transition': {
                'orientation': 'per_row', 'num_intermediate_factor': 4,
                'dropout_rate': 0.0, 'shared_dropout': True,
            },
            'outer_product_mean': {
                'orientation': 'per_row',
                'num_outer_channel': outer_channel,
                'dropout_rate': 0.0, 'shared_dropout': True,
            },
            'triangle_multiplication_outgoing': {
                'orientation': 'per_row',
                'num_intermediate_channel': tri_hidden,
                'gating': True, 'num_head': tri_heads,
                'inp_kernels': [],
                'dropout_rate': float(dropout.get('triangle', 0.1)),
                'shared_dropout': False,
            },
            'triangle_multiplication_incoming': {
                'orientation': 'per_column',
                'num_intermediate_channel': tri_hidden,
                'gating': True, 'num_head': tri_heads,
                'inp_kernels': [],
                'dropout_rate': float(dropout.get('triangle', 0.1)),
                'shared_dropout': False,
            },
            'triangle_attention_starting_node': {
                'orientation': 'per_row', 'num_head': tri_heads,
                'gating': True, 'inp_kernels': [],
                'dropout_rate': float(dropout.get('triangle', 0.1)),
                'shared_dropout': False,
                'chunk_size': chunk, 'eval_chunk_size': eval_chunk,
            },
            'triangle_attention_ending_node': {
                'orientation': 'per_column', 'num_head': tri_heads,
                'gating': True, 'inp_kernels': [],
                'dropout_rate': float(dropout.get('triangle', 0.1)),
                'shared_dropout': False,
                'chunk_size': chunk, 'eval_chunk_size': eval_chunk,
            },
            'pair_transition': {
                'orientation': 'per_row', 'num_intermediate_factor': 4,
                'dropout_rate': 0.0, 'shared_dropout': True,
            },
        },
    })


# Single residue authority is SeparatedAminoAcidFeature above.

class PairEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        feat_dim = config.pair_channel
        self.feat_dim = int(feat_dim)
        self.pair_distance_chunk_size = max(
            1, int(getattr(config, 'pair_distance_chunk_size', 8))
        )
        self.pair_distance_chunk_size_eval = max(
            self.pair_distance_chunk_size,
            int(getattr(
                config, 'pair_distance_chunk_size_eval',
                self.pair_distance_chunk_size,
            )),
        )
        self.dgram_config = config.prev_pos
        self.num_bins = self.dgram_config.num_bins
        self.max_num_atoms = 14
        self.max_aa_types = RESTYPE_NUM + 3
        self.max_relpos = 32
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*self.max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, self.max_num_atoms*self.max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            Linear(self.max_num_atoms*self.max_num_atoms, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
        )

        self.dgram_embed = nn.Embedding(self.num_bins, feat_dim)

        infeat_dim = feat_dim * 4
        self.out_mlp = nn.Sequential(
            Linear(infeat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
            nn.ReLU(),
            Linear(feat_dim, feat_dim, init='linear', bias=True),
        )

    def _project_pair_components(self, feat_aapair, feat_relpos, feat_dist, feat_dgram):
        """Apply the donor concat-MLP without materializing a 4C tensor.

        For the first linear layer,
            W[a;r;d;g] + b
        is exactly
            W_a a + W_r r + W_d d + W_g g + b.
        The parameter matrix is unchanged; only the execution order is factorized
        to reduce peak memory.
        """
        first = self.out_mlp[0]
        C = self.feat_dim
        W = first.weight
        b = first.bias
        out = F.linear(feat_aapair, W[:, 0:C], b)
        out = out + F.linear(feat_relpos, W[:, C:2*C], None)
        out = out + F.linear(feat_dist, W[:, 2*C:3*C], None)
        out = out + F.linear(feat_dgram, W[:, 3*C:4*C], None)
        for layer in self.out_mlp[1:]:
            out = layer(out)
        return out

    def forward(self, batch, seq, atom14_positions, atom14_gt_exists):
        """Build dense relational pair states with chunked exact geometry kernels.

        The scientific feature definition is unchanged from the AbX-style donor:
        amino-acid pair, IMGT relative position, 14x14 atom distances, and
        pseudo-beta distogram.  Only the row dimension is chunked so the large
        [B,L,L,14,14,3] / [B,14L,14L] temporaries are never materialized.
        """
        geometry_mask = batch.get('geometry_condition_mask', batch['fixed_mask'])
        fixed = batch['mask'].bool() & geometry_mask.bool()
        B, L = fixed.shape
        aa = seq.long()
        chain_ids = batch['chain_id']
        residx = batch['residx']
        coords = atom14_positions
        ca_exists = atom14_gt_exists[..., ATOM14_ORDER['CA']].to(coords.dtype)

        pseudo_beta = pseudo_beta_fn_v2(aa, coords)
        disto_bins = dgram_from_positions(pseudo_beta, **self.dgram_config)

        pair_chunks = []
        # Chunking changes only execution order: every row/pair is still evaluated
        # by the same exact equations.  Training and eval can therefore use
        # different memory/speed points without changing the scientific model.
        chunk = (
            self.pair_distance_chunk_size
            if self.training
            else self.pair_distance_chunk_size_eval
        )

        # Full right-side atom bank is reused by every row chunk.
        rhs_atoms = coords.reshape(B, L * self.max_num_atoms, 3).float()

        for start in range(0, L, chunk):
            stop = min(L, start + chunk)
            K = stop - start

            aa_i = aa[:, start:stop]
            aa_pair = (
                aa_i[:, :, None] * self.max_aa_types
                + aa[:, None, :]
            )
            feat_aapair = self.aa_pair_embed(aa_pair)

            same_chain = (
                chain_ids[:, start:stop, None]
                == chain_ids[:, None, :]
            )
            relpos = torch.clamp(
                residx[:, start:stop, None] - residx[:, None, :],
                min=-self.max_relpos,
                max=self.max_relpos,
            )
            feat_relpos = self.relpos_embed(
                (relpos + self.max_relpos).long()
            )
            feat_relpos = feat_relpos * same_chain[..., None].to(
                feat_relpos.dtype
            )

            lhs_atoms = coords[:, start:stop].reshape(
                B, K * self.max_num_atoms, 3
            ).float()
            distance = torch.cdist(
                lhs_atoms,
                rhs_atoms,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            distance = distance.reshape(
                B, K, self.max_num_atoms, L, self.max_num_atoms
            ).permute(0, 1, 3, 2, 4)
            distance = (distance / 10.0).reshape(
                B, K, L, self.max_num_atoms * self.max_num_atoms
            ).to(coords.dtype)

            distance_coef = F.softplus(
                self.aapair_to_distcoef(aa_pair)
            )
            d_gauss = torch.exp(-distance_coef * distance.square())

            # Preserve the established AbX-style CA-resolved pair support.
            ca_pair = (
                ca_exists[:, start:stop, None, None]
                * ca_exists[:, None, :, None]
            )
            feat_dist = self.distance_embed(d_gauss * ca_pair)

            feat_dgram = self.dgram_embed(
                disto_bins[:, start:stop]
            )

            feat = self._project_pair_components(
                feat_aapair,
                feat_relpos,
                feat_dist,
                feat_dgram,
            )
            pair_mask = (
                fixed[:, start:stop, None]
                & fixed[:, None, :]
            )
            pair_chunks.append(
                feat * pair_mask[..., None].to(feat.dtype)
            )

        return torch.cat(pair_chunks, dim=1)


# ===== AbFlow donor timestep embedding =====
def pair_concat(pair_1, pair_2):
    assert pair_1.shape[0] == pair_2.shape[0] and pair_1.shape[-1] == pair_2.shape[-1]
    assert pair_1.device == pair_2.device
    device = pair_1.device
    batch_size = pair_1.shape[0]
    channel = pair_1.shape[-1]

    length_1 = pair_1.shape[1]
    length_2 = pair_2.shape[1]
    concat_dim1 = torch.cat(
        (
        pair_1, 
        torch.zeros((batch_size, length_2, length_1, channel), device=device)
        ), dim=1)
    
    concat_dim2 = torch.cat(
        (
        torch.zeros((batch_size, length_1, length_2, channel), device=device), 
        pair_2
        ), dim=1)
    pair_all = torch.cat([concat_dim1, concat_dim2], dim=2)
    return pair_all



def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    """
    From Fairseq.Build sinusoidal embeddings.This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # Zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):
    """
    A module for encoding diffusion timesteps and embedding them into sequence and pair representations.
    
    Returns:
        node_embed: [B, N,  seq_channel + index_embed_size]
        edge_embed: [B, N, N, pair_channel + 2*index_embed_size]
    """
    def __init__(self, model_conf):
        super(Embedder,self).__init__()
        self._embed_conf = model_conf
        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        
        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim = index_embed_size
        )


    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])


    def forward(self, seq_act, pair_act, batch):
        num_batch, num_res = batch['seq'].shape
        # node_feats = []
        t = batch['t'] 
        fixed_mask = batch['fixed_mask']

        # Embedding times t
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1,num_res,1)
        ) # [B, N, index_embed_size]
        #import ipdb; ipdb.set_trace()
        # Evoformer embedding
        pair_act = pair_act.reshape([num_batch, num_res**2, -1])

        seq_feats = [seq_act]
        pair_feats = [pair_act]
        seq_feats.append(prot_t_embed)
        pair_feats.append(self._cross_concat(prot_t_embed, num_batch, num_res))

        
        pair_feats = torch.cat(pair_feats, dim=-1).float()
        seq_feats = torch.cat(seq_feats, dim=-1).float()
        pair_feats = pair_feats.reshape([num_batch, num_res, num_res, -1])

        return seq_feats, pair_feats
    



# ===== AbFlow donor Seqformer operators: source-faithful local port =====
class Attention(nn.Module):
    def __init__(self, input_dim, key_dim, value_dim, output_dim, num_head,split_first=True, gating=True,inp_kernels=None, config=None):
        super().__init__()
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0

        self.key_dim, self.value_dim = key_dim, value_dim

        self.num_head = num_head
        
        self.split_first = split_first

        if self.split_first:
            self.proj_q = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_k = Linear(input_dim, key_dim, init='attn', bias=False, config=config)
            self.proj_v = Linear(input_dim, value_dim, init='attn', bias=False, config=config)
        else:
            assert (key_dim == value_dim)
            self.proj_in = Linear(input_dim, key_dim * 3, init='attn', bias=False, config=config)
        
        self.gating = gating
        if gating:
            self.gate= Linear(input_dim, value_dim, init='gate', config=config)

        self.proj_out = Linear(value_dim, output_dim, init='final', config=config)
         
        self.inp_kernels = inp_kernels
        if inp_kernels:
            self.inp_q = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_k = SpatialDepthWiseInception(key_dim // num_head, inp_kernels)
            self.inp_v = SpatialDepthWiseInception(value_dim // num_head, inp_kernels)

    def forward(self, q_data, k_data=None, bias=None, k_mask=None):
        """
        Arguments:
            q_data: (batch_size, N_seqs, N_queries, q_channel)
            k_data: (batch_size, N_seqs, N_keys, k_channel)
            k_mask: (batch_size, N_seqs, N_keys)
            bias  : (batch_size, N_queries, N_keys). shared by all seqs
        Returns:
            (b s l c)
        """
        key_dim, value_dim = self.key_dim // self.num_head, self.value_dim // self.num_head
        
        if self.split_first:
            assert (k_data is not None)
            q = self.proj_q(q_data) 
            k = self.proj_k(k_data)
            v = self.proj_v(k_data)
            q, k, v = map(lambda t: rearrange(t, 'b s l (h d) -> b s h l d', h = self.num_head), (q, k, v))
        else:
            assert (k_data is None)
            t = rearrange(self.proj_in(q_data), "... l (h d) -> ... h l d", h=self.num_head)
            q, k, v = torch.chunk(t, 3, dim=-1)
        
        if self.inp_kernels:
            q, k, v = map(lambda t: rearrange(t, 'b s h l d-> b (s h) l d'), (q, k, v))
            q = self.inp_q(q)
            k = self.inp_k(k)
            v = self.inp_v(v)
            q, k, v = map(lambda t: rearrange(t, 'b (s h) l d-> b s h l d', h = self.num_head), (q, k, v))
        
        q = q* key_dim**(-0.5)

        logits = torch.einsum('... h q d, ... h k d -> ... h q k', q, k)

        if bias is not None:
            logits = logits + rearrange(bias,  'b h q k -> b () h q k')

        if k_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            k_mask = rearrange(k_mask, 'b s k -> b s () () k')
            logits = logits.masked_fill(~k_mask.bool(), mask_value)

        weights = F.softmax(logits, dim = -1)
        weighted_avg = torch.einsum('b s h q k, b s h k d -> b s h q d', weights, v)
        weighted_avg = rearrange(weighted_avg, 'b s h q d -> b s q (h d)')
        
        if self.gating:
            gate_values = torch.sigmoid(self.gate(q_data))
            weighted_avg = weighted_avg * gate_values

        output = self.proj_out(weighted_avg)

        return output

class SeqAttentionWithPairBias(nn.Module):
    def __init__(self, config, num_in_seq_channel, num_in_pair_channel):
        super().__init__()
        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.seq_norm = LayerNorm(num_in_seq_channel)
        self.pair_norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)

        self.attn = Attention(
                input_dim=num_in_seq_channel,
                key_dim=num_in_seq_channel,
                value_dim=num_in_seq_channel,
                output_dim=num_in_seq_channel,
                num_head=c.num_head,
                split_first=False,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)

        self.config = config

    def forward(self, seq_act, pair_act, mask):
        """
        Arguments:
            seq_act: (b l c)
            pair_act: (b l l c)
            mask: (b l), padding mask
        Returns:
            (b l c)
        """
        mask = rearrange(mask, 'b l -> b () l')
        seq_act = self.seq_norm(seq_act)
        
        pair_act = self.pair_norm(pair_act)
        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')
        
        seq_act = rearrange(seq_act, 'b l c -> b () l c')
        seq_act = self.attn(q_data=seq_act, bias=bias, k_mask=mask)
        seq_act = rearrange(seq_act, 'b s l c -> (b s) l c')
        return seq_act

class Transition(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        intermediate_channel = num_in_channel * c.num_intermediate_factor
        self.transition = nn.Sequential(
                LayerNorm(num_in_channel),
                Linear(num_in_channel, intermediate_channel, init='linear', config=LoRA_conf),
                nn.ReLU(),
                Linear(intermediate_channel, num_in_channel, init='final', config=LoRA_conf),
                )

    def forward(self, act, mask):
        return self.transition(act)

# AF2 and ESM-FOLD have different implementations
# Here we just follow ESMFOLD
class OuterProductMean(nn.Module):
    def __init__(self, config, num_in_channel, num_out_channel):
        super().__init__()

        c = config
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)
        self.left_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_outer_channel, init='linear', config=LoRA_conf)

        self.out_proj = Linear(2 * c.num_outer_channel, num_out_channel, init='final', config=LoRA_conf)

    def forward(self, act, mask):
        """
        act: (b l c)
        mask: (b l)
        """
        mask = rearrange(mask, 'b l -> b l ()')
        act = self.norm(act)
        left_act = mask * self.left_proj(act)
        right_act = mask * self.right_proj(act)
        
        prod = left_act[:, None, :, :] * right_act[:, :, None, :]
        diff = left_act[:, None, :, :] - right_act[:, :, None, :]

        act = torch.cat([prod, diff], dim=-1)
        act = self.out_proj(act)

        return act

class TriangleMultiplication(nn.Module):
    def __init__(self, config, num_in_channel):
        super().__init__()
        c = config
        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None
        self.norm = LayerNorm(num_in_channel)

        self.left_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)
        self.right_proj = Linear(num_in_channel, c.num_intermediate_channel, init='linear', config=LoRA_conf)

        self.final_norm = LayerNorm(c.num_intermediate_channel)
        
        if c.gating:
            self.left_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.right_gate = Linear(num_in_channel, c.num_intermediate_channel, init='gate', config=LoRA_conf)
            self.final_gate = Linear(num_in_channel, num_in_channel, init='gate', config=LoRA_conf)
        
        self.proj_out = Linear(c.num_intermediate_channel, num_in_channel, init='final', config=LoRA_conf)

        
        if c.inp_kernels:
            self.inp_left = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)
            self.inp_right = SpatialDepthWiseInception(c.num_intermediate_channel // c.num_head, c.inp_kernels)

        self.config = c

    def forward(self, act, mask):
        """
        act: (b l l c)
        mask: (b l)
        """
        c = self.config

        #pair_mask = rearrange(mask, 'b l -> b l () ()') * rearrange(mask, 'b l -> b () l ()')
        pair_mask = mask[:,:,None,None] * mask[:,None,:,None]
        
        act = self.norm(act)

        input_act = act

        left_proj_act = self.left_proj(act)
        right_proj_act = self.right_proj(act)
        
        if c.inp_kernels:
            if c.orientation == 'per_row':
                equation = 'b i j (h d) -> b (i h) j d'
            else:
                equation = 'b i j (h d) -> b (j h) i d'

            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))

            left_proj_act = self.inp_left(left_proj_act)
            right_proj_act = self.inp_right(right_proj_act)
            
            if c.orientation == 'per_row':
                equation = 'b (i h) j d -> b i j (h d)'
            else:
                equation = 'b (j h) i d -> b i j (h d)'
            
            left_proj_act, right_proj_act = map(
                    lambda t: rearrange(t, equation, h = c.num_head), (left_proj_act, right_proj_act))
        
        left_proj_act = pair_mask * left_proj_act
        right_proj_act = pair_mask * right_proj_act
        
        if c.gating:
            left_gate_values = torch.sigmoid(self.left_gate(act))
            right_gate_values = torch.sigmoid(self.right_gate(act))

            left_proj_act = left_proj_act * left_gate_values
            right_proj_act = right_proj_act * right_gate_values

        if c.orientation == 'per_row':
            act = torch.einsum('b i k c, b j k c -> b i j c', left_proj_act, right_proj_act)
        elif c.orientation == 'per_column':
            act = torch.einsum('b k i c, b k j c -> b i j c', left_proj_act, right_proj_act)
        else:
            raise NotImplementedError(f'{self.orientation} not Implemented')

        act = self.final_norm(act)
        act = self.proj_out(act)
        
        if c.gating:
            gate_values = torch.sigmoid(self.final_gate(input_act))
            act = act * gate_values

        return act

class TriangleAttention(nn.Module):
    def __init__(self, config, num_in_pair_channel):
        super().__init__()
        c = config

        assert c.orientation in ['per_row', 'per_column']
        try:
            LoRA_conf = c.LoRA
        except:
            LoRA_conf = None

        self.norm = LayerNorm(num_in_pair_channel)
        self.proj_pair = Linear(num_in_pair_channel, c.num_head, init='linear', bias = False, config=LoRA_conf)
        self.attn = Attention(
                input_dim=num_in_pair_channel,
                key_dim=num_in_pair_channel,
                value_dim=num_in_pair_channel,
                output_dim=num_in_pair_channel,
                num_head=c.num_head,
                gating=c.gating,
                inp_kernels=c.inp_kernels,
                config=LoRA_conf)

        self.config = config

    def forward(self, pair_act, seq_mask):
        '''
        pair_act: (b l l c)
        seq_mask: (b l)
        '''
        c = self.config
        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        pair_act = self.norm(pair_act)
        seq_mask = rearrange(seq_mask, 'b l -> b () l')

        # V200_EXACT_TRIANGLE_CHUNKING
        # Triangle attention is independent along the outer ``s`` axis.  Chunk
        # only that axis while keeping the full q/k bias matrix. Concatenating
        # the chunks is algebraically identical to the dense donor call, but the
        # cubic logits/softmax temporary becomes [B,chunk,H,L,L] instead of
        # [B,L,H,L,L].  No approximation, sparsification or attention deletion.
        bias = rearrange(self.proj_pair(pair_act), 'b i j h -> b h i j')
        chunk = max(1, int(
            c.chunk_size
            if self.training
            else getattr(c, 'eval_chunk_size', c.chunk_size)
        ))
        if int(pair_act.shape[1]) > chunk:
            outs = []
            for start in range(0, int(pair_act.shape[1]), chunk):
                stop = min(start + chunk, int(pair_act.shape[1]))
                pc = pair_act[:, start:stop]
                outs.append(self.attn(
                    q_data=pc, k_data=pc, bias=bias, k_mask=seq_mask
                ))
            pair_act = torch.cat(outs, dim=1)
        else:
            pair_act = self.attn(
                q_data=pair_act, k_data=pair_act, bias=bias, k_mask=seq_mask
            )

        if c.orientation == 'per_column':
            pair_act = rearrange(pair_act, 'b i j c -> b j i c')

        return pair_act

class SeqformerIteration(nn.Module):
    def __init__(self, config, seq_channel, pair_channel):
        super().__init__()
        c = config

        self.seq_attn = SeqAttentionWithPairBias(c.seq_attention_with_pair_bias, seq_channel, pair_channel)
        self.seq_transition = Transition(c.seq_transition, seq_channel)
        self.outer_product_mean = OuterProductMean(c.outer_product_mean, seq_channel, pair_channel)
        
        self.triangle_multiplication_outgoing = TriangleMultiplication(c.triangle_multiplication_outgoing, pair_channel)
        self.triangle_multiplication_incoming = TriangleMultiplication(c.triangle_multiplication_incoming, pair_channel)
        self.triangle_attention_starting_node = TriangleAttention(c.triangle_attention_starting_node, pair_channel)
        self.triangle_attention_ending_node = TriangleAttention(c.triangle_attention_ending_node, pair_channel)
        self.pair_transition = Transition(c.pair_transition, pair_channel)

        self.config = config

    def forward(self, seq_act, pair_act, seq_mask):
        """
        seq_act: (b l c)
        pair_act: (b l l c)
        seq_mask: (b l)
        """
        c = self.config

        def dropout_fn(input_act, act, config):
            if self.training and config.dropout_rate > 0.:
                if config.shared_dropout:
                    if config.orientation == 'per_row':
                        broadcast_dim = 1
                    else:
                        broadcast_dim = 2
                else:
                    broadcast_dim = None
                act = apply_dropout(act, config.dropout_rate,
                        is_training=True, broadcast_dim=broadcast_dim)
            return input_act + act
        
        seq_act = dropout_fn(
                seq_act, self.seq_attn(seq_act, pair_act, seq_mask), c.seq_attention_with_pair_bias)
        seq_act = seq_act + self.seq_transition(seq_act, seq_mask)
        
        pair_act = pair_act + self.outer_product_mean(seq_act, seq_mask)
        
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_outgoing(pair_act, seq_mask), c.triangle_multiplication_outgoing)
        pair_act = dropout_fn(
                pair_act, self.triangle_multiplication_incoming(pair_act, seq_mask), c.triangle_multiplication_incoming)

        pair_act = dropout_fn(
                pair_act, self.triangle_attention_starting_node(pair_act, seq_mask), c.triangle_attention_starting_node)

        pair_act = dropout_fn(
                pair_act, self.triangle_attention_ending_node(pair_act, seq_mask), c.triangle_attention_ending_node)
        pair_act = pair_act + self.pair_transition(pair_act, seq_mask)
        
        return seq_act, pair_act

class Seqformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config

        self.activation_checkpoint = bool(c.activation_checkpoint)
        self.blocks = nn.ModuleList([
            SeqformerIteration(
                c.seqformer,
                c.seq_channel + c.index_embed_size,
                c.pair_channel + 2 * c.index_embed_size,
            )
            for _ in range(c.seqformer_num_block)
        ])

    def forward(self, seq_act, pair_act, mask, is_recycling=True):
        checkpoint_enabled = bool(
            self.training and not is_recycling and self.activation_checkpoint
        )
        for block in self.blocks:
            block_fn = fn.partial(block, seq_mask=mask)
            if checkpoint_enabled:
                # AbFlow v4_l3 has one block, so the donor's historical it>0 gate
                # never checkpointed anything.  Since V200 the persistent AbFlow
                # trunk runs once per outer forward (outside the three R05 physical
                # rounds).  Checkpointing this one block preserves equations/RNG/
                # gradients while dropping its retained intra-block activations.
                seq_act, pair_act = checkpoint(
                    block_fn, seq_act, pair_act, preserve_rng_state=True
                )
            else:
                seq_act, pair_act = block_fn(seq_act, pair_act)
        return seq_act, pair_act

class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim,
                kernel_size=(kernel_size,),
                # padding=(kernel_size - 1,),
                padding=kernel_size//2,
                groups=head_dim)
    
    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        #if self.kernel_size>1:
        #    x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x

class SpatialDepthWiseInception(nn.Module):
    def __init__(self, head_dim, kernels):
        super().__init__()
       
        assert len(kernels) > 1 and  kernels[0] == 1

        self.convs = torch.nn.ModuleList([SpatialDepthWiseConvolution(head_dim, kernel_size=k) for k in kernels[1:]])
        self.kernels = kernels
    def forward(self, x):
        # x: (batch, num_heads, len, head_dim)
        
        assert x.shape[1] % len(self.kernels) == 0
        group_num_head = x.shape[1] // len(self.kernels)
        
        outputs = [x[:,:group_num_head]]

        for i, conv in enumerate(self.convs):
            outputs.append(conv(x[:,group_num_head*(i+1):group_num_head*(i+2)]))

        outputs = torch.cat(outputs, dim=1)

        return outputs

# ===== AbFlow donor boundary-local classes =====

class SinglePairEncoder(nn.Module):
    """Persistent dense single/pair representation used once per outer R05 step."""

    def __init__(self, representation_config):
        super().__init__()
        c = seqformer_config(representation_config, False, False)
        self.config = c
        self.use_time_embedding = bool(c.time_embed)

        self.proj_rel_pos = nn.Embedding(
            c.max_relative_feature * 2 + 2, c.pair_channel
        )
        self.pair_embedding = PairEmbedding(c)
        self.seqformer = Seqformer(c)
        self.time_embedder = Embedder(c)

    def forward(self, batch, residue_feature):
        c = self.config
        seq = batch['seq_t']
        mask = batch['mask'].bool()
        seq_pos = batch['residx']
        antibody_len = batch['antibody_len']
        B, L = seq.shape

        pos_index = torch.arange(L, device=seq.device)[None, :]
        ab_mask = (pos_index < antibody_len[:, None]) & mask
        ag_mask = (~ab_mask) & mask

        seq_act = residue_feature.encode_single_pair_base(seq, ag_mask)

        offset = seq_pos[:, None, :] - seq_pos[:, :, None]
        rel = torch.clip(
            offset + c.max_relative_feature,
            min=0, max=2 * c.max_relative_feature,
        ) + 1
        same_group = (
            (ab_mask[:, :, None] & ab_mask[:, None, :])
            | (ag_mask[:, :, None] & ag_mask[:, None, :])
        )
        pair_act = self.proj_rel_pos(rel.long())
        pair_act = pair_act * same_group[..., None].to(pair_act.dtype)

        seq_act = seq_act + residue_feature.encode_single_pair_residue(
            batch,
            batch['seq_t'],
            batch['atom14_gt_positions'],
            batch['atom14_gt_exists'],
            batch['torsion_angles_sin_cos'],
        )
        pair_act = pair_act + self.pair_embedding(
            batch,
            batch['seq_t'],
            batch['atom14_gt_positions'],
            batch['atom14_gt_exists'],
        )


        if self.use_time_embedding:
            seq_act, pair_act = self.time_embedder(seq_act, pair_act, batch)
        else:
            iz = int(c.index_embed_size)
            seq_act = torch.cat(
                [seq_act, seq_act.new_zeros((B, L, iz))], dim=-1
            )
            pair_act = torch.cat(
                [pair_act, pair_act.new_zeros((B, L, L, 2 * iz))],
                dim=-1,
            )

        seq_act, pair_act = self.seqformer(
            seq_act, pair_act, mask=mask, is_recycling=False
        )
        return seq_act, pair_act


class DistogramHead(nn.Module):
    """AbX donor distogram head adapted only to the active compact pair width.

    The donor equation is unchanged: a zero-initialized linear projection is
    symmetrized across (i,j)/(j,i).  ``pair_dim`` is supplied by the selected
    compact R28/R29/R30 representation rather than hard-coded donor widths.
    """
    def __init__(self, pair_dim, num_bins=64, first_break=2.3125, last_break=21.6875):
        super().__init__()
        self.input_dim = int(pair_dim)
        self.num_bins = int(num_bins)
        if self.input_dim <= 0:
            raise ValueError(f"distogram pair_dim must be positive, got {self.input_dim}")
        self.register_buffer(
            'breaks',
            torch.linspace(first_break, last_break, steps=self.num_bins - 1),
            persistent=False,
        )
        self.proj = Linear(self.input_dim, self.num_bins, init='final')

    def forward(self, pair):
        if int(pair.shape[-1]) != self.input_dim:
            raise RuntimeError(
                "AbFlow distogram width contract violated: "
                f"z.shape[-1]={int(pair.shape[-1])}, head.input_dim={self.input_dim}. "
                "The auxiliary head must use the same active AbFlow width profile as the trunk."
            )
        x = self.proj(pair)
        return (x + rearrange(x, 'b i j c -> b j i c')) * 0.5




__all__ = [
    "AminoAcidFeature", "SeparatedAminoAcidFeature", "ProteinFeature",
    "EdgeConstructor", "GMEdgeConstructor", "SeperatedCoordNormalizer",
    "SinglePairEncoder", "PairEmbedding", "Seqformer", "SeqformerIteration",
    "OuterProductMean", "DistogramHead", "get_timestep_embedding",
    "pseudo_beta_fn_v2", "_abflow_ca_fill_observed_mask",
    "_atom14_chemical_mask", "_atom14_exists_from_seq",
    "_torsions_from_atom14", "seqformer_config", "_knn_edges",
]
