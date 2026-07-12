#!/usr/bin/python
# -*- coding:utf-8 -*-
import math, time, os
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



def _env_str(name, default):
    value = os.environ.get(name, None)
    if value is None or value == "":
        return default
    return value


def _env_float(name, default):
    value = os.environ.get(name, None)
    if value is None or value == "":
        return default
    return float(value)


def _env_flag(name, default=False):
    value = os.environ.get(name, None)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}



def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """Sinusoidal embedding for continuous flow time t in [0, 1].

    This is the same style of time embedding used in diffusion models and in
    the uploaded AbX Seqformer.  It lets AbFlow learn f_theta(X_t, t, c)
    instead of forcing one network to average over all noise/flow times.

    Args:
        timesteps: [B] tensor with values in [0, 1].
        embedding_dim: output channel dimension.
        max_positions: frequency scale.
    Returns:
        [B, embedding_dim] sinusoidal embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float() * max_positions
    half_dim = embedding_dim // 2
    if half_dim <= 1:
        emb = timesteps[:, None]
        return F.pad(emb, (0, max(0, embedding_dim - 1)))[:, :embedding_dim]
    freq = math.log(max_positions) / (half_dim - 1)
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -freq)
    emb = timesteps[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb

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
        # Analytic-score-consistent endpoint Flow Matching
        # =========================================================
        # The network predicts a clean endpoint X_1^theta. Following the
        # analytic parameterization used in AbX, the coordinate score is
        # induced from the known forward kernel; no independent score head is
        # introduced.
        #
        # Core principles of this version:
        #   1. X_0/S_0 always come from the reference distribution.
        #   2. X_pep/S_pep are conditions only and never overwrite X_t/S_t.
        #   3. The endpoint objective is defined exactly once per complex.
        #   4. Endpoint and analytic DSM targets replace one another per complex;
        #      they are never stacked for the same sample.
        #   5. Analytic DSM is applied only to CA translation, whose reference
        #      kernel is exactly isotropic Gaussian around the antigen center.
        #   6. The real transport path reaches t=1; min_sigma protects only
        #      analytic score/bridge denominators.
        self.scorefm_eps = 1e-8
        self.scorefm_min_sigma = _env_float(
            "ABFLOW_SCOREFM_MIN_SIGMA", 1e-2
        )
        if not (0.0 < self.scorefm_min_sigma < 1.0):
            raise ValueError(
                "ABFLOW_SCOREFM_MIN_SIGMA must be in (0, 1)."
            )

        # Coordinate objective:
        #   endpoint:
        #       Unique per-complex clean-endpoint SmoothL1 at every t.
        #   analytic_core:
        #       Use analytic CA-score DSM only inside a bounded time interval;
        #       use endpoint reconstruction outside that interval. This avoids
        #       both low-t score degeneracy and late-time score singularity.
        self.scorefm_loss_mode = _env_str(
            "ABFLOW_SCOREFM_LOSS_MODE", "endpoint"
        ).lower()
        if self.scorefm_loss_mode in {"off", "none", "base"}:
            self.scorefm_loss_mode = "endpoint"
        if self.scorefm_loss_mode in {"core", "dtm_core", "hybrid"}:
            self.scorefm_loss_mode = "analytic_core"
        if self.scorefm_loss_mode not in {"endpoint", "analytic_core"}:
            raise ValueError(
                "Unknown ABFLOW_SCOREFM_LOSS_MODE="
                f"{self.scorefm_loss_mode}. Choose from endpoint, analytic_core."
            )

        self.scorefm_dsm_t_min = _env_float(
            "ABFLOW_SCOREFM_DSM_T_MIN", 0.2
        )
        self.scorefm_dsm_t_max = _env_float(
            "ABFLOW_SCOREFM_DSM_T_MAX", 0.8
        )
        if not (
            0.0 <= self.scorefm_dsm_t_min
            < self.scorefm_dsm_t_max <= 1.0
        ):
            raise ValueError(
                "Require 0 <= ABFLOW_SCOREFM_DSM_T_MIN < "
                "ABFLOW_SCOREFM_DSM_T_MAX <= 1."
            )

        # State-path controls.
        self.scorefm_per_sample_t = _env_flag(
            "ABFLOW_SCOREFM_PER_SAMPLE_T", True
        )
        self.scorefm_t_sampling = _env_str(
            "ABFLOW_SCOREFM_T_SAMPLING", "uniform"
        ).lower()
        self.scorefm_state_path = _env_flag(
            "ABFLOW_SCOREFM_STATE_PATH", True
        )

        # Node-level time conditioning. Pair-time conditioning is deliberately
        # deferred so that the present experiments remain attributable.
        self.scorefm_time_embed = _env_flag(
            "ABFLOW_SCOREFM_TIME_EMBED", True
        )
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, embed_size),
        )

        # Minimal sampler set: residual is a conservative ablation; bridge is
        # the endpoint-parameterized FM sampler used by default.
        self.scorefm_sampler_mode = _env_str(
            "ABFLOW_SCOREFM_SAMPLER_MODE", "bridge"
        ).lower()
        if self.scorefm_sampler_mode not in {"residual", "bridge"}:
            raise ValueError(
                "Unknown ABFLOW_SCOREFM_SAMPLER_MODE="
                f"{self.scorefm_sampler_mode}. Choose from residual, bridge."
            )

        # =========================================================
        # Peptide information as condition, never as source-state injection
        # =========================================================
        # Coordinate proposal conditioning is represented only in scalar hidden
        # space. The directional CA displacement is expressed in the proposal's
        # local N-CA-C frame, which is stable even when the current flow state is
        # highly noisy at low t. Distance statistics use log1p compression to
        # limit dynamic range. The resulting scalar features are invariant to global
        # SE(3) transformations. Crucially, X_pep never directly updates
        # interface_X; it conditions f_theta instead of acting as a post-hoc
        # coordinate correction.
        self.coord_pep_as_condition = _env_flag(
            "ABFLOW_COORD_PEP_AS_CONDITION", False
        )
        self.coord_pep_condition_dim = 6
        if self.coord_pep_as_condition:
            # Input: current hidden state H_0 plus six E(3)-invariant features:
            #   local-frame CA displacement (3),
            #   CA distance, mean backbone distance, RMS backbone distance (3).
            self.coord_pep_condition_adapter = nn.Sequential(
                nn.Linear(embed_size + self.coord_pep_condition_dim, embed_size),
                nn.SiLU(),
                nn.Linear(embed_size, embed_size),
            )
            # Zero-start residual adapter: the initial model is exactly REF.
            nn.init.zeros_(self.coord_pep_condition_adapter[-1].weight)
            nn.init.zeros_(self.coord_pep_condition_adapter[-1].bias)
        else:
            self.coord_pep_condition_adapter = None

        # S_pep is a proposal-token condition, not an ESM representation and
        # never a replacement for S_t.  Fusion is residue- and state-dependent:
        # H_0 already contains the current state and time embedding, so the
        # adapter can learn when the proposal token is useful instead of applying
        # one global scalar to every residue and every time.
        self.seq_input_mode = _env_str(
            "ABFLOW_SEQ_INPUT_MODE", "state"
        ).lower()
        if self.seq_input_mode not in {"state", "pep_condition"}:
            raise ValueError(
                "Unknown ABFLOW_SEQ_INPUT_MODE="
                f"{self.seq_input_mode}. Choose from state, pep_condition."
            )
        if self.seq_input_mode == "pep_condition":
            self.seq_pep_condition_embedding = nn.Embedding(
                num_classes, embed_size
            )
            self.seq_pep_condition_adapter = nn.Sequential(
                nn.Linear(2 * embed_size, embed_size),
                nn.SiLU(),
                nn.Linear(embed_size, embed_size),
            )
            nn.init.zeros_(self.seq_pep_condition_adapter[-1].weight)
            nn.init.zeros_(self.seq_pep_condition_adapter[-1].bias)
        else:
            self.seq_pep_condition_embedding = None
            self.seq_pep_condition_adapter = None

        self.seq_ce_weight = _env_float("ABFLOW_SEQ_CE_WEIGHT", 1.0)
        self.last_scorefm_losses = {}
        self.last_abflow_diagnostics = {}
        # Detached condition-strength diagnostics. These do not affect training.
        self._last_condition_diagnostics = {}
        self._latest_condition_diagnostics = {}


    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S
    
    @torch.no_grad()
    def _sample_categorical_path(self, clean_S, base_S, t_graph,
                                 interface_batch_id, corrupt_mask=None):
        """Sample a legal categorical interpolation q_t(S_t | S_1, S_0).

        With the same time convention as the coordinate path, t=0 is the base
        distribution and t=1 is clean. Each residue keeps the clean category
        with probability t and otherwise keeps its sampled base category.
        """
        t_graph = torch.as_tensor(
            t_graph, device=clean_S.device, dtype=torch.float32
        )
        if t_graph.dim() == 0 or t_graph.numel() == 1:
            keep_prob = t_graph.reshape(1).expand_as(clean_S)
        else:
            keep_prob = t_graph[interface_batch_id]
        keep_clean = torch.rand(
            clean_S.shape, device=clean_S.device
        ) < keep_prob.clamp(0.0, 1.0)
        sampled = torch.where(keep_clean, clean_S, base_S).long()
        if corrupt_mask is None:
            return sampled
        corrupt_mask = corrupt_mask.to(device=clean_S.device, dtype=torch.bool)
        return torch.where(corrupt_mask, sampled, clean_S).long()
    
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
            #print(e)
            raise RuntimeError("Failed to align epitope-antibody edges.") from e
        
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
        

    def _sample_flow_times(self, batch_size, device, dtype=torch.float32):
        """Sample continuous flow times for training.

        t is a real path time in [0, 1]. We do not shrink the endpoint here.
        Numerical stability is handled separately by sigma_score=max(1-t, sigma_min).
        """
        n = int(batch_size) if getattr(self, 'scorefm_per_sample_t', False) else 1
        mode = getattr(self, 'scorefm_t_sampling', 'uniform')

        if mode == 'uniform':
            t = torch.rand(n, device=device, dtype=dtype)

        elif mode in {'low_t', 'low', 'square'}:
            u = torch.rand(n, device=device, dtype=dtype)
            t = u ** 2

        elif mode in {'stratified', 'strat'}:
            if n <= 1:
                t = torch.rand(n, device=device, dtype=dtype)
            else:
                base = (torch.arange(n, device=device, dtype=dtype) +
                        torch.rand(n, device=device, dtype=dtype)) / float(n)
                perm = torch.randperm(n, device=device)
                t = base[perm]

        else:
            raise ValueError(
                f"Unknown ABFLOW_SCOREFM_T_SAMPLING={mode}. "
                "Choose from uniform, low_t, stratified."
            )

        return t.clamp(min=0.0, max=1.0)
    
    def _time_for_interface(self, t_graph, interface_batch_id, ref_tensor):
        """Broadcast graph-level time to [N_interface, 1, 1]."""
        if t_graph is None:
            return None
        t_graph = torch.as_tensor(t_graph, device=ref_tensor.device, dtype=ref_tensor.dtype)
        if t_graph.dim() == 0 or t_graph.numel() == 1:
            return t_graph.reshape(1, 1, 1)
        return t_graph[interface_batch_id].reshape(-1, 1, 1)

    def _flow_time_embedding_for_residues(self, flow_t, batch_id, H_0):
        """Create residue-wise time embeddings aligned with H_0."""
        if flow_t is None or not getattr(self, 'scorefm_time_embed', False):
            return None
        flow_t = torch.as_tensor(flow_t, device=H_0.device, dtype=H_0.dtype)
        if flow_t.dim() == 0 or flow_t.numel() == 1:
            n_graph = int(batch_id.max().item()) + 1 if batch_id.numel() > 0 else 1
            flow_t = flow_t.reshape(1).expand(n_graph)

        t_emb = get_timestep_embedding(flow_t, H_0.shape[-1]).to(dtype=H_0.dtype, device=H_0.device)
        t_emb = self.flow_time_mlp(t_emb)
        return t_emb[batch_id]


    def _build_coord_pep_condition_for_residues(
            self, pep_X_model, interface_X, paratope_mask,
            pep_coord_valid=None):
        """Build dynamic proposal-coordinate condition features.

        X_pep is condition only: this function never modifies interface_X.

        Direction:
            The displacement from the current CA to the proposal CA is projected
            into the proposal N-CA-C local frame. Under any global proper
            rotation/translation, the frame and displacement transform together,
            so the projected components are SE(3)-invariant.

        Magnitude:
            The signed local displacement is compressed radially so its norm is
            log1p(CA distance). CA, mean-backbone and RMS-backbone distances are
            also compressed with log1p. This preserves direction and near-range
            sensitivity while preventing a poor proposal from dominating the
            hidden-state adapter through extreme raw distances.

        Robustness:
            Invalid proposal residues are masked. Degenerate proposal frames use
            distance-only conditioning by setting directional components to zero.

        Features per paratope residue:
            1-3) radially log-compressed proposal-local CA displacement;
            4)   log1p(CA distance);
            5)   log1p(mean backbone distance);
            6)   log1p(RMS backbone distance).
        """
        if (
            not self.coord_pep_as_condition
            or self.coord_pep_condition_adapter is None
            or pep_X_model is None
        ):
            return None, None

        if pep_X_model.shape != interface_X.shape:
            raise ValueError(
                "pep_X_model/interface_X shape mismatch: "
                f"{tuple(pep_X_model.shape)} vs {tuple(interface_X.shape)}"
            )
        if interface_X.shape[1] < 3:
            raise ValueError(
                "Coordinate conditioning requires N/CA/C channels."
            )

        n_int = int(interface_X.shape[0])
        if int(paratope_mask.sum().item()) != n_int:
            raise ValueError(
                "paratope/interface size mismatch: "
                f"{int(paratope_mask.sum().item())} vs {n_int}."
            )
        if pep_coord_valid is None:
            valid_int = torch.ones(
                n_int, device=interface_X.device, dtype=torch.bool
            )
        else:
            valid_int = torch.as_tensor(
                pep_coord_valid,
                device=interface_X.device,
                dtype=torch.bool,
            ).reshape(-1)
            if valid_int.numel() != n_int:
                raise ValueError(
                    "pep_coord_valid length mismatch: "
                    f"expected {n_int}, got {valid_int.numel()}."
                )

        delta = pep_X_model - interface_X
        ca_delta = delta[:, 1]
        ca_dist = torch.norm(ca_delta, dim=-1, keepdim=True)

        # Stable proposal-local N-CA-C frame.
        n_vec = pep_X_model[:, 0] - pep_X_model[:, 1]
        c_vec = pep_X_model[:, 2] - pep_X_model[:, 1]

        c_norm = torch.norm(c_vec, dim=-1, keepdim=True)
        e1 = F.normalize(c_vec, dim=-1, eps=self.scorefm_eps)

        n_orth = (
            n_vec
            - (n_vec * e1).sum(dim=-1, keepdim=True) * e1
        )
        n_orth_norm = torch.norm(n_orth, dim=-1, keepdim=True)
        e2 = F.normalize(n_orth, dim=-1, eps=self.scorefm_eps)
        e3 = F.normalize(
            torch.cross(e1, e2, dim=-1),
            dim=-1,
            eps=self.scorefm_eps,
        )

        local_delta = torch.stack(
            [
                (ca_delta * e1).sum(dim=-1),
                (ca_delta * e2).sum(dim=-1),
                (ca_delta * e3).sum(dim=-1),
            ],
            dim=-1,
        )

        # Parameter-free radial dynamic-range compression.  The three signed
        # proposal-local components retain their direction, while their joint
        # magnitude changes from d to log(1+d).  This avoids letting a very poor
        # proposal dominate the residual adapter through an arbitrarily large
        # raw displacement, without introducing a peptide-prior weight.
        ca_dist_safe = ca_dist.clamp_min(self.scorefm_eps)
        local_delta = (
            local_delta
            * (torch.log1p(ca_dist) / ca_dist_safe)
        )

        frame_valid = (
            torch.isfinite(c_norm.squeeze(-1))
            & torch.isfinite(n_orth_norm.squeeze(-1))
            & (c_norm.squeeze(-1) > 1e-4)
            & (n_orth_norm.squeeze(-1) > 1e-4)
        )
        direction_valid = valid_int & frame_valid
        local_delta = torch.where(
            direction_valid.unsqueeze(-1),
            local_delta,
            torch.zeros_like(local_delta),
        )

        # N/CA/C/O are available independently of the sampled side-chain token.
        n_bb = min(4, interface_X.shape[1])
        bb_dist = torch.norm(delta[:, :n_bb], dim=-1)
        mean_bb_dist = bb_dist.mean(dim=-1, keepdim=True)
        rms_bb_dist = torch.sqrt(
            (bb_dist ** 2).mean(dim=-1, keepdim=True)
            + self.scorefm_eps
        )

        distance_feat = torch.cat(
            [ca_dist, mean_bb_dist, rms_bb_dist],
            dim=-1,
        )
        distance_feat = torch.log1p(
            distance_feat.clamp_min(0.0)
        )

        feat_int = torch.cat(
            [local_delta, distance_feat],
            dim=-1,
        )
        feat_int = torch.nan_to_num(
            feat_int, nan=0.0, posinf=0.0, neginf=0.0
        )
        feat_int = (
            feat_int
            * valid_int.unsqueeze(-1).to(feat_int.dtype)
        )

        n_res = int(paratope_mask.shape[0])
        feat_full = interface_X.new_zeros(
            (n_res, self.coord_pep_condition_dim)
        )
        mask_full = torch.zeros(
            n_res, device=interface_X.device, dtype=torch.bool
        )
        feat_full[paratope_mask] = feat_int
        mask_full[paratope_mask] = valid_int
        return feat_full, mask_full

    def _build_seq_pep_condition_for_residues(
            self, S_pep, paratope_mask, ref_tensor):
        """Build residue-level sequence proposal conditions.

        S_pep is never written into S_t. Valid proposal tokens are embedded and
        placed on the corresponding paratope residues through functional tensor
        construction. Invalid or missing tokens contribute exactly zero.

        The returned tensor has the same hidden width and dtype as ref_tensor.
        """
        if self.seq_pep_condition_embedding is None or S_pep is None:
            return None, None

        n_int = int(paratope_mask.sum().item())
        if S_pep.numel() != n_int:
            raise ValueError(
                "S_pep/paratope size mismatch: "
                f"expected {n_int}, got {S_pep.numel()}."
            )

        pep_S = S_pep.to(
            device=ref_tensor.device, dtype=torch.long
        ).reshape(-1)
        valid_int = torch.logical_and(
            pep_S >= 0, pep_S < self.num_classes
        )
        if not valid_int.any():
            return None, None

        n_res = int(paratope_mask.shape[0])
        par_idx = paratope_mask.nonzero(
            as_tuple=False
        ).reshape(-1)
        valid_idx = par_idx[valid_int]

        # Build full residue-aligned token/mask tensors without modifying an
        # embedding output in place. Invalid positions use token 0 but are
        # multiplied by a zero mask, so they contribute no forward value or
        # embedding gradient.
        full_tokens = torch.zeros(
            n_res, device=ref_tensor.device, dtype=torch.long
        )
        full_tokens = full_tokens.index_copy(
            0, valid_idx, pep_S[valid_int]
        )

        cond_mask = torch.zeros(
            n_res, device=ref_tensor.device, dtype=torch.bool
        )
        cond_mask = cond_mask.index_fill(0, valid_idx, True)

        cond_emb = self.seq_pep_condition_embedding(
            full_tokens
        ).to(
            device=ref_tensor.device,
            dtype=ref_tensor.dtype,
        )
        cond_emb = (
            cond_emb
            * cond_mask.unsqueeze(-1).to(cond_emb.dtype)
        )
        return cond_emb, cond_mask

    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask,
                        batch_id, round_idx, memory_H=None, smooth_prob=None,
                        smooth_mask=None, flow_t=None,
                        coord_pep_condition=None,
                        coord_pep_condition_mask=None,
                        seq_pep_condition=None,
                        seq_pep_condition_mask=None):
        # embeddings, hidden state, (internal edges, external edges),
        # (A : c*d, w : c*1)
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(
            X, S, batch_id, self.k_neighbors, residue_pos,
            smooth_prob=smooth_prob, smooth_mask=smooth_mask
        )

        time_emb = self._flow_time_embedding_for_residues(
            flow_t, batch_id, H_0
        )
        if time_emb is not None:
            H_0 = H_0 + time_emb

        # Reset detached condition-strength diagnostics for this refinement round.
        zero_diag = H_0.detach().new_tensor(0.0)
        self._last_condition_diagnostics = {
            "coord_condition_residual_ratio": zero_diag,
            "seq_condition_residual_ratio": zero_diag,
            "coord_condition_valid_rate": zero_diag,
            "seq_condition_valid_rate": zero_diag,
        }

        # Coordinate proposal enters only through a zero-start residual feature
        # adapter.  H_0 already contains the current state and time embedding,
        # making the fusion residue-, context-, and time-dependent.
        if (
            coord_pep_condition is not None
            and coord_pep_condition_mask is not None
            and self.coord_pep_condition_adapter is not None
        ):
            cond_feat = coord_pep_condition.to(
                device=H_0.device, dtype=H_0.dtype
            )
            if cond_feat.shape != (
                H_0.shape[0], self.coord_pep_condition_dim
            ):
                raise ValueError(
                    "coordinate condition shape mismatch: expected "
                    f"{(H_0.shape[0], self.coord_pep_condition_dim)}, "
                    f"got {tuple(cond_feat.shape)}."
                )
            cond_mask = coord_pep_condition_mask.to(
                device=H_0.device, dtype=torch.bool
            )
            coord_residual = self.coord_pep_condition_adapter(
                torch.cat([H_0, cond_feat], dim=-1)
            )
            coord_mask_f = cond_mask.unsqueeze(-1).to(H_0.dtype)
            masked_coord_residual = coord_residual * coord_mask_f

            with torch.no_grad():
                if cond_mask.any():
                    base_rms = torch.sqrt(
                        (H_0[cond_mask].detach() ** 2).mean()
                        + self.scorefm_eps
                    )
                    residual_rms = torch.sqrt(
                        (coord_residual[cond_mask].detach() ** 2).mean()
                        + self.scorefm_eps
                    )
                    self._last_condition_diagnostics[
                        "coord_condition_residual_ratio"
                    ] = residual_rms / base_rms.clamp_min(self.scorefm_eps)
                    self._last_condition_diagnostics[
                        "coord_condition_valid_rate"
                    ] = cond_mask.float().mean()

            H_0 = H_0 + masked_coord_residual

        # Sequence proposal is fused through a residue- and time-dependent
        # zero-start adapter, rather than a single global scalar shared by all
        # samples, residues and times.
        if (
            seq_pep_condition is not None
            and seq_pep_condition_mask is not None
            and self.seq_pep_condition_adapter is not None
        ):
            seq_cond = seq_pep_condition.to(
                device=H_0.device, dtype=H_0.dtype
            )
            if seq_cond.shape != H_0.shape:
                raise ValueError(
                    "sequence condition shape mismatch: expected "
                    f"{tuple(H_0.shape)}, got {tuple(seq_cond.shape)}."
                )
            seq_mask = seq_pep_condition_mask.to(
                device=H_0.device, dtype=torch.bool
            )
            seq_residual = self.seq_pep_condition_adapter(
                torch.cat([H_0, seq_cond], dim=-1)
            )
            seq_mask_f = seq_mask.unsqueeze(-1).to(H_0.dtype)
            masked_seq_residual = seq_residual * seq_mask_f

            with torch.no_grad():
                if seq_mask.any():
                    base_rms = torch.sqrt(
                        (H_0[seq_mask].detach() ** 2).mean()
                        + self.scorefm_eps
                    )
                    residual_rms = torch.sqrt(
                        (seq_residual[seq_mask].detach() ** 2).mean()
                        + self.scorefm_eps
                    )
                    self._last_condition_diagnostics[
                        "seq_condition_residual_ratio"
                    ] = residual_rms / base_rms.clamp_min(self.scorefm_eps)
                    self._last_condition_diagnostics[
                        "seq_condition_valid_rate"
                    ] = seq_mask.float().mean()

            H_0 = H_0 + masked_seq_residual

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
        
        local_X[local_is_ab] = interface_X
        # prepare local complex edges
        local_ctx_edges = self.batch_constants['local_ctx_edges']  # [2, Ec]
        local_inter_edges = self.batch_constants['local_inter_edges']  # [2, Ei]
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        # Context and interaction edges are both derived from the current state.
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni))
        # For interaction edges, optionally use the learned distance predictor.
        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src_H, dst_H = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) +\
                          self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))  # perm-invariant
            p_edge_dist = p_edge_dist.squeeze(-1)
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
        # is_ctx = row_segment_ids == col_segment_ids
        # is_inter = torch.logical_not(is_ctx)
        
        row_is_ag = row_segment_ids == self.aa_feature.ag_seg_id
        col_is_ag = col_segment_ids == self.aa_feature.ag_seg_id
        is_inter = torch.logical_xor(row_is_ag, col_is_ag)
        is_ctx = torch.logical_not(is_inter)

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
    
    def _raw_interface_to_model_frame(self, interface_X, paratope_mask, batch_id):
        """Convert raw paratope coordinates into internal shadow frame.

        `_forward` centers the antigen/antibody and normalizes coordinates before
        message passing. Shadow paratope coordinates are later uncentered with
        `_type=4`, i.e. by adding the antigen center. Therefore an externally
        supplied raw X_t must be represented internally as:

            X_t_model = (X_t_raw - antigen_center) / std.
        """
        interface_batch_id = batch_id[paratope_mask]
        ag_centers = self.normalizer.ag_centers[interface_batch_id]
        return self.normalizer.normalize(interface_X - ag_centers.unsqueeze(1))

    def _reference_ca_mean(self, X, S, paratope_mask, batch_id):
        """Reference mean for CA translation under init_interface().

        init_interface samples each paratope CA as:
            antigen_center + N(0, I_3).
        Therefore the conditional CA score is analytically available with
        mean=antigen_center and identity covariance.
        """
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]
        return ag_centers[batch_id[paratope_mask]]

    def _analytic_ca_score_from_clean(
            self, Xt, clean_X, t, sigma_t, source_ca_mean):
        """Analytic CA translation score for the linear reference bridge.

        Path:
            X_t^CA = sigma_t X_0^CA + t X_1^CA,
            X_0^CA ~ N(source_ca_mean, I).

        Hence:
            p_t(X_t^CA | X_1^CA, c)
              = N(sigma_t * source_ca_mean + t * X_1^CA,
                  sigma_t^2 I)

            score = -(X_t^CA - sigma_t*mu_0 - t*X_1^CA) / sigma_t^2.

        Only CA translation is used here. The full-atom initialization is
        atom-correlated, so pretending that all atom channels are isotropic
        Gaussian would be mathematically inconsistent.
        """
        t = torch.as_tensor(t, device=Xt.device, dtype=Xt.dtype)
        sigma_t = torch.as_tensor(
            sigma_t, device=Xt.device, dtype=Xt.dtype
        )

        if t.dim() == 3:
            t_ca = t[:, 0, :]
        else:
            t_ca = t.reshape(-1, 1)

        if sigma_t.dim() == 3:
            sigma_ca = sigma_t[:, 0, :]
        else:
            sigma_ca = sigma_t.reshape(-1, 1)

        ca_idx = 1 if Xt.shape[1] > 1 else 0
        Xt_ca = Xt[:, ca_idx]
        clean_ca = clean_X[:, ca_idx]
        mean_t = (
            sigma_ca * source_ca_mean
            + t_ca * clean_ca
        )
        return -(
            Xt_ca - mean_t
        ) / (sigma_ca.pow(2) + self.scorefm_eps)

    def _interface_valid_graph_mask(
            self, interface_batch_id, n_graph, device):
        valid = torch.zeros(
            n_graph, device=device, dtype=torch.bool
        )
        if interface_batch_id.numel() > 0:
            valid[torch.unique(interface_batch_id)] = True
        return valid

    def _masked_residue_mse_per_graph(
            self, diff, atom_mask, interface_batch_id):
        """Per-complex normalized vector MSE.

        Reduction order:
            xyz -> atom channels -> residues -> complex.
        """
        if interface_batch_id.numel() == 0:
            return (
                diff.new_zeros(1),
                torch.zeros(1, device=diff.device, dtype=torch.bool),
            )

        n_graph = int(interface_batch_id.max().item()) + 1
        atom_mask_f = atom_mask.to(diff.dtype)

        atom_sq = (diff ** 2).sum(dim=-1) * atom_mask_f
        per_res = atom_sq.sum(dim=-1) / (
            3.0 * atom_mask_f.sum(dim=-1).clamp_min(1.0)
        )

        per_graph = scatter_mean(
            per_res,
            interface_batch_id,
            dim=0,
            dim_size=n_graph,
        )
        valid_graph = self._interface_valid_graph_mask(
            interface_batch_id, n_graph, diff.device
        )
        return per_graph, valid_graph

    def _masked_residue_smooth_l1_per_graph(
            self, pred, target, atom_mask, interface_batch_id):
        """Unique per-complex endpoint objective.

        This replaces both the old global interface loss and the duplicated
        auxiliary x1 loss. Each complex contributes one normalized value,
        independent of CDR length.
        """
        if interface_batch_id.numel() == 0:
            return (
                pred.new_zeros(1),
                torch.zeros(1, device=pred.device, dtype=torch.bool),
            )

        n_graph = int(interface_batch_id.max().item()) + 1
        atom_mask_f = atom_mask.to(pred.dtype)

        err = F.smooth_l1_loss(
            pred, target, reduction="none"
        ).sum(dim=-1)
        err = err * atom_mask_f
        per_res = err.sum(dim=-1) / (
            3.0 * atom_mask_f.sum(dim=-1).clamp_min(1.0)
        )

        per_graph = scatter_mean(
            per_res,
            interface_batch_id,
            dim=0,
            dim_size=n_graph,
        )
        valid_graph = self._interface_valid_graph_mask(
            interface_batch_id, n_graph, pred.device
        )
        return per_graph, valid_graph

    def _scorefm_time_per_graph(
            self, t, interface_batch_id, ref_tensor):
        """Convert scalar/graph/interface time to one value per complex."""
        if interface_batch_id.numel() == 0:
            return (
                ref_tensor.new_zeros(1),
                torch.zeros(
                    1, device=ref_tensor.device, dtype=torch.bool
                ),
            )

        n_graph = int(interface_batch_id.max().item()) + 1
        valid_graph = self._interface_valid_graph_mask(
            interface_batch_id, n_graph, ref_tensor.device
        )

        t_tensor = torch.as_tensor(
            t, device=ref_tensor.device, dtype=ref_tensor.dtype
        )
        if t_tensor.dim() == 0 or t_tensor.numel() == 1:
            t_graph = t_tensor.reshape(1).expand(n_graph)
        else:
            t_flat = t_tensor.reshape(-1)
            if t_flat.numel() == n_graph:
                t_graph = t_flat
            elif t_flat.numel() == interface_batch_id.numel():
                t_graph = scatter_mean(
                    t_flat,
                    interface_batch_id,
                    dim=0,
                    dim_size=n_graph,
                )
            else:
                raise ValueError(
                    "t must be scalar, graph-level [B], or "
                    "interface-level [N_int]. "
                    f"Got {t_flat.numel()} values for "
                    f"{n_graph} graphs and "
                    f"{interface_batch_id.numel()} residues."
                )

        return t_graph.clamp(0.0, 1.0), valid_graph

    def _coordinate_training_objective(
            self, *, Xt, X1, pred_clean_X, atom_mask,
            interface_batch_id, t, sigma_t, source_ca_mean):
        """Single non-redundant coordinate objective.

        endpoint mode:
            Per-complex endpoint SmoothL1 for every sample.

        analytic_core mode:
            Use analytic CA-score DSM only in the bounded interval
            [dsm_t_min, dsm_t_max]. Outside this interval use the endpoint
            objective. The switch is per complex and the two objectives are
            never added together for the same sample.
        """
        endpoint_per_graph, endpoint_valid = (
            self._masked_residue_smooth_l1_per_graph(
                pred_clean_X, X1, atom_mask, interface_batch_id
            )
        )

        if endpoint_valid.any():
            endpoint_loss = endpoint_per_graph[
                endpoint_valid
            ].mean()
        else:
            endpoint_loss = pred_clean_X.new_tensor(0.0)

        zero = endpoint_loss.detach() * 0.0

        if self.scorefm_loss_mode == "endpoint":
            details = {
                "scorefm_total": endpoint_loss.detach(),
                "scorefm_endpoint": endpoint_loss.detach(),
                "scorefm_dsm": zero,
                "scorefm_dsm_rate": zero,
            }
            return endpoint_loss, details

        gt_score_ca = self._analytic_ca_score_from_clean(
            Xt, X1, t, sigma_t, source_ca_mean
        ).detach()
        pred_score_ca = self._analytic_ca_score_from_clean(
            Xt, pred_clean_X, t, sigma_t, source_ca_mean
        )

        if sigma_t.dim() == 3:
            sigma_ca = sigma_t[:, 0, :]
        else:
            sigma_ca = sigma_t.reshape(-1, 1)

        # AbX-style score scaling: sigma_t * score residual. DSM is used
        # only where t/(1-t) is bounded by the configured interval.
        scaled_score_diff = (
            sigma_ca * (pred_score_ca - gt_score_ca)
        ).unsqueeze(1)

        ca_idx = 1 if atom_mask.shape[1] > 1 else 0
        ca_mask = atom_mask[:, ca_idx:ca_idx + 1]

        dsm_per_graph, dsm_valid = self._masked_residue_mse_per_graph(
            scaled_score_diff, ca_mask, interface_batch_id
        )

        t_graph, t_valid = self._scorefm_time_per_graph(
            t, interface_batch_id, pred_clean_X
        )
        valid = endpoint_valid & dsm_valid & t_valid

        if dsm_valid.any():
            dsm_loss = dsm_per_graph[dsm_valid].mean()
        else:
            dsm_loss = pred_clean_X.new_tensor(0.0)

        if not valid.any():
            details = {
                "scorefm_total": endpoint_loss.detach(),
                "scorefm_endpoint": endpoint_loss.detach(),
                "scorefm_dsm": dsm_loss.detach(),
                "scorefm_dsm_rate": zero,
            }
            return endpoint_loss, details

        use_dsm = (
            (t_graph >= self.scorefm_dsm_t_min)
            & (t_graph <= self.scorefm_dsm_t_max)
        )

        per_graph = torch.where(
            use_dsm,
            dsm_per_graph,
            endpoint_per_graph,
        )
        total = per_graph[valid].mean()

        details = {
            "scorefm_total": total.detach(),
            "scorefm_endpoint": endpoint_loss.detach(),
            "scorefm_dsm": dsm_loss.detach(),
            "scorefm_dsm_rate": use_dsm[valid].float().mean().detach(),
        }
        return total, details

    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                 surface, residue_pos, template, lengths, init_noise=None,
                 interface_init=None, sequence_init=None, flow_t=None):
        """Evaluate f_theta(X_t, S_t, t, conditions).

        X_pep/S_pep are conditions only. They never overwrite the explicit
        generated states interface_init/sequence_init.
        """
        batch_id = self.batch_constants['batch_id']

        X = X.clone()
        S = S.clone()
        surface = surface.clone()

        has_interface_state = interface_init is not None
        has_sequence_state = sequence_init is not None

        X, S = self.init_mask(X, S, cmask, smask, template)

        if has_interface_state:
            expected_shape = X[paratope_mask].shape
            if interface_init.shape != expected_shape:
                raise ValueError(
                    f"interface_init shape mismatch: expected {tuple(expected_shape)}, "
                    f"got {tuple(interface_init.shape)}."
                )
            X[paratope_mask] = interface_init.to(
                device=X.device, dtype=X.dtype
            )

        if has_sequence_state:
            expected_shape = S[paratope_mask].shape
            if sequence_init.shape != expected_shape:
                raise ValueError(
                    f"sequence_init shape mismatch: expected {tuple(expected_shape)}, "
                    f"got {tuple(sequence_init.shape)}."
                )
            S[paratope_mask] = sequence_init.to(
                device=S.device, dtype=torch.long
            )

        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)
        surface = self.normalizer.normalize(surface)
        X = self.aa_feature.update_global_coordinates(X, S)

        if has_interface_state:
            interface_X = self._raw_interface_to_model_frame(
                interface_init, paratope_mask, batch_id
            )
            interface_S = S[paratope_mask].clone()
        else:
            interface_X, interface_S = self.init_interface(
                X, S, paratope_mask, batch_id, init_noise
            )

        # Convert X_pep once to the internal frame. Its relation to the current
        # interface state is recomputed after every refinement round. Proposal
        # validity is tracked per residue so missing/invalid proposal coordinates
        # cannot silently become a condition.
        pep_X_model = None
        pep_coord_valid = None
        if (
            self.coord_pep_as_condition
            and X_pep is not None
            and X_pep.shape == interface_X.shape
        ):
            pep_X_raw = X_pep.to(device=X.device, dtype=X.dtype)
            proposal_backbone = pep_X_raw[:, :3]
            pep_coord_valid = (
                torch.isfinite(proposal_backbone).all(dim=-1).all(dim=-1)
                & (
                    proposal_backbone.abs()
                    .sum(dim=-1)
                    .sum(dim=-1)
                    > self.scorefm_eps
                )
            )
            if pep_coord_valid.any():
                pep_X_model = self._raw_interface_to_model_frame(
                    pep_X_raw, paratope_mask, batch_id
                )

        if self.seq_pep_condition_embedding is not None:
            seq_ref_tensor = interface_X.new_zeros(
                (paratope_mask.shape[0],
                 self.seq_pep_condition_embedding.embedding_dim)
            )
        else:
            seq_ref_tensor = interface_X.new_zeros(
                (paratope_mask.shape[0], 1)
            )
        seq_pep_condition, seq_pep_condition_mask = (
            self._build_seq_pep_condition_for_residues(
                S_pep, paratope_mask, seq_ref_tensor
            )
        )

        r_pred_S_logits, pred_S_dist = [], None
        r_interface_X = [interface_X.clone()]
        r_edge_dist = []
        memory_H = None
        condition_diag_rounds = []

        for round_idx in range(self.round):
            (
                coord_pep_condition,
                coord_pep_condition_mask,
            ) = self._build_coord_pep_condition_for_residues(
                pep_X_model,
                interface_X,
                paratope_mask,
                pep_coord_valid=pep_coord_valid,
            )

            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(
                X, S, residue_pos, interface_X, surface, paratope_mask,
                batch_id, round_idx, memory_H, pred_S_dist, smask,
                flow_t=flow_t,
                coord_pep_condition=coord_pep_condition,
                coord_pep_condition_mask=coord_pep_condition_mask,
                seq_pep_condition=seq_pep_condition,
                seq_pep_condition_mask=seq_pep_condition_mask,
            )

            condition_diag_rounds.append({
                key: value.detach()
                for key, value in self._last_condition_diagnostics.items()
            })

            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_edge_dist.append(edge_dist)

            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_global_coordinates(X, S)

            if not self.struct_only:
                S = S.clone()
                if round_idx == self.round - 1:
                    S[smask] = torch.argmax(
                        pred_S_logits[smask], dim=-1
                    )
                else:
                    pred_S_dist = torch.softmax(
                        pred_S_logits[smask], dim=-1
                    )

        if condition_diag_rounds:
            keys = condition_diag_rounds[0].keys()
            self._latest_condition_diagnostics = {
                key: torch.stack(
                    [round_diag[key] for round_diag in condition_diag_rounds]
                ).mean()
                for key in keys
            }
        else:
            self._latest_condition_diagnostics = {}

        interface_batch_id = self.batch_constants['interface_batch_id']
        if self.struct_only:
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()
        else:
            prmsd = None

        pred_X = self.normalizer.unnormalize(pred_X)
        pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X_i in enumerate(r_interface_X):
            interface_X_i = self.normalizer.unnormalize(interface_X_i)
            interface_X_i = self.normalizer.uncentering(
                interface_X_i, interface_batch_id, _type=4
            )
            r_interface_X[i] = interface_X_i

        self.normalizer.clear_cache()
        return H, S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd

    def forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, xloss_mask, context_ratio=0):
        '''
        :param X: [N, n_channel, 3], Cartesian coordinates
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training, probability of keeping ground-truth sequence context among originally masked positions.
        '''
        # import ipdb; ipdb.set_trace()
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            if X_pep is not None:
                X_pep = X_pep[:, :4]
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
        
        gt_interface_X = true_X[paratope_mask]
        batch_size = int(self.batch_constants['batch_size'].item()) if torch.is_tensor(self.batch_constants['batch_size']) else int(self.batch_constants['batch_size'])
        interface_batch_id = self.batch_constants['interface_batch_id']
        state_path = bool(self.scorefm_state_path)

        if state_path:
            # Sample X_0/S_0 from the reference initialization used at inference.
            # Peptide-derived information never overwrites this generated state.
            interface_X, interface_S = self.init_interface(
                X, S, paratope_mask, batch_id
            )
            source_ca_mean = self._reference_ca_mean(
                X, S, paratope_mask, batch_id
            )

            # Continuous flow time. Avoid sigma_t = 1 - t being too small because
            # the analytic score contains 1 / sigma_t^2.
            t_graph = self._sample_flow_times(batch_size, device=X.device, dtype=X.dtype)

            # Real path weight: must use the true endpoint geometry.
            # X_t = (1 - t) X_0 + t X_1 reaches exactly X_1 at t=1.
            base_weight_graph = 1.0 - t_graph

            # Score denominator: numerically protected only for score/velocity loss.
            sigma_score_graph = base_weight_graph.clamp_min(self.scorefm_min_sigma)

            t_int = self._time_for_interface(t_graph, interface_batch_id, interface_X)
            base_weight_int = self._time_for_interface(base_weight_graph, interface_batch_id, interface_X)
            sigma_score_int = self._time_for_interface(sigma_score_graph, interface_batch_id, interface_X)

            Xt = base_weight_int * interface_X + t_int * gt_interface_X

            if not self.struct_only:
                St = self._sample_categorical_path(
                    true_S[paratope_mask], interface_S, t_graph, interface_batch_id,
                    corrupt_mask=smask[paratope_mask],
                )
                sequence_state_for_model = St
            else:
                St = interface_S
                sequence_state_for_model = None
        else:
            # Non-state evaluator: no explicit X_t/S_t/t is injected.
            interface_X = None
            interface_S = None
            Xt = None
            St = None
            t_int = None
            sigma_score_int = None
            t_graph = X.new_zeros(1)
            sequence_state_for_model = None
            source_ca_mean = None

        # get results
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            interface_init=Xt if state_path else None,
            sequence_init=sequence_state_for_model if state_path else None,
            flow_t=t_graph if state_path else None
        )

        # sequence negative log likelihood
        snll = X.new_tensor(0.0)
        total = X.new_tensor(0.0)
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                if mask.any():
                    snll = snll + F.cross_entropy(
                        logits[mask], true_S[mask], reduction='sum'
                    )
                    total = total + mask.sum()
            snll = snll / total.clamp_min(1.0)

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss

        # 1. Unique coordinate objective for the shadow paratope.
        # The previous implementation added a global interface loss and a second
        # x1 auxiliary loss for the same endpoint error. Here the endpoint is
        # supervised exactly once, with per-complex normalization.
        interface_atom_pos = self.aa_feature._construct_atom_pos(
            true_S[paratope_mask]
        )
        interface_atom_mask = (
            interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        )

        if state_path:
            interface_loss, scorefm_details = (
                self._coordinate_training_objective(
                    Xt=Xt,
                    X1=gt_interface_X,
                    pred_clean_X=r_interface_X[-1],
                    atom_mask=interface_atom_mask,
                    interface_batch_id=interface_batch_id,
                    t=t_int,
                    sigma_t=sigma_score_int,
                    source_ca_mean=source_ca_mean,
                )
            )
        else:
            endpoint_per_graph, endpoint_valid = (
                self._masked_residue_smooth_l1_per_graph(
                    r_interface_X[-1],
                    gt_interface_X,
                    interface_atom_mask,
                    interface_batch_id,
                )
            )
            if endpoint_valid.any():
                interface_loss = endpoint_per_graph[
                    endpoint_valid
                ].mean()
            else:
                interface_loss = pred_X.new_tensor(0.0)

            zero = interface_loss.detach() * 0.0
            scorefm_details = {
                "scorefm_total": interface_loss.detach(),
                "scorefm_endpoint": interface_loss.detach(),
                "scorefm_dsm": zero,
                "scorefm_dsm_rate": zero,
            }

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
        loss = (
            self.seq_ce_weight * snll
            + struct_loss
            + dock_loss
            + (0 if pdev_loss is None else pdev_loss)
        )

        self._clean_batch_constants()

        # AAR and conditioning diagnostics.
        with torch.no_grad():
            if smask.any():
                aa_hit = pred_S[smask] == true_S[smask]
                aar = aa_hit.float().mean()
            else:
                aar = X.new_tensor(0.0)

            diag = {
                "seq_ce_weight": torch.as_tensor(self.seq_ce_weight, device=X.device),
                "scorefm_state_path": torch.as_tensor(
                    1.0 if state_path else 0.0, device=X.device
                ),
                "coord_pep_as_condition": torch.as_tensor(
                    1.0 if getattr(self, "coord_pep_as_condition", False) else 0.0,
                    device=X.device
                ),
                "seq_input_mode_state": torch.as_tensor(
                    1.0 if self.seq_input_mode == "state" else 0.0, device=X.device
                ),
                "seq_input_mode_pep_condition": torch.as_tensor(
                    1.0 if self.seq_input_mode == "pep_condition" else 0.0, device=X.device
                ),
                "t_mean": t_graph.detach().float().mean(),
                "t_min": t_graph.detach().float().min(),
                "t_max": t_graph.detach().float().max(),
            }
            for key, value in self._latest_condition_diagnostics.items():
                diag[key] = value.detach()

            valid_pep = (
                S_pep is not None
                and S_pep.numel() == int(paratope_mask.sum().item())
            )
            if valid_pep and smask[paratope_mask].any():
                pep_full = torch.empty_like(S)
                pep_full.copy_(S)
                pep_full[paratope_mask] = S_pep.to(device=S.device, dtype=torch.long)
                pep_mask = smask
                pred_pep_hit = pred_S[pep_mask] == pep_full[pep_mask]
                pep_native_hit = pep_full[pep_mask] == true_S[pep_mask]
                diag["seq_pred_vs_pep_aar"] = pred_pep_hit.float().mean()
                diag["seq_pep_vs_native_aar"] = pep_native_hit.float().mean()

            # Measure the proposal's own coordinate quality.  Without this
            # diagnostic, an improvement or degradation from coordinate
            # conditioning cannot be attributed to the condition mechanism
            # versus the quality of X_pep itself.
            valid_coord_pep = (
                X_pep is not None
                and X_pep.shape == gt_interface_X.shape
            )
            if valid_coord_pep:
                pep_raw = X_pep.to(
                    device=gt_interface_X.device,
                    dtype=gt_interface_X.dtype,
                )
                proposal_backbone = pep_raw[:, :3]
                proposal_valid = (
                    torch.isfinite(proposal_backbone)
                    .all(dim=-1)
                    .all(dim=-1)
                    & (
                        proposal_backbone.abs()
                        .sum(dim=-1)
                        .sum(dim=-1)
                        > self.scorefm_eps
                    )
                )
                if proposal_valid.any():
                    ca_idx = 1 if X_pep.shape[1] > 1 else 0
                    pep_ca = pep_raw[:, ca_idx]
                    native_ca = gt_interface_X[:, ca_idx]
                    pep_ca_sq = ((pep_ca - native_ca) ** 2).sum(dim=-1)

                    valid_graph_id = interface_batch_id[proposal_valid]
                    n_graph = int(interface_batch_id.max().item()) + 1
                    pep_ca_sum = torch.zeros(
                        n_graph,
                        device=pep_ca_sq.device,
                        dtype=pep_ca_sq.dtype,
                    )
                    pep_ca_count = torch.zeros(
                        n_graph,
                        device=pep_ca_sq.device,
                        dtype=pep_ca_sq.dtype,
                    )
                    pep_ca_sum.scatter_add_(
                        0,
                        valid_graph_id,
                        pep_ca_sq[proposal_valid],
                    )
                    pep_ca_count.scatter_add_(
                        0,
                        valid_graph_id,
                        torch.ones_like(pep_ca_sq[proposal_valid]),
                    )
                    valid_graph = pep_ca_count > 0
                    pep_ca_mse_graph = (
                        pep_ca_sum
                        / pep_ca_count.clamp_min(1.0)
                    )
                    diag["coord_pep_to_native_ca_rmsd"] = torch.sqrt(
                        pep_ca_mse_graph[valid_graph].clamp_min(0.0)
                    ).mean()
                    diag["coord_pep_valid_rate"] = (
                        proposal_valid.float().mean()
                    )

            self.last_abflow_diagnostics = {
                k: v.detach() if torch.is_tensor(v) else v for k, v in diag.items()
            }

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss)


    def _sampling_time_grid(self, n_steps, device, dtype):
        """Return true interval boundaries [0, ..., 1].

        Velocity is evaluated at left endpoints t_i<1. The final model readout
        is queried at t=1 without evaluating an analytic score denominator.
        """
        n_steps = max(1, int(n_steps))
        return torch.linspace(
            0.0, 1.0, steps=n_steps + 1,
            device=device, dtype=dtype
        )

    @torch.no_grad()
    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
               surface, residue_pos, template, lengths, n_steps=10,
               init_noise=None, return_hidden=False, show_progress=False,
               progress_desc=None):
        if not bool(getattr(self, "scorefm_state_path", True)):
            return self.struct_sample(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                init_noise=init_noise, return_hidden=return_hidden
            )

        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]
            if X_pep is not None:
                X_pep = X_pep[:, :4]

        gen_X, gen_S = X.clone(), S.clone()
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size_raw = self.batch_constants['batch_size']
        batch_size = (
            int(batch_size_raw.item())
            if torch.is_tensor(batch_size_raw)
            else int(batch_size_raw)
        )
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.full(
            (batch_size,), 1e10, dtype=torch.float, device=X.device
        )
        interface_cmask = paratope_mask[cmask]

        interface_X, interface_S = self.init_interface(
            X, S, paratope_mask, batch_id, init_noise=init_noise
        )
        time_grid = self._sampling_time_grid(
            n_steps, device=X.device, dtype=X.dtype
        )
        Xt = interface_X.clone()
        St = interface_S.clone()

        step_iter = range(n_steps)
        if show_progress:
            step_iter = tqdm(
                step_iter, total=n_steps,
                desc=progress_desc or 'Sampling ODE',
                leave=False, dynamic_ncols=True
            )

        for i in step_iter:
            t = time_grid[i]
            t_next = time_grid[i + 1]
            dt = t_next - t
            flow_t_graph = t.reshape(1).expand(batch_size)
            if show_progress and hasattr(step_iter, 'set_postfix'):
                step_iter.set_postfix(t=f'{float(t):.2f}')

            sequence_state_for_model = St if not self.struct_only else None
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                interface_init=Xt,
                sequence_init=sequence_state_for_model,
                flow_t=flow_t_graph
            )
            pred_clean_X = r_interface_X[-1]

            raw_residual = pred_clean_X - Xt
            if self.scorefm_sampler_mode == "residual":
                dX = raw_residual
            elif self.scorefm_sampler_mode == "bridge":
                sigma_t = (1.0 - t).clamp_min(
                    self.scorefm_min_sigma
                )
                dX = raw_residual / sigma_t
            else:
                raise ValueError(
                    f"Unknown sampler mode: {self.scorefm_sampler_mode}"
                )

            Xt = Xt + dX * dt

            if not self.struct_only:
                cur_logits = r_pred_S_logits[-1][0][paratope_mask]
                cur_logits = cur_logits - cur_logits.max(
                    dim=-1, keepdim=True
                )[0]
                cur_probs = F.softmax(cur_logits, dim=-1)
                refresh_prob = min(
                    1.0,
                    float(dt) / max(1e-8, 1.0 - float(t))
                )
                proposed_S = torch.multinomial(
                    cur_probs.clamp_min(1e-8), num_samples=1
                ).squeeze(-1)
                refresh = (
                    torch.rand(St.shape, device=St.device) < refresh_prob
                )
                refresh = refresh & smask[paratope_mask]
                St = torch.where(refresh, proposed_S, St)

        # Never mutate caller-owned input tensors during final readout.
        X_state = X.clone()
        S_state = S.clone()
        X_state[paratope_mask] = Xt
        S_state[paratope_mask] = St

        final_t = time_grid[-1].detach()
        final_flow_t_graph = final_t.reshape(1).expand(batch_size)
        sequence_state_for_model = St if not self.struct_only else None
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(
            X_state, S_state, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            interface_init=Xt,
            sequence_init=sequence_state_for_model,
            flow_t=final_flow_t_graph
        )

        if not self.struct_only:
            S_logits = r_pred_S_logits[-1][0][smask]
            if S_logits.shape[0] > 0:
                S_probs = torch.softmax(
                    S_logits, dim=-1
                ).max(dim=-1)[0]
                nlls = -torch.log(S_probs.clamp_min(1e-8))
                metric = scatter_mean(
                    nlls, s_batch_id, dim=0, dim_size=batch_size
                )
            else:
                metric = best_metric.new_zeros(batch_size)
        else:
            metric = scatter_mean(
                prmsd[interface_cmask], interface_batch_id,
                dim=0, dim_size=batch_size
            )

        update = metric < best_metric
        cupdate = cmask & update[batch_id]
        supdate = smask & update[batch_id]
        best_metric[update] = metric[update]
        gen_X[cupdate] = pred_X[cupdate]
        if not self.struct_only:
            gen_S[supdate] = pred_S[supdate]

        interface_X_final = r_interface_X[-1]
        for b in range(batch_size):
            if not update[b]:
                continue
            is_cur_graph = batch_id == b
            current_paratope = is_cur_graph & paratope_mask
            ori_cdr = gen_X[current_paratope][:, :4]
            pred_cdr = interface_X_final[
                interface_batch_id == b
            ][:, :4]
            _, R, trans = kabsch_torch(
                ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3)
            )
            is_cur_ab = is_cur_graph & is_ab
            gen_X[is_cur_ab] = torch.matmul(
                gen_X[is_cur_ab], R.T
            ) + trans

        self._clean_batch_constants()
        if return_hidden:
            return gen_X, gen_S, best_metric, H
        return gen_X, gen_S, best_metric

    def struct_sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths, init_noise=None, return_hidden=False):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            if X_pep is not None:
                X_pep = X_pep[:, :4]
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        batch_size = int(batch_size.item()) if torch.is_tensor(batch_size) else int(batch_size)
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
                metric = scatter_mean(nlls, s_batch_id, dim=0, dim_size=batch_size)  # [batch_size]
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id, dim=0, dim_size=batch_size)  # [batch_size]

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
            return gen_X, gen_S, best_metric, H
        return gen_X, gen_S, best_metric

    def sample_many(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths,
                    n_samples=5, n_steps=20, return_hidden=False, show_progress=False):
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
                    template, lengths, n_steps=n_steps, init_noise=init_noise, return_hidden=True,
                    show_progress=show_progress, progress_desc=f'Sample {i + 1}/{n_samples} ODE'
                )
                list_H.append(H)
            else:
                gen_X, gen_S, metric = self.sample(
                    X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos,
                    template, lengths, n_steps=n_steps, init_noise=init_noise,
                    show_progress=show_progress, progress_desc=f'Sample {i + 1}/{n_samples} ODE'
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