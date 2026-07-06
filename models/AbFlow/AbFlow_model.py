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
        self.scorefm_t_threshold = 0.2

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
        self.use_scorefm = True  # current Score-FM code path

        # self.timing_stats = {
        #     'surface_processing': 0.0,
        #     'sme_encoding': 0.0,
        #     'count': 0
        # }
        
        # =========================================================
        # Ablation controls
        # =========================================================
        # Loss modes:
        #   off        : no new DTM/ScoreFM objective, only base AbFlow losses
        #   x1         : clean endpoint reconstruction only
        #   dsm        : analytic score DSM only
        #   hybrid     : thresholded DSM/x1 hybrid + x1
        #   velocity   : velocity identity only
        #   x1_vel     : x1 + velocity
        #   dtm_core   : x1 + hybrid DSM/x1 + velocity, no geometry regularizers
        #   geom_only  : local/contact/clash only
        #   no_contact : full but contact loss disabled
        #   no_clash   : full but clash losses disabled
        #   no_geom    : same as dtm_core
        #   full       : current full objective
        self.scorefm_loss_mode = _env_str("ABFLOW_SCOREFM_LOSS_MODE", "full").lower()

        # Path/time controls.
        # ABFLOW_SCOREFM_PER_SAMPLE_T=on follows the AbX practice: every complex
        # receives its own continuous time t instead of sharing one scalar for the
        # whole batch. This reduces timestep-gradient variance and prevents a batch
        # from being dominated by a single noise level.
        self.scorefm_per_sample_t = _env_flag("ABFLOW_SCOREFM_PER_SAMPLE_T", True)

        # t sampling schedule for the coordinate path:
        #   uniform    : t ~ U(0, 1-sigma_min)
        #   low_t      : t = (1-sigma_min) * u^2, biases training toward harder
        #                low-t states that contain less native endpoint information
        #   stratified : approximately covers the whole time interval in each batch
        self.scorefm_t_sampling = _env_str("ABFLOW_SCOREFM_T_SAMPLING", "uniform").lower()

        # Flow-time conditioning. When enabled, a sinusoidal embedding of t is
        # added to the initial residue feature H_0 before the AbFlow encoder.
        # This is the minimal AbFlow analogue of AbX's time-conditioned Seqformer.
        self.scorefm_time_embed = _env_flag("ABFLOW_SCOREFM_TIME_EMBED", False)
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, embed_size),
        )

        # bridge:
        #     dX = pred_clean_X + sigma_t * pred_score
        #        = (pred_clean_X - X_t) / sigma_t
        #     This recovers the FM velocity X_1 - X_0 when pred_clean_X = X_1.
        # residual:
        #     dX = pred_clean_X - X_t
        #     Conservative endpoint residual, not the exact FM velocity.
        # damped_bridge:
        #     dX = sigma_t^p * bridge_velocity
        #     Reduces bridge magnitude near late time if p > 0.
        # blend:
        #     Interpolates between residual and bridge velocity.
        self.scorefm_sampler_mode = _env_str("ABFLOW_SCOREFM_SAMPLER_MODE", "bridge").lower()
        self.scorefm_bridge_damping_power = _env_float("ABFLOW_SCOREFM_DAMPING_POWER", 0.0)
        self.scorefm_bridge_blend = _env_float("ABFLOW_SCOREFM_BRIDGE_BLEND", 1.0)

        # A single generative state is used for both global and shadow-interface
        # paths. Peptide predictions may condition the initial distribution, but
        # they never overwrite X_t/S_t inside _forward.
        self.coord_pep_prior_weight = _env_float(
            "ABFLOW_COORD_PEP_PRIOR_WEIGHT", 0.5
        )
        # Coordinate prior mode:
        #   blend:
        #       Existing behavior. X0 <- (1-rho) X_noise + rho X_pep.
        #       With rho=1, this becomes the deterministic hard prior X0=X_pep.
        #   conditional_gaussian:
        #       Peptide-informed Gaussian base. X0 <- X_pep + sigma_pep * eps.
        #       This keeps stochasticity while centering the base distribution
        #       around the paratope prior.
        self.coord_pep_prior_mode = _env_str(
            "ABFLOW_COORD_PEP_PRIOR_MODE", "blend"
        ).lower()
        if self.coord_pep_prior_mode not in {"blend", "conditional_gaussian"}:
            raise ValueError(
                "Unknown ABFLOW_COORD_PEP_PRIOR_MODE="
                f"{self.coord_pep_prior_mode}. Choose from blend, conditional_gaussian."
            )
        self.coord_pep_prior_sigma = _env_float(
            "ABFLOW_COORD_PEP_PRIOR_SIGMA", 1.0
        )
        self.seq_pep_prior_weight = _env_float(
            "ABFLOW_SEQ_PEP_PRIOR_WEIGHT", 0.5
        )
        # state: use the sampled categorical state S_t as the sequence input.
        # pep_condition: emulate original AbFlow persistent peptide conditioning;
        #                S_pep is visible at every _forward call even when S_t
        #                exists. This intentionally breaks pure state-only input.
        self.seq_input_mode = _env_str("ABFLOW_SEQ_INPUT_MODE", "state").lower()
        if self.seq_input_mode not in {"state", "pep_condition"}:
            raise ValueError(
                "Unknown ABFLOW_SEQ_INPUT_MODE="
                f"{self.seq_input_mode}. Choose from state, pep_condition."
            )
        self.seq_ce_weight = _env_float("ABFLOW_SEQ_CE_WEIGHT", 1.0)
        self.last_abflow_diagnostics = {}

        # Allow component weights to be overridden from shell scripts.
        self.scorefm_loss_weight = _env_float("ABFLOW_SCOREFM_LOSS_WEIGHT", self.scorefm_loss_weight)
        self.scorefm_velocity_weight = _env_float("ABFLOW_SCOREFM_VELOCITY_WEIGHT", self.scorefm_velocity_weight)
        self.scorefm_dsm_weight = _env_float("ABFLOW_SCOREFM_DSM_WEIGHT", self.scorefm_dsm_weight)
        self.scorefm_x1_weight = _env_float("ABFLOW_SCOREFM_X1_WEIGHT", self.scorefm_x1_weight)
        self.scorefm_local_dist_weight = _env_float("ABFLOW_SCOREFM_LOCAL_DIST_WEIGHT", self.scorefm_local_dist_weight)
        self.scorefm_interface_contact_weight = _env_float("ABFLOW_SCOREFM_CONTACT_WEIGHT", self.scorefm_interface_contact_weight)
        self.scorefm_inter_clash_weight = _env_float("ABFLOW_SCOREFM_INTER_CLASH_WEIGHT", self.scorefm_inter_clash_weight)
        self.scorefm_intra_clash_weight = _env_float("ABFLOW_SCOREFM_INTRA_CLASH_WEIGHT", self.scorefm_intra_clash_weight)

        if self.scorefm_loss_mode in {"off", "none", "base"}:
            self.use_scorefm = False


    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S
    
    def replace_pep(self, X, S, paratope_mask, X_pep, S_pep,
                    replace_seq=True, replace_struct=True):
        """Legacy endpoint conditioning used only when no explicit state exists."""
        if (
            replace_seq
            and getattr(self, 'pep_seq', True)
            and S_pep is not None
            and S_pep.numel() == int(paratope_mask.sum().item())
        ):
            S[paratope_mask] = S_pep
        if (
            replace_struct
            and getattr(self, 'pep_struct', True)
            and X_pep is not None
            and X_pep.shape == X[paratope_mask].shape
            and bool(torch.any(X_pep != 0))
        ):
            X[paratope_mask] = X_pep
        return X, S

    @torch.no_grad()
    def _condition_initial_interface(self, interface_X, interface_S, X_pep, S_pep):
        """Condition the base distribution without creating a second state path.

        Coordinate prior:
            X_0 <- (1-rho_x) X_noise + rho_x X_pep
        or, when ABFLOW_COORD_PEP_PRIOR_MODE=conditional_gaussian:
            X_0 <- X_pep + sigma_pep * eps

        Sequence prior:
            pi_0 = (1-rho_s) Uniform + rho_s delta(S_pep)
            S_0 ~ pi_0

        The returned X_0/S_0 remain the only generative state consumed by the
        network. Setting either weight to zero recovers the unconditioned base.
        """
        if (
            getattr(self, 'pep_struct', True)
            and X_pep is not None
            and X_pep.shape == interface_X.shape
            and bool(torch.any(X_pep != 0))
        ):
            pep_X = X_pep.to(device=interface_X.device, dtype=interface_X.dtype)
            if self.coord_pep_prior_mode == "conditional_gaussian":
                sigma = max(0.0, float(self.coord_pep_prior_sigma))
                interface_X = pep_X + sigma * torch.randn_like(interface_X)
            else:
                rho_x = max(0.0, min(1.0, float(self.coord_pep_prior_weight)))
                interface_X = (1.0 - rho_x) * interface_X + rho_x * pep_X

        if (
            not self.struct_only
            and getattr(self, 'pep_seq', True)
            and S_pep is not None
            and S_pep.shape == interface_S.shape
        ):
            rho_s = max(0.0, min(1.0, float(self.seq_pep_prior_weight)))
            pep_S = S_pep.to(device=interface_S.device, dtype=torch.long)
            valid = torch.logical_and(pep_S >= 0, pep_S < self.num_classes)
            use_pep = torch.logical_and(
                valid,
                torch.rand(interface_S.shape, device=interface_S.device) < rho_s
            )
            interface_S = torch.where(use_pep, pep_S, interface_S)

        return interface_X, interface_S

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
        

    def _sample_flow_times(self, batch_size, device, dtype=torch.float32):
        """Sample continuous flow times for DTM/ScoreFM path training.

        Returns:
            t_graph: [B] if per-sample time is enabled, otherwise [1]. Values
            are upper-bounded by 1 - sigma_min so analytic scores never divide
            by a vanishing sigma_t.
        """
        n = int(batch_size) if getattr(self, 'scorefm_per_sample_t', False) else 1
        max_t = 1.0 - float(getattr(self, 'scorefm_min_sigma', 5e-2))
        mode = getattr(self, 'scorefm_t_sampling', 'uniform')
        if mode == 'uniform':
            u = torch.rand(n, device=device, dtype=dtype)
            t = max_t * u
        elif mode in {'low_t', 'low', 'square'}:
            u = torch.rand(n, device=device, dtype=dtype)
            t = max_t * (u ** 2)
        elif mode in {'stratified', 'strat'}:
            # Stratification is only meaningful for per-sample t. For n=1 it
            # reduces to uniform sampling over the full interval.
            if n == 1:
                t = max_t * torch.rand(n, device=device, dtype=dtype)
            else:
                base = (torch.arange(n, device=device, dtype=dtype) + torch.rand(n, device=device, dtype=dtype)) / float(n)
                # Randomly permute bins so graph order never correlates with t.
                perm = torch.randperm(n, device=device)
                t = max_t * base[perm]
        else:
            raise ValueError(
                f"Unknown ABFLOW_SCOREFM_T_SAMPLING={mode}. "
                "Choose from uniform, low_t, stratified."
            )
        return t.clamp(min=0.0, max=max_t)

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
        if not hasattr(self, 'flow_time_mlp'):
            raise RuntimeError(
                "ABFLOW_SCOREFM_TIME_EMBED is enabled but this checkpoint/model "
                "does not contain flow_time_mlp. Train with AbFlow_model_v7_dtm.py "
                "from scratch, or instantiate the new model and load old weights with strict=False."
            )
        t_emb = get_timestep_embedding(flow_t, H_0.shape[-1]).to(dtype=H_0.dtype, device=H_0.device)
        t_emb = self.flow_time_mlp(t_emb)
        return t_emb[batch_id]

    def message_passing(self, X, S, residue_pos, interface_X, surf, paratope_mask, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None, flow_t=None):
        # embeddings, hidden state, (internal edges, external edges), (A : c * d, w : c * 1)
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask)

        # Minimal AbFlow time conditioning. AbX injects batch['t'] into both
        # sequence and pair features before its Seqformer; here we add an
        # equivalent graph-level time signal to AbFlow's residue embeddings.
        # This keeps the modification local and makes the ablation clean.
        time_emb = self._flow_time_embedding_for_residues(flow_t, batch_id, H_0)
        if time_emb is not None:
            H_0 = H_0 + time_emb

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
        # Replace the generated region before constructing any geometric KNN
        # graph. This makes intra-paratope and paratope-antigen edges depend on
        # the same current state X_t.
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

        `t` and `sigma_t` can be scalars or [N_interface,1,1] tensors.
        """
        t = torch.as_tensor(t, device=Xt.device, dtype=Xt.dtype)
        sigma_t = torch.as_tensor(sigma_t, device=Xt.device, dtype=Xt.dtype)
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
        ag_mask_full = torch.logical_and(
            segment_ids == self.aa_feature.ag_seg_id,
            true_S != self.aa_feature.boa_idx
        )

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
        ag_mask_full = torch.logical_and(
            segment_ids == self.aa_feature.ag_seg_id,
            true_S != self.aa_feature.boa_idx
        )

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
        """Ablation-aware coordinate DTM / ScoreFM objective.

        The goal is to identify which part of the new objective helps or hurts:
          - x1: clean endpoint reconstruction
          - dsm: analytically induced score matching
          - velocity: score-to-velocity identity
          - geometry: local distance/contact/clash feasibility terms
        """
        mode = self.scorefm_loss_mode
        zero = pred_clean_X.new_tensor(0.0)

        # Fast exit for pure AbFlow baseline under the v4 code path.
        if mode in {"off", "none", "base"} or (not self.use_scorefm) or self.scorefm_loss_weight == 0:
            details = {
                "scorefm_total": zero.detach(),
                "scorefm_dsm": zero.detach(),
                "scorefm_x1": zero.detach(),
                "scorefm_hybrid": zero.detach(),
                "scorefm_velocity": zero.detach(),
                "scorefm_local_dist": zero.detach(),
                "scorefm_interface_contact": zero.detach(),
                "scorefm_inter_clash": zero.detach(),
                "scorefm_intra_clash": zero.detach(),
            }
            return zero, details

        # 1. Analytic scores induced by the same Gaussian coordinate path.
        gt_score = self._coord_score_from_clean(Xt, X1, t, sigma_t).detach()
        pred_score = self._coord_score_from_clean(Xt, pred_clean_X, t, sigma_t)

        # 2. ABX-style scaled DSM.
        # score_scaling = 1 / sigma_t, so (pred_score - gt_score) / score_scaling
        # equals sigma_t * score residual. This avoids excessive late-time score scale.
        score_scaling = 1.0 / sigma_t.clamp_min(self.scorefm_min_sigma)
        dsm_diff = (pred_score - gt_score) / score_scaling
        dsm_loss = self._masked_residue_mse(dsm_diff, atom_mask, interface_batch_id)

        # 3. Clean endpoint reconstruction.
        x1_loss = self._masked_residue_smooth_l1(pred_clean_X, X1, atom_mask, interface_batch_id)

        # 4. Hybrid DSM/x1 objective.
        # For high t, score matching becomes meaningful; for low t, x1 is more stable.
        # With per-sample t, use the fraction of high-t interface residues as a
        # smooth mixture weight so the loss remains a scalar and remains stable.
        t_tensor = torch.as_tensor(t, device=pred_clean_X.device, dtype=pred_clean_X.dtype)
        high_t_weight = (t_tensor.reshape(-1) > pred_clean_X.new_tensor(self.scorefm_t_threshold)).to(pred_clean_X.dtype).mean()
        score_or_x1 = high_t_weight * dsm_loss + (1.0 - high_t_weight) * x1_loss

        # 5. Velocity identity for the same path:
        # v_theta = Xhat_1 + sigma_t * s_theta
        # true velocity for X_t = sigma_t X_0 + t X_1 is X_1 - X_0.
        pred_v = pred_clean_X + sigma_t * pred_score
        true_v = X1 - X0
        velocity_loss = self._masked_residue_smooth_l1(pred_v, true_v, atom_mask, interface_batch_id)

        # 6. Geometry terms are only computed when needed.
        need_geometry = mode in {
            "full",
            "geom_only",
            "no_contact",
            "no_clash",
        }

        if need_geometry:
            local_dist_loss = self._local_ca_distance_loss(pred_clean_X, X1, interface_batch_id)
            interface_contact_loss = self._interface_contact_bce_loss(
                pred_clean_X, X1, true_X, true_S, paratope_mask, batch_id,
                segment_ids, interface_batch_id, atom_mask)
            inter_clash_loss = self._interface_clash_loss(
                pred_clean_X, true_X, true_S, batch_id, segment_ids,
                interface_batch_id, atom_mask)
            intra_clash_loss = self._intra_paratope_clash_loss(
                pred_clean_X, atom_mask, interface_batch_id)
        else:
            local_dist_loss = zero
            interface_contact_loss = zero
            inter_clash_loss = zero
            intra_clash_loss = zero

        # 7. Select objective according to ablation mode.
        if mode == "x1":
            objective = self.scorefm_x1_weight * x1_loss

        elif mode == "dsm":
            objective = self.scorefm_dsm_weight * dsm_loss

        elif mode == "hybrid":
            objective = (
                self.scorefm_dsm_weight * score_or_x1
                + self.scorefm_x1_weight * x1_loss
            )

        elif mode == "velocity":
            objective = self.scorefm_velocity_weight * velocity_loss

        elif mode == "x1_vel":
            objective = (
                self.scorefm_x1_weight * x1_loss
                + self.scorefm_velocity_weight * velocity_loss
            )

        elif mode in {"dtm_core", "no_geom"}:
            objective = (
                self.scorefm_velocity_weight * velocity_loss
                + self.scorefm_dsm_weight * score_or_x1
                + self.scorefm_x1_weight * x1_loss
            )

        elif mode == "geom_only":
            objective = (
                self.scorefm_local_dist_weight * local_dist_loss
                + self.scorefm_interface_contact_weight * interface_contact_loss
                + self.scorefm_inter_clash_weight * inter_clash_loss
                + self.scorefm_intra_clash_weight * intra_clash_loss
            )

        elif mode == "no_contact":
            objective = (
                self.scorefm_velocity_weight * velocity_loss
                + self.scorefm_dsm_weight * score_or_x1
                + self.scorefm_x1_weight * x1_loss
                + self.scorefm_local_dist_weight * local_dist_loss
                + self.scorefm_inter_clash_weight * inter_clash_loss
                + self.scorefm_intra_clash_weight * intra_clash_loss
            )

        elif mode == "no_clash":
            objective = (
                self.scorefm_velocity_weight * velocity_loss
                + self.scorefm_dsm_weight * score_or_x1
                + self.scorefm_x1_weight * x1_loss
                + self.scorefm_local_dist_weight * local_dist_loss
                + self.scorefm_interface_contact_weight * interface_contact_loss
            )

        elif mode == "full":
            objective = (
                self.scorefm_velocity_weight * velocity_loss
                + self.scorefm_dsm_weight * score_or_x1
                + self.scorefm_x1_weight * x1_loss
                + self.scorefm_local_dist_weight * local_dist_loss
                + self.scorefm_interface_contact_weight * interface_contact_loss
                + self.scorefm_inter_clash_weight * inter_clash_loss
                + self.scorefm_intra_clash_weight * intra_clash_loss
            )

        else:
            raise ValueError(
                f"Unknown ABFLOW_SCOREFM_LOSS_MODE={mode}. "
                "Choose from off, x1, dsm, hybrid, velocity, x1_vel, "
                "dtm_core, geom_only, no_contact, no_clash, no_geom, full."
            )

        total = self.scorefm_loss_weight * objective

        details = {
            "scorefm_total": total.detach(),
            "scorefm_dsm": dsm_loss.detach(),
            "scorefm_x1": x1_loss.detach(),
            "scorefm_hybrid": score_or_x1.detach(),
            "scorefm_velocity": velocity_loss.detach(),
            "scorefm_local_dist": local_dist_loss.detach(),
            "scorefm_interface_contact": interface_contact_loss.detach(),
            "scorefm_inter_clash": inter_clash_loss.detach(),
            "scorefm_intra_clash": intra_clash_loss.detach(),
        }
        return total, details

    def _forward(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                 surface, residue_pos, template, lengths, init_noise=None,
                 interface_init=None, sequence_init=None, flow_t=None):
        
        batch_id = self.batch_constants['batch_id']
        X = X.clone()
        S = S.clone()
        surface = surface.clone()

        # mask sequence and initialize coordinates with template
        X, S = self.init_mask(X, S, cmask, smask, template)
        
        # Peptide coordinates are legacy conditioning only when X_t is absent.
        # Sequence conditioning is controlled separately:
        #   state         -> model sees the sampled categorical state S_t
        #   pep_condition -> model sees S_pep persistently, matching original AbFlow
        X, S = self.replace_pep(
            X, S, paratope_mask, X_pep, S_pep,
            replace_seq=(
                sequence_init is None or self.seq_input_mode == "pep_condition"
            ),
            replace_struct=interface_init is None,
        )
        if interface_init is not None:
            X[paratope_mask] = interface_init.to(device=X.device, dtype=X.dtype)
        if sequence_init is not None and self.seq_input_mode == "state":
            S[paratope_mask] = sequence_init.to(device=S.device, dtype=torch.long)

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
            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(
                X, S, residue_pos, interface_X, surface, paratope_mask, batch_id,
                t, memory_H, pred_S_dist, smask, flow_t=flow_t
            )
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

        # Sample X_0 from exactly the same initialization distribution used at
        # inference. Do not align or permute X_0 with the native X_1, because
        # the native structure is unavailable at generation time.
        interface_X, interface_S = self.init_interface(
            X, S, paratope_mask, batch_id
        )
        interface_X, interface_S = self._condition_initial_interface(
            interface_X, interface_S, X_pep, S_pep
        )
        
        # Continuous flow time. Avoid sigma_t = 1 - t being too small because
        # the analytic score contains 1 / sigma_t^2.
        # Following AbX-style diffusion training, optionally sample one t per
        # complex instead of one scalar for the whole batch. This is a clean
        # ablation controlled by ABFLOW_SCOREFM_PER_SAMPLE_T.
        batch_size = int(self.batch_constants['batch_size'].item()) if torch.is_tensor(self.batch_constants['batch_size']) else int(self.batch_constants['batch_size'])
        interface_batch_id = self.batch_constants['interface_batch_id']
        t_graph = self._sample_flow_times(batch_size, device=X.device, dtype=X.dtype)
        sigma_graph = (1.0 - t_graph).clamp_min(self.scorefm_min_sigma)
        t_int = self._time_for_interface(t_graph, interface_batch_id, interface_X)
        sigma_int = self._time_for_interface(sigma_graph, interface_batch_id, interface_X)

        # Coordinate-only flow path: X_t = (1 - t) X_0 + t X_1.
        # Sequence labels remain discrete and are handled by the CE loss; never
        # linearly interpolate amino-acid indices.
        Xt = sigma_int * interface_X + t_int * gt_interface_X
        St = self._sample_categorical_path(
            true_S[paratope_mask], interface_S, t_graph, interface_batch_id,
            corrupt_mask=smask[paratope_mask],
        )

        # get results
        # X_t and S_t are the actual state seen by every model branch.
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(
            X, S, cmask, smask, paratope_mask, X_pep, S_pep,
            surface, residue_pos, template, lengths,
            interface_init=Xt, sequence_init=St, flow_t=t_graph
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
            X0=interface_X,
            X1=gt_interface_X,
            pred_clean_X=r_interface_X[-1],
            atom_mask=interface_atom_mask,
            true_X=true_X,
            true_S=true_S,
            paratope_mask=paratope_mask,
            batch_id=batch_id,
            segment_ids=self.batch_constants['segment_ids'],
            interface_batch_id=self.batch_constants['interface_batch_id'],
            t=t_int,
            sigma_t=sigma_int,
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
        loss = self.seq_ce_weight * snll + struct_loss + dock_loss + flow_loss + (0 if pdev_loss is None else pdev_loss)
        # loss = snll + struct_loss + dock_loss + flow_loss + (0 if pdev_loss is None else pdev_loss)

        self._clean_batch_constants()

        # AAR and sequence-conditioning diagnostics.
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]
            diag = {
                "seq_ce_weight": torch.as_tensor(self.seq_ce_weight, device=X.device),
                "coord_prior_mode_blend": torch.as_tensor(
                    1.0 if self.coord_pep_prior_mode == "blend" else 0.0, device=X.device
                ),
                "coord_prior_mode_conditional_gaussian": torch.as_tensor(
                    1.0 if self.coord_pep_prior_mode == "conditional_gaussian" else 0.0, device=X.device
                ),
                "coord_pep_prior_sigma": torch.as_tensor(
                    self.coord_pep_prior_sigma, device=X.device
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
            self.last_abflow_diagnostics = {
                k: v.detach() if torch.is_tensor(v) else v for k, v in diag.items()
            }

        return loss, (snll, aar), (struct_loss, *struct_loss_details), (dock_loss, interface_loss, ed_loss, r_ed_losses), (pdev_loss, prmsd_loss)

    def sample(self, X, S, cmask, smask, paratope_mask, X_pep, S_pep, surface, residue_pos, template, lengths,
               n_steps=10, init_noise=None, return_hidden=False, show_progress=False, progress_desc=None):
        
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            X_pep = X_pep[:, :4]
        
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

        interface_X, interface_S = self.init_interface(
            X, S, paratope_mask, batch_id, init_noise=init_noise
        )
        interface_X, interface_S = self._condition_initial_interface(
            interface_X, interface_S, X_pep, S_pep
        )
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
            flow_t_graph = t.reshape(1).expand(batch_size)
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
                interface_init=Xt, sequence_init=St, flow_t=flow_t_graph
            )

            # Score-factorized velocity. Given predicted clean endpoint Xhat_1,
            # induced score s_theta = -(X_t - t Xhat_1) / sigma_t^2 and
            # v_theta = Xhat_1 + sigma_t * s_theta = (Xhat_1 - X_t) / sigma_t.
            sigma_t = (1.0 - t).clamp_min(self.scorefm_min_sigma)
            pred_clean_X = r_interface_X[-1]
            pred_score = self._coord_score_from_clean(Xt, pred_clean_X, t, sigma_t)

            # Sampler ablation:
            # residual:      original conservative endpoint residual update
            # bridge:        score-to-velocity bridge, v = Xhat_1 + sigma_t * score
            # damped_bridge: bridge velocity damped by sigma_t^p
            # blend:         interpolation between residual and bridge
            raw_residual = pred_clean_X - Xt
            bridge_velocity = pred_clean_X + sigma_t * pred_score

            sampler_mode = self.scorefm_sampler_mode
            if sampler_mode == "residual":
                dX = raw_residual

            elif sampler_mode == "bridge":
                dX = bridge_velocity

            elif sampler_mode == "damped_bridge":
                damping = sigma_t.clamp_min(self.scorefm_min_sigma).pow(
                    self.scorefm_bridge_damping_power
                )
                dX = damping * bridge_velocity

            elif sampler_mode == "blend":
                alpha = float(self.scorefm_bridge_blend)
                alpha = max(0.0, min(1.0, alpha))
                dX = (1.0 - alpha) * raw_residual + alpha * bridge_velocity

            else:
                raise ValueError(
                    f"Unknown ABFLOW_SCOREFM_SAMPLER_MODE={sampler_mode}. "
                    "Choose from residual, bridge, damped_bridge, blend."
                )
            if not self.struct_only:
                cur_logits = r_pred_S_logits[-1][0][paratope_mask]
                cur_logits = cur_logits - cur_logits.max(dim=-1, keepdim=True)[0]
                cur_probs = F.softmax(cur_logits, dim=-1)
                
            # Euler coordinate step.
            Xt = Xt + dX * dt
            if not self.struct_only:
                # Categorical stochastic interpolation. For q_t =
                # t*delta(clean)+(1-t)*pi_0, moving from t to t+dt refreshes a
                # residue from the predicted clean distribution with
                # probability dt/(1-t), otherwise retaining its current state.
                refresh_prob = min(1.0, dt / max(1e-8, 1.0 - float(t)))
                proposed_S = torch.multinomial(
                    cur_probs.clamp_min(1e-8), num_samples=1
                ).squeeze(-1)
                refresh = torch.rand(
                    St.shape, device=St.device
                ) < refresh_prob
                St = torch.where(refresh, proposed_S, St)
        
        X[paratope_mask] = Xt
        S[paratope_mask] = St
            
        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            # Use the final ODE state Xt as shadow-interface input instead of
            # reinitializing from noise.
            final_flow_t_graph = torch.full((batch_size,), 1.0 - self.scorefm_min_sigma, device=X.device, dtype=X.dtype)
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(
                X, S, cmask, smask, paratope_mask, X_pep, S_pep,
                surface, residue_pos, template, lengths,
                interface_init=Xt, sequence_init=St,
                flow_t=final_flow_t_graph
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
            X_pep = X_pep[:, :4]
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
