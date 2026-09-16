#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Core AbFlow representation boundary and R3 flow-matching components.

`NativeTrunk` owns the padded single/pair representation adapter.
`AbFlowR3Matcher` owns Cartesian R3 path/score/velocity algebra.
The top-level `AbFlowModel` imports these components instead of defining them.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import (
    CDR_TO_ENUM, UNKNOWN_AB_REGION_INDEX, imgt_region_index, normalize_regions,
)
from utils.nn_utils import (
    DistogramHead, SinglePairEncoder, _abflow_ca_fill_observed_mask,
    _atom14_chemical_mask, _atom14_exists_from_seq, _torsions_from_atom14,
    pseudo_beta_fn_v2,
)

class NativeTrunk(nn.Module):
    """Dense single/pair representation with an R05-only boundary adapter.

    The relational operators (compact residue geometry, PairEmbedding, time embedding
    and one-block Seqformer) are adapted from the supplied AbX implementation for R05.
    Only data routing is adapted:

    * AbFlow is flat/ragged, AbFlow is padded [B,L,...]; this class packs/unpacks.
    * ESM is intentionally absent because the project has a frozen no-ESM
      boundary. No Seqformer operator is removed.
    * AbFlow recycling is deliberately disabled. R05's three physical refinement
      rounds remain the sole recurrent mechanism in this experiment family.
    * ``fixed_mask`` retains the donor's information barrier: design-region structural
      features are masked in ResidueEmbedding/PairEmbedding. This is important
      here because R05 already exposes the current flow state through its native
      full-atom radial geometry; copying clean/native design geometry into the
      donor trunk would leak X1, while redundantly re-encoding the shadow Xt would
      create a dual-coordinate-state ambiguity between ctx and local branches.
    """
    # Widths are instance-level because  supports both donor and localized
    # profiles.  ``single_dim`` / ``pair_dim`` are derived from trunk.config in
    # __init__ and are the single source of truth for all downstream consumers.

    def __init__(self, representation_config, distogram_config=None,
                 forward_seed=271828):
        super().__init__()
        self.representation_config = representation_config
        self.trunk = SinglePairEncoder(representation_config)
        geometry_cfg = representation_config.get("geometry", {})
        self.torsion_norm_eps = float(geometry_cfg.get("torsion_norm_eps", 1e-12))
        self.ca_fill_tol2 = float(geometry_cfg.get("ca_fill_tol2", 1e-12))

        # R79 correctness barrier: once H3 geometry is opened to the relational
        # trunk, the generated task domain must not inherit native atom-resolution
        # metadata through xloss_mask.  Fixed scaffold/antigen remain observed
        # conditions; H3 atom support is inferred from the CURRENT model-visible
        # sequence/coordinates using the existing AbFlow CA-fill convention.
        round_state_cfg = representation_config.get("round_state_conditioning", {})
        self.task_atom_observation_source = str(
            round_state_cfg.get(
                "task_atom_observation_source", "native_xloss_mask"
            ) or "native_xloss_mask"
        ).strip().lower()
        if self.task_atom_observation_source not in {
            "native_xloss_mask", "current_state_ca_fill"
        }:
            raise ValueError(
                "round_state_conditioning.task_atom_observation_source must be "
                "'native_xloss_mask' or 'current_state_ca_fill', got "
                f"{self.task_atom_observation_source!r}."
            )

        self.single_dim = int(
            self.trunk.config.seq_channel + self.trunk.config.index_embed_size
        )
        self.pair_dim = int(
            self.trunk.config.pair_channel + 2 * self.trunk.config.index_embed_size
        )
        distogram_config = distogram_config or {}
        self.enable_distogram = float(
            distogram_config.get("weight", 0.0)
        ) > 0.0
        self.distogram_pair_scope = distogram_config.get(
            "pair_scope", "all_resolved"
        )
        self.distogram_bins = int(distogram_config.get("bins", 64))
        self.distogram_min = float(distogram_config.get("min_bin", 2.3125))
        self.distogram_max = float(distogram_config.get("max_bin", 21.6875))

        self.forward_seed = int(forward_seed)
        self.register_buffer(
            "_training_forward_call",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.distogram_head = (
            DistogramHead(
                pair_dim=self.pair_dim,
                num_bins=self.distogram_bins,
                first_break=self.distogram_min,
                last_break=self.distogram_max,
            )
            if self.enable_distogram else None
        )
        self.last_diagnostics = {}

    def _ordered_nodes(self, valid_mask, batch_id, is_antigen):
        """Use the complete antibody plus the dataset-defined epitope."""
        nodes, antibody_lens, antigen_counts = [], [], []
        B = int(batch_id.max().item()) + 1 if batch_id.numel() else 0
        for gid in range(B):
            g = valid_mask & (batch_id == gid)
            ab = torch.nonzero(g & (~is_antigen), as_tuple=False).flatten()
            ag = torch.nonzero(g & is_antigen, as_tuple=False).flatten()
            nodes.append(torch.cat([ab, ag], dim=0))
            antibody_lens.append(int(ab.numel()))
            antigen_counts.append(int(ag.numel()))
        return nodes, antibody_lens, antigen_counts

    @staticmethod
    def _remap_chain_ids(seg, mask):
        out = seg.new_zeros(seg.shape)
        for b in range(seg.shape[0]):
            seen = []
            for j in range(int(mask[b].sum().item())):
                value = int(seg[b, j].item())
                if value not in seen:
                    seen.append(value)
                out[b, j] = seen.index(value)
        return out

    @staticmethod
    def _cdr_definition(residue_index, chain_id, mask, antibody_len, cdr_type, design_mask):
        """Map packed antibody residues through the canonical IMGT authority.

        Runtime packing decides only which packed chain is heavy/light.  The
        numerical IMGT ranges and region ids live exclusively in ``configs.py``.
        """
        out = residue_index.new_full(residue_index.shape, UNKNOWN_AB_REGION_INDEX)
        B = residue_index.shape[0]
        for b in range(B):
            ab_n = min(int(antibody_len[b].item()), int(mask[b].sum().item()))
            for j in range(ab_n):
                cid = int(chain_id[b, j].item())
                chain_kind = "H" if cid == 0 else "L"
                out[b, j] = imgt_region_index(
                    int(residue_index[b, j].item()), chain_kind
                )

        names = normalize_regions(cdr_type)
        if len(names) == 1 and names[0] in CDR_TO_ENUM:
            out = torch.where(
                design_mask,
                torch.full_like(out, CDR_TO_ENUM[names[0]]),
                out,
            )
        return out

    def _pack(self, X, S, segment_ids, residue_pos, batch_id, valid_mask,
              is_antigen, design_mask, flow_t, cdr_type, atom_observed_mask=None,
              aux_task_mask=None, condition_design_geometry=False):
        nodes, ab_lens, antigen_counts = self._ordered_nodes(
            valid_mask, batch_id, is_antigen,
        )
        B = len(nodes)
        L = max([int(idx.numel()) for idx in nodes] or [0])
        if L == 0:
            raise RuntimeError("AbFlow trunk received no biological residues")
        if X.shape[1] != 14:
            raise ValueError(
                f" single/pair port requires the donor's formal 14-slot full-atom state; got {tuple(X.shape)}"
            )

        Sp = S.new_full((B, L), 20)
        Xp = X.new_zeros((B, L, 14, 3))
        Obsp = X.new_zeros((B, L, 14)) if atom_observed_mask is not None else None
        Seg = segment_ids.new_zeros((B, L))
        Rp = torch.zeros((B, L), device=X.device, dtype=torch.long)
        M = torch.zeros((B, L), device=X.device, dtype=torch.bool)
        Design = torch.zeros_like(M)
        AuxTask = torch.zeros_like(M)
        IsAg = torch.zeros_like(M)
        GI = S.new_full((B, L), -1)
        if aux_task_mask is not None:
            if tuple(aux_task_mask.shape) != (int(X.shape[0]),):
                raise ValueError(
                    f"AbFlow aux_task_mask shape mismatch: {tuple(aux_task_mask.shape)} "
                    f"vs residue count {(int(X.shape[0]),)}"
                )
        if atom_observed_mask is not None:
            if tuple(atom_observed_mask.shape) != tuple(X.shape[:-1]):
                raise ValueError(
                    f"AbFlow xloss_mask shape mismatch: {tuple(atom_observed_mask.shape)} "
                    f"vs X {tuple(X.shape)}"
                )
        for b, idx in enumerate(nodes):
            n = int(idx.numel())
            if n == 0:
                continue
            Sp[b, :n] = S[idx].long().clamp(0, 22)
            Xp[b, :n] = X[idx]
            if Obsp is not None:
                Obsp[b, :n] = atom_observed_mask[idx].to(dtype=Xp.dtype)
            Seg[b, :n] = segment_ids[idx]
            rp = residue_pos[idx]
            rp = rp[..., 0] if rp.dim() > 1 else rp
            # torch<=1.11 CUDA does not implement round() for Long tensors.
            # R05 residue_pos is normally already integer-valued; round only a
            # floating representation before the final semantic cast.
            if torch.is_floating_point(rp):
                rp = torch.round(rp)
            Rp[b, :n] = rp.to(dtype=torch.long)
            M[b, :n] = True
            Design[b, :n] = design_mask[idx].bool()
            if aux_task_mask is not None:
                AuxTask[b, :n] = aux_task_mask[idx].bool()
            IsAg[b, :n] = is_antigen[idx].bool()
            GI[b, :n] = idx

        Chain = self._remap_chain_ids(Seg, M)
        antibody_len = torch.tensor(ab_lens, device=X.device, dtype=torch.long)
        Cdr = self._cdr_definition(Rp, Chain, M, antibody_len, cdr_type, Design)
        #  formal atom-observation contract. AbFlow stores unresolved atom
        # slots as a copy of CA and records the true observation state in
        # xloss_mask; donor AbFlow stores unresolved atom14 coordinates as zero.
        # Translate only this private donor input representation. Parent R05 X,
        # EGNN geometry, losses and coordinate authority are untouched.
        # Do not let native H3 atom-observation/missingness become an
        # information side-channel after R79 opens H3 relational geometry.
        # The fixed context keeps its dataset observation mask because it is
        # observed at inference.  H3 replaces that mask by one inferred solely
        # from the current model-visible state.
        observation_for_encoder = Obsp
        if (
            bool(condition_design_geometry)
            and self.task_atom_observation_source == "current_state_ca_fill"
        ):
            if aux_task_mask is None:
                raise RuntimeError(
                    "current_state_ca_fill requires aux_task_mask so the H3 "
                    "information barrier has an explicit task domain."
                )
            current_state_observed = _abflow_ca_fill_observed_mask(
                Sp, Xp, tol2=self.ca_fill_tol2
            ).to(dtype=Xp.dtype)
            observation_for_encoder = (
                current_state_observed
                if Obsp is None
                else torch.where(
                    AuxTask[..., None], current_state_observed, Obsp
                )
            )

        Exists = _atom14_exists_from_seq(
            Sp, Xp, observation_for_encoder, tol2=self.ca_fill_tol2
        ) * M[..., None].to(Xp.dtype)
        Xp_donor = torch.where(Exists.bool()[..., None], Xp, torch.zeros_like(Xp))

        # Keep the original clean-condition barrier as a separate semantic mask.
        # R79 exposes geometry ONLY on the physical H3 authority domain supplied by
        # aux_task_mask (paratope_mask), never on all cmask/design rows.  The caller
        # has already replaced those H3 coordinates by the current model-visible
        # physical state (outer Xt for round 0; analytic endpoint thereafter).
        # Therefore no native X1 is revealed and no second Cartesian authority exists.
        Fixed = M & (~Design)
        if bool(condition_design_geometry):
            if aux_task_mask is None:
                raise RuntimeError(
                    "round-state geometry conditioning requires aux_task_mask "
                    "(formal R79: paratope/H3 physical-authority domain)."
                )
            outside_design = AuxTask & (~Design)
            if bool(outside_design.any()):
                raise RuntimeError(
                    "round-state geometry domain mismatch: aux_task_mask must be "
                    "a subset of design_mask so physical and relational H3 domains agree."
                )
            GeometryCondition = Fixed | AuxTask
        else:
            GeometryCondition = Fixed
        Tors = _torsions_from_atom14(
            Sp, Xp_donor, Chain, GeometryCondition, atom_exists=Exists.bool(),
            epsilon=self.torsion_norm_eps,
        )

        t = flow_t
        if t is None:
            t = Xp.new_zeros(B)
        elif t.dim() == 0:
            t = t.expand(B)
        elif t.numel() != B:
            t = torch.stack([t[batch_id == gid].reshape(-1)[0] for gid in range(B)])

        batch = {
            'seq': Sp, 'seq_t': Sp, 'mask': M, 'fixed_mask': Fixed,
            'geometry_condition_mask': GeometryCondition,
            'chain_id': Chain, 'residx': Rp, 'cdr_def': Cdr,
            'atom14_gt_positions': Xp_donor, 'atom14_gt_exists': Exists,
            'torsion_angles_sin_cos': Tors, 't': t.to(Xp.dtype),
            'antibody_len': antibody_len, 'is_recycling': False,
            '_design_mask': Design, '_aux_task_mask': AuxTask, '_is_antigen': IsAg,
        }
        return batch, GI, nodes, antigen_counts


    def prepare_layout(self, X_ref, S, segment_ids, residue_pos, batch_id,
                       valid_mask, is_antigen, design_mask, flow_t, cdr_type,
                       atom_observed_mask=None, aux_task_mask=None,
                       condition_design_geometry=False):
        """Prepare the round-invariant NativeTrunk packing layout once.

        R79/R80 refresh Single/Pair three times at one fixed outer ``t``.  Across
        those macro rounds sequence ids, chain/residue ids, task masks, padded
        topology and global<->padded lookup are invariant; only H3 coordinates
        (and therefore atom observability/torsions/geometric features) change.
        Rebuilding the static layout three times adds Python work and CUDA
        synchronization but no new mathematics.  This method hoists exactly
        those invariant operations while leaving every geometry-dependent tensor
        to ``_pack_prepared``.
        """
        nodes, ab_lens, antigen_counts = self._ordered_nodes(
            valid_mask, batch_id, is_antigen,
        )
        B = len(nodes)
        L = max([int(idx.numel()) for idx in nodes] or [0])
        if L == 0:
            raise RuntimeError("AbFlow trunk received no biological residues")
        if X_ref.shape[1] != 14:
            raise ValueError(
                f" single/pair port requires the donor's formal 14-slot full-atom state; got {tuple(X_ref.shape)}"
            )
        if aux_task_mask is not None and tuple(aux_task_mask.shape) != (int(X_ref.shape[0]),):
            raise ValueError(
                f"AbFlow aux_task_mask shape mismatch: {tuple(aux_task_mask.shape)} "
                f"vs residue count {(int(X_ref.shape[0]),)}"
            )
        if atom_observed_mask is not None and tuple(atom_observed_mask.shape) != tuple(X_ref.shape[:-1]):
            raise ValueError(
                f"AbFlow xloss_mask shape mismatch: {tuple(atom_observed_mask.shape)} "
                f"vs X {tuple(X_ref.shape)}"
            )

        Sp = S.new_full((B, L), 20)
        Obsp = X_ref.new_zeros((B, L, 14)) if atom_observed_mask is not None else None
        Seg = segment_ids.new_zeros((B, L))
        Rp = torch.zeros((B, L), device=X_ref.device, dtype=torch.long)
        M = torch.zeros((B, L), device=X_ref.device, dtype=torch.bool)
        Design = torch.zeros_like(M)
        AuxTask = torch.zeros_like(M)
        IsAg = torch.zeros_like(M)
        GI = S.new_full((B, L), -1)

        pack_global_parts, pack_batch_parts, pack_local_parts = [], [], []
        for b, idx in enumerate(nodes):
            n = int(idx.numel())
            if n == 0:
                continue
            pack_global_parts.append(idx.long())
            pack_batch_parts.append(torch.full(
                (n,), b, device=idx.device, dtype=torch.long))
            pack_local_parts.append(torch.arange(
                n, device=idx.device, dtype=torch.long))

        if pack_global_parts:
            PG = torch.cat(pack_global_parts, dim=0)
            PB = torch.cat(pack_batch_parts, dim=0)
            PL = torch.cat(pack_local_parts, dim=0)
            Sp[PB, PL] = S[PG].long().clamp(0, 22)
            if Obsp is not None:
                Obsp[PB, PL] = atom_observed_mask[PG].to(dtype=X_ref.dtype)
            Seg[PB, PL] = segment_ids[PG]
            rp = residue_pos[PG]
            rp = rp[..., 0] if rp.dim() > 1 else rp
            if torch.is_floating_point(rp):
                rp = torch.round(rp)
            Rp[PB, PL] = rp.to(dtype=torch.long)
            M[PB, PL] = True
            Design[PB, PL] = design_mask[PG].bool()
            if aux_task_mask is not None:
                AuxTask[PB, PL] = aux_task_mask[PG].bool()
            IsAg[PB, PL] = is_antigen[PG].bool()
            GI[PB, PL] = PG.to(dtype=GI.dtype)
        else:
            PG = torch.empty(0, device=X_ref.device, dtype=torch.long)
            PB = torch.empty(0, device=X_ref.device, dtype=torch.long)
            PL = torch.empty(0, device=X_ref.device, dtype=torch.long)

        Chain = self._remap_chain_ids(Seg, M)
        antibody_len = torch.tensor(ab_lens, device=X_ref.device, dtype=torch.long)
        Cdr = self._cdr_definition(Rp, Chain, M, antibody_len, cdr_type, Design)

        Fixed = M & (~Design)
        if bool(condition_design_geometry):
            if aux_task_mask is None:
                raise RuntimeError(
                    "round-state geometry conditioning requires aux_task_mask "
                    "(formal R80: paratope/H3 physical-authority domain)."
                )
            outside_design = AuxTask & (~Design)
            if bool(outside_design.any()):
                raise RuntimeError(
                    "round-state geometry domain mismatch: aux_task_mask must be "
                    "a subset of design_mask so physical and relational H3 domains agree."
                )
            GeometryCondition = Fixed | AuxTask
        else:
            GeometryCondition = Fixed

        t = flow_t
        if t is None:
            t = X_ref.new_zeros(B)
        elif t.dim() == 0:
            t = t.expand(B)
        elif t.numel() != B:
            t = torch.stack([
                t[batch_id == gid].reshape(-1)[0] for gid in range(B)
            ])
        t = t.to(X_ref.dtype)

        node_graph, node_local = self._node_lookup(GI, M, int(S.shape[0]))
        return {
            'B': B, 'L': L, 'nodes': nodes,
            'pack_global': PG, 'pack_batch': PB, 'pack_local': PL,
            'seq': Sp, 'native_observed': Obsp, 'segment': Seg,
            'residx': Rp, 'mask': M, 'design': Design, 'aux_task': AuxTask,
            'is_antigen': IsAg, 'global_index': GI, 'chain': Chain,
            'cdr_def': Cdr, 'antibody_len': antibody_len,
            'fixed': Fixed, 'geometry_condition': GeometryCondition,
            't': t, 'antigen_counts': antigen_counts,
            'node_graph': node_graph, 'node_local': node_local,
            'valid_mask_global': valid_mask.bool(),
            'condition_design_geometry': bool(condition_design_geometry),
        }

    def _pack_prepared(self, X, layout):
        """Fill only round-dependent geometry into a prepared static layout."""
        B, L = int(layout['B']), int(layout['L'])
        PB, PL, PG = layout['pack_batch'], layout['pack_local'], layout['pack_global']
        Xp = X.new_zeros((B, L, 14, 3))
        if PG.numel():
            Xp[PB, PL] = X[PG]

        Sp = layout['seq']
        M = layout['mask']
        Design = layout['design']
        AuxTask = layout['aux_task']
        Fixed = layout['fixed']
        GeometryCondition = layout['geometry_condition']
        Obsp = layout['native_observed']

        observation_for_encoder = Obsp
        if (
            layout['condition_design_geometry']
            and self.task_atom_observation_source == 'current_state_ca_fill'
        ):
            current_state_observed = _abflow_ca_fill_observed_mask(
                Sp, Xp, tol2=self.ca_fill_tol2
            ).to(dtype=Xp.dtype)
            observation_for_encoder = (
                current_state_observed
                if Obsp is None
                else torch.where(AuxTask[..., None], current_state_observed, Obsp)
            )

        Exists = _atom14_exists_from_seq(
            Sp, Xp, observation_for_encoder, tol2=self.ca_fill_tol2
        ) * M[..., None].to(Xp.dtype)
        Xp_donor = torch.where(Exists.bool()[..., None], Xp, torch.zeros_like(Xp))
        Tors = _torsions_from_atom14(
            Sp, Xp_donor, layout['chain'], GeometryCondition,
            atom_exists=Exists.bool(), epsilon=self.torsion_norm_eps,
        )
        batch = {
            'seq': Sp, 'seq_t': Sp, 'mask': M, 'fixed_mask': Fixed,
            'geometry_condition_mask': GeometryCondition,
            'chain_id': layout['chain'], 'residx': layout['residx'],
            'cdr_def': layout['cdr_def'],
            'atom14_gt_positions': Xp_donor, 'atom14_gt_exists': Exists,
            'torsion_angles_sin_cos': Tors, 't': layout['t'],
            'antibody_len': layout['antibody_len'], 'is_recycling': False,
            '_design_mask': Design, '_aux_task_mask': AuxTask,
            '_is_antigen': layout['is_antigen'],
        }
        return batch

    def forward_prepared(self, layout, X, residue_feature, round_idx=-1):
        """Run exact NativeTrunk math using a round-invariant prepared layout."""
        batch = self._pack_prepared(X, layout)

        if self.training:
            call_index = int(self._training_forward_call.item())
            self._training_forward_call.add_(1)
            rank = (
                int(torch.distributed.get_rank())
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
            donor_seed = self.forward_seed + 1_000_003 * rank + call_index
            rng_devices = (
                [int(X.device.index)]
                if X.is_cuda and X.device.index is not None else []
            )
            with torch.random.fork_rng(devices=rng_devices):
                torch.default_generator.manual_seed(donor_seed)
                if X.is_cuda:
                    with torch.cuda.device(X.device):
                        torch.cuda.manual_seed(donor_seed)
                s, z = self.trunk(batch, residue_feature)
        else:
            s, z = self.trunk(batch, residue_feature)

        N = int(layout['node_graph'].shape[0])
        global_s = s.new_zeros((N, s.shape[-1]))
        PG, PB, PL = layout['pack_global'], layout['pack_batch'], layout['pack_local']
        if PG.numel():
            global_s[PG] = s[PB, PL]
        logits = self.distogram_head(z) if self.distogram_head is not None else None
        with torch.no_grad():
            geom_mask = batch['geometry_condition_mask'].bool()
            design = batch['_design_mask'].bool()
            task = batch['_aux_task_mask'].bool()
            task_count = task.sum().to(dtype=X.dtype)
            other_design = design & (~task)
            other_design_count = other_design.sum().to(dtype=X.dtype)
            self.last_diagnostics = {
                'relational_single_rms': torch.sqrt(
                    global_s.float().pow(2).mean() + 1e-8).to(X.dtype),
                'relational_pair_rms': torch.sqrt(
                    z.float().pow(2).mean() + 1e-8).to(X.dtype),
                'relational_task_geometry_visible_fraction': (
                    (geom_mask & task).sum().to(dtype=X.dtype)
                    / task_count.clamp_min(1.0)
                ),
                'relational_non_task_design_geometry_visible_fraction': (
                    (geom_mask & other_design).sum().to(dtype=X.dtype)
                    / other_design_count.clamp_min(1.0)
                ),
            }
        return {
            'single_global': global_s,
            'pair_dense': z,
            'node_graph': layout['node_graph'],
            'node_local': layout['node_local'],
            'global_index': layout['global_index'],
            'mask': batch['mask'],
            'seq_padded': batch['seq'],
            'atom_exists_padded': batch['atom14_gt_exists'].bool(),
            'design_padded': batch['_design_mask'].bool(),
            'is_antigen_padded': batch['_is_antigen'].bool(),
            'distogram_logits': logits,
            'biological_mask': layout['valid_mask_global'],
            'diag': self.last_diagnostics,
        }

    @staticmethod
    def _node_lookup(GI, mask, n_global):
        graph = GI.new_full((n_global,), -1)
        local = GI.new_full((n_global,), -1)
        for b in range(mask.shape[0]):
            n = int(mask[b].sum().item())
            if n == 0:
                continue
            idx = GI[b, :n].long()
            graph[idx] = b
            local[idx] = torch.arange(n, device=GI.device, dtype=GI.dtype)
        return graph, local

    @staticmethod
    def gather_pair(z, query_edges_global, node_graph, node_local):
        """Restrict dense z[i,j] to the exact directed R05 edge order.

        R05 uses ``col -> row`` message flow.  Seqformer pair state ``z[i,j]``
        uses the first index as the receiving/query residue and the second as
        the sending/key residue, so an R05 edge (row=i, col=j) consumes z[i,j]
        without transposition or symmetrization.
        """
        if query_edges_global.numel() == 0:
            return z.new_zeros((0, z.shape[-1]))
        row, col = query_edges_global.long()
        gr, gc = node_graph[row], node_graph[col]
        lr, lc = node_local[row], node_local[col]
        valid = (gr >= 0) & (gr == gc) & (lr >= 0) & (lc >= 0)
        out = z.new_zeros((row.numel(), z.shape[-1]))
        if bool(valid.any()):
            out[valid] = z[gr[valid], lr[valid], lc[valid]]
        return out

    def forward(self, X, S, segment_ids, residue_pos, batch_id, valid_mask,
                is_antigen, design_mask, flow_t, cdr_type, residue_feature,
                round_idx=-1, atom_observed_mask=None, aux_task_mask=None,
                condition_design_geometry=False):
        """Compatibility wrapper; R80 model uses prepare_layout+forward_prepared."""
        layout = self.prepare_layout(
            X_ref=X, S=S, segment_ids=segment_ids, residue_pos=residue_pos,
            batch_id=batch_id, valid_mask=valid_mask, is_antigen=is_antigen,
            design_mask=design_mask, flow_t=flow_t, cdr_type=cdr_type,
            atom_observed_mask=atom_observed_mask,
            aux_task_mask=aux_task_mask,
            condition_design_geometry=condition_design_geometry,
        )
        return self.forward_prepared(
            layout=layout, X=X, residue_feature=residue_feature,
            round_idx=round_idx,
        )


    def distogram_loss_from_native(self, state, true_X, true_S, collect_audit=False):
        """AbX-style distogram supervision on the live dense pair state.

        The head/targets follow the supplied AbX implementation: symmetric
        logits, pseudo-beta targets and 64 bins over 2.3125--21.6875 Angstrom.
        For the support mask we adopt the standard Boltz-style non-self rule so
        trivial i==j zero-distance labels do not dilute relational supervision.
        ``all_resolved`` remains the formal R29/R30 scope; DA statistics are
        observation-only and never rebalance the objective.
        """
        zero = true_X.sum() * 0.0
        if not self.enable_distogram or state.get('distogram_logits') is None:
            return zero, {}

        GI, token_mask = state['global_index'], state['mask']
        logits = state['distogram_logits']
        B, L = token_mask.shape
        Xp = true_X.new_zeros((B, L, 14, 3))
        Sp = true_S.new_full((B, L), 20)
        Obs = torch.zeros((B, L, 14), device=true_X.device, dtype=torch.bool)
        packed_obs = state['atom_exists_padded'].bool()
        for b in range(B):
            n = int(token_mask[b].sum().item())
            if n:
                idx = GI[b, :n].long()
                Xp[b, :n] = true_X[idx]
                Sp[b, :n] = true_S[idx].long().clamp(0, 22)
                Obs[b, :n] = packed_obs[b, :n]

        pseudo_beta, pseudo_beta_mask = pseudo_beta_fn_v2(Sp, Xp, Obs)
        pseudo_beta_mask = pseudo_beta_mask.bool() & token_mask
        boundaries = torch.linspace(
            self.distogram_min, self.distogram_max, self.distogram_bins - 1,
            device=logits.device, dtype=pseudo_beta.dtype,
        )
        d2 = torch.sum(
            (pseudo_beta[:, :, None, :] - pseudo_beta[:, None, :, :]).square(),
            dim=-1, keepdim=True,
        )
        target = torch.sum(d2 > boundaries.square(), dim=-1).long()
        ce = F.cross_entropy(
            logits.reshape(-1, self.distogram_bins),
            target.reshape(-1), reduction='none',
        ).reshape(B, L, L)

        pair_mask = pseudo_beta_mask[:, :, None] & pseudo_beta_mask[:, None, :]
        pair_mask = pair_mask & (~torch.eye(
            L, device=logits.device, dtype=torch.bool
        )[None, :, :])
        design = state['design_padded'].bool() & pseudo_beta_mask
        if self.distogram_pair_scope == 'design_anchored':
            pair_mask = pair_mask & (design[:, :, None] | design[:, None, :])
        elif self.distogram_pair_scope != 'all_resolved':
            raise ValueError(
                f'unknown distogram pair_scope={self.distogram_pair_scope!r}'
            )

        denom = pair_mask.sum(dim=(-1, -2)).to(ce.dtype).clamp_min(1.0)
        per_graph = (ce * pair_mask.to(ce.dtype)).sum(dim=(-1, -2)) / denom
        loss = per_graph.mean()

        audit = {}
        if collect_audit:
            with torch.no_grad():
                antigen = state['is_antigen_padded'].bool() & pseudo_beta_mask
                da = pair_mask & design[:, :, None] & antigen[:, None, :]
                da_n = da.sum()
                da_ce = (
                    (ce * da.to(ce.dtype)).sum() / da_n.clamp_min(1).to(ce.dtype)
                )
                probs = torch.softmax(logits.float(), dim=-1)
                last_contact_bin = int((boundaries < 8.0).sum().item())
                contact_prob = probs[..., :last_contact_bin + 1].sum(dim=-1)
                native_contact = d2.squeeze(-1) < (8.0 ** 2)
                soft_tp = (contact_prob * native_contact.float() * da.float()).sum()
                soft_pred = (contact_prob * da.float()).sum()
                audit = {
                    'distogram_valid_pairs': pair_mask.sum().to(loss.dtype),
                    'distogram_da_ce': da_ce.to(loss.dtype),
                    'distogram_da_contact_precision': (
                        soft_tp / soft_pred.clamp_min(1e-8)
                    ).to(loss.dtype),
                    'distogram_head_weight_rms': (
                        self.distogram_head.proj.weight.detach().float()
                        .square().mean().sqrt()
                    ).to(loss.dtype),
                }
        return loss, audit


def design_region_smooth_lddt_loss(
    pred_X, true_X, valid_atom_mask, design_residue_mask, batch_id,
    is_antigen_mask=None, cutoff=15.0, collect_audit=False,
):
    """Task-localized Boltz smooth-lDDT with an exact O(N_design*N) kernel.

    The formal objective is identical to the symmetric full pair mask that keeps
    all resolved non-self pairs within the native-distance cutoff and requires at
    least one endpoint to be a design atom.  Because the score is symmetric, the
    same numerator/denominator can be evaluated from design rows only: design--
    context pairs receive weight 2 (for both directions), while design--design
    ordered pairs already appear in both directions across the design rows.
    This reduces memory from O(N^2) to O(N_design*N) without changing the loss
    or its gradient with respect to generated design coordinates.
    """
    if is_antigen_mask is None:
        is_antigen_mask = torch.zeros_like(design_residue_mask, dtype=torch.bool)
    cutoff = float(cutoff)
    graph_losses = []
    rel_values = {'intra': [], 'scaffold': [], 'antigen': []}
    support_weight_total = pred_X.detach().new_zeros(())

    for gid_t in torch.unique(batch_id):
        graph = batch_id == gid_t
        pred = pred_X[graph].reshape(-1, 3).float()
        true = true_X[graph].reshape(-1, 3).float()
        valid = valid_atom_mask[graph].bool().reshape(-1)
        nr, na = valid_atom_mask[graph].shape
        design = design_residue_mask[graph].bool()[:, None].expand(nr, na).reshape(-1)
        antigen = is_antigen_mask[graph].bool()[:, None].expand(nr, na).reshape(-1)
        scaffold = (~design) & (~antigen)

        all_idx = torch.nonzero(valid, as_tuple=False).flatten()
        design_idx = torch.nonzero(valid & design, as_tuple=False).flatten()
        if all_idx.numel() == 0 or design_idx.numel() == 0:
            continue

        true_d = torch.cdist(true[design_idx], true[all_idx])
        pred_d = torch.cdist(pred[design_idx], pred[all_idx])
        delta = (pred_d - true_d).abs()
        score = 0.25 * (
            torch.sigmoid(0.5 - delta)
            + torch.sigmoid(1.0 - delta)
            + torch.sigmoid(2.0 - delta)
            + torch.sigmoid(4.0 - delta)
        )

        support = true_d < cutoff
        support = support & (design_idx[:, None] != all_idx[None, :])
        col_is_design = design[all_idx]
        weights = torch.where(
            col_is_design[None, :],
            torch.ones_like(score),
            torch.full_like(score, 2.0),
        )
        weighted_support = support.to(score.dtype) * weights
        denom = weighted_support.sum()
        support_weight_total = support_weight_total + denom.detach().to(pred_X.dtype)
        if bool(support.any()):
            graph_losses.append(
                1.0 - (score * weighted_support).sum() / denom.clamp_min(1.0)
            )

        if collect_audit:
            with torch.no_grad():
                relation_cols = {
                    'intra': design[all_idx],
                    'scaffold': scaffold[all_idx],
                    'antigen': antigen[all_idx],
                }
                for name, col_mask in relation_cols.items():
                    rel = support & col_mask[None, :]
                    if bool(rel.any()):
                        rel_values[name].append(1.0 - score[rel].mean().detach())

    zero = pred_X.sum() * 0.0
    total = torch.stack(graph_losses).mean().to(pred_X.dtype) if graph_losses else zero

    def relation_mean(name):
        return (
            torch.stack(rel_values[name]).mean().to(pred_X.dtype)
            if rel_values[name] else zero.detach()
        )

    return total, {
        'intra': relation_mean('intra'),
        'scaffold': relation_mean('scaffold'),
        'antigen': relation_mean('antigen'),
        'support_weight': support_weight_total,
    }


class AbFlowR3Matcher:
    def __init__(self, *, transport_fraction=0.05, path_min_sigma=0.0, eps=1e-8):
        self.transport_fraction = float(transport_fraction)
        self.path_min_sigma = float(path_min_sigma)
        self.eps = float(eps)
        if not (0.0 < self.transport_fraction <= 1.0):
            raise ValueError('transport_fraction must be in (0, 1].')
        if self.path_min_sigma < 0.0:
            raise ValueError('path_min_sigma must be non-negative.')
        if self.eps <= 0.0:
            raise ValueError('eps must be positive.')

    def graph_g_from_transport(self, transport):
        """FoldFlow g calibrated to preserve S01 midpoint RMS.

        S01 midpoint vector RMS was eta * D, with eta=transport_fraction.
        FoldFlow-R3 uses sigma(t)=sqrt(g^2 t(1-t)+sigma_min^2) per coordinate.
        Choose g so the *total* midpoint vector RMS matches eta*D whenever the
        requested width is above the numerical sigma floor.
        """
        target_mid_coord = self.transport_fraction * transport / math.sqrt(3.0)
        floor = torch.as_tensor(
            self.path_min_sigma, device=transport.device, dtype=transport.dtype
        )
        dynamic_sq = (target_mid_coord.square() - floor.square()).clamp_min(0.0)
        return 2.0 * torch.sqrt(dynamic_sq)

    def sigma_t(self, t_graph, g_graph):
        """FoldFlow R3 temporal width: sqrt(g^2 t(1-t)+min_sigma^2)."""
        t = torch.as_tensor(t_graph, device=g_graph.device, dtype=g_graph.dtype)
        return torch.sqrt(
            (g_graph.square() * t * (1.0 - t)).clamp_min(0.0)
            + self.path_min_sigma ** 2
        )

    @staticmethod
    def linear_mean(x0, x1, t_int):
        return (1.0 - t_int) * x0 + t_int * x1

    @staticmethod
    def clean_conditional_velocity(x0, x1):
        """FoldFlow/CFM Euclidean target: u_t = x1 - x0."""
        return x1 - x0

    @staticmethod
    def endpoint_target_for_velocity(xt, velocity, t_int):
        """Encode a velocity target through the donor's endpoint parameterization."""
        return xt + (1.0 - t_int) * velocity

    # ============================================================
    # v64: genuine F01 R3 score / canonical Gaussian Score-Flow
    # ============================================================
    def score_sigma_t(self, t_graph, g_graph, score_min_sigma=1e-2):
        """Numerically safe sigma used only for analytic score evaluation.

        The physical F01 path is NOT changed. Formal Score-Flow profiles keep
        path_min_sigma=0.  The floor protects the DSM denominator only.
        """
        sigma = self.sigma_t(t_graph, g_graph)
        return sigma.clamp_min(float(score_min_sigma))

    def global_conditional_score(
            self, z_t, z0, z1, t_graph, g_graph,
            score_min_sigma=1e-2):
        """Exact conditional score in F01's stochastic global-translation R3.

        q_t(z_t | z0,z1)
          = N((1-t)z0 + t z1, sigma_t^2 I_3)

        score = grad_{z_t} log q_t
              = -(z_t - mu_t) / sigma_t^2.

        IMPORTANT:
        F01 broadcasts ONE graph-level 3D shift to all H3 atoms.  Therefore its
        full all-atom covariance is rank-3/singular.  This score is deliberately
        defined only in the actual non-degenerate R3 translation subspace.
        """
        t = torch.as_tensor(
            t_graph, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        if t.numel() == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])
        mu = (1.0 - t[:, None]) * z0 + t[:, None] * z1
        sigma = self.score_sigma_t(
            t, g_graph.to(z_t.dtype), score_min_sigma
        )
        return -(z_t - mu) / sigma[:, None].square()

    def scaled_score_residual(
            self, z_t, z0, true_z1, pred_z1, t_graph, g_graph,
            score_min_sigma=1e-2):
        """endpoint-style scaled DSM residual under the SAME F01 corruption kernel.

        The network remains endpoint-parameterized:
            pred_z1 -> analytic pred score.
        No independent score head is introduced.
        """
        target = self.global_conditional_score(
            z_t, z0, true_z1, t_graph, g_graph, score_min_sigma
        ).detach()
        pred = self.global_conditional_score(
            z_t, z0, pred_z1, t_graph, g_graph, score_min_sigma
        )
        sigma = self.score_sigma_t(
            t_graph, g_graph.to(z_t.dtype), score_min_sigma
        )
        scaled = sigma[:, None] * (pred - target)
        return scaled, pred, target

    def canonical_log_sigma_derivative(self, t_graph):
        """d log(sigma_t) / dt for sigma_t=g*sqrt(t(1-t)), g>0.

        For formal F01 path_min_sigma=0:
            d log sigma / dt = (1-2t) / [2 t (1-t)].

        Crucially, g cancels.  This is the key reason the inference-time
        canonical Score-Flow velocity does NOT need native-endpoint g.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free canonical Score-Flow requires "
                "ABFLOW_R3_PATH_MIN_SIGMA=0."
            )
        t = torch.as_tensor(t_graph)
        t_safe = t.clamp(min=self.eps, max=1.0 - self.eps)
        return (1.0 - 2.0 * t_safe) / (
            2.0 * t_safe * (1.0 - t_safe)
        )

    def canonical_global_velocity_gfree(
            self, z_t, z0, z1, t_graph):
        """Canonical Gaussian conditional-flow velocity in global R3.

        For
            mu_t=(1-t)z0+t z1,
            sigma_t=g sqrt(t(1-t)),
        the Gaussian FM field is
            u_t = mu_dot + (sigma_dot/sigma)(z_t-mu_t)
                = mu_dot - sigma*sigma_dot*score_t.

        Because sigma_dot/sigma is independent of g, this velocity is
        inference-available without native endpoint scale.
        """
        t = torch.as_tensor(
            t_graph, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        if t.numel() == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])
        mu = (1.0 - t[:, None]) * z0 + t[:, None] * z1
        mean_velocity = z1 - z0
        k = self.canonical_log_sigma_derivative(t).to(z_t.dtype)
        return mean_velocity + k[:, None] * (z_t - mu)

    def exact_global_scoreflow_step_gfree(
            self, z_t, z0, pred_z1, t, t_next):
        """Piecewise-exact canonical Score-Flow step with NO inference-time g.

        Freeze pred_z1 on [t,t_next].  If r_t=z_t-mu_t, then
            dr/dt = (sigma_dot/sigma) r,
        hence
            r_next = (sigma_next/sigma_t) r_t.

        With sigma=g*sqrt(t(1-t)) and path_min_sigma=0:
            sigma_next/sigma_t
              = sqrt[t_next(1-t_next) / (t(1-t))],
        so g cancels exactly.

        Boundary handling is analytic, not a numerical hack:
        - at t=0 the source lies on the path mean => r_0=0;
        - at t_next=1, sigma_next=0 => z_1=pred_z1.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free exact Score-Flow step requires "
                "ABFLOW_R3_PATH_MIN_SIGMA=0."
            )
        if z_t.numel() == 0:
            return z_t, {}

        n = z_t.shape[0]
        t0 = torch.as_tensor(
            t, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        t1 = torch.as_tensor(
            t_next, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        if t0.numel() == 1:
            t0 = t0.expand(n)
        if t1.numel() == 1:
            t1 = t1.expand(n)

        mu0 = (1.0 - t0[:, None]) * z0 + t0[:, None] * pred_z1
        mu1 = (1.0 - t1[:, None]) * z0 + t1[:, None] * pred_z1

        base0 = (t0 * (1.0 - t0)).clamp_min(0.0)
        base1 = (t1 * (1.0 - t1)).clamp_min(0.0)
        residual = z_t - mu0

        ratio = torch.zeros_like(base0)
        interior = base0 > self.eps
        ratio[interior] = torch.sqrt(
            base1[interior] / base0[interior]
        )

        # At t=0 the formal path residual is exactly zero.
        residual = torch.where(
            interior[:, None], residual, torch.zeros_like(residual)
        )
        z_next = mu1 + ratio[:, None] * residual

        return z_next, {
            "residual_ratio": ratio,
            "residual_norm": torch.linalg.norm(residual, dim=-1),
            "mean_step_norm": torch.linalg.norm(mu1 - mu0, dim=-1),
        }

    # ============================================================
    # v85: F01 single-field endpoint-like canonical carrier
    # ============================================================
    @staticmethod
    def _broadcast_time_like(t, ref):
        tt = torch.as_tensor(t, device=ref.device, dtype=ref.dtype)
        if tt.dim() == 0 or tt.numel() == 1:
            shape = [1] * ref.dim()
            return tt.reshape(*shape)
        tt = tt.reshape(-1)
        if tt.numel() != ref.shape[0]:
            raise ValueError(
                f"Expected scalar time or {ref.shape[0]} leading times, "
                f"got {tt.numel()}."
            )
        return tt.reshape(ref.shape[0], *([1] * (ref.dim() - 1)))

    def canonical_carrier_target_gfree(
            self, x_t, x0, x1, t, boundary_eps=5e-2):
        """Endpoint-like carrier for the F01 canonical Score--Flow field.

        F01 stochastic state:
            x_t = mu_t + g*sqrt(t(1-t))*eps
            mu_t = (1-t)x0 + t x1

        Canonical conditional probability-flow velocity:
            u* = (x1-x0) + k(t)(x_t-mu_t),
            k(t)=(1-2t)/(2t(1-t)).

        Existing AbFlow bridge parameterization decodes a coordinate carrier Y
        as u=(Y-x_t)/(1-t).  Therefore the exact carrier target is
            Y* = x_t + (1-t)u*
               = x1 + (x_t-mu_t)/(2t).

        g cancels.  ``boundary_eps`` is used only to keep the unused t->0
        expression finite before the caller replaces that boundary with the
        clean Endpoint target.  It is not a loss weight and does not alter the
        physical F01 corruption path.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free F01 canonical carrier requires path_min_sigma=0."
            )
        t_b = self._broadcast_time_like(t, x_t)
        mu = (1.0 - t_b) * x0 + t_b * x1
        residual = x_t - mu
        t_safe = t_b.clamp_min(float(boundary_eps))
        return x1 + residual / (2.0 * t_safe)

    def endpoint_from_canonical_carrier_gfree(
            self, x_t, x0, carrier, t, boundary_eps=5e-2):
        """Invert the v85 canonical carrier to the endpoint it implies.

        From Y = X1 + (Xt-mu_t)/(2t),
            X1 = 2Y - [Xt-(1-t)X0]/t.
        This uses only inference-known Xt, X0, t and network carrier Y.
        """
        t_b = self._broadcast_time_like(t, x_t)
        t_safe = t_b.clamp_min(float(boundary_eps))
        return 2.0 * carrier - (x_t - (1.0 - t_b) * x0) / t_safe

    def exact_carrier_scoreflow_step_gfree(
            self, x_t, x0, carrier, t, t_next, canonical_t_min=5e-2):
        """Matched F01 step for the v85 single-field coordinate carrier.

        Boundary region t<canonical_t_min:
            ``carrier`` is trained as clean Endpoint, so use the historical
            endpoint bridge update.

        Canonical region:
            1. invert carrier -> predicted clean endpoint;
            2. freeze that endpoint over [t,t_next];
            3. evolve the Gaussian residual exactly with
               sqrt[t_next(1-t_next)/(t(1-t))].

        The residual ratio contains no g, hence no native-dependent path width
        is required at inference.  At t_next=1 the ratio is exactly zero and
        the state lands on the endpoint implied by the carrier.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free exact carrier Score--Flow step requires "
                "path_min_sigma=0."
            )
        t_b = self._broadcast_time_like(t, x_t)
        tn_b = self._broadcast_time_like(t_next, x_t)
        active = t_b >= float(canonical_t_min)

        # Historical Endpoint bridge for the mathematically singular early
        # boundary.  This is exact for the target used in that region.
        one_minus_t = (1.0 - t_b).clamp_min(self.eps)
        bridge_alpha = (tn_b - t_b) / one_minus_t
        endpoint_next = x_t + bridge_alpha * (carrier - x_t)

        pred_x1 = self.endpoint_from_canonical_carrier_gfree(
            x_t, x0, carrier, t_b, boundary_eps=canonical_t_min
        )
        mu0 = (1.0 - t_b) * x0 + t_b * pred_x1
        mu1 = (1.0 - tn_b) * x0 + tn_b * pred_x1
        residual = x_t - mu0
        base0 = (t_b * (1.0 - t_b)).clamp_min(0.0)
        base1 = (tn_b * (1.0 - tn_b)).clamp_min(0.0)
        ratio = torch.zeros_like(base0)
        interior = base0 > self.eps
        ratio = torch.where(
            interior, torch.sqrt(base1 / base0.clamp_min(self.eps)), ratio
        )
        canonical_next = mu1 + ratio * residual
        x_next = torch.where(active, canonical_next, endpoint_next)

        with torch.no_grad():
            return x_next, {
                "canonical_active_rate": active.to(x_t.dtype).mean(),
                "canonical_residual_rms": torch.sqrt(
                    residual.pow(2).mean().clamp_min(0.0)
                ),
                "canonical_ratio_mean": ratio.mean(),
            }

    # ============================================================
    # v86: boundary-regular preconditioned canonical carrier
    # ============================================================
    def boundary_regular_carrier_target_gfree(
            self, x_t, x0, x1, t):
        """Boundary-regular chart of the same F01 canonical Score--Flow field.

        The v85 natural carrier
            Y*_t = X1 + (Xt-mu_t)/(2t)
        is an exact bridge-coordinate representation of the conditional
        probability-flow velocity but is badly conditioned near t=0.

        v86 applies the parameter-free homotopy lambda(t)=t between the clean
        Endpoint and the natural canonical carrier:
            P*_t = (1-t) X1 + t Y*_t
                 = X1 + 0.5 (Xt-mu_t).

        IMPORTANT:
        - this changes only the neural output coordinate chart;
        - the physical F01 stochastic path is unchanged;
        - the decoded endpoint is still used by the same canonical Gaussian
          residual evolution;
        - no score head, flow head, auxiliary loss, or t-threshold is added.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free F01 boundary-regular carrier requires "
                "path_min_sigma=0."
            )
        t_b = self._broadcast_time_like(t, x_t)
        mu = (1.0 - t_b) * x0 + t_b * x1
        residual = x_t - mu
        return x1 + 0.5 * residual

    def endpoint_from_boundary_regular_carrier_gfree(
            self, x_t, x0, carrier, t):
        """Decode the endpoint implied by the v86 boundary-regular carrier.

        From
            P = X1 + 0.5[Xt-(1-t)X0-tX1]
              = (1-t/2)X1 + 0.5[Xt-(1-t)X0],
        therefore
            X1 = [P - 0.5(Xt-(1-t)X0)] / (1-t/2).

        The denominator lies in [0.5, 1] for t in [0,1], hence there is no
        source-side 1/t inversion singularity.
        """
        t_b = self._broadcast_time_like(t, x_t)
        denom = (1.0 - 0.5 * t_b).clamp_min(0.5)
        known = 0.5 * (x_t - (1.0 - t_b) * x0)
        return (carrier - known) / denom

    def exact_boundary_regular_scoreflow_step_gfree(
            self, x_t, x0, carrier, t, t_next):
        """Matched sampler for the v86 boundary-regular carrier.

        1. Decode carrier -> endpoint estimate with a nonsingular transform.
        2. Freeze the decoded endpoint over [t,t_next].
        3. Evolve the F01 Gaussian residual with the exact g-free ratio
              sqrt[t_next(1-t_next)/(t(1-t))].

        At t=0 the formal residual is exactly zero, so the first interval starts
        from the decoded mean. At t_next=1 the ratio is zero and the state lands
        exactly on the decoded endpoint. No hard Endpoint/canonical switch is
        used anywhere.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free boundary-regular Score--Flow step requires "
                "path_min_sigma=0."
            )
        if x_t.numel() == 0:
            return x_t, {}

        t_b = self._broadcast_time_like(t, x_t)
        tn_b = self._broadcast_time_like(t_next, x_t)
        pred_x1 = self.endpoint_from_boundary_regular_carrier_gfree(
            x_t, x0, carrier, t_b
        )

        mu0 = (1.0 - t_b) * x0 + t_b * pred_x1
        mu1 = (1.0 - tn_b) * x0 + tn_b * pred_x1
        residual = x_t - mu0

        base0 = (t_b * (1.0 - t_b)).clamp_min(0.0)
        base1 = (tn_b * (1.0 - tn_b)).clamp_min(0.0)
        interior = base0 > self.eps
        ratio = torch.zeros_like(base0)
        ratio = torch.where(
            interior,
            torch.sqrt(base1 / base0.clamp_min(self.eps)),
            ratio,
        )
        residual = torch.where(
            interior, residual, torch.zeros_like(residual)
        )
        x_next = mu1 + ratio * residual

        with torch.no_grad():
            return x_next, {
                "boundary_regular_residual_rms": torch.sqrt(
                    residual.pow(2).mean().clamp_min(0.0)
                ),
                "boundary_regular_ratio_mean": ratio.mean(),
                "boundary_regular_endpoint_rms": torch.sqrt(
                    pred_x1.pow(2).mean().clamp_min(0.0)
                ),
            }

    # ============================================================
    # v87: C1 smoothstep source-anchored canonical carrier
    # ============================================================
    @staticmethod
    def c1_smoothstep_lambda(t):
        """Unique cubic Hermite homotopy with zero endpoint slopes.

        lambda(0)=0, lambda'(0)=0, lambda(1)=1, lambda'(1)=0,
        hence lambda(t)=3t^2-2t^3.
        """
        return t.square() * (3.0 - 2.0 * t)

    def c1_smoothstep_carrier_target_gfree(self, x_t, x0, x1, t):
        """One-forward C1 chart of the same F01 canonical Score--Flow field.

        Natural canonical carrier:
            Y*=X1+r/(2t), r=x_t-mu_t.

        Use the parameter-free cubic Hermite homotopy
            lambda(t)=3t^2-2t^3
        between Endpoint and the natural canonical chart:
            P*=(1-lambda)X1+lambda Y*
              =X1+c(t)r,
            c(t)=lambda/(2t)=t(3-2t)/2.

        Therefore dP*/dx_t=c(t)I -> 0 at the source, while c(t) is about
        0.54--0.56 on t in [0.6,0.8], preserving the strong interior response
        observed for U02/U03. No threshold, new head, auxiliary loss or second
        network query is introduced.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free F01 C1 smoothstep carrier requires path_min_sigma=0."
            )
        t_b = self._broadcast_time_like(t, x_t)
        mu = (1.0 - t_b) * x0 + t_b * x1
        residual = x_t - mu
        gain = 0.5 * t_b * (3.0 - 2.0 * t_b)
        return x1 + gain * residual

    def endpoint_from_c1_smoothstep_carrier_gfree(
            self, x_t, x0, carrier, t):
        """Stable endpoint decode for the v87 C1 smoothstep chart.

        P=X1+c(t)[x_t-(1-t)x0-tX1]
         =[1-c(t)t]X1+c(t)[x_t-(1-t)x0].

        Since c(t)t=lambda(t)/2 and lambda in [0,1], the denominator
        1-lambda/2 lies in [0.5,1].
        """
        t_b = self._broadcast_time_like(t, x_t)
        lam = self.c1_smoothstep_lambda(t_b)
        gain = 0.5 * t_b * (3.0 - 2.0 * t_b)
        denom = (1.0 - 0.5 * lam).clamp_min(0.5)
        known = gain * (x_t - (1.0 - t_b) * x0)
        return (carrier - known) / denom

    def exact_c1_smoothstep_scoreflow_step_gfree(
            self, x_t, x0, carrier, t, t_next):
        """Matched F01 canonical interval step for the v87 C1 chart.

        The chart changes only neural conditioning. Decode X1, then evolve the
        exact same adaptive-g F01 Gaussian residual using the g-free ratio.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free C1 smoothstep Score--Flow step requires path_min_sigma=0."
            )
        if x_t.numel() == 0:
            return x_t, {}
        t_b = self._broadcast_time_like(t, x_t)
        tn_b = self._broadcast_time_like(t_next, x_t)
        pred_x1 = self.endpoint_from_c1_smoothstep_carrier_gfree(
            x_t, x0, carrier, t_b
        )
        mu0 = (1.0 - t_b) * x0 + t_b * pred_x1
        mu1 = (1.0 - tn_b) * x0 + tn_b * pred_x1
        residual = x_t - mu0
        base0 = (t_b * (1.0 - t_b)).clamp_min(0.0)
        base1 = (tn_b * (1.0 - tn_b)).clamp_min(0.0)
        interior = base0 > self.eps
        ratio = torch.zeros_like(base0)
        ratio = torch.where(
            interior,
            torch.sqrt(base1 / base0.clamp_min(self.eps)),
            ratio,
        )
        residual = torch.where(interior, residual, torch.zeros_like(residual))
        x_next = mu1 + ratio * residual
        with torch.no_grad():
            gain = 0.5 * t_b * (3.0 - 2.0 * t_b)
            lam = self.c1_smoothstep_lambda(t_b)
            return x_next, {
                "c1_smoothstep_gain_mean": gain.mean(),
                "c1_smoothstep_lambda_mean": lam.mean(),
                "c1_smoothstep_residual_rms": torch.sqrt(
                    residual.pow(2).mean().clamp_min(0.0)
                ),
                "c1_smoothstep_ratio_mean": ratio.mean(),
                "c1_smoothstep_decode_denom_min": (1.0 - 0.5 * lam).min(),
            }

    # ============================================================
    # v68: direct-Flow / dual-field Score-Flow coupling
    # ============================================================
    def exact_global_step_from_scaled_score(
            self, z_t, z0, mean_velocity, scaled_score,
            t, t_next, score_min_sigma=1e-2, score_active=None):
        """Exact global-R3 interval step using a learned *scaled score* q=sigma*s.

        The direct Flow field predicts the mean transport
            d_theta ~= z1-z0
            z1_flow = z0 + d_theta.

        For the Gaussian path
            q_t = sigma_t s_t = -(z_t-mu_t)/sigma_t,
        the score-implied mean satisfies
            mu_t = z_t + sigma_t q_t.

        Relative to the Flow-implied Gaussian score q_flow, an independent
        learned q_score therefore implies an endpoint correction
            delta z1 = sigma_hat / t * (q_score-q_flow).

        This is exact when:
          1) sigma_hat equals the path sigma,
          2) q_score is the true scaled score.

        To avoid the t=0 singularity and extrapolating an untrained score head,
        score_active can disable score correction outside the training window.
        The actual interval transport then uses the already-tested g-free exact
        Gaussian residual ratio, so no oracle g is needed after z1_corrected is
        formed.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "v68 exact global Score-Flow requires "
                "ABFLOW_R3_PATH_MIN_SIGMA=0."
            )
        if z_t.numel() == 0:
            return z_t, {}

        n = z_t.shape[0]
        dtype = z_t.dtype
        device = z_t.device

        t0 = torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
        t1 = torch.as_tensor(t_next, device=device, dtype=dtype).reshape(-1)
        if t0.numel() == 1:
            t0 = t0.expand(n)
        if t1.numel() == 1:
            t1 = t1.expand(n)

        d_theta = mean_velocity.to(dtype)
        z1_flow = z0 + d_theta

        # Inference-available scale estimated ONLY from the predicted Flow.
        transport_hat = torch.linalg.norm(d_theta, dim=-1)
        g_hat = self.graph_g_from_transport(transport_hat)
        sigma_hat = self.score_sigma_t(
            t0, g_hat.to(dtype), score_min_sigma
        )

        mu_flow = (1.0 - t0[:, None]) * z0 + t0[:, None] * z1_flow
        q_flow = -(z_t - mu_flow) / sigma_hat[:, None]

        if score_active is None:
            active = t0 > self.eps
        else:
            active = torch.as_tensor(
                score_active, device=device, dtype=torch.bool
            ).reshape(-1)
            if active.numel() == 1:
                active = active.expand(n)
            active = active & (t0 > self.eps)

        # Score gives a correction to the endpoint implied by Flow.
        delta_q = scaled_score.to(dtype) - q_flow
        correction = torch.zeros_like(delta_q)
        correction[active] = (
            sigma_hat[active, None]
            * delta_q[active]
            / t0[active, None].clamp_min(self.eps)
        )
        z1_corrected = z1_flow + correction

        z_next, diag = self.exact_global_scoreflow_step_gfree(
            z_t=z_t,
            z0=z0,
            pred_z1=z1_corrected,
            t=t0,
            t_next=t1,
        )
        diag.update({
            "score_active_rate": active.float().mean(),
            "score_endpoint_correction_rms": torch.sqrt(
                correction.pow(2).mean().clamp_min(0.0)
            ),
            "flow_transport_mean": transport_hat.mean(),
        })
        return z_next, diag


    # ============================================================
    # v70: SF²M-consistent stochastic global-R3 identities
    # ============================================================
    def sf2m_log_sigma_derivative(self, t_graph, t_eps=1e-2):
        """d log sigma_t / dt for sigma_t = g*sqrt(t(1-t)).

        This is exactly the coefficient used by Schrodinger-bridge CFM:
            k(t) = (1-2t) / [2t(1-t)].

        `t_eps` is a numerical sampling boundary, not a loss weight.
        Formal v69 profiles sample t in [t_eps, 1-t_eps].
        """
        t = torch.as_tensor(t_graph)
        t_safe = t.clamp(min=float(t_eps), max=1.0-float(t_eps))
        return (1.0 - 2.0*t_safe) / (
            2.0*t_safe*(1.0-t_safe)
        )

    def sf2m_global_probability_flow(
            self, z_t, z0, z1, t_graph, t_eps=1e-2):
        """Conditional probability-flow ODE target for the F01 Gaussian R3 path.

        For
            z_t = mu_t + sigma_t * eps
            mu_t = (1-t)z0 + t z1
            sigma_t = g sqrt(t(1-t)),

        the exact conditional probability-flow field is
            u_t^o = (z1-z0) + (sigma_dot/sigma)(z_t-mu_t).

        This matches the Schrodinger-bridge CFM formula used by SF²M.
        """
        t = torch.as_tensor(
            t_graph, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        if t.numel() == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])
        mu = (1.0-t[:,None])*z0 + t[:,None]*z1
        k = self.sf2m_log_sigma_derivative(
            t, t_eps=t_eps
        ).to(z_t.dtype)
        return (z1-z0) + k[:,None]*(z_t-mu)

    def sf2m_scoreflow_correction(
            self, z_t, z0, z1, t_graph, t_eps=1e-2):
        """Exact score-induced correction inside the Gaussian probability flow.

        For
            z_t = mu_t + sigma_t * eps,
            mu_t = (1-t) z0 + t z1,
            sigma_t = g * sqrt(t(1-t)),

        raw conditional score:
            s_t = -(z_t-mu_t) / sigma_t^2.

        The exact Gaussian probability flow decomposes as
            u_t^o = mu_dot - sigma_t * sigma_dot_t * s_t
                  = (z1-z0) + k(t) * (z_t-mu_t),
        where
            k(t) = sigma_dot/sigma
                 = (1-2t)/(2t(1-t)).

        Therefore the score-induced FLOW correction is
            c_t = -sigma_t*sigma_dot_t*s_t
                = k(t)*(z_t-mu_t).

        This quantity has VELOCITY units and avoids the raw-score 1/sigma
        magnitude explosion.  It is the formal v72 F03 supervision target.
        """
        t = torch.as_tensor(
            t_graph, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        if t.numel() == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])
        mu = (1.0 - t[:, None]) * z0 + t[:, None] * z1
        k = self.sf2m_log_sigma_derivative(t, t_eps=t_eps).to(z_t.dtype)
        return k[:, None] * (z_t - mu)

    def sf2m_recover_mean_displacement(
            self, z_t, z0, probability_flow, t_graph, t_eps=1e-2):
        """Recover d=z1-z0 from the canonical stochastic probability-flow field.

        Starting from
            v = d + k(t)[z_t-z0-t d],
        solve exactly:
            d = [v-k(t)(z_t-z0)]/[1-k(t)t].

        For the Brownian-bridge schedule,
            1-k(t)t = 1/[2(1-t)] > 0,
        so the inversion is unique on t in (0,1).
        """
        t = torch.as_tensor(
            t_graph, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        if t.numel() == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])
        t_safe = t.clamp(min=float(t_eps), max=1.0-float(t_eps))
        k = self.sf2m_log_sigma_derivative(
            t_safe, t_eps=t_eps
        ).to(z_t.dtype)
        denom = (1.0-k*t_safe).clamp_min(self.eps)
        return (
            probability_flow
            - k[:,None]*(z_t-z0)
        ) / denom[:,None]

    @staticmethod
    def sf2m_scaled_score_from_state(z_t, mu_t, sigma_t, eps=1e-8):
        """Return q = sigma*s = -(z_t-mu_t)/sigma.

        This is the unit-variance score target used by SF²M's weighting idea.
        For z_t=mu_t+sigma_t*epsilon, q*=-epsilon.
        """
        sigma = torch.as_tensor(
            sigma_t, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1).clamp_min(float(eps))
        return -(z_t-mu_t)/sigma[:,None]

    def sf2m_time_only_score_weight(self, t_graph, t_eps=1e-2):
        """Time-only lambda(t)=2*sqrt(t(1-t)) for raw-score regression.

        The factor 2 normalizes lambda(0.5)=1.  Critically, lambda does NOT
        contain g_graph, because g_graph depends on the latent endpoint pair in
        the donor's task-adaptive F01 path.  This preserves the standard
        conditional-score-matching optimum for the marginal score.
        """
        t = torch.as_tensor(t_graph)
        t_safe = t.clamp(min=float(t_eps), max=1.0-float(t_eps))
        return 2.0*torch.sqrt((t_safe*(1.0-t_safe)).clamp_min(0.0))

    @staticmethod
    def sf2m_forward_sde_drift(probability_flow, raw_score, diffusion_g):
        """SF²M forward SDE drift:
            b_plus = v_probability_flow + 1/2 g^2 score.
        """
        g = torch.as_tensor(
            diffusion_g,
            device=probability_flow.device,
            dtype=probability_flow.dtype,
        ).reshape(-1)
        return probability_flow + 0.5*g[:,None].square()*raw_score



    # ============================================================
    # v103: endpoint-centered Cartesian Flow / Score--Flow algebra
    # ============================================================
    @staticmethod
    def direct_cartesian_cfm_velocity(x0, x1):
        """FoldFlow/CFM Euclidean target on the donor's full Cartesian state.

        For X_t=(1-t)X0+tX1, the exact conditional velocity is X1-X0.
        No endpoint carrier and no (1-t) reweighting are introduced.
        """
        return x1 - x0

    @staticmethod
    def canonical_brownian_k(t, eps=1e-8):
        """d log sigma / dt for sigma(t)=g*sqrt(t(1-t))."""
        tt = torch.as_tensor(t)
        tt = tt.clamp(min=float(eps), max=1.0-float(eps))
        return (1.0 - 2.0 * tt) / (2.0 * tt * (1.0 - tt))

    @staticmethod
    def foldflow_scaled_g_to_raw(*, g_scaled=0.1, coordinate_scaling=0.1):
        """Convert FoldFlow's scaled R3 diffusion amplitude to Angstroms.

        FoldFlow first applies ``x_scaled = coordinate_scaling * x_raw`` and
        defines ``g`` in that scaled space.  Therefore a path implemented on
        raw Angstrom coordinates must use

            g_raw = g_scaled / coordinate_scaling.

        With the published/default FoldFlow convention g_scaled=0.1 and
        coordinate_scaling=0.1, this is exactly 1.0 Angstrom.
        """
        scale = float(coordinate_scaling)
        if scale <= 0.0:
            raise ValueError("coordinate_scaling must be positive.")
        g = float(g_scaled)
        if g <= 0.0:
            raise ValueError("g_scaled must be positive.")
        return g / scale

    @staticmethod
    def foldflow_scaled_sigma_to_raw(
            t, *, g_scaled=0.1, coordinate_scaling=0.1,
            min_sigma_scaled=0.0):
        """FoldFlow-style R3 width converted from model units to Angstroms.

        FoldFlow uses coordinate_scaling=0.1 and g=0.1 in scaled translation
        coordinates.  For the paired PCS-RC->native Score--Flow path we keep
        clean endpoints, hence the *physical* path floor defaults to zero.
        Numerical score floors remain a separate concern.
        """
        tt = torch.as_tensor(t)
        sigma_scaled = torch.sqrt(
            (float(g_scaled) ** 2 * tt * (1.0 - tt)).clamp_min(0.0)
            + float(min_sigma_scaled) ** 2
        )
        return sigma_scaled / float(coordinate_scaling)

    def canonical_cartesian_scoreflow_velocity(
            self, x_t, x0, x1, t):
        """Canonical probability-flow target for a clean-endpoint bridge.

        x_t = mu_t + sigma_t * eps,  mu_t=(1-t)x0+t x1
        u*  = (x1-x0) + (d log sigma/dt) * (x_t-mu_t)

        The stochastic residual may live in a lower-dimensional protein-aware
        translation support; the formula is valid on that support and the
        deterministic mean transport still acts on every full-atom coordinate.
        """
        tb = self._broadcast_time_like(t, x_t)
        mu = (1.0 - tb) * x0 + tb * x1
        k = self.canonical_brownian_k(tb, eps=self.eps).to(x_t.dtype)
        return (x1 - x0) + k * (x_t - mu)
