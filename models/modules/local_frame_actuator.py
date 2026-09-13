#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Local-frame full-atom geometry actuator for AbFlow.

The representation trunk remains invariant/relational. Geometry is changed only
through residue-local actions, following the AlphaFold/AbX StructureModule
factorization:

    invariant state -> local rigid update -> equivariant Cartesian action.

For full-atom AbFlow, the backbone N/CA/C/O channels share one residue rigid
update. Side-chain channels additionally receive a local-frame residual so the
formal 14-slot Cartesian target remains expressive without restoring the
edge-distance-times-scalar feedback path of AM-EGNN.
"""

import math
import torch
import torch.nn as nn


def _safe_unit(v, eps):
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / n.clamp_min(eps), n


def _quat_vec_to_rot(vec):
    """AlphaFold-style local quaternion update q = normalize([1, vec])."""
    one = torch.ones_like(vec[..., :1])
    q = torch.cat([one, vec], dim=-1)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return torch.stack([
        ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy),
        2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx),
        2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz,
    ], dim=-1).reshape(*vec.shape[:-1], 3, 3)


def _quantile(x, q, zero):
    if x.numel() == 0:
        return zero
    return torch.quantile(x.detach().float(), q).to(dtype=zero.dtype, device=zero.device)


class LocalFrameFullAtomActuator(nn.Module):
    """Equivariant local-frame actuator with zero-identity initialization.

    No coordinate clipping, no tanh authority bound, no learned distance
    multiplier and no hand-chosen displacement scale are used.
    """

    def __init__(self, hidden_nf, channel_nf, frame_eps=1.0e-6,
                 coordinate_scale=0.1, backbone_channels=4):
        super().__init__()
        self.hidden_nf = int(hidden_nf)
        self.channel_nf = int(channel_nf)
        self.frame_eps = float(frame_eps)
        self.coordinate_scale = float(coordinate_scale)
        if self.coordinate_scale <= 0.0:
            raise ValueError('coordinate_scale must be > 0')
        self.angstrom_per_internal = 1.0 / self.coordinate_scale
        self.backbone_channels = int(backbone_channels)

        self.state_norm = nn.LayerNorm(self.hidden_nf)
        self.rigid_head = nn.Linear(self.hidden_nf, 6)
        nn.init.zeros_(self.rigid_head.weight)
        nn.init.zeros_(self.rigid_head.bias)

        # Full-atom extension: backbone channels are governed only by the shared
        # rigid action. Side-chain slots get an atom-local residual expressed in
        # the same N-CA-C frame. The final projection is zero-initialized, so the
        # complete actuator is exactly identity at initialization.
        self.atom_norm = nn.LayerNorm(self.channel_nf)
        self.atom_hidden = nn.Linear(self.hidden_nf + self.channel_nf, self.hidden_nf)
        self.atom_out = nn.Linear(self.hidden_nf, 3)
        nn.init.zeros_(self.atom_out.weight)
        nn.init.zeros_(self.atom_out.bias)
        self.act = nn.SiLU()
        self.last_diagnostics = {}

    def _frame(self, coord, atom_mask):
        if coord.shape[1] < 3:
            raise ValueError('local N-CA-C frame requires at least 3 atom channels')
        n = coord[:, 0]
        ca = coord[:, 1]
        c = coord[:, 2]
        e1, n1 = _safe_unit(c - ca, self.frame_eps)
        plane = torch.cross(e1, n - ca, dim=-1)
        e3, n3 = _safe_unit(plane, self.frame_eps)
        e2 = torch.cross(e3, e1, dim=-1)
        R = torch.stack([e1, e2, e3], dim=-1)  # columns: local -> global
        has_atoms = atom_mask[:, :3].all(dim=-1)
        valid = has_atoms & (n1.squeeze(-1) > self.frame_eps) & (n3.squeeze(-1) > self.frame_eps)
        return R, ca, valid

    @staticmethod
    def _backbone_pair_distances(x, mask, n_backbone):
        vals, valid = [], []
        for a in range(n_backbone):
            for b in range(a + 1, n_backbone):
                vals.append(torch.linalg.norm(x[:, a] - x[:, b], dim=-1))
                valid.append(mask[:, a] & mask[:, b])
        if not vals:
            return x.new_zeros((x.shape[0], 0)), mask.new_zeros((x.shape[0], 0))
        return torch.stack(vals, dim=-1), torch.stack(valid, dim=-1)

    def forward(self, h, coord, channel_attr, channel_weights, movable_mask,
                capture_diagnostics=False):
        if coord.dim() != 3 or coord.shape[-1] != 3:
            raise ValueError(f'coord must be [N,C,3], got {tuple(coord.shape)}')
        if h.shape[0] != coord.shape[0]:
            raise ValueError('h/coord residue dimension mismatch')
        atom_mask = channel_weights != 0
        movable_mask = movable_mask.to(device=coord.device, dtype=torch.bool)
        if movable_mask.shape != (coord.shape[0],):
            raise ValueError('movable_mask must be [N]')

        R, origin, frame_valid = self._frame(coord, atom_mask)
        # Keep the frame differentiable. The supplied AbX variants disagree on
        # rotation stop-gradient (folding.py detaches between iterations whereas
        # score_network.py does not), so V216 does not introduce that ambiguous
        # optimization choice as a new scientific factor.
        R_action = R
        active_residue = movable_mask & frame_valid

        h_norm = self.state_norm(h)
        rigid = self.rigid_head(h_norm)
        rot_vec, trans_local = rigid[..., :3], rigid[..., 3:]
        delta_rot = _quat_vec_to_rot(rot_vec)
        I = torch.eye(3, device=coord.device, dtype=coord.dtype).expand(coord.shape[0], 3, 3)

        centered = coord - origin[:, None, :]
        local = torch.einsum('naj,njk->nak', centered, R_action)
        # Exact zero-identity form: (R_delta - I)r + t. This avoids reconstructing
        # x from a numerically estimated frame when the zero-initialized head emits
        # the identity action.
        rigid_delta_local = torch.einsum('nij,naj->nai', delta_rot - I, local)
        rigid_delta_local = rigid_delta_local + trans_local[:, None, :]

        atom_h = h_norm[:, None, :].expand(-1, coord.shape[1], -1)
        atom_a = self.atom_norm(channel_attr)
        atom_delta_local = self.atom_out(self.act(self.atom_hidden(
            torch.cat([atom_h, atom_a], dim=-1)
        )))
        channel_index = torch.arange(coord.shape[1], device=coord.device)
        sidechain_slot = channel_index[None, :] >= min(self.backbone_channels, coord.shape[1])
        sidechain_active = sidechain_slot & atom_mask & active_residue[:, None]
        atom_delta_local = torch.where(
            sidechain_active[..., None], atom_delta_local, torch.zeros_like(atom_delta_local)
        )

        delta_local = rigid_delta_local + atom_delta_local
        delta_global = torch.einsum('nij,naj->nai', R_action, delta_local)
        active_atom = active_residue[:, None] & atom_mask
        delta_global = torch.where(active_atom[..., None], delta_global, torch.zeros_like(delta_global))
        out = coord + delta_global

        if capture_diagnostics:
            with torch.no_grad():
                zero = coord.new_zeros(())
                trans_norm = torch.linalg.norm(trans_local, dim=-1)[active_residue]
                qnorm = torch.linalg.norm(rot_vec, dim=-1)
                rot_angle = (2.0 * torch.atan(qnorm) * (180.0 / math.pi))[active_residue]
                side_norm = torch.linalg.norm(atom_delta_local, dim=-1)[sidechain_active]
                update_norm = torch.linalg.norm(delta_global, dim=-1)[active_atom]
                fixed_atom = (~movable_mask)[:, None] & atom_mask
                fixed_abs = delta_global[fixed_atom].abs().amax() if bool(fixed_atom.any()) else zero

                RtR = torch.matmul(R.transpose(-1, -2), R)
                eye = torch.eye(3, device=coord.device, dtype=coord.dtype)[None]
                ortho = (RtR - eye).abs().amax(dim=(-2, -1))
                ortho = ortho[frame_valid]

                nb = min(self.backbone_channels, coord.shape[1])
                before_d, bb_valid = self._backbone_pair_distances(coord, atom_mask, nb)
                after_d, _ = self._backbone_pair_distances(out, atom_mask, nb)
                bb_valid = bb_valid & active_residue[:, None]
                bb_err = (after_d - before_d).abs()[bb_valid]

                def stats(prefix, x, scale=1.0):
                    return {
                        f'{prefix}_p50': _quantile(x * scale, 0.50, zero),
                        f'{prefix}_p95': _quantile(x * scale, 0.95, zero),
                        f'{prefix}_p99': _quantile(x * scale, 0.99, zero),
                        f'{prefix}_max': (x * scale).amax().to(zero) if x.numel() else zero,
                    }

                diag = {
                    'frame_valid_fraction': frame_valid.float().mean().to(zero),
                    'movable_frame_valid_fraction': (
                        frame_valid[movable_mask].float().mean().to(zero)
                        if bool(movable_mask.any()) else zero
                    ),
                    'frame_orthogonality_error_max': ortho.amax().to(zero) if ortho.numel() else zero,
                    'fixed_atom_update_absmax_A': fixed_abs.to(zero) * self.angstrom_per_internal,
                    'backbone_rigid_distance_error_max_A': bb_err.amax().to(zero) * self.angstrom_per_internal if bb_err.numel() else zero,
                    'rigid_head_weight_rms': self.rigid_head.weight.detach().float().square().mean().sqrt().to(zero),
                    'atom_head_weight_rms': self.atom_out.weight.detach().float().square().mean().sqrt().to(zero),
                }
                diag.update(stats('translation_norm_internal', trans_norm))
                diag.update(stats('translation_norm_A', trans_norm, self.angstrom_per_internal))
                diag.update(stats('rotation_angle_deg', rot_angle))
                diag.update(stats('sidechain_residual_norm_A', side_norm, self.angstrom_per_internal))
                diag.update(stats('atom_update_norm_A', update_norm, self.angstrom_per_internal))
                diag['coord_update_absmax'] = delta_global.detach().abs().amax() if delta_global.numel() else zero
                self.last_diagnostics = diag
        else:
            self.last_diagnostics = {}
        return out

class HierarchicalLocalFrameFullAtomActuator(nn.Module):
    """Evidence-backed hierarchical full-atom local-frame actuator (V217).

    Factorization:
        invariant residue state
          -> zero-init coarse local rigid action (AbX/AlphaFold-style)
          -> zero-init atom-local internal correction in the residue frame
          -> equivariant atom14 Cartesian endpoint update.

    The internal correction is active for every observed atom except CA.  CA is
    the residue-frame origin, so its endpoint displacement is represented by the
    coarse translation.  N/C/O and side-chain atoms may change their local
    coordinates, restoring the full atom14 endpoint expressivity that a
    rigid-only backbone actuator lacks.

    No coordinate clipping, tanh authority bound, learned edge-distance
    multiplier, trust radius, or hand-chosen displacement scale is introduced.
    """

    def __init__(self, hidden_nf, channel_nf, frame_eps=1.0e-6,
                 coordinate_scale=0.1, ca_channel=1, backbone_channels=4):
        super().__init__()
        self.hidden_nf = int(hidden_nf)
        self.channel_nf = int(channel_nf)
        self.frame_eps = float(frame_eps)
        self.coordinate_scale = float(coordinate_scale)
        if self.coordinate_scale <= 0.0:
            raise ValueError('coordinate_scale must be > 0')
        self.angstrom_per_internal = 1.0 / self.coordinate_scale
        self.ca_channel = int(ca_channel)
        self.backbone_channels = int(backbone_channels)

        self.state_norm = nn.LayerNorm(self.hidden_nf)
        self.rigid_head = nn.Linear(self.hidden_nf, 6)
        nn.init.zeros_(self.rigid_head.weight)
        nn.init.zeros_(self.rigid_head.bias)

        # FAMPNN-style local Cartesian internal state, adapted to AbFlow atom14.
        # The current atom-local coordinate is an explicit input because the
        # denoiser must know the current full-atom state it is correcting.
        self.atom_norm = nn.LayerNorm(self.channel_nf)
        self.atom_hidden = nn.Linear(
            self.hidden_nf + self.channel_nf + 3, self.hidden_nf
        )
        self.atom_out = nn.Linear(self.hidden_nf, 3)
        nn.init.zeros_(self.atom_out.weight)
        nn.init.zeros_(self.atom_out.bias)
        self.act = nn.SiLU()
        self.last_diagnostics = {}

    def _frame(self, coord, atom_mask):
        if coord.shape[1] < 3:
            raise ValueError('local N-CA-C frame requires at least 3 atom channels')
        n = coord[:, 0]
        ca = coord[:, self.ca_channel]
        c = coord[:, 2]
        e1, n1 = _safe_unit(c - ca, self.frame_eps)
        plane = torch.cross(e1, n - ca, dim=-1)
        e3, n3 = _safe_unit(plane, self.frame_eps)
        e2 = torch.cross(e3, e1, dim=-1)
        R = torch.stack([e1, e2, e3], dim=-1)  # columns: local -> global
        has_atoms = atom_mask[:, [0, self.ca_channel, 2]].all(dim=-1)
        valid = (
            has_atoms
            & (n1.squeeze(-1) > self.frame_eps)
            & (n3.squeeze(-1) > self.frame_eps)
        )
        return R, ca, valid

    @staticmethod
    def _pair_distances(x, mask, n_channels):
        vals, valid = [], []
        for a in range(n_channels):
            for b in range(a + 1, n_channels):
                vals.append(torch.linalg.norm(x[:, a] - x[:, b], dim=-1))
                valid.append(mask[:, a] & mask[:, b])
        if not vals:
            return x.new_zeros((x.shape[0], 0)), mask.new_zeros((x.shape[0], 0))
        return torch.stack(vals, dim=-1), torch.stack(valid, dim=-1)

    @staticmethod
    def _apply_local_action(coord, R, origin, delta_rot, trans_local,
                            internal_delta_local, active_residue, atom_mask):
        """Apply a hierarchical action with exact zero-identity semantics.

        Let r be current atom coordinates in the current residue frame.  The
        updated frame is R' = R DeltaR and the updated origin is
        o' = o + R t.  Internal coordinates become r' = r + delta_r.  Thus

            x' = o' + R' r'.

        Rewriting x'-x avoids reconstructing unchanged atoms when all heads are
        zero and gives exact identity at initialization.
        """
        centered = coord - origin[:, None, :]
        local = torch.einsum('naj,njk->nak', centered, R)
        local_next = local + internal_delta_local
        rotated_local_next = torch.einsum('nij,naj->nai', delta_rot, local_next)
        delta_parent = rotated_local_next - local + trans_local[:, None, :]
        delta_global = torch.einsum('nij,naj->nai', R, delta_parent)
        active_atom = active_residue[:, None] & atom_mask
        delta_global = torch.where(
            active_atom[..., None], delta_global, torch.zeros_like(delta_global)
        )
        return coord + delta_global, delta_global, local

    def forward(self, h, coord, channel_attr, channel_weights, movable_mask,
                capture_diagnostics=False):
        if coord.dim() != 3 or coord.shape[-1] != 3:
            raise ValueError(f'coord must be [N,C,3], got {tuple(coord.shape)}')
        if h.shape[0] != coord.shape[0]:
            raise ValueError('h/coord residue dimension mismatch')
        if channel_attr.shape[:2] != coord.shape[:2]:
            raise ValueError('channel_attr must share [N,C] with coord')
        atom_mask = channel_weights != 0
        movable_mask = movable_mask.to(device=coord.device, dtype=torch.bool)
        if movable_mask.shape != (coord.shape[0],):
            raise ValueError('movable_mask must be [N]')
        if not (0 <= self.ca_channel < coord.shape[1]):
            raise ValueError('CA channel is outside the coordinate channel axis')

        R, origin, frame_valid = self._frame(coord, atom_mask)
        active_residue = movable_mask & frame_valid

        h_norm = self.state_norm(h)
        rigid = self.rigid_head(h_norm)
        rot_vec, trans_local = rigid[..., :3], rigid[..., 3:]
        delta_rot = _quat_vec_to_rot(rot_vec)

        centered = coord - origin[:, None, :]
        local = torch.einsum('naj,njk->nak', centered, R)
        atom_h = h_norm[:, None, :].expand(-1, coord.shape[1], -1)
        atom_a = self.atom_norm(channel_attr)
        atom_delta_local = self.atom_out(self.act(self.atom_hidden(
            torch.cat([atom_h, atom_a, local], dim=-1)
        )))

        # CA is the local-frame origin.  Its internal coordinate is fixed to zero;
        # its endpoint motion is represented exactly by the coarse translation.
        channel_index = torch.arange(coord.shape[1], device=coord.device)
        internal_slot = channel_index[None, :] != self.ca_channel
        internal_active = internal_slot & atom_mask & active_residue[:, None]
        atom_delta_local = torch.where(
            internal_active[..., None], atom_delta_local,
            torch.zeros_like(atom_delta_local)
        )

        out, delta_global, _ = self._apply_local_action(
            coord, R, origin, delta_rot, trans_local,
            atom_delta_local, active_residue, atom_mask,
        )

        if capture_diagnostics:
            with torch.no_grad():
                zero = coord.new_zeros(())
                trans_norm = torch.linalg.norm(trans_local, dim=-1)[active_residue]
                qnorm = torch.linalg.norm(rot_vec, dim=-1)
                rot_angle = (
                    2.0 * torch.atan(qnorm) * (180.0 / math.pi)
                )[active_residue]

                internal_norm = torch.linalg.norm(atom_delta_local, dim=-1)
                internal_all = internal_norm[internal_active]
                nb = min(self.backbone_channels, coord.shape[1])
                bb_slot = (channel_index[None, :] < nb) & internal_slot
                bb_active = bb_slot & atom_mask & active_residue[:, None]
                sc_slot = channel_index[None, :] >= nb
                sc_active = sc_slot & atom_mask & active_residue[:, None]
                internal_bb = internal_norm[bb_active]
                internal_sc = internal_norm[sc_active]

                update_norm = torch.linalg.norm(delta_global, dim=-1)
                active_atom = active_residue[:, None] & atom_mask
                update_active = update_norm[active_atom]
                fixed_atom = (~movable_mask)[:, None] & atom_mask
                fixed_abs = (
                    delta_global[fixed_atom].abs().amax()
                    if bool(fixed_atom.any()) else zero
                )

                RtR = torch.matmul(R.transpose(-1, -2), R)
                eye = torch.eye(3, device=coord.device, dtype=coord.dtype)[None]
                ortho = (RtR - eye).abs().amax(dim=(-2, -1))[frame_valid]

                # Coarse component alone must be exactly rigid.  Final backbone
                # geometry may change by design through internal local correction.
                zero_internal = torch.zeros_like(atom_delta_local)
                rigid_out, _, _ = self._apply_local_action(
                    coord, R, origin, delta_rot, trans_local,
                    zero_internal, active_residue, atom_mask,
                )
                before_d, pair_valid = self._pair_distances(coord, atom_mask, nb)
                rigid_d, _ = self._pair_distances(rigid_out, atom_mask, nb)
                final_d, _ = self._pair_distances(out, atom_mask, nb)
                pair_valid = pair_valid & active_residue[:, None]
                rigid_err = (rigid_d - before_d).abs()[pair_valid]
                final_internal_change = (final_d - rigid_d).abs()[pair_valid]

                ca_delta_local = atom_delta_local[:, self.ca_channel]
                ca_internal_abs = (
                    ca_delta_local.abs().amax() if ca_delta_local.numel() else zero
                )

                def stats(prefix, x, scale=1.0):
                    return {
                        f'{prefix}_p50': _quantile(x * scale, 0.50, zero),
                        f'{prefix}_p95': _quantile(x * scale, 0.95, zero),
                        f'{prefix}_p99': _quantile(x * scale, 0.99, zero),
                        f'{prefix}_max': (
                            (x * scale).amax().to(zero) if x.numel() else zero
                        ),
                    }

                diag = {
                    'frame_valid_fraction': frame_valid.float().mean().to(zero),
                    'movable_frame_valid_fraction': (
                        frame_valid[movable_mask].float().mean().to(zero)
                        if bool(movable_mask.any()) else zero
                    ),
                    'frame_orthogonality_error_max': (
                        ortho.amax().to(zero) if ortho.numel() else zero
                    ),
                    'fixed_atom_update_absmax_A': (
                        fixed_abs.to(zero) * self.angstrom_per_internal
                    ),
                    'ca_internal_update_absmax_A': (
                        ca_internal_abs.to(zero) * self.angstrom_per_internal
                    ),
                    'coarse_rigid_distance_error_max_A': (
                        rigid_err.amax().to(zero) * self.angstrom_per_internal
                        if rigid_err.numel() else zero
                    ),
                    'rigid_head_weight_rms': (
                        self.rigid_head.weight.detach().float().square().mean().sqrt().to(zero)
                    ),
                    'internal_head_weight_rms': (
                        self.atom_out.weight.detach().float().square().mean().sqrt().to(zero)
                    ),
                }
                diag.update(stats('translation_norm_internal', trans_norm))
                diag.update(stats('translation_norm_A', trans_norm, self.angstrom_per_internal))
                diag.update(stats('rotation_angle_deg', rot_angle))
                diag.update(stats('internal_residual_norm_A', internal_all, self.angstrom_per_internal))
                diag.update(stats('backbone_internal_residual_norm_A', internal_bb, self.angstrom_per_internal))
                diag.update(stats('sidechain_internal_residual_norm_A', internal_sc, self.angstrom_per_internal))
                diag.update(stats('final_backbone_internal_distance_change_A', final_internal_change, self.angstrom_per_internal))
                diag.update(stats('atom_update_norm_A', update_active, self.angstrom_per_internal))
                diag['coord_update_absmax'] = (
                    delta_global.detach().abs().amax() if delta_global.numel() else zero
                )
                self.last_diagnostics = diag
        else:
            self.last_diagnostics = {}
        return out

