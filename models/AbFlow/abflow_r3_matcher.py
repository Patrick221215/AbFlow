#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Patrick221215 <1427584833@qq.com>
# Date: 2026-08-18 16:49:04
# LastEditors: Patrick221215 <1427584833@qq.com>
# LastEditTime: 2026-08-24 15:45:30
# FilePath: /cjm/project/AbFlow/models/AbFlow/abflow_r3_matcher.py

"""FoldFlow-R3-inspired stochastic conditional paths for AbFlow.

This module preserves only the Euclidean R3 translation component used by the
historical PCS_RC_LC_R1_FF_R3_GLOBAL_ENDPOINT baseline.

No SO(3)/SE(3) rotational or manifold branch, optimal transport, ESM,
IPA replacement, or whole-complex center-of-mass recentering is introduced.
"""

from __future__ import annotations

import math

import torch


class AbFlowR3Matcher:
    """R3 stochastic-path helper for proposal-conditioned AbFlow training."""

    def __init__(
        self,
        *,
        transport_fraction=0.05,
        path_min_sigma=0.0,
        eps=1e-8,
    ):
        self.transport_fraction = float(transport_fraction)
        self.path_min_sigma = float(path_min_sigma)
        self.eps = float(eps)

        if not (0.0 < self.transport_fraction <= 1.0):
            raise ValueError("transport_fraction must be in (0, 1].")

        if self.path_min_sigma < 0.0:
            raise ValueError("path_min_sigma must be non-negative.")

        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")

    def graph_g_from_transport(self, transport):
        """Convert graph-level transport distance to the Brownian path width.

        The midpoint coordinate-wise RMS is calibrated to

            transport_fraction * transport / sqrt(3).

        For sigma_t^2 = g^2 * t * (1-t) + sigma_min^2,
        t = 0.5 gives sigma_mid^2 = g^2 / 4 + sigma_min^2.
        """
        target_mid_coord = (
            self.transport_fraction
            * transport
            / math.sqrt(3.0)
        )

        floor = torch.as_tensor(
            self.path_min_sigma,
            device=transport.device,
            dtype=transport.dtype,
        )

        dynamic_sq = (
            target_mid_coord.square() - floor.square()
        ).clamp_min(0.0)

        return 2.0 * torch.sqrt(dynamic_sq)

    def sigma_t(self, t_graph, g_graph):
        """Return the stochastic-path standard deviation at continuous time t."""
        t = torch.as_tensor(
            t_graph,
            device=g_graph.device,
            dtype=g_graph.dtype,
        )

        variance = (
            g_graph.square() * t * (1.0 - t)
            + self.path_min_sigma ** 2
        )

        return torch.sqrt(variance.clamp_min(0.0))

    @staticmethod
    def linear_mean(x0, x1, t_int):
        """Linear conditional-path mean: mu_t = (1-t) x0 + t x1."""
        return (1.0 - t_int) * x0 + t_int * x1

    @staticmethod
    def clean_conditional_velocity(x0, x1):
        """Clean linear-path conditional velocity: u_t = x1 - x0."""
        return x1 - x0

    @staticmethod
    def endpoint_target_for_velocity(xt, velocity, t_int):
        """Recover the endpoint implied by a velocity at state x_t.

        For a linear conditional path,

            x1 = x_t + (1-t) * u_t.
        """
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
        """ABX-style scaled DSM residual under the SAME F01 corruption kernel.

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
    # v79: current--osmotic Brownian-bridge decomposition
    # ============================================================
    def brownian_current_osmotic_fields(
            self, z_t, z0, pred_z1, t_graph):
        """Decompose a Brownian bridge into current and osmotic velocities.

        For the fixed-endpoint Brownian bridge
            dz_t = b_plus(z_t,t) dt + g dW_t,
            b_plus = (z1-z_t)/(1-t),
        the time-reversed drift is
            b_minus = (z_t-z0)/t.

        Nelson/Schrodinger decomposition gives
            v_current  = 0.5 * (b_plus + b_minus),
            v_osmotic  = 0.5 * (b_plus - b_minus)
                       = 0.5 * g^2 * score.

        Thus Flow and Score are not two learned heads: they are the symmetric
        and antisymmetric parts of the same bidirectional bridge dynamics.
        The decomposition is defined only for interior t in (0,1); callers
        should use the existing analytic boundary rules at t=0/1.
        """
        t = torch.as_tensor(
            t_graph, device=z_t.device, dtype=z_t.dtype
        ).reshape(-1)
        if t.numel() == 1 and z_t.shape[0] > 1:
            t = t.expand(z_t.shape[0])
        t_safe = t.clamp(min=self.eps, max=1.0 - self.eps)
        b_plus = (pred_z1 - z_t) / (1.0 - t_safe)[:, None]
        b_minus = (z_t - z0) / t_safe[:, None]
        current = 0.5 * (b_plus + b_minus)
        osmotic = 0.5 * (b_plus - b_minus)
        interior = (t > self.eps) & (t < 1.0 - self.eps)
        return {
            "forward_drift": b_plus,
            "backward_drift": b_minus,
            "current_velocity": current,
            "osmotic_velocity": osmotic,
            "interior": interior,
        }

    def exact_fixedg_brownian_bridge_step(
            self, z_t, pred_z1, t, t_next, g_graph, noise=None):
        """Exact conditional transition of a fixed-g Brownian bridge.

        With endpoint pred_z1 frozen on [t,t_next], the Brownian bridge
            dz = (pred_z1-z)/(1-t) dt + g dW
        has exact transition
            E[z_next|z_t] = z_t + a (pred_z1-z_t),
            a = (t_next-t)/(1-t),
            Var[z_next|z_t] = g^2 (t_next-t)(1-t_next)/(1-t).

        This provides a train/inference-matched stochastic sampler when g is
        fixed and known at inference.  No score head is needed: equivalently,
        b_plus = v_current + v_osmotic and v_osmotic=(g^2/2) score.
        """
        if z_t.numel() == 0:
            return z_t, {}
        n = z_t.shape[0]
        t0 = torch.as_tensor(t, device=z_t.device, dtype=z_t.dtype).reshape(-1)
        t1 = torch.as_tensor(t_next, device=z_t.device, dtype=z_t.dtype).reshape(-1)
        if t0.numel() == 1:
            t0 = t0.expand(n)
        if t1.numel() == 1:
            t1 = t1.expand(n)
        g = torch.as_tensor(g_graph, device=z_t.device, dtype=z_t.dtype).reshape(-1)
        if g.numel() == 1:
            g = g.expand(n)
        remain = (1.0 - t0).clamp_min(self.eps)
        delta = (t1 - t0).clamp_min(0.0)
        alpha = (delta / remain).clamp(0.0, 1.0)
        mean = z_t + alpha[:, None] * (pred_z1 - z_t)
        var = g.square() * delta * (1.0 - t1).clamp_min(0.0) / remain
        std = torch.sqrt(var.clamp_min(0.0))
        if noise is None:
            noise = torch.randn_like(z_t)
        z_next = mean + std[:, None] * noise
        final = t1 >= 1.0 - self.eps
        z_next = torch.where(final[:, None], pred_z1, z_next)
        return z_next, {
            "bridge_alpha": alpha,
            "bridge_std": std,
            "stochastic_step_rms": torch.sqrt(
                (std.square()).mean().clamp_min(0.0)
            ),
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
        AbFlow's task-adaptive F01 path.  This preserves the standard
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


