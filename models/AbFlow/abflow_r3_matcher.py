#!/usr/bin/python
# -*- coding:utf-8 -*-
"""FoldFlow-R3-inspired stochastic conditional paths for AbFlow.

Only Euclidean translation ideas are absorbed.  No SO(3), SE(3), OT, ESM,
IPA replacement, or whole-complex COM recentering is introduced.

AbFlow time convention:
    t=0 : PCS-RC proposal/source X0
    t=1 : native endpoint X1

The current AbFlow network stays endpoint-parameterized.  For an arbitrary
velocity target u*, its equivalent endpoint-like target is
    Y* = Xt + (1-t) u*.
This lets us test FoldFlow's R3 conditional-flow target without adding a new
velocity head or changing the sampler interface.
"""
import math
import torch


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
        """Encode a velocity target through AbFlow's endpoint parameterization."""
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
    # v93 / U11: source-flat Hermite phase-matched canonical carrier
    # ============================================================
    @staticmethod
    def source_flat_hermite_gain(t, transition_t=0.5):
        """Return the carrier residual gain c(t) and lambda(t)=2 t c(t).

        Scientific constraints
        ----------------------
        The carrier family is
            P_t = X1 + c(t) [Xt - mu_t],
            mu_t = (1-t) X0 + t X1.

        Empirical evidence motivating U11:
          * U03 (c=1/2) improved absolute H3 placement.
          * U02's natural canonical carrier c=1/(2t) retained the strongest
            sequence/interface performance in the contraction phase.
          * U05 suppressed source sensitivity toward zero and did not preserve
            U02's interior carrier response.

        We therefore match U03 at the source in both value and first derivative,
        and match U02 at the intrinsic Gaussian phase boundary T=1/2 in both
        value and first derivative:
            c(0)=1/2,      c'(0)=0,
            c(T)=1/(2T),   c'(T)=-1/(2T^2).

        The unique cubic on [0,T] is
            c(t) = 1/2
                 + (4-3T)/(2T^3) t^2
                 + (T-3/2)/T^4 t^3.

        Formal U11 fixes T=1/2 (not a tuned hyperparameter), giving
            c(t)=1/2 + 10 t^2 - 16 t^3,    t <= 1/2
            c(t)=1/(2t),                    t >= 1/2.

        lambda(t)=2tc(t) is in [0,1] for T=1/2, so endpoint decoding uses
        denominator 1-lambda/2 in [1/2,1].  The chart is C1-matched to the
        exact U02 natural canonical chart at t=1/2 and has no U02 t=0.20 seam.
        """
        T = float(transition_t)
        if abs(T - 0.5) > 1e-12:
            raise ValueError(
                "Formal source-flat Hermite carrier fixes transition_t=0.5; "
                "do not tune this value."
            )
        tt = torch.as_tensor(t)
        early = tt <= T

        # Source-flat cubic in c-space.  For T=1/2:
        # c=0.5 + 10 t^2 - 16 t^3.
        c_early = 0.5 + 10.0 * tt.square() - 16.0 * tt.pow(3)
        t_safe = tt.clamp_min(torch.finfo(tt.dtype).eps)
        c_late = 0.5 / t_safe
        gain = torch.where(early, c_early, c_late)
        lam = 2.0 * tt * gain

        # Numerical guard only; the analytic formal schedule already satisfies
        # lambda in [0,1].
        lam = lam.clamp(0.0, 1.0)
        return gain, lam

    def source_flat_hermite_carrier_target_gfree(
            self, x_t, x0, x1, t, transition_t=0.5):
        """U11 carrier target with one semantic quantity at every refinement round.

        P*=X1+c(t)(Xt-mu_t), where c(t) is source-flat U03-matched for the
        source phase and is EXACTLY U02's natural canonical gain for t>=1/2.

        This changes only the neural output chart.  The F01 stochastic state,
        PCS-RC source, full-atom backbone and canonical Gaussian sampler physics
        are unchanged.  No independent score/flow head, auxiliary loss, hard
        t=0.20 switch or round-index-dependent target is introduced.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free U11 carrier requires path_min_sigma=0."
            )
        t_b = self._broadcast_time_like(t, x_t)
        mu = (1.0 - t_b) * x0 + t_b * x1
        residual = x_t - mu
        gain, _ = self.source_flat_hermite_gain(
            t_b, transition_t=transition_t
        )
        return x1 + gain * residual

    def endpoint_from_source_flat_hermite_carrier_gfree(
            self, x_t, x0, carrier, t, transition_t=0.5):
        """Decode the clean endpoint implied by the U11 carrier.

        From
            P = X1 + c[Xt-(1-t)X0-tX1]
              = (1-ct)X1 + c[Xt-(1-t)X0],
        therefore
            X1 = [P-c(Xt-(1-t)X0)] / (1-ct).

        Since lambda=2tc is in [0,1], 1-ct=1-lambda/2 is in [1/2,1].
        """
        t_b = self._broadcast_time_like(t, x_t)
        gain, lam = self.source_flat_hermite_gain(
            t_b, transition_t=transition_t
        )
        denom = (1.0 - 0.5 * lam).clamp_min(0.5)
        known = gain * (x_t - (1.0 - t_b) * x0)
        return (carrier - known) / denom

    def exact_source_flat_hermite_scoreflow_step_gfree(
            self, x_t, x0, carrier, t, t_next, transition_t=0.5):
        """Matched canonical Gaussian interval step for U11.

        1) Decode the endpoint from the U11 carrier.
        2) Freeze that endpoint on [t,t_next].
        3) Evolve the exact F01 Gaussian residual with
             sqrt[t_next(1-t_next)/(t(1-t))].

        This is the same g-free canonical residual-ratio physics used by U03/U05
        after decoding.  At t_next=1 it returns the decoded endpoint exactly.
        """
        if abs(float(self.path_min_sigma)) > self.eps:
            raise ValueError(
                "g-free U11 Score--Flow step requires path_min_sigma=0."
            )
        if x_t.numel() == 0:
            return x_t, {}

        t_b = self._broadcast_time_like(t, x_t)
        tn_b = self._broadcast_time_like(t_next, x_t)
        pred_x1 = self.endpoint_from_source_flat_hermite_carrier_gfree(
            x_t, x0, carrier, t_b, transition_t=transition_t
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
            gain, lam = self.source_flat_hermite_gain(
                t_b, transition_t=transition_t
            )
            return x_next, {
                "source_flat_hermite_gain_mean": gain.mean(),
                "source_flat_hermite_lambda_mean": lam.mean(),
                "source_flat_hermite_residual_rms": torch.sqrt(
                    residual.pow(2).mean().clamp_min(0.0)
                ),
                "source_flat_hermite_ratio_mean": ratio.mean(),
                "source_flat_hermite_decode_denom_min": (
                    1.0 - 0.5 * lam
                ).min(),
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


