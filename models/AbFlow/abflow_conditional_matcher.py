#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Centralized full-atom conditional Flow-Matching algebra for AbFlow.

This module deliberately contains no neural network and no random-number calls.
It is a single source of truth for the deterministic path/sampler algebra shared
by endpoint FM, SATC and inference.  Keeping it RNG-free is important because
Stage-2 experiments must not change the stochastic call order of the original
v54 implementation simply because the formulas were refactored.
"""

from __future__ import annotations

import torch


class AbFlowConditionalMatcher:
    """Linear proposal-to-endpoint conditional flow in full-atom Cartesian space.

    Time convention (kept identical to current AbFlow):
        t = 0: proposal/source X0
        t = 1: clean/native endpoint X1

    The network remains endpoint-parameterized.  It predicts X1_hat and the
    bridge velocity is induced analytically as
        v_theta(Xt, t) = (X1_hat - Xt) / max(1-t, min_sigma).

    No independent score head or velocity head is introduced here.
    """

    def __init__(self, min_sigma: float = 1e-2, eps: float = 1e-8):
        min_sigma = float(min_sigma)
        eps = float(eps)
        if not (0.0 < min_sigma < 1.0):
            raise ValueError("min_sigma must be in (0, 1).")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.min_sigma = min_sigma
        self.eps = eps

    @staticmethod
    def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Deterministic linear conditional path: Xt=(1-t)X0+tX1."""
        return (1.0 - t) * x0 + t * x1

    @staticmethod
    def clean_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Exact conditional velocity of the linear path: u*=X1-X0."""
        return x1 - x0

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Numerically protected bridge denominator; the true path still ends at t=1."""
        return (1.0 - t).clamp_min(self.min_sigma)

    def endpoint_velocity(
        self,
        xt: torch.Tensor,
        endpoint: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert an endpoint prediction into the bridge vector field."""
        return (endpoint - xt) / self.sigma(t)

    def correction_target(
        self,
        gamma: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Analytic pull-back correction for Zt=mu_t+gamma*eps."""
        gamma = torch.as_tensor(gamma, device=eps.device, dtype=eps.dtype)
        return -(gamma / self.sigma(t)) * eps

    def bridge_step(
        self,
        xt: torch.Tensor,
        endpoint: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """One deterministic Euler step of the endpoint-induced bridge."""
        return xt + self.endpoint_velocity(xt, endpoint, t) * dt

    @staticmethod
    def categorical_refresh_probability(t: torch.Tensor, dt: torch.Tensor) -> float:
        """CTMC refresh probability for the same linear categorical path."""
        t_value = float(t)
        dt_value = float(dt)
        return min(1.0, dt_value / max(1e-8, 1.0 - t_value))
