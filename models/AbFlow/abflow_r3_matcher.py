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
