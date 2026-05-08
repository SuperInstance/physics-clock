"""PhysicsClock — Bayesian temporal inference from physics models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .estimator import TimeEstimate
from .models import PhysicsModel


class PhysicsClock:
    """
    Infer time from physics observations.

    Each physics model provides an independent clock signal.
    The joint probability P(t | all observations) gives precise time.
    """

    def __init__(self, models: list[PhysicsModel] | None = None):
        self.models: list[PhysicsModel] = models or []

    def add_model(self, model: PhysicsModel) -> None:
        """Add a physics clock model."""
        self.models.append(model)

    def _log_likelihood_grid(
        self, observations: dict[str, Any], t_grid: np.ndarray
    ) -> np.ndarray:
        """Compute log P(all observations | t) for each t in grid."""
        log_likes = np.zeros(len(t_grid))
        for model in self.models:
            for i, t in enumerate(t_grid):
                ll = model.likelihood(observations, t)
                log_likes[i] += math.log(max(ll, 1e-300))
        return log_likes

    def infer_time(
        self,
        observations: dict[str, Any],
        t_min: float = 0.0,
        t_max: float = 3600.0,
        n_points: int = 10000,
        prior: str = "uniform",
    ) -> TimeEstimate:
        """Bayesian temporal inference from all physics observations.

        Computes P(t | all observations) ∝ Π P(obs_i | t) × P(t)
        and returns the peak as best estimate with uncertainty from
        the curvature of the log-posterior.
        """
        if not self.models:
            return TimeEstimate(
                timestamp=0.0, uncertainty=float("inf"), n_clocks=0, sources={}
            )

        t_grid = np.linspace(t_min, t_max, n_points)

        # Log-likelihood from each model
        log_posterior = self._log_likelihood_grid(observations, t_grid)

        # Prior
        if prior == "jeffreys":
            # Jeffreys prior: p(t) ∝ 1/t (improper, use log-spaced grid instead)
            log_posterior -= np.log(np.maximum(t_grid, 1e-10))
        # uniform prior: no adjustment

        # Find peak
        best_idx = np.argmax(log_posterior)
        best_t = t_grid[best_idx]

        # Estimate uncertainty from curvature (second derivative of log-posterior)
        dt = t_grid[1] - t_grid[0]
        if 1 <= best_idx < n_points - 1:
            d2 = log_posterior[best_idx + 1] - 2 * log_posterior[best_idx] + log_posterior[best_idx - 1]
            d2 /= dt ** 2
            if d2 < -1e-10:
                sigma = 1.0 / math.sqrt(-d2)
            else:
                # Fall back to grid resolution
                sigma = dt * 10
        else:
            sigma = dt * 10

        # Per-model contributions
        sources: dict[str, Any] = {}
        for model in self.models:
            name = model.__class__.__name__
            est, unc = model.infer(observations)
            sources[name] = {"estimate": est, "uncertainty": unc}

        return TimeEstimate(
            timestamp=float(best_t),
            uncertainty=float(sigma),
            n_clocks=len(self.models),
            sources=sources,
        )

    def verify_temporal_parity(self, reported: dict[str, Any]) -> bool:
        """Check that reported timing matches physics expectations.

        Simplified check: compare reported eval_time against expected
        from thermal + complexity model.
        """
        from .parity import RealityParity

        parity = RealityParity()
        result = parity.check(reported)
        return result.honest
