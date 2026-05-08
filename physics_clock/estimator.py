"""TimeEstimate — result of temporal inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeEstimate:
    """Best estimate of time from Bayesian fusion of physics clocks."""

    timestamp: float  # best estimate (seconds since epoch)
    uncertainty: float  # 1σ uncertainty (seconds)
    n_clocks: int  # how many independent clocks contributed
    sources: dict[str, Any] = field(default_factory=dict)

    @property
    def precision(self) -> str:
        if self.uncertainty < 1e-6:
            return "microsecond"
        if self.uncertainty < 1e-3:
            return "millisecond"
        if self.uncertainty < 1:
            return "sub-second"
        if self.uncertainty < 60:
            return "second"
        return "minute"
