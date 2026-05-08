"""physics-clock — Temporal Inference from Physics Models."""

from .clock import PhysicsClock
from .estimator import TimeEstimate
from .models import (
    AbsorptionClock,
    DopplerClock,
    PhysicsModel,
    PropagationClock,
    SiliconClock,
    SoundSpeedClock,
    ThermalClock,
)
from .parity import ParityResult, RealityParity

__all__ = [
    "PhysicsClock",
    "TimeEstimate",
    "PhysicsModel",
    "SoundSpeedClock",
    "AbsorptionClock",
    "ThermalClock",
    "PropagationClock",
    "DopplerClock",
    "SiliconClock",
    "RealityParity",
    "ParityResult",
]
