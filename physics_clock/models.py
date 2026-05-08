"""Physics clock models — each provides an independent clock signal."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .estimator import TimeEstimate


class PhysicsModel(ABC):
    """Base class for all physics clock models."""

    @abstractmethod
    def likelihood(self, observation: Any, t: float) -> float:
        """Return P(observation | t) — likelihood of seeing this observation at time t."""

    @abstractmethod
    def infer(self, observations: dict[str, Any]) -> tuple[float, float]:
        """Return (estimate, uncertainty) from observations."""

    def contribute(self, observations: dict[str, Any]) -> tuple[float, float]:
        """Convenience: infer + return (estimate, uncertainty)."""
        return self.infer(observations)


class SoundSpeedClock(PhysicsModel):
    """UNESCO/Chen-Millero sound speed → depth-dependent clock.

    Sound speed in seawater depends on temperature, salinity, pressure (depth).
    By measuring sound speed, we can infer depth → infer time for a
    diving vehicle with known dive profile.
    """

    # Simplified Chen-Millero coefficients
    # Real formula is more complex; this captures the physics
    C0 = 1449.05  # base speed m/s at 0°C, 35ppt, 0m depth

    def sound_speed(self, temperature: float, salinity: float = 35.0, depth: float = 0.0) -> float:
        """Compute sound speed using simplified Chen-Millero equation."""
        t = temperature
        s = salinity
        d = depth
        c = (
            self.C0
            + 4.57 * t
            - 0.0521 * t ** 2
            + 0.00023 * t ** 3
            + 1.34 * (s - 35.0)
            + 0.016 * d
        )
        return c

    def likelihood(self, observation: dict, t: float) -> float:
        """P(measured_speed | t) assuming linear dive profile."""
        expected_depth = observation.get("descent_rate", 0.5) * t
        expected_temp = observation.get("surface_temp", 20.0) - 0.1 * expected_depth
        expected_speed = self.sound_speed(expected_temp, observation.get("salinity", 35.0), expected_depth)
        measured = observation.get("measured_speed", 1500.0)
        sigma = observation.get("speed_sigma", 1.0)
        return math.exp(-0.5 * ((measured - expected_speed) / sigma) ** 2)

    def infer(self, observations: dict) -> tuple[float, float]:
        """Infer elapsed time from sound speed measurement."""
        measured_speed = observations.get("measured_speed", 1500.0)
        descent_rate = observations.get("descent_rate", 0.5)
        surface_temp = observations.get("surface_temp", 20.0)
        salinity = observations.get("salinity", 35.0)

        # Solve for depth that gives measured speed
        # Iterative: c(T(d), S, d) = measured_speed
        # Start from a reasonable initial guess
        c_surface = self.sound_speed(surface_temp, salinity, 0.0)
        # Rough estimate: ~0.01 m/s per meter depth
        depth = max(0, (measured_speed - c_surface) / 0.015)
        for _ in range(100):
            temp = surface_temp - 0.1 * depth
            c = self.sound_speed(temp, salinity, depth)
            if abs(c - measured_speed) < 0.01:
                break
            # dc/dd = d/d(depth) of sound_speed
            # = 0.016 (pressure term) + d/d(depth)[-0.0521*T^2 + 4.57*T + ...] × dT/dd
            # dT/dd = -0.1, so contribution ≈ -0.1 × (4.57 - 0.1042*T)
            dc_dd = 0.016 - 0.1 * (4.57 - 0.1042 * temp)
            if abs(dc_dd) < 1e-6:
                break
            depth += (measured_speed - c) / dc_dd
            depth = max(0, depth)

        elapsed = depth / descent_rate if descent_rate > 0 else 0.0
        uncertainty = elapsed * 0.05  # 5% uncertainty
        return elapsed, uncertainty


class AbsorptionClock(PhysicsModel):
    """Francois-Garrison absorption → range-dependent clock.

    Acoustic absorption in seawater is frequency-dependent.
    By measuring signal attenuation, we can infer range → time.
    """

    # Simplified absorption coefficient (dB/km) for ~10 kHz
    ALPHA_BASE = 1.0  # dB/km at 10 kHz, 20°C, 35ppt

    def absorption_coeff(self, frequency_khz: float = 10.0, temperature: float = 20.0) -> float:
        """Simplified Francois-Garrison absorption."""
        # Frequency dependence: α ∝ f^2 at low frequencies
        return self.ALPHA_BASE * (frequency_khz / 10.0) ** 2

    def likelihood(self, observation: dict, t: float) -> float:
        """P(measured_attenuation | t) assuming known propagation speed."""
        c = observation.get("sound_speed", 1500.0)
        range_m = c * t
        range_km = range_m / 1000.0
        alpha = self.absorption_coeff(
            observation.get("frequency_khz", 10.0),
            observation.get("temperature", 20.0),
        )
        expected_loss = alpha * range_km
        measured = observation.get("measured_attenuation_db", 0.0)
        sigma = observation.get("attenuation_sigma", 0.5)
        return math.exp(-0.5 * ((measured - expected_loss) / sigma) ** 2)

    def infer(self, observations: dict) -> tuple[float, float]:
        """Infer elapsed time from absorption measurement."""
        measured_att = observations.get("measured_attenuation_db", 0.0)
        freq = observations.get("frequency_khz", 10.0)
        c = observations.get("sound_speed", 1500.0)

        alpha = self.absorption_coeff(freq)
        if alpha <= 0:
            return 0.0, float("inf")

        range_km = measured_att / alpha
        range_m = range_km * 1000.0
        elapsed = range_m / c

        uncertainty = elapsed * 0.08  # 8% uncertainty
        return elapsed, uncertainty


class ThermalClock(PhysicsModel):
    """Silicon gate delay → temperature-dependent clock.

    eval_time(T) = t_base × (1 + α × (T - T_ref))
    α ≈ 0.15 µs/°C for typical constraint evaluator on ESP32.

    Given a known t_base (calibration), measure temperature → infer
    expected eval time, or measure eval time → infer temperature.
    For temporal inference: track temperature drift → elapsed time.
    """

    ALPHA = 0.15e-6  # 0.15 µs/°C gate delay coefficient
    T_REF = 25.0  # reference temperature

    def eval_time_at_temp(self, t_base: float, temperature: float) -> float:
        """Compute expected eval time at given temperature."""
        return t_base * (1 + self.ALPHA * 1e6 * (temperature - self.T_REF))

    def likelihood(self, observation: dict, t: float) -> float:
        """P(temperature_reading | t) assuming Newtonian cooling/heating."""
        t_base_temp = observation.get("initial_temp", self.T_REF)
        ambient = observation.get("ambient_temp", self.T_REF)
        tau = observation.get("thermal_tau", 60.0)  # time constant in seconds
        expected_temp = ambient + (t_base_temp - ambient) * math.exp(-t / tau)
        measured = observation.get("temperature", self.T_REF)
        sigma = observation.get("temp_sigma", 0.1)
        return math.exp(-0.5 * ((measured - expected_temp) / sigma) ** 2)

    def infer(self, observations: dict) -> tuple[float, float]:
        """Infer elapsed time from thermal drift."""
        measured_temp = observations.get("temperature", self.T_REF)
        initial_temp = observations.get("initial_temp", self.T_REF)
        ambient_temp = observations.get("ambient_temp", self.T_REF)
        tau = observations.get("thermal_tau", 60.0)

        if initial_temp == ambient_temp:
            # No drift → can't infer time
            return 0.0, float("inf")

        # Newton's law: T(t) = T_amb + (T_0 - T_amb) * exp(-t/τ)
        # Solve for t: t = -τ * ln((T - T_amb) / (T_0 - T_amb))
        ratio = (measured_temp - ambient_temp) / (initial_temp - ambient_temp)
        ratio = max(1e-10, min(ratio, 1.0))  # clamp for numerical safety

        elapsed = -tau * math.log(ratio)
        # Uncertainty grows with elapsed time
        uncertainty = elapsed * 0.10  # 10% uncertainty
        return max(0.0, elapsed), uncertainty


class PropagationClock(PhysicsModel):
    """Signal propagation → distance-dependent clock.

    delay = distance / c (speed of light or sound)
    """

    SPEED_OF_LIGHT = 299_792_458.0  # m/s
    SPEED_OF_SOUND_AIR = 343.0  # m/s at 20°C
    SPEED_OF_SOUND_WATER = 1500.0  # m/s typical

    def likelihood(self, observation: dict, t: float) -> float:
        """P(measured_delay | t) given known propagation distance."""
        c = observation.get("propagation_speed", self.SPEED_OF_LIGHT)
        distance = observation.get("distance", 0.0)
        expected_delay = distance / c
        measured = observation.get("measured_delay", 0.0)
        sigma = observation.get("delay_sigma", 1e-9)
        return math.exp(-0.5 * ((measured - expected_delay) / sigma) ** 2)

    def infer(self, observations: dict) -> tuple[float, float]:
        """Infer elapsed time from propagation delay."""
        measured_delay = observations.get("measured_delay", 0.0)
        c = observations.get("propagation_speed", self.SPEED_OF_LIGHT)
        distance = observations.get("distance", 0.0)

        # The propagation delay IS the elapsed time
        elapsed = measured_delay
        uncertainty = elapsed * 0.01  # 1% — very precise
        return elapsed, uncertainty


class DopplerClock(PhysicsModel):
    """Doppler shift → velocity-dependent clock.

    Δf/f = v/c → integrate velocity over time for elapsed time estimate.
    """

    def likelihood(self, observation: dict, t: float) -> float:
        """P(doppler_shift | t) assuming known motion profile."""
        f0 = observation.get("source_frequency", 1000.0)
        velocity = observation.get("velocity", 0.0)
        c = observation.get("medium_speed", 343.0)
        expected_shift = f0 * velocity / c
        measured = observation.get("measured_shift", 0.0)
        sigma = observation.get("shift_sigma", 0.1)
        return math.exp(-0.5 * ((measured - expected_shift) / sigma) ** 2)

    def infer(self, observations: dict) -> tuple[float, float]:
        """Infer elapsed time from Doppler shift (requires velocity profile)."""
        velocity = observations.get("velocity", 0.0)
        distance = observations.get("distance_traveled", 0.0)

        if abs(velocity) < 1e-10:
            return 0.0, float("inf")

        elapsed = distance / abs(velocity)
        uncertainty = elapsed * 0.05  # 5%
        return elapsed, uncertainty


class SiliconClock(PhysicsModel):
    """Gate delay fingerprint → chip-specific clock.

    Each chip has unique gate delay profile (PUF-like).
    Timing at known temperature = chip identity + temporal state.
    """

    def likelihood(self, observation: dict, t: float) -> float:
        """P(measured_timing | t) given chip fingerprint."""
        t_base = observation.get("base_delay_ns", 100.0)
        temperature = observation.get("temperature", 25.0)
        fingerprint = observation.get("chip_fingerprint", 1.0)
        alpha = 0.0015  # per °C
        expected = t_base * fingerprint * (1 + alpha * (temperature - 25.0))
        measured = observation.get("measured_delay_ns", 100.0)
        sigma = observation.get("timing_sigma", 0.5)
        return math.exp(-0.5 * ((measured - expected) / sigma) ** 2)

    def infer(self, observations: dict) -> tuple[float, float]:
        """Infer time from silicon timing drift.

        Given known clock frequency drift over time (aging), estimate elapsed time.
        """
        measured_freq = observations.get("measured_frequency_hz", 0.0)
        initial_freq = observations.get("initial_frequency_hz", measured_freq)
        aging_rate = observations.get("aging_rate_ppb_per_day", 0.0)

        if aging_rate == 0 or initial_freq == 0:
            return 0.0, float("inf")

        # Frequency drift: Δf/f = aging_rate × t_days
        drift_ppb = (measured_freq - initial_freq) / initial_freq * 1e9
        days_elapsed = drift_ppb / aging_rate

        elapsed = days_elapsed * 86400.0
        uncertainty = abs(elapsed) * 0.20  # 20% — aging is noisy
        return elapsed, uncertainty
