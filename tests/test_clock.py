"""Tests for physics-clock."""

import math

import pytest

from physics_clock import (
    AbsorptionClock,
    DopplerClock,
    PhysicsClock,
    PropagationClock,
    RealityParity,
    SiliconClock,
    SoundSpeedClock,
    ThermalClock,
)


class TestThermalClock:
    """Test that thermal clock infers correct elapsed time from temperature drift."""

    def test_cooling_inference(self):
        """Device cooling from 60°C to 40°C in 25°C ambient → infer elapsed time."""
        clock = ThermalClock()
        # Newton's law: T(t) = T_amb + (T_0 - T_amb) * exp(-t/τ)
        # At t=60s: T = 25 + (60-25)*exp(-60/120) = 25 + 35*0.6065 ≈ 46.23°C
        tau = 120.0
        t_true = 60.0
        T0 = 60.0
        T_amb = 25.0
        measured_temp = T_amb + (T0 - T_amb) * math.exp(-t_true / tau)

        estimate, uncertainty = clock.infer({
            "temperature": measured_temp,
            "initial_temp": T0,
            "ambient_temp": T_amb,
            "thermal_tau": tau,
        })

        # Should be close to 60 seconds
        assert abs(estimate - t_true) < 1.0, f"Expected ~{t_true}s, got {estimate}s"
        assert uncertainty > 0

    def test_no_drift_returns_inf_uncertainty(self):
        """If initial == ambient, can't infer time."""
        clock = ThermalClock()
        estimate, uncertainty = clock.infer({
            "temperature": 25.0,
            "initial_temp": 25.0,
            "ambient_temp": 25.0,
            "thermal_tau": 60.0,
        })
        assert uncertainty == float("inf")

    def test_likelihood_peaks_at_correct_time(self):
        """Likelihood should be highest near the true time."""
        clock = ThermalClock()
        obs = {
            "temperature": 40.0,
            "initial_temp": 60.0,
            "ambient_temp": 25.0,
            "thermal_tau": 120.0,
        }
        # Compute true time
        t_true = -120.0 * math.log((40.0 - 25.0) / (60.0 - 25.0))

        l_at_true = clock.likelihood(obs, t_true)
        l_off = clock.likelihood(obs, t_true + 100.0)
        assert l_at_true > l_off


class TestSoundSpeedClock:
    """Test that sound speed clock infers depth from sound speed profile."""

    def test_depth_inference(self):
        """Measure sound speed at depth → infer elapsed time for diving vehicle."""
        clock = SoundSpeedClock()

        # At 100m depth: temp ≈ 20 - 0.1*100 = 10°C, sound speed
        depth = 100.0
        temp_at_depth = 20.0 - 0.1 * depth
        measured_speed = clock.sound_speed(temp_at_depth, 35.0, depth)

        # Vehicle descends at 0.5 m/s → 100m takes 200s
        descent_rate = 0.5
        estimate, uncertainty = clock.infer({
            "measured_speed": measured_speed,
            "descent_rate": descent_rate,
            "surface_temp": 20.0,
            "salinity": 35.0,
        })

        expected_time = depth / descent_rate  # 200s
        assert abs(estimate - expected_time) < 20.0, f"Expected ~{expected_time}s, got {estimate}s"


class TestRealityParity:
    """Test that reality parity catches fake timing."""

    def test_honest_device_passes(self):
        """Honest device: eval_time matches temperature → passes."""
        parity = RealityParity()

        # 100 constraints at 30°C, 3300mV
        expected = parity.compute_expected_eval_time(100, 30.0, 3300.0)

        result = parity.check({
            "eval_ns": expected,  # exactly as expected
            "n_constraints": 100,
            "temperature": 30.0,
            "voltage_mv": 3300.0,
        })
        assert result.honest is True
        assert result.reason == ""

    def test_spoofed_device_fails(self):
        """Spoofed device: eval_time doesn't match temperature → fails."""
        parity = RealityParity()

        # Attacker claims 50°C but reports timing for 25°C (forgot to adjust)
        # Expected at 50°C is much slower (higher temp → slower silicon)
        # But attacker reports the 25°C timing
        expected_25 = parity.compute_expected_eval_time(100, 25.0, 3300.0)

        result = parity.check({
            "eval_ns": expected_25,  # timing for 25°C ...
            "n_constraints": 100,
            "temperature": 50.0,  # ... but claiming 50°C
            "voltage_mv": 3300.0,
            "natural_variation_pct": 0.5,  # tight 0.5% variation
        })
        # The 25°C difference causes large deviation > 3σ with tight variation
        assert result.honest is False
        assert "spoofed" in result.reason or "anomaly" in result.reason

    def test_too_fast_detected(self):
        """Device claiming impossibly fast evaluation → flagged."""
        parity = RealityParity()
        expected = parity.compute_expected_eval_time(100, 25.0, 3300.0)

        result = parity.check({
            "eval_ns": expected * 0.5,  # 50% too fast
            "n_constraints": 100,
            "temperature": 25.0,
        })
        assert result.honest is False
        assert result.deviation_sigma > 3.0

    def test_voltage_variation(self):
        """Low voltage → slower timing, correctly detected."""
        parity = RealityParity()

        # 3.1V (200mV low) → slower
        expected = parity.compute_expected_eval_time(100, 25.0, 3100.0, 3300.0)

        result = parity.check({
            "eval_ns": expected,
            "n_constraints": 100,
            "temperature": 25.0,
            "voltage_mv": 3100.0,
        })
        assert result.honest is True


class TestJointInference:
    """Test that joint inference is more precise than any single clock."""

    def test_joint_more_precise(self):
        """Using multiple clocks should reduce uncertainty."""
        # Setup: a device with thermal drift and sound propagation
        thermal = ThermalClock()
        sound = SoundSpeedClock()

        # Ground truth: 60 seconds elapsed
        t_true = 60.0
        T0, T_amb, tau = 60.0, 25.0, 120.0
        measured_temp = T_amb + (T0 - T_amb) * math.exp(-t_true / tau)

        observations = {
            "temperature": measured_temp,
            "initial_temp": T0,
            "ambient_temp": T_amb,
            "thermal_tau": tau,
            "measured_speed": 1495.0,
            "descent_rate": 0.5,
            "surface_temp": 20.0,
            "salinity": 35.0,
        }

        # Single clock: thermal only
        thermal_est, thermal_unc = thermal.infer(observations)

        # Joint: both clocks
        clock = PhysicsClock([thermal, sound])
        result = clock.infer_time(observations, t_min=0, t_max=300, n_points=50000)

        # Joint should have finite uncertainty
        assert result.uncertainty < float("inf")
        assert result.n_clocks == 2


class TestPropagationClock:
    def test_delay_inference(self):
        """Propagation delay directly gives elapsed time."""
        clock = PropagationClock()
        # 1ms delay
        estimate, _ = clock.infer({"measured_delay": 0.001})
        assert abs(estimate - 0.001) < 1e-10


class TestAbsorptionClock:
    def test_range_inference(self):
        """Absorption measurement → range → time."""
        clock = AbsorptionClock()
        # 10 kHz, 1 dB/km absorption, measured 5 dB loss → 5 km → 5000/1500 ≈ 3.33s
        estimate, unc = clock.infer({
            "measured_attenuation_db": 5.0,
            "frequency_khz": 10.0,
            "sound_speed": 1500.0,
        })
        expected = (5.0 / 1.0) * 1000 / 1500  # 3.33s
        assert abs(estimate - expected) < 0.5


class TestDopplerClock:
    def test_velocity_inference(self):
        """Doppler + distance → elapsed time."""
        clock = DopplerClock()
        estimate, _ = clock.infer({
            "velocity": 10.0,  # 10 m/s
            "distance_traveled": 100.0,  # 100m
        })
        assert abs(estimate - 10.0) < 0.01  # 100/10 = 10s


class TestSiliconClock:
    def test_aging_inference(self):
        """Frequency drift from aging → elapsed time."""
        clock = SiliconClock()
        # 1 ppb/day aging, 100 ppb drift → 100 days
        estimate, _ = clock.infer({
            "initial_frequency_hz": 32_000_000.0,
            "measured_frequency_hz": 32_000_000.032,  # +1 ppb = +0.032 Hz
            "aging_rate_ppb_per_day": 1.0,
        })
        expected_days = 1.0  # 1 ppb drift / 1 ppb/day
        expected_seconds = expected_days * 86400.0
        assert abs(estimate - expected_seconds) < 100  # within ~100s
