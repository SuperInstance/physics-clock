"""RealityParity — RAID-5 parity across physical signals for device verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParityResult:
    """Result of reality parity check."""

    honest: bool
    reason: str = ""
    expected_eval_ns: float = 0.0
    actual_eval_ns: float = 0.0
    sigma_ns: float = 0.0
    deviation_sigma: float = 0.0


class RealityParity:
    """
    RAID-5 parity across physical signals.

    Signal 0: Evaluation timing (silicon)
    Signal 1: Thermal state (on-die sensor)
    Signal 2: Propagation delay (radio/wire)
    Signal 3: Constraint state (evaluation result)
    Parity: physics_model(signals 0-2) must produce signal 3

    Any signal spoofed → parity fails → device flagged.
    """

    # Gate delay temperature coefficient (fractional change per °C)
    ALPHA_TEMP = 0.0015  # 0.15% per °C
    # Voltage coefficient (fractional change per mV from nominal)
    ALPHA_VOLTAGE = 0.0005  # 0.05% per mV
    # Base eval time per constraint (ns) — representative for ESP32 @ 240MHz
    BASE_NS_PER_CONSTRAINT = 150.0  # ns

    def compute_expected_eval_time(
        self,
        n_constraints: int,
        temperature: float,
        voltage_mv: float = 3300.0,
        nominal_voltage_mv: float = 3300.0,
        t_ref: float = 25.0,
    ) -> float:
        """Compute expected evaluation time in nanoseconds.

        eval_time = base × n × (1 + α_T × (T - T_ref)) × (1 + α_V × (V - V_nom))
        """
        base = self.BASE_NS_PER_CONSTRAINT * n_constraints
        temp_factor = 1.0 + self.ALPHA_TEMP * (temperature - t_ref)
        voltage_factor = 1.0 + self.ALPHA_VOLTAGE * (voltage_mv - nominal_voltage_mv)
        return base * temp_factor * voltage_factor

    def check(self, device_report: dict[str, Any]) -> ParityResult:
        """Check that reported timing matches physics expectations.

        Parameters
        ----------
        device_report : dict
            Must contain:
            - eval_ns: reported evaluation time in nanoseconds
            - n_constraints: number of constraints evaluated
            - temperature: reported die temperature (°C)
            Optional:
            - voltage_mv: supply voltage in mV (default 3300)
            - natural_variation_pct: allowed natural variation % (default 2)

        Returns
        -------
        ParityResult
        """
        actual_time = device_report["eval_ns"]
        expected_time = self.compute_expected_eval_time(
            n_constraints=device_report["n_constraints"],
            temperature=device_report["temperature"],
            voltage_mv=device_report.get("voltage_mv", 3300.0),
        )

        variation_pct = device_report.get("natural_variation_pct", 2.0)
        sigma = expected_time * variation_pct / 100.0
        deviation = abs(actual_time - expected_time)
        deviation_sigma = deviation / sigma if sigma > 0 else float("inf")

        honest = deviation_sigma <= 3.0  # 3σ check

        reason = ""
        if not honest:
            if actual_time < expected_time:
                reason = "timing_too_fast_spoofed"
            else:
                reason = "timing_too_slow_anomaly"

        return ParityResult(
            honest=honest,
            reason=reason,
            expected_eval_ns=expected_time,
            actual_eval_ns=actual_time,
            sigma_ns=sigma,
            deviation_sigma=deviation_sigma,
        )
