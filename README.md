# physics-clock — Temporal Inference from Physics

No RTC. No NTP. No GPS. The physics IS the clock.

Infers elapsed time from physical observations using Bayesian temporal
inference across multiple independent physics models. Each model provides
an independent clock signal. The joint probability gives precise time.

## How It Works
1. Observe physical quantities (temperature, sound speed, signal propagation, etc.)
2. Each physics model gives P(observation | t)
3. Bayesian fusion: P(t | all observations) ∝ Π P(obs_i | t) × P(t)
4. Peak of posterior = best time estimate

## Physics Clock Models

| Model | Physical Signal | Clock Source |
|---|---|---|
| `SoundSpeedClock` | UNESCO/Chen-Millero sound speed | Depth-dependent timing |
| `AbsorptionClock` | Francois-Garrison absorption | Range-dependent timing |
| `ThermalClock` | Silicon gate delay vs temperature | Newtonian cooling/heating |
| `PropagationClock` | Signal propagation delay | Distance / speed |
| `DopplerClock` | Doppler frequency shift | Velocity integration |
| `SiliconClock` | Gate delay fingerprint (PUF) | Crystal aging drift |

## Quick Start

```python
from physics_clock import PhysicsClock, ThermalClock, SoundSpeedClock

clock = PhysicsClock([
    ThermalClock(),
    SoundSpeedClock(),
])

result = clock.infer_time({
    "temperature": 40.0,
    "initial_temp": 60.0,
    "ambient_temp": 25.0,
    "thermal_tau": 120.0,
    "measured_speed": 1495.0,
    "descent_rate": 0.5,
    "surface_temp": 20.0,
    "salinity": 35.0,
}, t_min=0, t_max=300)

print(f"Elapsed: {result.timestamp:.1f}s ± {result.uncertainty:.1f}s")
print(f"Precision: {result.precision}")
print(f"Clocks used: {result.n_clocks}")
```

## Use Cases
- **Underwater vehicles**: infer time from sound speed + absorption + thermocline
- **Robot fleets**: infer time from silicon timing + thermal + constraint load
- **IoT networks**: infer time without NTP (physics is the sync protocol)
- **Security**: verify device honesty through reality parity (can't fake physics)

## Security: Reality Parity

The timing of constraint evaluation IS attestation:

```python
from physics_clock import RealityParity

parity = RealityParity()
result = parity.check({
    "eval_ns": 15000,        # reported eval time
    "n_constraints": 100,     # constraints evaluated
    "temperature": 30.0,      # reported die temperature
    "voltage_mv": 3300.0,     # supply voltage
})

if not result.honest:
    print(f"Device flagged: {result.reason}")
    print(f"Expected {result.expected_eval_ns:.0f}ns, got {result.actual_eval_ns:.0f}ns")
    print(f"Deviation: {result.deviation_sigma:.1f}σ")
```

- `eval_time = f(complexity, temperature, voltage)`
- If reported timing doesn't match reported temperature → device is lying
- No cryptographic keys needed. The physics can't be spoofed.

## Composable With
- **cocapn-schemas**: temporal fingerprint tile format
- **fleet-constraint-kernel**: evaluation timing as temporal signal
- **fleet-proto**: Rust version of physics clock types
- **temporal-auth**: full authentication using physics-clock

## Install

```bash
pip install physics-clock
```

## License
Apache 2.0
