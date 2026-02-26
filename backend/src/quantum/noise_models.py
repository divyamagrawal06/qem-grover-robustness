"""Noise model construction for Grover robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    QuantumError,
    depolarizing_error,
    thermal_relaxation_error,
)


ONE_QUBIT_GATES = ["id", "rz", "sx", "x", "h"]
TWO_QUBIT_GATES = ["cx"]


@dataclass(slots=True)
class NoiseParameters:
    """Noise knobs exposed to the API and experiment scripts."""

    single_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0
    readout_error: float = 0.0
    enable_thermal_relaxation: bool = False
    t1_seconds: float = 120e-6
    t2_seconds: float = 80e-6
    one_qubit_gate_time_seconds: float = 50e-9
    two_qubit_gate_time_seconds: float = 300e-9

    def cleaned(self) -> "NoiseParameters":
        """Return a validated, clipped copy."""
        return NoiseParameters(
            single_qubit_depolarizing=_clip_probability(self.single_qubit_depolarizing),
            two_qubit_depolarizing=_clip_probability(self.two_qubit_depolarizing),
            readout_error=_clip_probability(self.readout_error),
            enable_thermal_relaxation=bool(self.enable_thermal_relaxation),
            t1_seconds=max(self.t1_seconds, 1e-12),
            t2_seconds=max(self.t2_seconds, 1e-12),
            one_qubit_gate_time_seconds=max(self.one_qubit_gate_time_seconds, 1e-12),
            two_qubit_gate_time_seconds=max(self.two_qubit_gate_time_seconds, 1e-12),
        )

    def is_noisy(self) -> bool:
        """True if any non-trivial error source is enabled."""
        cleaned = self.cleaned()
        return (
            cleaned.single_qubit_depolarizing > 0.0
            or cleaned.two_qubit_depolarizing > 0.0
            or cleaned.readout_error > 0.0
            or cleaned.enable_thermal_relaxation
        )


def build_noise_model(params: NoiseParameters) -> NoiseModel | None:
    """Build a composite Aer noise model from configured parameters."""
    cleaned = params.cleaned()
    if not cleaned.is_noisy():
        return None

    noise_model = NoiseModel()

    one_qubit_error = _build_one_qubit_error(cleaned)
    if one_qubit_error is not None:
        noise_model.add_all_qubit_quantum_error(one_qubit_error, ONE_QUBIT_GATES)

    two_qubit_error = _build_two_qubit_error(cleaned)
    if two_qubit_error is not None:
        noise_model.add_all_qubit_quantum_error(two_qubit_error, TWO_QUBIT_GATES)

    if cleaned.readout_error > 0.0:
        readout = ReadoutError(
            [
                [1.0 - cleaned.readout_error, cleaned.readout_error],
                [cleaned.readout_error, 1.0 - cleaned.readout_error],
            ]
        )
        noise_model.add_all_qubit_readout_error(readout)

    return noise_model


def _build_one_qubit_error(params: NoiseParameters) -> QuantumError | None:
    errors: list[QuantumError] = []

    if params.single_qubit_depolarizing > 0.0:
        errors.append(depolarizing_error(params.single_qubit_depolarizing, 1))

    if params.enable_thermal_relaxation:
        errors.append(
            thermal_relaxation_error(
                params.t1_seconds,
                params.t2_seconds,
                params.one_qubit_gate_time_seconds,
            )
        )

    return _compose_errors(errors)


def _build_two_qubit_error(params: NoiseParameters) -> QuantumError | None:
    errors: list[QuantumError] = []

    if params.two_qubit_depolarizing > 0.0:
        errors.append(depolarizing_error(params.two_qubit_depolarizing, 2))

    if params.enable_thermal_relaxation:
        one_qubit_thermal = thermal_relaxation_error(
            params.t1_seconds,
            params.t2_seconds,
            params.two_qubit_gate_time_seconds,
        )
        errors.append(one_qubit_thermal.tensor(one_qubit_thermal))

    return _compose_errors(errors)


def _compose_errors(errors: list[QuantumError]) -> QuantumError | None:
    if not errors:
        return None

    composed = errors[0]
    for error in errors[1:]:
        composed = composed.compose(error)
    return composed


def _clip_probability(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
