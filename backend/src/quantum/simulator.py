"""Simulation utilities for ideal, noisy, and mitigated Grover experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from src.quantum.grover_ideal import build_grover_circuit, optimal_grover_iterations
from src.quantum.metrics import normalize_probability_vector, success_probability
from src.quantum.noise_models import NoiseParameters, build_noise_model


MitigationFn = Callable[[np.ndarray, float], np.ndarray]


@dataclass(slots=True)
class GroverRunResult:
    """Single simulation output used by API responses."""

    distribution: np.ndarray
    counts: dict[str, int]
    success_probability: float
    depth: int
    gate_count: int
    iterations: int


@dataclass(slots=True)
class SweepPoint:
    """One point in an iteration or noise sweep."""

    x_value: float
    ideal_success: float
    noisy_success: float
    mitigated_success: float | None = None


def simulate_grover_run(
    n_qubits: int,
    marked_states: list[int],
    shots: int,
    num_iterations: int | None = None,
    noise: NoiseParameters | None = None,
) -> GroverRunResult:
    """Run one Grover circuit with optional noise and return distribution metrics."""
    if shots < 1:
        raise ValueError("shots must be >= 1")

    if num_iterations is None:
        num_iterations = optimal_grover_iterations(n_qubits=n_qubits, n_marked=len(marked_states))

    circuit = build_grover_circuit(
        n_qubits=n_qubits,
        marked_states=marked_states,
        num_iterations=num_iterations,
    )

    noise_model = build_noise_model(noise or NoiseParameters())
    counts = _run_counts(circuit=circuit, shots=shots, noise_model=noise_model)
    distribution = counts_to_probability_vector(counts=counts, n_qubits=n_qubits, shots=shots)

    return GroverRunResult(
        distribution=distribution,
        counts=counts,
        success_probability=success_probability(distribution, marked_states),
        depth=circuit.depth(),
        gate_count=circuit.size(),
        iterations=num_iterations,
    )


def run_iteration_sweep(
    n_qubits: int,
    marked_states: list[int],
    shots: int,
    max_iterations: int | None,
    noise: NoiseParameters,
    mitigation_fn: MitigationFn | None = None,
) -> list[SweepPoint]:
    """Return success probability vs Grover iteration count."""
    if max_iterations is None:
        optimal = optimal_grover_iterations(n_qubits=n_qubits, n_marked=len(marked_states))
        max_iterations = max(2, 2 * optimal)

    points: list[SweepPoint] = []
    for iterations in range(1, max_iterations + 1):
        ideal = simulate_grover_run(
            n_qubits=n_qubits,
            marked_states=marked_states,
            shots=shots,
            num_iterations=iterations,
            noise=NoiseParameters(),
        )
        noisy = simulate_grover_run(
            n_qubits=n_qubits,
            marked_states=marked_states,
            shots=shots,
            num_iterations=iterations,
            noise=noise,
        )

        mitigated_success: float | None = None
        if mitigation_fn is not None:
            mitigated = normalize_probability_vector(
                mitigation_fn(noisy.distribution, float(iterations))
            )
            mitigated_success = success_probability(mitigated, marked_states)

        points.append(
            SweepPoint(
                x_value=float(iterations),
                ideal_success=ideal.success_probability,
                noisy_success=noisy.success_probability,
                mitigated_success=mitigated_success,
            )
        )

    return points


def run_noise_strength_sweep(
    n_qubits: int,
    marked_states: list[int],
    shots: int,
    num_iterations: int | None,
    noise_strength_values: list[float],
    base_noise: NoiseParameters,
    mitigation_fn: MitigationFn | None = None,
) -> list[SweepPoint]:
    """Return Grover success probability vs depolarizing noise strength."""
    points: list[SweepPoint] = []
    if num_iterations is None:
        num_iterations = optimal_grover_iterations(n_qubits=n_qubits, n_marked=len(marked_states))

    ideal = simulate_grover_run(
        n_qubits=n_qubits,
        marked_states=marked_states,
        shots=shots,
        num_iterations=num_iterations,
        noise=NoiseParameters(),
    )

    for strength in noise_strength_values:
        noisy_noise = NoiseParameters(
            single_qubit_depolarizing=strength,
            two_qubit_depolarizing=min(1.0, strength * 2.0),
            readout_error=base_noise.readout_error,
            enable_thermal_relaxation=base_noise.enable_thermal_relaxation,
            t1_seconds=base_noise.t1_seconds,
            t2_seconds=base_noise.t2_seconds,
            one_qubit_gate_time_seconds=base_noise.one_qubit_gate_time_seconds,
            two_qubit_gate_time_seconds=base_noise.two_qubit_gate_time_seconds,
        )

        noisy = simulate_grover_run(
            n_qubits=n_qubits,
            marked_states=marked_states,
            shots=shots,
            num_iterations=num_iterations,
            noise=noisy_noise,
        )

        mitigated_success: float | None = None
        if mitigation_fn is not None:
            mitigated = normalize_probability_vector(
                mitigation_fn(noisy.distribution, float(strength))
            )
            mitigated_success = success_probability(mitigated, marked_states)

        points.append(
            SweepPoint(
                x_value=float(strength),
                ideal_success=ideal.success_probability,
                noisy_success=noisy.success_probability,
                mitigated_success=mitigated_success,
            )
        )

    return points


def counts_to_probability_vector(counts: dict[str, int], n_qubits: int, shots: int) -> np.ndarray:
    """Convert Aer counts to a dense probability vector ordered by basis index."""
    n_states = 2 ** n_qubits
    probabilities = np.zeros(n_states, dtype=np.float64)

    for bitstring, count in counts.items():
        state_index = int(bitstring.replace(" ", ""), 2)
        probabilities[state_index] = count / shots

    return normalize_probability_vector(probabilities)


def basis_labels(n_qubits: int) -> list[str]:
    """Return computational basis labels in binary order."""
    return [format(index, f"0{n_qubits}b") for index in range(2 ** n_qubits)]


def _run_counts(circuit, shots: int, noise_model=None) -> dict[str, int]:
    backend = AerSimulator(noise_model=noise_model) if noise_model is not None else AerSimulator()
    compiled = transpile(circuit, backend, optimization_level=1)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    if isinstance(counts, list):
        counts = counts[0]

    return dict(counts)
