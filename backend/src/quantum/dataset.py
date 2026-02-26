"""Dataset generation for supervised quantum error mitigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.quantum.grover_ideal import optimal_grover_iterations
from src.quantum.noise_models import NoiseParameters
from src.quantum.simulator import simulate_grover_run


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for noisy/ideal pair generation."""

    n_qubits: int
    sample_count: int = 256
    shots: int = 10_000
    ideal_shots: int = 25_000
    single_qubit_depolarizing_min: float = 0.0
    single_qubit_depolarizing_max: float = 0.12
    readout_error_min: float = 0.0
    readout_error_max: float = 0.08
    include_thermal_relaxation: bool = True
    rng_seed: int = 7


def generate_noisy_ideal_dataset(config: DatasetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate aligned noisy and ideal probability distribution pairs."""
    if config.sample_count < 2:
        raise ValueError("sample_count must be >= 2")
    if config.ideal_shots < config.shots:
        raise ValueError("ideal_shots must be >= shots")

    rng = np.random.default_rng(config.rng_seed)
    n_states = 2 ** config.n_qubits

    noisy_samples = np.zeros((config.sample_count, n_states), dtype=np.float32)
    ideal_samples = np.zeros((config.sample_count, n_states), dtype=np.float32)

    for row in range(config.sample_count):
        marked_state = int(rng.integers(0, n_states))
        optimal_iterations = optimal_grover_iterations(config.n_qubits, n_marked=1)
        iteration_delta = int(rng.integers(-1, 2))
        iterations = max(1, optimal_iterations + iteration_delta)

        single_noise = float(
            rng.uniform(
                config.single_qubit_depolarizing_min,
                config.single_qubit_depolarizing_max,
            )
        )
        two_noise = min(1.0, single_noise * 2.0)
        readout_noise = float(
            rng.uniform(config.readout_error_min, config.readout_error_max)
        )

        noisy_noise = NoiseParameters(
            single_qubit_depolarizing=single_noise,
            two_qubit_depolarizing=two_noise,
            readout_error=readout_noise,
            enable_thermal_relaxation=config.include_thermal_relaxation,
        )

        ideal_run = simulate_grover_run(
            n_qubits=config.n_qubits,
            marked_states=[marked_state],
            shots=config.ideal_shots,
            num_iterations=iterations,
            noise=NoiseParameters(),
        )
        noisy_run = simulate_grover_run(
            n_qubits=config.n_qubits,
            marked_states=[marked_state],
            shots=config.shots,
            num_iterations=iterations,
            noise=noisy_noise,
        )

        noisy_samples[row] = noisy_run.distribution.astype(np.float32)
        ideal_samples[row] = ideal_run.distribution.astype(np.float32)

    return noisy_samples, ideal_samples


def generate_noisy_ideal_dataset_with_context(
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired distributions and context vectors for noise-aware QEM."""
    if config.sample_count < 2:
        raise ValueError("sample_count must be >= 2")
    if config.ideal_shots < config.shots:
        raise ValueError("ideal_shots must be >= shots")

    rng = np.random.default_rng(config.rng_seed)
    n_states = 2 ** config.n_qubits
    context_dim = 6

    noisy_samples = np.zeros((config.sample_count, n_states), dtype=np.float32)
    ideal_samples = np.zeros((config.sample_count, n_states), dtype=np.float32)
    contexts = np.zeros((config.sample_count, context_dim), dtype=np.float32)

    for row in range(config.sample_count):
        marked_state = int(rng.integers(0, n_states))
        optimal_iterations = optimal_grover_iterations(config.n_qubits, n_marked=1)
        iteration_delta = int(rng.integers(-1, 2))
        iterations = max(1, optimal_iterations + iteration_delta)

        single_noise = float(
            rng.uniform(
                config.single_qubit_depolarizing_min,
                config.single_qubit_depolarizing_max,
            )
        )
        two_noise = min(1.0, single_noise * 2.0)
        readout_noise = float(
            rng.uniform(config.readout_error_min, config.readout_error_max)
        )

        noisy_noise = NoiseParameters(
            single_qubit_depolarizing=single_noise,
            two_qubit_depolarizing=two_noise,
            readout_error=readout_noise,
            enable_thermal_relaxation=config.include_thermal_relaxation,
        )

        ideal_run = simulate_grover_run(
            n_qubits=config.n_qubits,
            marked_states=[marked_state],
            shots=config.ideal_shots,
            num_iterations=iterations,
            noise=NoiseParameters(),
        )
        noisy_run = simulate_grover_run(
            n_qubits=config.n_qubits,
            marked_states=[marked_state],
            shots=config.shots,
            num_iterations=iterations,
            noise=noisy_noise,
        )

        noisy_samples[row] = noisy_run.distribution.astype(np.float32)
        ideal_samples[row] = ideal_run.distribution.astype(np.float32)
        contexts[row] = np.array(
            [
                single_noise,
                two_noise,
                readout_noise,
                1.0 if config.include_thermal_relaxation else 0.0,
                iterations / max(1.0, float(optimal_iterations * 2)),
                marked_state / max(1.0, float(n_states - 1)),
            ],
            dtype=np.float32,
        )

    return noisy_samples, ideal_samples, contexts
