"""Quantum circuit builders, noise models, and simulator helpers."""

from src.quantum.dataset import (
    DatasetConfig,
    generate_noisy_ideal_dataset,
    generate_noisy_ideal_dataset_with_context,
)
from src.quantum.grover_ideal import build_grover_circuit, optimal_grover_iterations
from src.quantum.metrics import (
    distribution_fidelity,
    mean_absolute_error,
    normalize_probability_vector,
    success_probability,
)
from src.quantum.noise_models import NoiseParameters, build_noise_model
from src.quantum.simulator import (
    GroverRunResult,
    SweepPoint,
    basis_labels,
    run_iteration_sweep,
    run_noise_strength_sweep,
    simulate_grover_run,
)

__all__ = [
    "DatasetConfig",
    "GroverRunResult",
    "NoiseParameters",
    "SweepPoint",
    "basis_labels",
    "build_grover_circuit",
    "build_noise_model",
    "distribution_fidelity",
    "generate_noisy_ideal_dataset",
    "generate_noisy_ideal_dataset_with_context",
    "mean_absolute_error",
    "normalize_probability_vector",
    "optimal_grover_iterations",
    "run_iteration_sweep",
    "run_noise_strength_sweep",
    "simulate_grover_run",
    "success_probability",
]
