"""Metrics for noisy and mitigated Grover probability distributions."""

from __future__ import annotations

import numpy as np


EPSILON = 1e-12


def normalize_probability_vector(probabilities: np.ndarray) -> np.ndarray:
    """Return clipped and normalized probabilities for 1D or 2D tensors."""
    vector = np.asarray(probabilities, dtype=np.float64)
    vector = np.clip(vector, 0.0, None)

    if vector.ndim == 1:
        total = float(vector.sum())
        if total <= EPSILON:
            return np.full_like(vector, 1.0 / len(vector))
        return vector / total

    if vector.ndim == 2:
        totals = vector.sum(axis=1, keepdims=True)
        normalized = np.empty_like(vector)

        valid_rows = totals[:, 0] > EPSILON
        normalized[valid_rows] = vector[valid_rows] / totals[valid_rows]
        normalized[~valid_rows] = 1.0 / vector.shape[1]
        return normalized

    raise ValueError("normalize_probability_vector expects 1D or 2D input")


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute MAE between two distributions."""
    return float(np.mean(np.abs(normalize_probability_vector(a) - normalize_probability_vector(b))))


def distribution_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Classical fidelity between two probability distributions."""
    pa = normalize_probability_vector(a)
    pb = normalize_probability_vector(b)
    fidelity = float(np.square(np.sum(np.sqrt(pa * pb))))
    return max(0.0, min(1.0, fidelity))


def success_probability(distribution: np.ndarray, marked_states: list[int]) -> float:
    """Compute probability mass on the marked states."""
    probabilities = normalize_probability_vector(distribution)
    return float(sum(probabilities[state] for state in marked_states))
