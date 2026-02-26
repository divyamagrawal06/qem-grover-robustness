"""QEM training and inference service with per-qubit model registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock

import numpy as np
from torch import nn

from src.ml.model import TrainConfig, TrainSummary, mitigate_distribution, train_qem_model
from src.quantum.dataset import DatasetConfig, generate_noisy_ideal_dataset_with_context
from src.quantum.metrics import normalize_probability_vector


@dataclass(slots=True)
class QEMArtifact:
    """Trained model and metadata."""

    n_qubits: int
    model: nn.Module
    summary: TrainSummary
    trained_at_utc: str
    sample_count: int
    model_type: str
    context_dim: int


class QEMService:
    """Stateful service used by FastAPI routes for QEM operations."""

    def __init__(self) -> None:
        self._artifacts: dict[int, QEMArtifact] = {}
        self._lock = Lock()

    def train_model(
        self,
        n_qubits: int,
        sample_count: int,
        shots: int,
        train_config: TrainConfig,
        include_thermal_relaxation: bool = True,
    ) -> QEMArtifact:
        """Generate training pairs and fit a model for a fixed qubit count."""
        dataset_config = DatasetConfig(
            n_qubits=n_qubits,
            sample_count=sample_count,
            shots=shots,
            ideal_shots=max(shots * 2, 20_000),
            include_thermal_relaxation=include_thermal_relaxation,
            rng_seed=train_config.seed,
        )
        noisy_samples, ideal_samples, contexts = generate_noisy_ideal_dataset_with_context(
            dataset_config
        )
        model, summary = train_qem_model(
            noisy_samples,
            ideal_samples,
            train_config,
            context_samples=contexts,
        )

        artifact = QEMArtifact(
            n_qubits=n_qubits,
            model=model,
            summary=summary,
            trained_at_utc=datetime.now(timezone.utc).isoformat(),
            sample_count=sample_count,
            model_type=train_config.model_type,
            context_dim=contexts.shape[1],
        )

        with self._lock:
            self._artifacts[n_qubits] = artifact

        return artifact

    def mitigate(
        self,
        n_qubits: int,
        noisy_distribution: np.ndarray,
        context_features: np.ndarray | None = None,
        auto_train_if_missing: bool = False,
        sample_count: int = 192,
        shots: int = 10_000,
        include_thermal_relaxation: bool = True,
    ) -> np.ndarray:
        """Mitigate one distribution using a trained model for the same qubit count."""
        artifact = self.get_artifact(n_qubits)
        if artifact is None and auto_train_if_missing:
            artifact = self.train_model(
                n_qubits=n_qubits,
                sample_count=sample_count,
                shots=shots,
                train_config=TrainConfig(),
                include_thermal_relaxation=include_thermal_relaxation,
            )

        if artifact is None:
            return normalize_probability_vector(noisy_distribution)

        return mitigate_distribution(
            artifact.model,
            noisy_distribution,
            context_features=context_features,
        )

    def get_artifact(self, n_qubits: int) -> QEMArtifact | None:
        """Fetch trained model metadata for a qubit count."""
        with self._lock:
            return self._artifacts.get(n_qubits)

    def status(self) -> list[QEMArtifact]:
        """List all trained models."""
        with self._lock:
            return list(self._artifacts.values())
