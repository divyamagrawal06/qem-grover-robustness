"""PyTorch models and training utilities for quantum error mitigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.quantum.metrics import distribution_fidelity, mean_absolute_error, normalize_probability_vector


class QEMMLP(nn.Module):
    """Simple MLP denoiser from noisy to ideal probability vectors."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)


class QEMAutoencoder(nn.Module):
    """Denoising autoencoder baseline for QEM."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 48,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        logits = self.decoder(latent)
        return torch.softmax(logits, dim=-1)


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for QEM model fitting."""

    model_type: str = "mlp"
    hidden_dim: int = 128
    latent_dim: int = 48
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    train_split: float = 0.85
    seed: int = 7


@dataclass(slots=True)
class TrainSummary:
    """Training summary returned to API and scripts."""

    train_loss: float
    val_loss: float
    baseline_mae: float
    mitigated_mae: float
    mae_reduction_pct: float
    baseline_fidelity: float
    mitigated_fidelity: float


def train_qem_model(
    noisy_samples: np.ndarray,
    ideal_samples: np.ndarray,
    config: TrainConfig,
    context_samples: np.ndarray | None = None,
) -> tuple[nn.Module, TrainSummary]:
    """Train QEM model and return final metrics."""
    if noisy_samples.shape != ideal_samples.shape:
        raise ValueError("noisy_samples and ideal_samples must have the same shape")

    if noisy_samples.ndim != 2:
        raise ValueError("training samples must be a 2D array [samples, features]")
    if context_samples is not None:
        if context_samples.ndim != 2:
            raise ValueError("context_samples must be a 2D array [samples, context_features]")
        if context_samples.shape[0] != noisy_samples.shape[0]:
            raise ValueError("context_samples and noisy_samples must share the same sample count")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cpu")
    context_dim = 0 if context_samples is None else context_samples.shape[1]
    model = _build_model(
        input_dim=noisy_samples.shape[1] + context_dim,
        output_dim=noisy_samples.shape[1],
        config=config,
    ).to(device)

    noisy_tensor = torch.tensor(noisy_samples, dtype=torch.float32)
    ideal_tensor = torch.tensor(ideal_samples, dtype=torch.float32)
    context_tensor = (
        torch.tensor(context_samples, dtype=torch.float32)
        if context_samples is not None
        else None
    )
    model_input_tensor = _compose_model_input(noisy_tensor, context_tensor)

    train_loader, val_loader = _build_loaders(model_input_tensor, ideal_tensor, config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_loss = 0.0
    val_loss = 0.0
    for _ in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for noisy_batch, ideal_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            ideal_batch = ideal_batch.to(device)

            optimizer.zero_grad()
            prediction = model(noisy_batch)
            loss = criterion(prediction, ideal_batch)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_batches += 1

        if train_batches:
            train_loss /= train_batches

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for noisy_batch, ideal_batch in val_loader:
                noisy_batch = noisy_batch.to(device)
                ideal_batch = ideal_batch.to(device)
                prediction = model(noisy_batch)
                val_loss += float(criterion(prediction, ideal_batch).item())
                val_batches += 1

        if val_batches:
            val_loss /= val_batches

    model.eval()
    with torch.no_grad():
        mitigated_tensor = model(model_input_tensor.to(device)).cpu().numpy()

    baseline_mae = mean_absolute_error(noisy_samples, ideal_samples)
    mitigated_mae = mean_absolute_error(mitigated_tensor, ideal_samples)
    baseline_fidelity = _average_fidelity(noisy_samples, ideal_samples)
    mitigated_fidelity = _average_fidelity(mitigated_tensor, ideal_samples)

    mae_reduction = 100.0 * (baseline_mae - mitigated_mae) / max(baseline_mae, 1e-12)

    summary = TrainSummary(
        train_loss=train_loss,
        val_loss=val_loss,
        baseline_mae=baseline_mae,
        mitigated_mae=mitigated_mae,
        mae_reduction_pct=mae_reduction,
        baseline_fidelity=baseline_fidelity,
        mitigated_fidelity=mitigated_fidelity,
    )
    return model, summary


def mitigate_distribution(
    model: nn.Module,
    noisy_distribution: np.ndarray,
    context_features: np.ndarray | None = None,
) -> np.ndarray:
    """Apply trained model to a single noisy probability distribution."""
    input_vector = normalize_probability_vector(np.asarray(noisy_distribution, dtype=np.float32))
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
    if context_features is not None:
        context_vector = np.asarray(context_features, dtype=np.float32)
        context_tensor = torch.tensor(context_vector, dtype=torch.float32).unsqueeze(0)
        input_tensor = _compose_model_input(input_tensor, context_tensor)

    with torch.no_grad():
        output_tensor = model(input_tensor)
    output = output_tensor.squeeze(0).cpu().numpy()
    return normalize_probability_vector(output)


def _build_model(input_dim: int, output_dim: int, config: TrainConfig) -> nn.Module:
    model_type = config.model_type.lower()
    if model_type == "mlp":
        return QEMMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
        )
    if model_type == "autoencoder":
        return QEMAutoencoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        )
    raise ValueError("model_type must be 'mlp' or 'autoencoder'")


def _build_loaders(
    model_input_tensor: torch.Tensor,
    ideal_tensor: torch.Tensor,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
    sample_count = model_input_tensor.shape[0]
    indices = torch.randperm(sample_count)
    split = max(1, min(sample_count - 1, int(sample_count * config.train_split)))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = TensorDataset(model_input_tensor[train_idx], ideal_tensor[train_idx])
    val_dataset = TensorDataset(model_input_tensor[val_idx], ideal_tensor[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def _average_fidelity(predictions: np.ndarray, targets: np.ndarray) -> float:
    values = [
        distribution_fidelity(predictions[idx], targets[idx])
        for idx in range(predictions.shape[0])
    ]
    return float(np.mean(values))


def _compose_model_input(
    noisy_tensor: torch.Tensor,
    context_tensor: torch.Tensor | None,
) -> torch.Tensor:
    if context_tensor is None:
        return noisy_tensor
    return torch.cat([noisy_tensor, context_tensor], dim=1)
