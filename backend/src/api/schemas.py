"""Pydantic request/response schemas for the FastAPI interface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NoiseConfig(BaseModel):
    """API-facing noise parameters."""

    single_qubit_depolarizing: float = Field(default=0.01, ge=0.0, le=1.0)
    two_qubit_depolarizing: float = Field(default=0.02, ge=0.0, le=1.0)
    readout_error: float = Field(default=0.01, ge=0.0, le=1.0)
    enable_thermal_relaxation: bool = True
    t1_seconds: float = Field(default=120e-6, gt=0.0)
    t2_seconds: float = Field(default=80e-6, gt=0.0)
    one_qubit_gate_time_seconds: float = Field(default=50e-9, gt=0.0)
    two_qubit_gate_time_seconds: float = Field(default=300e-9, gt=0.0)


class GroverRunRequest(BaseModel):
    """Parameters for single Grover simulation run."""

    n_qubits: int = Field(default=4, ge=2, le=8)
    marked_state: int = Field(default=5, ge=0)
    shots: int = Field(default=10_000, ge=100, le=100_000)
    num_iterations: int | None = Field(default=None, ge=1, le=64)
    apply_mitigation: bool = True
    auto_train_if_missing: bool = True
    noise: NoiseConfig = Field(default_factory=NoiseConfig)


class GroverRunResponse(BaseModel):
    """Simulation result payload consumed by the frontend dashboard."""

    n_qubits: int
    marked_state: int
    shots: int
    iterations: int
    basis_labels: list[str]
    ideal_distribution: list[float]
    noisy_distribution: list[float]
    mitigated_distribution: list[float] | None
    ideal_success_probability: float
    noisy_success_probability: float
    mitigated_success_probability: float | None
    noisy_mae: float
    mitigated_mae: float | None
    noisy_fidelity: float
    mitigated_fidelity: float | None
    depth: int
    gate_count: int


class IterationSweepRequest(BaseModel):
    """Request for success-vs-iterations sweep."""

    n_qubits: int = Field(default=4, ge=2, le=8)
    marked_state: int = Field(default=5, ge=0)
    shots: int = Field(default=6_000, ge=100, le=100_000)
    max_iterations: int | None = Field(default=None, ge=2, le=64)
    include_mitigation: bool = True
    noise: NoiseConfig = Field(default_factory=NoiseConfig)


class IterationSweepPoint(BaseModel):
    """One chart point for success vs iteration count."""

    iteration: int
    ideal_success_probability: float
    noisy_success_probability: float
    mitigated_success_probability: float | None = None


class NoiseSweepRequest(BaseModel):
    """Request for success-vs-noise sweep."""

    n_qubits: int = Field(default=4, ge=2, le=8)
    marked_state: int = Field(default=5, ge=0)
    shots: int = Field(default=6_000, ge=100, le=100_000)
    num_iterations: int | None = Field(default=None, ge=1, le=64)
    noise_min: float = Field(default=0.0, ge=0.0, le=1.0)
    noise_max: float = Field(default=0.12, ge=0.0, le=1.0)
    steps: int = Field(default=10, ge=3, le=40)
    include_mitigation: bool = True
    noise: NoiseConfig = Field(default_factory=NoiseConfig)


class NoiseSweepPoint(BaseModel):
    """One chart point for success vs depolarizing strength."""

    noise_strength: float
    ideal_success_probability: float
    noisy_success_probability: float
    mitigated_success_probability: float | None = None


class QEMTrainRequest(BaseModel):
    """Model-training request for a specific qubit count."""

    n_qubits: int = Field(default=4, ge=2, le=8)
    sample_count: int = Field(default=256, ge=64, le=2_048)
    shots: int = Field(default=10_000, ge=500, le=100_000)
    include_thermal_relaxation: bool = True
    model_type: str = Field(default="mlp", pattern="^(mlp|autoencoder)$")
    hidden_dim: int = Field(default=128, ge=16, le=1024)
    latent_dim: int = Field(default=48, ge=8, le=512)
    epochs: int = Field(default=60, ge=5, le=500)
    batch_size: int = Field(default=32, ge=8, le=512)
    learning_rate: float = Field(default=1e-3, gt=0.0, lt=1.0)
    seed: int = Field(default=7, ge=0, le=999_999)


class QEMTrainResponse(BaseModel):
    """Training summary response."""

    n_qubits: int
    model_type: str
    sample_count: int
    trained_at_utc: str
    train_loss: float
    val_loss: float
    baseline_mae: float
    mitigated_mae: float
    mae_reduction_pct: float
    baseline_fidelity: float
    mitigated_fidelity: float


class QEMStatusItem(BaseModel):
    """Status of one trained QEM model."""

    n_qubits: int
    model_type: str
    sample_count: int
    trained_at_utc: str
    mae_reduction_pct: float
    baseline_mae: float
    mitigated_mae: float


class QEMStatusResponse(BaseModel):
    """Status endpoint payload."""

    models: list[QEMStatusItem]
