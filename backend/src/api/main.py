"""FastAPI application for Grover robustness simulation and QEM inference."""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    GroverRunRequest,
    GroverRunResponse,
    IterationSweepPoint,
    IterationSweepRequest,
    NoiseSweepPoint,
    NoiseSweepRequest,
    QEMStatusItem,
    QEMStatusResponse,
    QEMTrainRequest,
    QEMTrainResponse,
)
from src.ml import QEMService, TrainConfig
from src.quantum import (
    NoiseParameters,
    basis_labels,
    build_grover_circuit,
    distribution_fidelity,
    mean_absolute_error,
    optimal_grover_iterations,
    run_iteration_sweep,
    run_noise_strength_sweep,
    simulate_grover_run,
    success_probability,
)


app = FastAPI(
    title="QEM-Grover Robustness API",
    description="API for noisy Grover simulation, failure-regime sweeps, and QEM mitigation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qem_service = QEMService()


@app.get("/", tags=["health"])
async def health_check() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok", "project": "qem-grover-robustness"}


@app.get("/api/grover/ideal", tags=["grover"])
async def grover_ideal(n_qubits: int = 3, marked_state: int = 5) -> dict[str, object]:
    """Return circuit metadata for a noiseless Grover setup."""
    _validate_marked_state(n_qubits, marked_state)
    circuit = build_grover_circuit(n_qubits=n_qubits, marked_states=[marked_state])
    return {
        "n_qubits": n_qubits,
        "marked_state": marked_state,
        "depth": circuit.depth(),
        "gate_count": circuit.size(),
        "circuit_ascii": circuit.draw(output="text").single_string(),
    }


@app.post("/api/grover/run", response_model=GroverRunResponse, tags=["grover"])
async def run_grover(request: GroverRunRequest) -> GroverRunResponse:
    """Run one ideal/noisy Grover simulation and optionally apply QEM mitigation."""
    _validate_marked_state(request.n_qubits, request.marked_state)
    marked_states = [request.marked_state]

    ideal_run = simulate_grover_run(
        n_qubits=request.n_qubits,
        marked_states=marked_states,
        shots=request.shots,
        num_iterations=request.num_iterations,
        noise=NoiseParameters(),
    )

    noisy_run = simulate_grover_run(
        n_qubits=request.n_qubits,
        marked_states=marked_states,
        shots=request.shots,
        num_iterations=request.num_iterations,
        noise=_noise_params_from_request(request),
    )

    mitigated_distribution: np.ndarray | None = None
    mitigated_success: float | None = None
    mitigated_mae: float | None = None
    mitigated_fidelity: float | None = None

    if request.apply_mitigation:
        context_features = _build_context_features(
            n_qubits=request.n_qubits,
            marked_state=request.marked_state,
            iterations=noisy_run.iterations,
            single_noise=request.noise.single_qubit_depolarizing,
            two_noise=request.noise.two_qubit_depolarizing,
            readout_error=request.noise.readout_error,
            thermal_enabled=request.noise.enable_thermal_relaxation,
        )
        mitigated_distribution = qem_service.mitigate(
            n_qubits=request.n_qubits,
            noisy_distribution=noisy_run.distribution,
            context_features=context_features,
            auto_train_if_missing=request.auto_train_if_missing,
            sample_count=192,
            shots=request.shots,
            include_thermal_relaxation=request.noise.enable_thermal_relaxation,
        )
        mitigated_success = success_probability(mitigated_distribution, marked_states)
        mitigated_mae = mean_absolute_error(mitigated_distribution, ideal_run.distribution)
        mitigated_fidelity = distribution_fidelity(mitigated_distribution, ideal_run.distribution)

    return GroverRunResponse(
        n_qubits=request.n_qubits,
        marked_state=request.marked_state,
        shots=request.shots,
        iterations=noisy_run.iterations,
        basis_labels=basis_labels(request.n_qubits),
        ideal_distribution=ideal_run.distribution.tolist(),
        noisy_distribution=noisy_run.distribution.tolist(),
        mitigated_distribution=(
            mitigated_distribution.tolist() if mitigated_distribution is not None else None
        ),
        ideal_success_probability=ideal_run.success_probability,
        noisy_success_probability=noisy_run.success_probability,
        mitigated_success_probability=mitigated_success,
        noisy_mae=mean_absolute_error(noisy_run.distribution, ideal_run.distribution),
        mitigated_mae=mitigated_mae,
        noisy_fidelity=distribution_fidelity(noisy_run.distribution, ideal_run.distribution),
        mitigated_fidelity=mitigated_fidelity,
        depth=noisy_run.depth,
        gate_count=noisy_run.gate_count,
    )


@app.post(
    "/api/grover/sweep/iterations",
    response_model=list[IterationSweepPoint],
    tags=["grover"],
)
async def sweep_iterations(request: IterationSweepRequest) -> list[IterationSweepPoint]:
    """Compute success probability vs iteration count."""
    _validate_marked_state(request.n_qubits, request.marked_state)

    mitigation_fn = None
    if request.include_mitigation:
        def mitigation_fn(distribution: np.ndarray, iteration_value: float) -> np.ndarray:
            context_features = _build_context_features(
                n_qubits=request.n_qubits,
                marked_state=request.marked_state,
                iterations=int(iteration_value),
                single_noise=request.noise.single_qubit_depolarizing,
                two_noise=request.noise.two_qubit_depolarizing,
                readout_error=request.noise.readout_error,
                thermal_enabled=request.noise.enable_thermal_relaxation,
            )
            return qem_service.mitigate(
                n_qubits=request.n_qubits,
                noisy_distribution=distribution,
                context_features=context_features,
                auto_train_if_missing=True,
                sample_count=160,
                shots=request.shots,
                include_thermal_relaxation=request.noise.enable_thermal_relaxation,
            )

    points = run_iteration_sweep(
        n_qubits=request.n_qubits,
        marked_states=[request.marked_state],
        shots=request.shots,
        max_iterations=request.max_iterations,
        noise=_noise_params_from_request(request),
        mitigation_fn=mitigation_fn,
    )

    return [
        IterationSweepPoint(
            iteration=int(point.x_value),
            ideal_success_probability=point.ideal_success,
            noisy_success_probability=point.noisy_success,
            mitigated_success_probability=point.mitigated_success,
        )
        for point in points
    ]


@app.post(
    "/api/grover/sweep/noise",
    response_model=list[NoiseSweepPoint],
    tags=["grover"],
)
async def sweep_noise(request: NoiseSweepRequest) -> list[NoiseSweepPoint]:
    """Compute success probability vs depolarizing noise strength."""
    _validate_marked_state(request.n_qubits, request.marked_state)
    if request.noise_max <= request.noise_min:
        raise HTTPException(status_code=422, detail="noise_max must be greater than noise_min")

    noise_values = np.linspace(request.noise_min, request.noise_max, request.steps).tolist()

    mitigation_fn = None
    if request.include_mitigation:
        def mitigation_fn(distribution: np.ndarray, noise_strength: float) -> np.ndarray:
            single_noise = float(noise_strength)
            two_noise = min(1.0, single_noise * 2.0)
            context_features = _build_context_features(
                n_qubits=request.n_qubits,
                marked_state=request.marked_state,
                iterations=request.num_iterations,
                single_noise=single_noise,
                two_noise=two_noise,
                readout_error=request.noise.readout_error,
                thermal_enabled=request.noise.enable_thermal_relaxation,
            )
            return qem_service.mitigate(
                n_qubits=request.n_qubits,
                noisy_distribution=distribution,
                context_features=context_features,
                auto_train_if_missing=True,
                sample_count=160,
                shots=request.shots,
                include_thermal_relaxation=request.noise.enable_thermal_relaxation,
            )

    points = run_noise_strength_sweep(
        n_qubits=request.n_qubits,
        marked_states=[request.marked_state],
        shots=request.shots,
        num_iterations=request.num_iterations,
        noise_strength_values=noise_values,
        base_noise=_noise_params_from_request(request),
        mitigation_fn=mitigation_fn,
    )

    return [
        NoiseSweepPoint(
            noise_strength=point.x_value,
            ideal_success_probability=point.ideal_success,
            noisy_success_probability=point.noisy_success,
            mitigated_success_probability=point.mitigated_success,
        )
        for point in points
    ]


@app.post("/api/qem/train", response_model=QEMTrainResponse, tags=["qem"])
async def train_qem(request: QEMTrainRequest) -> QEMTrainResponse:
    """Train QEM model on generated noisy/ideal paired data."""
    train_config = TrainConfig(
        model_type=request.model_type,
        hidden_dim=request.hidden_dim,
        latent_dim=request.latent_dim,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        seed=request.seed,
    )

    artifact = qem_service.train_model(
        n_qubits=request.n_qubits,
        sample_count=request.sample_count,
        shots=request.shots,
        train_config=train_config,
        include_thermal_relaxation=request.include_thermal_relaxation,
    )

    return QEMTrainResponse(
        n_qubits=artifact.n_qubits,
        model_type=artifact.model_type,
        sample_count=artifact.sample_count,
        trained_at_utc=artifact.trained_at_utc,
        train_loss=artifact.summary.train_loss,
        val_loss=artifact.summary.val_loss,
        baseline_mae=artifact.summary.baseline_mae,
        mitigated_mae=artifact.summary.mitigated_mae,
        mae_reduction_pct=artifact.summary.mae_reduction_pct,
        baseline_fidelity=artifact.summary.baseline_fidelity,
        mitigated_fidelity=artifact.summary.mitigated_fidelity,
    )


@app.get("/api/qem/status", response_model=QEMStatusResponse, tags=["qem"])
async def qem_status() -> QEMStatusResponse:
    """List trained QEM models currently loaded in memory."""
    artifacts = qem_service.status()
    return QEMStatusResponse(
        models=[
            QEMStatusItem(
                n_qubits=artifact.n_qubits,
                model_type=artifact.model_type,
                sample_count=artifact.sample_count,
                trained_at_utc=artifact.trained_at_utc,
                mae_reduction_pct=artifact.summary.mae_reduction_pct,
                baseline_mae=artifact.summary.baseline_mae,
                mitigated_mae=artifact.summary.mitigated_mae,
            )
            for artifact in artifacts
        ]
    )


def _validate_marked_state(n_qubits: int, marked_state: int) -> None:
    if marked_state >= 2 ** n_qubits:
        raise HTTPException(
            status_code=422,
            detail=f"marked_state must be in [0, {2 ** n_qubits - 1}] for n_qubits={n_qubits}",
        )


def _noise_params_from_request(request: GroverRunRequest | IterationSweepRequest | NoiseSweepRequest) -> NoiseParameters:
    noise = request.noise
    return NoiseParameters(
        single_qubit_depolarizing=noise.single_qubit_depolarizing,
        two_qubit_depolarizing=noise.two_qubit_depolarizing,
        readout_error=noise.readout_error,
        enable_thermal_relaxation=noise.enable_thermal_relaxation,
        t1_seconds=noise.t1_seconds,
        t2_seconds=noise.t2_seconds,
        one_qubit_gate_time_seconds=noise.one_qubit_gate_time_seconds,
        two_qubit_gate_time_seconds=noise.two_qubit_gate_time_seconds,
    )


def _build_context_features(
    n_qubits: int,
    marked_state: int,
    iterations: int | None,
    single_noise: float,
    two_noise: float,
    readout_error: float,
    thermal_enabled: bool,
) -> np.ndarray:
    optimal_iterations = optimal_grover_iterations(n_qubits=n_qubits, n_marked=1)
    if iterations is None:
        iterations = optimal_iterations

    normalized_iteration = iterations / max(1.0, float(optimal_iterations * 2))
    normalized_marked_state = marked_state / max(1.0, float(2**n_qubits - 1))
    return np.array(
        [
            single_noise,
            two_noise,
            readout_error,
            1.0 if thermal_enabled else 0.0,
            normalized_iteration,
            normalized_marked_state,
        ],
        dtype=np.float32,
    )
