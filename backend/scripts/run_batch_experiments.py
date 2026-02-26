"""Batch Grover robustness experiments with optional QEM mitigation.

Example:
    python backend/scripts/run_batch_experiments.py --shots 10000 --include-mitigation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.ml import QEMService, TrainConfig
from src.quantum import NoiseParameters, optimal_grover_iterations, run_iteration_sweep, run_noise_strength_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch Grover robustness experiments")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--min-qubits", type=int, default=2)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--noise-max", type=float, default=0.15)
    parser.add_argument("--noise-steps", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--include-mitigation", action="store_true")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    noise_values = np.linspace(0.0, args.noise_max, args.noise_steps).tolist()
    qem_service = QEMService()

    iteration_rows: list[dict[str, float | int | None]] = []
    noise_rows: list[dict[str, float | int | None]] = []
    threshold_rows: list[dict[str, float | int | None]] = []

    for n_qubits in range(args.min_qubits, args.max_qubits + 1):
        marked_state = min(1, 2**n_qubits - 1)
        base_noise = NoiseParameters(
            single_qubit_depolarizing=0.03,
            two_qubit_depolarizing=0.06,
            readout_error=0.02,
            enable_thermal_relaxation=True,
        )

        mitigation_fn = None
        if args.include_mitigation:
            qem_service.train_model(
                n_qubits=n_qubits,
                sample_count=args.train_samples,
                shots=args.shots,
                train_config=TrainConfig(epochs=args.epochs),
                include_thermal_relaxation=True,
            )
            mitigation_fn = lambda distribution, _x_value, n=n_qubits, marked=marked_state: qem_service.mitigate(
                n_qubits=n,
                noisy_distribution=distribution,
                context_features=np.array(
                    [
                        base_noise.single_qubit_depolarizing,
                        base_noise.two_qubit_depolarizing,
                        base_noise.readout_error,
                        1.0 if base_noise.enable_thermal_relaxation else 0.0,
                        optimal_grover_iterations(n) / max(1.0, float(optimal_grover_iterations(n) * 2)),
                        marked / max(1.0, float(2**n - 1)),
                    ],
                    dtype=np.float32,
                ),
                auto_train_if_missing=False,
            )

        max_iterations = max(2, optimal_grover_iterations(n_qubits) * 2)
        iteration_points = run_iteration_sweep(
            n_qubits=n_qubits,
            marked_states=[marked_state],
            shots=args.shots,
            max_iterations=max_iterations,
            noise=base_noise,
            mitigation_fn=mitigation_fn,
        )

        for point in iteration_points:
            iteration_rows.append(
                {
                    "n_qubits": n_qubits,
                    "iteration": int(point.x_value),
                    "ideal_success": point.ideal_success,
                    "noisy_success": point.noisy_success,
                    "mitigated_success": point.mitigated_success,
                }
            )

        noise_points = run_noise_strength_sweep(
            n_qubits=n_qubits,
            marked_states=[marked_state],
            shots=args.shots,
            num_iterations=optimal_grover_iterations(n_qubits),
            noise_strength_values=noise_values,
            base_noise=base_noise,
            mitigation_fn=mitigation_fn,
        )

        for point in noise_points:
            noise_rows.append(
                {
                    "n_qubits": n_qubits,
                    "noise_strength": point.x_value,
                    "ideal_success": point.ideal_success,
                    "noisy_success": point.noisy_success,
                    "mitigated_success": point.mitigated_success,
                }
            )

        random_guess = 1.0 / (2**n_qubits)
        threshold = next(
            (point.x_value for point in noise_points if point.noisy_success <= random_guess),
            None,
        )
        threshold_rows.append(
            {
                "n_qubits": n_qubits,
                "classical_random_baseline": random_guess,
                "noise_threshold": threshold,
            }
        )

    pd.DataFrame(iteration_rows).to_csv(args.output_dir / "iteration_sweep.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(args.output_dir / "noise_sweep.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.output_dir / "failure_thresholds.csv", index=False)

    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
