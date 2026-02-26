"""Train a QEM model from generated noisy/ideal Grover datasets.

Example:
    python backend/scripts/train_qem.py --n-qubits 4 --sample-count 512 --epochs 80
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from src.ml import QEMService, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QEM model")
    parser.add_argument("--n-qubits", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--model-type", choices=["mlp", "autoencoder"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/experiments/qem_model.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    service = QEMService()
    artifact = service.train_model(
        n_qubits=args.n_qubits,
        sample_count=args.sample_count,
        shots=args.shots,
        train_config=TrainConfig(
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        include_thermal_relaxation=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n_qubits": artifact.n_qubits,
            "model_type": artifact.model_type,
            "trained_at_utc": artifact.trained_at_utc,
            "state_dict": artifact.model.state_dict(),
            "summary": asdict(artifact.summary),
        },
        args.output,
    )

    print(f"Model saved to {args.output.resolve()}")
    print(
        "MAE reduction: "
        f"{artifact.summary.mae_reduction_pct:.2f}% "
        f"(baseline={artifact.summary.baseline_mae:.5f}, mitigated={artifact.summary.mitigated_mae:.5f})"
    )


if __name__ == "__main__":
    main()
