from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .config import ExperimentConfig, TrainingConfig, load_experiment_config
from .plots import plot_history
from .train import PolicyName, train_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regime-aware hedging experiment")
    parser.add_argument(
        "--model",
        choices=["residual", "deep", "asymptotic", "oracle", "zero"],
        default="residual",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default="outputs/run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_experiment_config(args.config) if args.config else ExperimentConfig()
    training = config.training
    if args.epochs is not None:
        training = replace(training, epochs=args.epochs)
    if args.batch_size is not None:
        training = replace(training, batch_size=args.batch_size)
    if args.device is not None:
        training = replace(training, device=args.device)
    config = replace(config, training=training)

    out_dir = Path(args.out)
    _, history = train_policy(args.model, config, out_dir)
    if history:
        latest = history[-1]
        print("Final metrics:")
        for key in [
            "mean_utility",
            "terminal_mse",
            "mean_cost",
            "mean_tracking_loss",
            "stress_utility",
            "auxiliary_loss",
        ]:
            if key in latest:
                print(f"  {key}: {latest[key]:.6g}")
    history_path = out_dir / "history.json"
    if history_path.exists():
        plot_history(history_path, out_dir / "history.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

