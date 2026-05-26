from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from regime_hedging.config import ExperimentConfig, MarketConfig, TrainingConfig
from regime_hedging.train import PolicyName, train_policy


HORIZONS = {
    "short": {"horizon": 0.25, "n_steps": 32},
    "medium": {"horizon": 1.00, "n_steps": 64},
    "long": {"horizon": 4.00, "n_steps": 128},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project comparison suite")
    parser.add_argument("--out", default="outputs/comparison_suite")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=6)
    parser.add_argument("--stress-weight", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["residual", "deep", "asymptotic", "oracle", "zero"],
        default=["residual", "deep", "asymptotic", "oracle"],
    )
    return parser.parse_args()


def make_config(horizon_name: str, args: argparse.Namespace) -> ExperimentConfig:
    horizon = HORIZONS[horizon_name]
    market = MarketConfig(
        horizon=horizon["horizon"],
        n_steps=horizon["n_steps"],
        seed=7 + horizon["n_steps"],
    )
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        eval_every=max(1, args.epochs // 5),
        device=args.device,
        seed=31 + horizon["n_steps"],
        stress_weight=args.stress_weight,
    )
    return ExperimentConfig(market=market, training=training)


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot = df.pivot(index="horizon", columns="model", values=metric)
    pivot = pivot.loc[[name for name in HORIZONS if name in pivot.index]]
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("horizon")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for horizon_name in HORIZONS:
        config = make_config(horizon_name, args)
        for model_name in args.models:
            run_dir = out_dir / horizon_name / model_name
            print(f"Running horizon={horizon_name} model={model_name}")
            _, history = train_policy(model_name, config, run_dir)
            final = dict(history[-1])
            final["horizon"] = horizon_name
            final["horizon_years"] = config.market.horizon
            final["n_steps"] = config.market.n_steps
            rows.append(final)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_metrics.csv", index=False)
    (out_dir / "comparison_metrics.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    for metric in ["mean_utility", "terminal_mse", "mean_cost", "stress_utility"]:
        plot_metric(df, metric, out_dir / f"{metric}.png")

    display_cols = [
        "horizon",
        "model",
        "mean_utility",
        "terminal_mse",
        "mean_cost",
        "stress_utility",
    ]
    print(df[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
