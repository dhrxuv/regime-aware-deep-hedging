from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history_json: str | Path, out_path: str | Path) -> None:
    history = pd.read_json(history_json)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(history["epoch"], history["mean_utility"])
    axes[0].set_title("Mean utility")
    axes[0].set_xlabel("epoch")

    axes[1].plot(history["epoch"], history["terminal_mse"])
    axes[1].set_title("Terminal MSE")
    axes[1].set_xlabel("epoch")

    axes[2].plot(history["epoch"], history["stress_utility"])
    axes[2].set_title("Stress utility")
    axes[2].set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
