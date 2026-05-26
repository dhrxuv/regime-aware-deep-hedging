from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import ExperimentConfig
from .train import PolicyName, evaluate_policy, make_policy


def evaluate_untrained_baselines(
    config: ExperimentConfig,
    names: Iterable[PolicyName] = ("zero", "asymptotic", "oracle"),
    seed: int = 1234,
) -> pd.DataFrame:
    rows = []
    for idx, name in enumerate(names):
        policy = make_policy(name, config).to(config.training.device)
        metrics = evaluate_policy(policy, config, seed + idx)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows)

