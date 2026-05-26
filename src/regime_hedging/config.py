from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class MarketConfig:
    """Parameters for the hidden-liquidity hedging environment."""

    n_steps: int = 64
    horizon: float = 1.0
    s0: float = 100.0

    mu_by_regime: tuple[float, ...] = (0.02, -0.01, 0.005)
    sigma_by_regime: tuple[float, ...] = (0.18, 0.42, 0.27)
    lambda_by_regime: tuple[float, ...] = (1.0e-4, 1.2e-3, 4.0e-4)
    transition_matrix: tuple[tuple[float, ...], ...] = (
        (0.965, 0.020, 0.015),
        (0.090, 0.850, 0.060),
        (0.080, 0.050, 0.870),
    )
    initial_regime_probs: tuple[float, ...] = (0.85, 0.05, 0.10)
    stress_regime: int = 1

    target_mean: float = 0.0
    target_kappa: float = 1.25
    target_vol: float = 0.30
    initial_target: float = 0.0
    initial_position: float = 0.0

    risk_aversion: float = 1.0
    cost_power: float = 2.0
    terminal_penalty: float = 0.20
    max_trade_rate: float = 8.0
    spread_noise: float = 0.25
    seed: int = 7

    @property
    def dt(self) -> float:
        return self.horizon / self.n_steps

    @property
    def n_regimes(self) -> int:
        return len(self.lambda_by_regime)

    @property
    def obs_dim(self) -> int:
        # time, previous return, noisy spread signal, previous abs return, target
        return 5

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.cost_power <= 1:
            raise ValueError("cost_power must be greater than 1")
        if self.max_trade_rate <= 0:
            raise ValueError("max_trade_rate must be positive")

        lengths = {
            len(self.mu_by_regime),
            len(self.sigma_by_regime),
            len(self.lambda_by_regime),
            len(self.initial_regime_probs),
            len(self.transition_matrix),
        }
        if len(lengths) != 1:
            raise ValueError("all regime parameter arrays must have equal length")

        transition = np.asarray(self.transition_matrix, dtype=float)
        if transition.shape != (self.n_regimes, self.n_regimes):
            raise ValueError("transition_matrix must be square with n_regimes rows")
        if np.any(transition < 0):
            raise ValueError("transition probabilities must be non-negative")
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("each transition row must sum to 1")

        initial_probs = np.asarray(self.initial_regime_probs, dtype=float)
        if np.any(initial_probs < 0) or not np.isclose(initial_probs.sum(), 1.0):
            raise ValueError("initial_regime_probs must be a probability vector")
        if not 0 <= self.stress_regime < self.n_regimes:
            raise ValueError("stress_regime is out of range")


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 48
    mlp_width: int = 64
    residual_scale: float = 0.35
    stress_gate_strength: float = 0.75
    use_auxiliary_regime_loss: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    batch_size: int = 1024
    learning_rate: float = 1.0e-3
    grad_clip: float = 1.0
    aux_weight: float = 0.02
    stress_weight: float = 0.35
    eval_every: int = 50
    eval_batches: int = 4
    seed: int = 11
    device: str = "cpu"


@dataclass(frozen=True)
class ExperimentConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    market = MarketConfig(**data.get("market", {}))
    model = ModelConfig(**data.get("model", {}))
    training = TrainingConfig(**data.get("training", {}))
    return ExperimentConfig(market=market, model=model, training=training)
