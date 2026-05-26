from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MarketConfig


@dataclass(frozen=True)
class NumpyPathBatch:
    regimes: np.ndarray
    price: np.ndarray
    returns: np.ndarray
    target: np.ndarray
    sigma: np.ndarray
    liquidity_cost: np.ndarray
    mu: np.ndarray
    spread_signal: np.ndarray

    @property
    def stress_mask(self) -> np.ndarray:
        return self.regimes == 1


def simulate_paths_np(
    config: MarketConfig,
    batch_size: int,
    seed: int | None = None,
) -> NumpyPathBatch:
    """Simulate hidden liquidity regimes and observable market signals."""

    rng = np.random.default_rng(config.seed if seed is None else seed)
    n_steps = config.n_steps
    dt = config.dt
    sqrt_dt = np.sqrt(dt)

    transition = np.asarray(config.transition_matrix, dtype=float)
    initial_probs = np.asarray(config.initial_regime_probs, dtype=float)

    regimes = np.empty((batch_size, n_steps), dtype=np.int64)
    regimes[:, 0] = rng.choice(config.n_regimes, size=batch_size, p=initial_probs)
    for step in range(1, n_steps):
        prev = regimes[:, step - 1]
        uniform = rng.random(batch_size)
        cumulative = np.cumsum(transition[prev], axis=1)
        regimes[:, step] = (uniform[:, None] > cumulative).sum(axis=1)

    mu_by = np.asarray(config.mu_by_regime, dtype=float)
    sigma_by = np.asarray(config.sigma_by_regime, dtype=float)
    lambda_by = np.asarray(config.lambda_by_regime, dtype=float)

    mu = mu_by[regimes]
    sigma = sigma_by[regimes]
    liquidity_cost = lambda_by[regimes]

    returns = mu * dt + sigma * sqrt_dt * rng.standard_normal((batch_size, n_steps))
    price = np.empty((batch_size, n_steps + 1), dtype=float)
    price[:, 0] = config.s0
    price[:, 1:] = config.s0 + np.cumsum(returns, axis=1)

    target = np.empty((batch_size, n_steps + 1), dtype=float)
    target[:, 0] = config.initial_target
    target_noise = rng.standard_normal((batch_size, n_steps))
    for step in range(n_steps):
        previous = target[:, step]
        drift = config.target_kappa * (config.target_mean - previous) * dt
        diffusion = config.target_vol * sqrt_dt * target_noise[:, step]
        target[:, step + 1] = previous + drift + diffusion

    centered_lambda = (liquidity_cost - lambda_by.mean()) / (lambda_by.std() + 1.0e-12)
    spread_signal = centered_lambda + config.spread_noise * rng.standard_normal(
        (batch_size, n_steps)
    )

    return NumpyPathBatch(
        regimes=regimes,
        price=price,
        returns=returns,
        target=target,
        sigma=sigma,
        liquidity_cost=liquidity_cost,
        mu=mu,
        spread_signal=spread_signal,
    )


def summarize_batch(batch: NumpyPathBatch, config: MarketConfig) -> dict[str, float]:
    """Return lightweight diagnostics for smoke tests and notebooks."""

    stress_share = float((batch.regimes == config.stress_regime).mean())
    mean_cost = float(batch.liquidity_cost.mean())
    mean_sigma = float(batch.sigma.mean())
    mean_abs_return = float(np.abs(batch.returns).mean())
    target_std = float(batch.target[:, -1].std())
    return {
        "stress_share": stress_share,
        "mean_cost": mean_cost,
        "mean_sigma": mean_sigma,
        "mean_abs_return": mean_abs_return,
        "terminal_target_std": target_std,
    }

