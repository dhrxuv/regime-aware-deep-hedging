from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import MarketConfig


@dataclass(frozen=True)
class TorchPathBatch:
    regimes: torch.Tensor
    price: torch.Tensor
    returns: torch.Tensor
    target: torch.Tensor
    sigma: torch.Tensor
    liquidity_cost: torch.Tensor
    mu: torch.Tensor
    spread_signal: torch.Tensor
    obs: torch.Tensor

    @property
    def stress_mask(self) -> torch.Tensor:
        return self.regimes == 1


def _tensor(values: tuple[float, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def simulate_paths_torch(
    config: MarketConfig,
    batch_size: int,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
) -> TorchPathBatch:
    """Differentiable-enough simulator for policy optimization.

    Regime draws are exogenous and do not need gradients. Policy gradients flow
    through positions, rates, transaction costs, and the objective.
    """

    device = torch.device(device)
    n_steps = config.n_steps
    dt = config.dt
    sqrt_dt = dt**0.5

    transition = torch.tensor(
        config.transition_matrix, dtype=torch.float32, device=device
    )
    initial_probs = _tensor(config.initial_regime_probs, device)

    regimes = torch.empty(batch_size, n_steps, dtype=torch.long, device=device)
    regimes[:, 0] = torch.multinomial(
        initial_probs, batch_size, replacement=True, generator=generator
    )
    for step in range(1, n_steps):
        prev_probs = transition[regimes[:, step - 1]]
        regimes[:, step] = torch.multinomial(
            prev_probs, 1, replacement=True, generator=generator
        ).squeeze(1)

    mu_by = _tensor(config.mu_by_regime, device)
    sigma_by = _tensor(config.sigma_by_regime, device)
    lambda_by = _tensor(config.lambda_by_regime, device)

    mu = mu_by[regimes]
    sigma = sigma_by[regimes]
    liquidity_cost = lambda_by[regimes]

    eps = torch.randn(batch_size, n_steps, device=device, generator=generator)
    returns = mu * dt + sigma * sqrt_dt * eps
    price = torch.empty(batch_size, n_steps + 1, device=device)
    price[:, 0] = config.s0
    price[:, 1:] = config.s0 + torch.cumsum(returns, dim=1)

    target = torch.empty(batch_size, n_steps + 1, device=device)
    target[:, 0] = config.initial_target
    target_noise = torch.randn(batch_size, n_steps, device=device, generator=generator)
    for step in range(n_steps):
        previous = target[:, step]
        drift = config.target_kappa * (config.target_mean - previous) * dt
        diffusion = config.target_vol * sqrt_dt * target_noise[:, step]
        target[:, step + 1] = previous + drift + diffusion

    lambda_mean = lambda_by.mean()
    lambda_std = lambda_by.std(unbiased=False).clamp_min(1.0e-12)
    spread_signal = (liquidity_cost - lambda_mean) / lambda_std
    spread_signal = spread_signal + config.spread_noise * torch.randn(
        batch_size, n_steps, device=device, generator=generator
    )

    prev_return = torch.zeros_like(returns)
    prev_return[:, 1:] = returns[:, :-1]
    scaled_prev_return = prev_return / sqrt_dt
    scaled_abs_return = prev_return.abs() / sqrt_dt
    time = torch.linspace(0.0, 1.0, n_steps, device=device).view(1, n_steps)
    time = time.expand(batch_size, -1)
    obs = torch.stack(
        [
            time,
            scaled_prev_return,
            spread_signal,
            scaled_abs_return,
            target[:, :-1],
        ],
        dim=-1,
    )

    return TorchPathBatch(
        regimes=regimes,
        price=price,
        returns=returns,
        target=target,
        sigma=sigma,
        liquidity_cost=liquidity_cost,
        mu=mu,
        spread_signal=spread_signal,
        obs=obs,
    )

