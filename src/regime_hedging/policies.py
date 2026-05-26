from __future__ import annotations

import torch
from torch import nn

from .config import MarketConfig


class BasePolicy(nn.Module):
    def initial_state(self, batch_size: int, device: torch.device):
        return None

    def forward_step(self, step: int, obs_t, phi_t, batch, hidden):
        raise NotImplementedError


class ZeroTradePolicy(BasePolicy):
    def forward_step(self, step: int, obs_t, phi_t, batch, hidden):
        return torch.zeros_like(phi_t), hidden, {}


class AsymptoticPolicy(BasePolicy):
    """Theory-motivated baseline for quadratic trading costs.

    For quadratic rate costs and quadratic tracking risk, the infinite-horizon
    LQ control suggests trading toward the frictionless target at speed roughly
    sqrt(gamma * sigma^2 / lambda). We use this as the leading-order baseline.
    """

    def __init__(self, market: MarketConfig, oracle: bool = False) -> None:
        super().__init__()
        self.market = market
        self.oracle = oracle
        self.register_buffer(
            "sigma_mean",
            torch.tensor(float(sum(market.sigma_by_regime) / market.n_regimes)),
        )
        self.register_buffer(
            "lambda_mean",
            torch.tensor(float(sum(market.lambda_by_regime) / market.n_regimes)),
        )

    def forward_step(self, step: int, obs_t, phi_t, batch, hidden):
        target_t = obs_t[:, 4]
        gap = target_t - phi_t
        if self.oracle:
            sigma = batch.sigma[:, step]
            liquidity_cost = batch.liquidity_cost[:, step]
        else:
            sigma = self.sigma_mean.to(phi_t.device)
            liquidity_cost = self.lambda_mean.to(phi_t.device)
        speed = torch.sqrt(
            self.market.risk_aversion * sigma.pow(2) / liquidity_cost.clamp_min(1.0e-8)
        )
        rate = speed * gap
        rate = torch.clamp(rate, -self.market.max_trade_rate, self.market.max_trade_rate)
        return rate, hidden, {"base_rate": rate}

