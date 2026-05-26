from __future__ import annotations

import torch
from torch import nn

from .config import MarketConfig, ModelConfig
from .policies import BasePolicy


def _mlp(in_dim: int, width: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, width),
        nn.SiLU(),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, out_dim),
    )


class DeepHedgingPolicy(BasePolicy):
    """Shared-network Deep Hedging baseline."""

    def __init__(self, market: MarketConfig, model: ModelConfig) -> None:
        super().__init__()
        self.market = market
        self.net = _mlp(market.obs_dim + 2, model.mlp_width, 1)

    def forward_step(self, step: int, obs_t, phi_t, batch, hidden):
        target_t = obs_t[:, 4]
        gap = target_t - phi_t
        features = torch.cat([obs_t, phi_t[:, None], gap[:, None]], dim=-1)
        raw_rate = self.net(features).squeeze(-1)
        rate = self.market.max_trade_rate * torch.tanh(raw_rate)
        return rate, hidden, {}


class RegimeAwareResidualPolicy(BasePolicy):
    """GRU belief encoder plus residual correction over an asymptotic baseline."""

    def __init__(self, market: MarketConfig, model: ModelConfig) -> None:
        super().__init__()
        self.market = market
        self.model = model
        self.encoder = nn.GRUCell(market.obs_dim, model.hidden_size)
        self.regime_head = nn.Linear(model.hidden_size, market.n_regimes)
        self.residual_net = _mlp(model.hidden_size + 3, model.mlp_width, 1)
        self.register_buffer(
            "sigma_by_regime",
            torch.tensor(market.sigma_by_regime, dtype=torch.float32),
        )
        self.register_buffer(
            "lambda_by_regime",
            torch.tensor(market.lambda_by_regime, dtype=torch.float32),
        )

    def initial_state(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.model.hidden_size, device=device)

    def forward_step(self, step: int, obs_t, phi_t, batch, hidden):
        if hidden is None:
            hidden = self.initial_state(obs_t.shape[0], obs_t.device)
        hidden = self.encoder(obs_t, hidden)

        target_t = obs_t[:, 4]
        gap = target_t - phi_t
        logits = self.regime_head(hidden)
        belief = torch.softmax(logits, dim=-1)
        sigma_hat = belief @ self.sigma_by_regime.to(obs_t.device)
        lambda_hat = belief @ self.lambda_by_regime.to(obs_t.device)

        speed = torch.sqrt(
            self.market.risk_aversion
            * sigma_hat.pow(2)
            / lambda_hat.clamp_min(1.0e-8)
        )
        base_rate = torch.clamp(
            speed * gap, -self.market.max_trade_rate, self.market.max_trade_rate
        )

        residual_features = torch.cat(
            [hidden, phi_t[:, None], gap[:, None], base_rate[:, None]], dim=-1
        )
        residual_raw = self.residual_net(residual_features).squeeze(-1)
        residual = (
            self.model.residual_scale
            * self.market.max_trade_rate
            * torch.tanh(residual_raw)
        )
        stress_belief = belief[:, self.market.stress_regime]
        liquidity_gate = 1.0 / (1.0 + self.model.stress_gate_strength * stress_belief)
        rate = torch.clamp(
            liquidity_gate * (base_rate + residual),
            -self.market.max_trade_rate,
            self.market.max_trade_rate,
        )
        aux = {
            "regime_logits": logits,
            "belief": belief,
            "base_rate": base_rate,
            "residual": residual,
            "liquidity_gate": liquidity_gate,
        }
        return rate, hidden, aux
