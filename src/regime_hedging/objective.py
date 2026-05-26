from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import MarketConfig
from .costs import convex_power_cost


@dataclass(frozen=True)
class RolloutResult:
    loss: torch.Tensor
    utility: torch.Tensor
    metrics: dict[str, float]
    positions: torch.Tensor | None = None
    rates: torch.Tensor | None = None
    beliefs: torch.Tensor | None = None


def rollout_policy(
    policy,
    batch,
    market: MarketConfig,
    aux_weight: float = 0.0,
    stress_weight: float = 0.0,
    return_paths: bool = False,
) -> RolloutResult:
    batch_size = batch.regimes.shape[0]
    device = batch.regimes.device
    dt = market.dt

    phi = torch.full(
        (batch_size,),
        float(market.initial_position),
        dtype=torch.float32,
        device=device,
    )
    hidden = policy.initial_state(batch_size, device)

    rewards = []
    costs = []
    tracking_losses = []
    carry_terms = []
    aux_losses = []
    positions = [phi]
    rates = []
    beliefs = []

    for step in range(market.n_steps):
        obs_t = batch.obs[:, step]
        rate, hidden, aux = policy.forward_step(step, obs_t, phi, batch, hidden)
        target_t = batch.target[:, step]
        sigma_t = batch.sigma[:, step]
        lambda_t = batch.liquidity_cost[:, step]
        mu_t = batch.mu[:, step]

        trading_cost = lambda_t * convex_power_cost(rate, market.cost_power)
        tracking = 0.5 * market.risk_aversion * (sigma_t * (phi - target_t)).pow(2)
        carry = phi * mu_t
        reward = (carry - tracking - trading_cost) * dt

        rewards.append(reward)
        costs.append(trading_cost * dt)
        tracking_losses.append(tracking * dt)
        carry_terms.append(carry * dt)
        rates.append(rate)

        if "regime_logits" in aux:
            aux_losses.append(F.cross_entropy(aux["regime_logits"], batch.regimes[:, step]))
        if "belief" in aux and return_paths:
            beliefs.append(aux["belief"])

        phi = phi + rate * dt
        positions.append(phi)

    reward_by_step = torch.stack(rewards, dim=1)
    utility = reward_by_step.sum(dim=1)
    terminal_error = phi - batch.target[:, -1]
    terminal_penalty = 0.5 * market.terminal_penalty * terminal_error.pow(2)
    utility = utility - terminal_penalty

    stress_mask = batch.regimes == market.stress_regime
    if stress_mask.any():
        stress_rewards = reward_by_step.masked_fill(~stress_mask, 0.0).sum(dim=1)
        stress_utility = stress_rewards.mean()
    else:
        stress_utility = torch.tensor(0.0, device=device)

    objective_loss = -utility.mean()
    if stress_weight > 0:
        objective_loss = objective_loss - stress_weight * stress_utility
    if aux_losses and aux_weight > 0:
        auxiliary_loss = torch.stack(aux_losses).mean()
        loss = objective_loss + aux_weight * auxiliary_loss
    else:
        auxiliary_loss = torch.tensor(0.0, device=device)
        loss = objective_loss

    metrics = {
        "loss": float(loss.detach().cpu()),
        "mean_utility": float(utility.mean().detach().cpu()),
        "utility_std": float(utility.std(unbiased=False).detach().cpu()),
        "mean_cost": float(torch.stack(costs, dim=1).sum(dim=1).mean().detach().cpu()),
        "mean_tracking_loss": float(
            torch.stack(tracking_losses, dim=1).sum(dim=1).mean().detach().cpu()
        ),
        "mean_carry": float(
            torch.stack(carry_terms, dim=1).sum(dim=1).mean().detach().cpu()
        ),
        "terminal_mse": float(terminal_error.pow(2).mean().detach().cpu()),
        "stress_utility": float(stress_utility.detach().cpu()),
        "auxiliary_loss": float(auxiliary_loss.detach().cpu()),
    }

    positions_tensor = torch.stack(positions, dim=1) if return_paths else None
    rates_tensor = torch.stack(rates, dim=1) if return_paths else None
    beliefs_tensor = torch.stack(beliefs, dim=1) if beliefs else None
    return RolloutResult(
        loss=loss,
        utility=utility,
        metrics=metrics,
        positions=positions_tensor,
        rates=rates_tensor,
        beliefs=beliefs_tensor,
    )
