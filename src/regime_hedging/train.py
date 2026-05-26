from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch

from .config import ExperimentConfig
from .models import DeepHedgingPolicy, RegimeAwareResidualPolicy
from .objective import rollout_policy
from .policies import AsymptoticPolicy, ZeroTradePolicy
from .torch_simulator import simulate_paths_torch

PolicyName = Literal["residual", "deep", "asymptotic", "oracle", "zero"]


def make_policy(name: PolicyName, config: ExperimentConfig):
    if name == "residual":
        return RegimeAwareResidualPolicy(config.market, config.model)
    if name == "deep":
        return DeepHedgingPolicy(config.market, config.model)
    if name == "asymptotic":
        return AsymptoticPolicy(config.market, oracle=False)
    if name == "oracle":
        return AsymptoticPolicy(config.market, oracle=True)
    if name == "zero":
        return ZeroTradePolicy()
    raise ValueError(f"unknown policy name: {name}")


@torch.no_grad()
def evaluate_policy(policy, config: ExperimentConfig, seed: int) -> dict[str, float]:
    policy.eval()
    generator = torch.Generator(device=config.training.device)
    generator.manual_seed(seed)

    totals: dict[str, float] = {}
    for _ in range(config.training.eval_batches):
        batch = simulate_paths_torch(
            config.market,
            config.training.batch_size,
            device=config.training.device,
            generator=generator,
        )
        result = rollout_policy(policy, batch, config.market, aux_weight=0.0)
        for key, value in result.metrics.items():
            totals[key] = totals.get(key, 0.0) + value

    return {key: value / config.training.eval_batches for key, value in totals.items()}


def train_policy(name: PolicyName, config: ExperimentConfig, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_json(out_dir / "config.json")

    torch.manual_seed(config.training.seed)
    policy = make_policy(name, config).to(config.training.device)
    params = [param for param in policy.parameters() if param.requires_grad]

    history: list[dict[str, float | int | str]] = []
    if not params:
        metrics = evaluate_policy(policy, config, config.training.seed + 10_000)
        record = {"epoch": 0, "model": name, **metrics}
        history.append(record)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return policy, history

    optimizer = torch.optim.Adam(params, lr=config.training.learning_rate)
    generator = torch.Generator(device=config.training.device)
    generator.manual_seed(config.training.seed)

    for epoch in range(1, config.training.epochs + 1):
        policy.train()
        batch = simulate_paths_torch(
            config.market,
            config.training.batch_size,
            device=config.training.device,
            generator=generator,
        )
        result = rollout_policy(
            policy,
            batch,
            config.market,
            aux_weight=config.training.aux_weight,
            stress_weight=config.training.stress_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        if config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, config.training.grad_clip)
        optimizer.step()

        if epoch == 1 or epoch % config.training.eval_every == 0 or epoch == config.training.epochs:
            eval_metrics = evaluate_policy(policy, config, config.training.seed + epoch)
            record = {"epoch": epoch, "model": name, **eval_metrics}
            history.append(record)
            (out_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

    torch.save(policy.state_dict(), out_dir / "policy.pt")
    return policy, history
