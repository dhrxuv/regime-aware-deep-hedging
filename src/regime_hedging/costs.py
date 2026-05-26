from __future__ import annotations

import torch


def convex_power_cost(rate: torch.Tensor, power: float) -> torch.Tensor:
    """G(v) = |v|^q / q for q > 1."""

    return rate.abs().pow(power) / power

