from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regime_hedging.config import MarketConfig
from regime_hedging.simulator import simulate_paths_np, summarize_batch


def main() -> int:
    config = MarketConfig(n_steps=32)
    batch = simulate_paths_np(config, batch_size=2048, seed=123)

    assert batch.regimes.shape == (2048, 32)
    assert batch.price.shape == (2048, 33)
    assert batch.target.shape == (2048, 33)
    assert batch.returns.shape == (2048, 32)
    assert batch.spread_signal.shape == (2048, 32)
    assert batch.liquidity_cost.min() > 0

    summary = summarize_batch(batch, config)
    assert 0.02 < summary["stress_share"] < 0.50
    assert summary["mean_abs_return"] > 0
    assert summary["terminal_target_std"] > 0

    print("Smoke test passed.")
    for key, value in summary.items():
        print(f"{key}: {value:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

