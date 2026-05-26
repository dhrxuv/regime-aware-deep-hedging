# Regime-Aware Deep Hedging

This project implements a **Regime-Aware Residual Deep Hedging** algorithm under transaction costs and hidden liquidity states, building upon the research framework of Shi, Xu, and Zhang (2023).

> **Core Research Focus:** Learn when not to trade by dynamically inferring latent market liquidity regimes.

Standard Deep Hedging and ST-Hedging algorithms assume that trading transaction costs are directly known or follow static, deterministic trajectories. In contrast, this extension models transaction costs as driven by a hidden Markov liquidity regime (e.g., normal, stressed, and recovering states). The hedger does not observe the current regime directly and must infer it purely from noisy market observations, such as recent asset returns, absolute returns, and a noisy spread proxy.

---

## Core Idea

The model is formulated as a **Regime-Aware Residual ST-Hedger**:

```text
trade_rate = asymptotic_baseline_rate + learned_residual_correction
```

1. **Asymptotic Baseline:** Encodes structural financial domain knowledge (e.g., target-chasing trading speeds) based on estimated liquidity and volatility, ensuring stable optimization.
2. **Latent Belief Recurrent Encoder:** A Gated Recurrent Unit (GRU) that processes recent noisy signals to compute posterior probabilities for each liquidity regime.
3. **Residual Correction & Liquidity Gate:** A feed-forward neural network learns state-dependent corrections to the baseline speed, while a specialized stress gate automatically dials back trading intensity when the probability of liquidity stress is high.

---

## Project Structure

```text
src/regime_hedging/
  config.py           Experiment, market, model, and training configurations
  simulator.py        Numerical NumPy simulator for fast validation and smoke tests
  torch_simulator.py  Differentiable PyTorch-based simulator for end-to-end rollouts
  costs.py            Convex trading friction/transaction cost functions
  policies.py         Non-learning benchmarks (asymptotic and oracle baselines)
  models.py           Deep Hedging policies and Regime-Aware Residual networks
  objective.py        Differentiable hedging rollouts, utility, and tracking metrics
  train.py            Training loops and evaluation utilities
  evaluate.py         Repeated evaluation helpers across horizons
  plots.py            Plotting helpers for comparing policy performances
  run_experiment.py   CLI script to run and evaluate single model training sessions
scripts/
  smoke_test.py       Runs fast NumPy validation checks without PyTorch
  run_comparison_suite.py  Runs full multi-horizon model comparison benchmarks
report/
  project_report.pdf  Detailed research report/paper (PDF)
```

---

## Installation

Install the package and its requirements in your Python environment:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Running the Code

### 1. Verification (Smoke Test)

Run the lightweight NumPy-based simulation check to verify your environment setup (does not require PyTorch):

```bash
python scripts/smoke_test.py
```

### 2. Run a Single Experiment

Train a specific model policy (e.g., `residual`, `deep`, or `asymptotic`) on the training environment:

```bash
python -m regime_hedging.run_experiment --model residual --epochs 200 --batch-size 512 --out outputs/residual_small
```

Compare with alternative configurations:

```bash
# Plain Deep Hedging
python -m regime_hedging.run_experiment --model deep --epochs 200 --batch-size 512 --out outputs/deep_small

# Asymptotic baseline (non-learning)
python -m regime_hedging.run_experiment --model asymptotic --out outputs/asymptotic
```

### 3. Run the Full Comparison Suite

Generate complete comparative benchmarks, tables, and figures across short, medium, and long trading horizons:

```bash
python scripts/run_comparison_suite.py --epochs 150 --batch-size 512 --out outputs/comparison_suite
```

Running the suite will populate the `outputs/comparison_suite/` directory with:
- `comparison_metrics.csv` / `comparison_metrics.json`
- `mean_utility.png` (Average utility comparison)
- `terminal_mse.png` (Terminal target tracking error)
- `mean_cost.png` (Accumulated transaction friction costs)
- `stress_utility.png` (Utility accumulated under simulated stress regimes)

---

## Research Report

The mathematical formulation, methodology, and detailed empirical observations are fully documented in the project paper:

* **Report PDF:** [report/project_report.pdf](file:///c:/Users/dhruv/Desktop/iqf/project/report/project_report.pdf)

---

## Reference Citation

This project is built upon the ST-Hedging framework presented in:

> Shi, X., Xu, D. & Zhang, Z. **Deep learning algorithms for hedging with frictions**. *Digital Finance* **5**, 113–147 (2023).  
> DOI: [10.1007/s42521-023-00075-z](https://doi.org/10.1007/s42521-023-00075-z)
