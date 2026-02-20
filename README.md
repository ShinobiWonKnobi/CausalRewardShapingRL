# Causal Reward Shaping for RL Trading

This repository contains my Minor Project exploring how to make Reinforcement Learning (RL) trading agents more robust to market regime shifts. 

Standard RL agents usually train on raw Profit/Loss (PnL) as their reward signal. The problem is that raw PnL is heavily confounded by the broader market—like the S&P 500's overall baseline drift and sudden shifts in the VIX volatility index. Because of this, agents often accidentally overfit to these "confounders" (e.g., they learn "low VIX = positive returns") rather than learning actual predictive alpha. 

This project solves that by applying **Causal Reward Shaping**. We use a moving Ordinary Least Squares (OLS) regression to calculate the linear relationship between the agent's PnL and our macro confounders, and subtract that effect out. The final, residualized  ("causal") reward forces the PPO agent to optimize strictly for independent skill.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate
The default training script runs an ablation study across **3 different initialization seeds** to ensure statistical robustness and prevent lucky weight initialization. Both a Baseline PPO (raw PnL) and a Causal PPO (adjusted reward) are trained:
```bash
python train.py --timesteps 50000
```
For a rapid test run:
```bash
python train.py --test
```

## 📁 Repository Structure

* `src/config.py`: Centralized configuration hyperparameters (lookbacks, PPO settings).
* `src/data/`: `fetcher.py` pulls SPY and VIX data from Yahoo Finance. `features.py` builds the state representations (z-score normalized MACD, RSI, volatility, etc.).
* `src/env/trading_env.py`: Custom Gymnasium environment for single-asset trading with transaction costs and drawdown penalties.
* `src/reward/calibrator.py`: The core causal engine. It implements the rolling OLS and the `CausalRewardWrapper` that intercepts and adjusts the reward signal before it hits the agent.
* `src/agents/trainer.py`: The Stable-Baselines3 PPO wrapper, managing multi-seed training loops and evaluation.
* `Research_Paper.tex`: Full IEEE-formatted LaTeX academic paper detailing the methodology and exact test metrics.

## 📊 Methodology & Results

**Causal Reward Formula:**
$$ R^{causal}_t = R_t - (\hat{\beta}_1 M_t + \hat{\beta}_2 \Delta V_t) $$
Where $M_t$ is the market return, $\Delta V_t$ is the VIX change, and the betas are recursively estimated over a 60-day window.

**Empirical Results (Average over 3 Seeds over 2023-2024 Test Set):**
- **Baseline PPO:** Often struggled to beat hold-and-wait strategies, succumbing to large drawdowns when market volatility spiked.
- **Causal PPO:** Consistently produced a significantly higher Sharpe ratio (+0.12 avg) and positive absolute returns, demonstrating genuine detachment from the VIX panic curve. (See `results/equity_curves.png` after running the code).

## 📚 Core References
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*.
- Ng, A. Y., et al. (1999). *Policy Invariance Under Reward Transformations*.

---
**License:** MIT License. Feel free to use for academic or research purposes!
