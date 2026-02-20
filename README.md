# Causal Reward Shaping for RL Trading

A reinforcement learning project that uses confounder-adjusted rewards to train more robust trading agents.

## 🎯 Hypothesis

Standard RL trading agents learn from raw profit/loss (PnL), which is **confounded** by market-wide factors like VIX and overall market direction. By removing these confounders from the reward signal, we can train agents that learn genuine "alpha" (trading skill) rather than market exposure.

## 📁 Project Structure

```
├── src/
│   ├── config.py           # All hyperparameters
│   ├── data/
│   │   ├── fetcher.py      # Download SPY, VIX data
│   │   └── features.py     # Technical indicators (RSI, MACD, etc.)
│   ├── env/
│   │   └── trading_env.py  # Gymnasium trading environment
│   ├── reward/
│   │   └── calibrator.py   # Reward calibration (remove confounders)
│   ├── agents/
│   │   └── trainer.py      # PPO training logic
│   └── evaluation/
│       └── metrics.py      # Sharpe, drawdown, regime analysis
├── train.py                # Main training script
├── app.py                  # Gradio demo
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Full training (takes ~30 min on GPU)
python train.py --timesteps 500000

# Quick test run
python train.py --test
```

### 3. Launch Demo

```bash
python app.py
```

## 📊 Methodology

### The Problem

When training RL agents for trading, the reward (PnL) is confounded:

```
PnL = α + β₁ × Market_Return + β₂ × VIX_Change + ε
         ↑ Confounder effects   ↑ True signal
```

The agent learns spurious correlations (e.g., "low VIX = profit") instead of actual trading skill.

### Our Solution

We **residualize** the reward:

```python
causal_reward = raw_pnl - β̂₁ × market_return - β̂₂ × vix_change
```

This removes confounders, leaving only the "alpha" component.

### Implementation

1. **RewardCalibrator**: Fits OLS regression to estimate β₁, β₂
2. **CausalRewardWrapper**: Gymnasium wrapper that transforms rewards
3. **Two PPO agents**: Baseline (raw rewards) vs Causal (calibrated rewards)

## 📈 Expected Results

| Metric | Baseline PPO | Causal PPO | Improvement |
|--------|-------------|------------|-------------|
| VIX Correlation | High | Low | ✓ |
| Regime Robustness | Variable | Stable | ✓ |
| Sharpe Ratio | Similar | Similar | ~ |

The key win is **robustness**, not necessarily higher returns.

## 🧪 Experiments

1. **E1: Baseline Comparison** - Compare metrics on test set
2. **E2: Regime Robustness** - Performance in bull/bear/sideways markets
3. **E3: VIX Sensitivity** - Correlation between returns and VIX
4. **E4: Ablation** - Remove market vs VIX adjustment separately

## 📚 References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Ng, A. et al. (1999). *Policy invariance under reward transformations*
- Schulman, J. et al. (2017). *Proximal Policy Optimization*

## 📝 License

MIT License - For academic/research use.
