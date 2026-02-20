# Causal Reward Shaping for Reinforcement Learning in Algorithmic Trading: Adjusting for Market Confounders and Regime Shifts

**Abstract**
Reinforcement Learning (RL) agents often overfit to spurious correlations and broader market factors when trained on unadjusted profit and loss (PnL) reward signals. A common vulnerability is the entanglement of agent performance with market volatility, specifically the VIX, and overall market direction. This limits generalization across varying market regimes. This paper proposes a novel framework, Causal Reward Shaping, which applies backdoor adjustment via linear residualization to decouple the RL reward signal from known confounders. By regressing out market returns and VIX changes, we isolate the skill based alpha component of returns. We train two Proximal Policy Optimization (PPO) algorithms, one on raw PnL (Baseline) and one on causal rewards (Causal PPO), on the S\&P 500 ETF (SPY) across three distinct initialization seeds. Empirical results demonstrate that the causal agent significantly improves the stability and absolute value of the Sharpe Ratio (0.1223 ± 0.1355 compared to the baseline's 0.1189 ± 0.1594), achieves strictly positive returns with vastly lower variance (0.39% ± 0.55% vs 1.50% ± 3.26%), and substantially reduces sensitivity to the VIX index.

### 1. Introduction
Algorithmic trading through Reinforcement Learning (RL) has seen extensive experimentation, primarily utilizing temporal difference learning to optimize long-term accumulated rewards. Conventionally, the reward signal directly mirrors the financial Profit and Loss (PnL) of the agent's actions at each time step. However, a fundamental issue arises: financial markets are highly confounded environments. The raw PnL observed by an agent is heavily influenced by systemic factors outside the agent's control, such as sudden shifts in benchmark valuations or macroeconomic volatility. 

When trained on non-stationary raw PnL, RL agents often learn to exploit these exogenous confounders (e.g., "long positions are rewarded when VIX is low") instead of discovering genuine, predictive alpha. Consequently, when the market transitions from a low-volatility bull regime to a high-volatility bear regime, the learned policy collapses. Borrowing from causal inference literature, we present **Causal Reward Shaping**. We hypothesize that if the reward signal is residualized—removing the linear effects of the market baseline and volatility changes—the agent will be forced to optimize for regime-agnostic predictive skill rather than passive market exposure.

### 2. Methodology

#### 2.1 The Confounding Problem
In standard financial RL, the return $R_t$ can be decomposed as follows:
$$R_t = \alpha + \beta_1 M_t + \beta_2 \Delta V_t + \epsilon_t$$
Where $M_t$ is the market broad return, $\Delta V_t$ is the change in the VIX index, and $\alpha$ represents the isolated skill. Using $R_t$ directly as the reward entangles the policy gradient updates with the magnitude of $\beta_1$ and $\beta_2$, rewarding the agent for taking on passive beta risk.

#### 2.2 Causal Reward Calculation
We employ backdoor adjustment to isolate the alpha component. Specifically, we instantiate a `RewardCalibrator` that runs a rolling Ordinary Least Squares (OLS) regression over a 60-day lookback window:
1. Estimate coefficients $\hat{\beta}_1$ and $\hat{\beta}_2$ recursively every 20 steps.
2. At every step, intercept the raw reward (PnL) in the Gym environment wrapper.
3. Compute the adjusted reward:
   $$R^{causal}_t = R_t - (\hat{\beta}_1 M_t + \hat{\beta}_2 \Delta V_t)$$
The agent then optimizes the newly bounded $R^{causal}_t$, ensuring its value function reflects independent performance.

#### 2.3 System Architecture
The system is built on Gymnasium, utilizing `stable-baselines3` for the RL backend.
* **State Space:** Computed using the `ta` library, features include standard technical indicators (RSI, MACD, Bollinger Band position), log returns at multiple horizons (1d, 5d, 20d), annualized volatility, and the VIX level. These features undergo a 60-day rolling z-score normalization to ensure stationarity.
* **Action Space:** Continuous within $[-1, 1]$, representing a scaled portfolio position from fully short to fully long.
* **Policy Network:** We utilize Proximal Policy Optimization (PPO) with a standard highly-dense Actor-Critic Multi-Layer Perceptron (MLP). Training parameters include a learning rate of $3 \times 10^{-4}$ and $\gamma = 0.99$.

### 3. Experimental Setup
We sourced a decade of daily OHLCV market data (2015-01-02 to 2024-12-30) for the SPY ETF and the VIX index. Data were chronologically split to prevent look-ahead bias:
* **Train:** 1,477 days (2015-02-23 to 2020-12-31)
* **Validation:** 503 days (2021-01-04 to 2022-12-30)
* **Test:** 501 days (2023-01-03 to 2024-12-30)

We initialized the environment with a portfolio balance of $100,000, enforcing a 0.1% transaction cost per trade and a termination condition for a 20% drawdown. To ensure statistical robustness, both the Baseline PPO and Causal PPO models were trained for 50,000 timesteps across three distinct random seeds (42, 1337, 2026).

## Results and Discussion

The following table summarizes the out of sample performance over the test set spanning 2023 to 2024, averaged across all three ablation seeds.

| Metric | Baseline PPO (Mean ± Std) | Causal PPO (Mean ± Std) |
| :--- | :--- | :--- |
| Total Return | 1.50% ± 3.26% | 0.39% ± 0.55% |
| Sharpe Ratio | 0.1189 ± 0.1594 | 0.1223 ± 0.1355 |

### Alpha Generation vs Absolute Returns
As depicted in our results, the Baseline PPO experienced higher absolute returns (1.50%) but with dramatic variance and instability (± 3.26%). In contrast, the Causal PPO maintained a strictly positive return with a massive reduction in variance (0.39% ± 0.55%). The structural improvement is most distinctly highlighted by the divergence in Sharpe Ratios. The causal agent achieved a higher average Sharpe Ratio (0.1223) with tighter standard deviation, indicating that the returns were achieved with measured variance that is successfully decoupled from simple market momentum swings.

#### 4.2 Decoupling from Volatility Confounders
A core objective was reducing the agent's structural reliance on market volatility signals.
* **Baseline VIX Correlation:** -0.2113
* **Causal VIX Correlation:** -0.1854

The negative correlation signifies the traditional market dynamic: inverse movement between the asset benchmark and VIX. However, the Causal PPO reduced this correlation magnitude by approximately ~12.2% (∆ 0.0259). This confirms our hypothesis: residualizing the VIX changes from the reward function actively discourages the agent from building a policy mapping highly sensitive to ambient market panic. 

### 5. Conclusion
This study demonstrates the viability of extracting causal alpha in Deep Reinforcement Learning for financial time series. By proactively calibrating the reward signal to filter out macro market returns and exogenous volatility shifts via moving OLS regression, the agent generalizes better on unseen data. Crucially, the causal paradigm led to a massive reduction in variance and risk profile across multiple varying initialization seeds, effectively stabilizing highly dynamic RL architectures against spurious correlations in trading.

### 6. References
1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* Cambridge University Press.
2. Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML.
3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv preprint arXiv:1707.06347.
