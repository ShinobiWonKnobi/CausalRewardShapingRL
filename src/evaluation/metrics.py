"""
Evaluation Metrics - Performance analysis and regime breakdown
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


def compute_metrics(
    balances: List[float],
    initial_balance: float = None
) -> Dict[str, float]:
    """
    Compute standard trading metrics from balance history.
    
    Args:
        balances: List of portfolio values over time
        initial_balance: Starting balance
        
    Returns:
        Dict with Sharpe, max drawdown, total return, etc.
    """
    initial_balance = initial_balance or config.env.initial_balance
    balances = np.array(balances)
    
    # Returns
    returns = np.diff(balances) / balances[:-1]
    total_return = (balances[-1] - initial_balance) / initial_balance
    
    # Sharpe (annualized, assuming daily data)
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max Drawdown
    peak = np.maximum.accumulate(balances)
    drawdown = (peak - balances) / peak
    max_drawdown = np.max(drawdown)
    
    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
    else:
        sortino = sharpe
    
    # Win rate
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0
    
    # Profit factor
    gains = returns[returns > 0].sum() if (returns > 0).any() else 0
    losses = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-8
    profit_factor = gains / losses
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_trades': len(returns),
        'final_balance': balances[-1]
    }


def classify_regime(
    returns: pd.Series,
    lookback: int = 60
) -> pd.Series:
    """
    Classify market regime based on rolling returns.
    
    Regimes:
        - 'bull': rolling return > 10% annualized
        - 'bear': rolling return < -10% annualized
        - 'sideways': otherwise
    """
    # Calculate rolling return
    rolling_return = returns.rolling(lookback).sum()
    
    # Annualize (assuming daily data)
    annualized_return = rolling_return * (252 / lookback)
    
    # Classify
    regime = pd.Series('sideways', index=returns.index)
    regime[annualized_return > 0.10] = 'bull'
    regime[annualized_return < -0.10] = 'bear'
    
    return regime


def regime_analysis(
    balances: List[float],
    dates: pd.DatetimeIndex,
    market_returns: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Analyze performance across different market regimes.
    
    Args:
        balances: Portfolio balances over time
        dates: Corresponding dates
        market_returns: Market (SPY) returns for regime classification
        
    Returns:
        Dict mapping regime name to metrics dict
    """
    # Classify regimes
    regime = classify_regime(market_returns)
    
    # Align data
    balances = np.array(balances)
    balance_returns = np.diff(balances) / balances[:-1]
    
    # Handle length mismatch
    min_len = min(len(balance_returns), len(regime) - 1)
    balance_returns = balance_returns[:min_len]
    regime_aligned = regime.iloc[1:min_len+1]
    
    # Compute metrics per regime
    results = {}
    for reg in ['bull', 'bear', 'sideways']:
        mask = (regime_aligned == reg).values
        if mask.sum() < 5:
            continue
            
        reg_returns = balance_returns[mask]
        
        # Sharpe for this regime
        if np.std(reg_returns) > 0:
            sharpe = np.mean(reg_returns) / np.std(reg_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        results[reg] = {
            'sharpe': sharpe,
            'mean_return': np.mean(reg_returns),
            'std_return': np.std(reg_returns),
            'n_days': mask.sum(),
            'total_return': np.prod(1 + reg_returns) - 1
        }
    
    return results


def compute_vix_sensitivity(
    agent_returns: np.ndarray,
    vix_changes: np.ndarray
) -> Dict[str, float]:
    """
    Compute correlation between agent returns and VIX.
    
    Lower correlation after calibration = success.
    """
    # Remove NaN
    valid = np.isfinite(agent_returns) & np.isfinite(vix_changes)
    agent_returns = agent_returns[valid]
    vix_changes = vix_changes[valid]
    
    if len(agent_returns) < 10:
        return {'correlation': 0.0, 'r_squared': 0.0}
    
    correlation = np.corrcoef(agent_returns, vix_changes)[0, 1]
    r_squared = correlation ** 2
    
    return {
        'correlation': correlation,
        'r_squared': r_squared
    }


def compare_agents(
    baseline_results: Dict[str, Any],
    causal_results: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create comparison table between baseline and causal agents.
    """
    metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    data = {
        'Metric': [],
        'Baseline PPO': [],
        'Causal PPO': [],
        'Improvement': []
    }
    
    for metric in metrics:
        baseline_val = baseline_results.get(metric, 0)
        causal_val = causal_results.get(metric, 0)
        
        if metric == 'max_drawdown':
            # Lower is better for drawdown
            improvement = baseline_val - causal_val
        else:
            improvement = causal_val - baseline_val
        
        data['Metric'].append(metric)
        data['Baseline PPO'].append(f"{baseline_val:.4f}")
        data['Causal PPO'].append(f"{causal_val:.4f}")
        data['Improvement'].append(f"{improvement:+.4f}")
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Simulate balance history
    n = 500
    returns = np.random.randn(n) * 0.01 + 0.0003  # Slight positive drift
    balances = [100000]
    for r in returns:
        balances.append(balances[-1] * (1 + r))
    
    # Compute metrics
    metrics = compute_metrics(balances)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test regime analysis
    dates = pd.date_range('2020-01-01', periods=n+1)
    market_returns = pd.Series(np.random.randn(n+1) * 0.01, index=dates)
    
    regime_results = regime_analysis(balances, dates, market_returns)
    print("\nRegime Analysis:")
    for regime, stats in regime_results.items():
        print(f"  {regime}: Sharpe={stats['sharpe']:.2f}, Days={stats['n_days']}")
