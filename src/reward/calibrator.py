"""
Reward Calibrator - Confounder adjustment for causal rewards
"""
import numpy as np
from typing import Optional, List, Tuple
import gymnasium as gym
from collections import deque

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class RewardCalibrator:
    """
    Calibrates rewards by removing confounder effects.
    
    Uses linear regression to estimate:
        PnL = α + β₁ × market_return + β₂ × vix_change + ε
    
    Causal reward = ε (the residual after removing confounder effects)
    """
    
    def __init__(
        self,
        lookback: Optional[int] = None,
        update_frequency: Optional[int] = None
    ):
        """
        Args:
            lookback: Number of steps to use for estimating betas
            update_frequency: How often to re-estimate betas
        """
        self.lookback = lookback or config.calibrator.lookback
        self.update_frequency = update_frequency or config.calibrator.update_frequency
        
        # Buffers for history
        self.rewards_buffer = deque(maxlen=self.lookback)
        self.market_returns_buffer = deque(maxlen=self.lookback)
        self.vix_changes_buffer = deque(maxlen=self.lookback)
        
        # Estimated coefficients
        self.beta_market = 0.0
        self.beta_vix = 0.0
        self.alpha = 0.0
        
        # Step counter for update frequency
        self.step_count = 0
        self.fitted = False
        
    def add_observation(
        self, 
        reward: float, 
        market_return: float, 
        vix_change: float
    ):
        """Add new observation to buffers."""
        self.rewards_buffer.append(reward)
        self.market_returns_buffer.append(market_return)
        self.vix_changes_buffer.append(vix_change)
        
        self.step_count += 1
        
        # Check if we should re-fit
        if (self.step_count % self.update_frequency == 0 and 
            len(self.rewards_buffer) >= 20):  # Minimum samples
            self._fit()
    
    def _fit(self):
        """Fit linear regression to estimate confounder effects."""
        rewards = np.array(self.rewards_buffer)
        market_returns = np.array(self.market_returns_buffer)
        vix_changes = np.array(self.vix_changes_buffer)
        
        # Handle NaN/Inf
        valid_mask = (
            np.isfinite(rewards) & 
            np.isfinite(market_returns) & 
            np.isfinite(vix_changes)
        )
        
        if valid_mask.sum() < 10:
            return  # Not enough valid data
        
        rewards = rewards[valid_mask]
        market_returns = market_returns[valid_mask]
        vix_changes = vix_changes[valid_mask]
        
        # Build design matrix [1, market_return, vix_change]
        X = np.column_stack([
            np.ones(len(rewards)),
            market_returns,
            vix_changes
        ])
        
        # OLS: β = (X'X)^(-1) X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ rewards
            betas = np.linalg.solve(XtX + 1e-6 * np.eye(3), Xty)  # Ridge for stability
            
            self.alpha = betas[0]
            self.beta_market = betas[1]
            self.beta_vix = betas[2]
            self.fitted = True
            
        except np.linalg.LinAlgError:
            # If matrix is singular, keep previous values
            pass
    
    def calibrate(
        self, 
        reward: float, 
        market_return: float, 
        vix_change: float
    ) -> float:
        """
        Return calibrated (residualized) reward.
        
        causal_reward = reward - β_market × market_return - β_vix × vix_change
        """
        if not self.fitted:
            # Before fitting, return raw reward
            return reward
        
        # Handle NaN in inputs
        if not np.isfinite(market_return):
            market_return = 0.0
        if not np.isfinite(vix_change):
            vix_change = 0.0
        
        adjustment = (
            self.beta_market * market_return + 
            self.beta_vix * vix_change
        )
        
        causal_reward = reward - adjustment
        
        return causal_reward
    
    def get_coefficients(self) -> dict:
        """Return current estimated coefficients."""
        return {
            'alpha': self.alpha,
            'beta_market': self.beta_market,
            'beta_vix': self.beta_vix,
            'fitted': self.fitted,
            'n_samples': len(self.rewards_buffer)
        }


class CausalRewardWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that transforms raw rewards to causal rewards.
    
    Removes the effect of market return and VIX change from the reward signal.
    """
    
    def __init__(
        self, 
        env: gym.Env,
        lookback: Optional[int] = None,
        update_frequency: Optional[int] = None
    ):
        super().__init__(env)
        self.calibrator = RewardCalibrator(lookback, update_frequency)
        
    def step(self, action):
        """Execute step and calibrate reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get confounders from info (set by TradingEnv)
        market_return = info.get('market_return', 0.0)
        vix_change = info.get('vix_change', 0.0)
        
        # Add to calibrator's history
        self.calibrator.add_observation(reward, market_return, vix_change)
        
        # Get calibrated reward
        causal_reward = self.calibrator.calibrate(reward, market_return, vix_change)
        
        # Store both rewards in info
        info['raw_reward'] = reward
        info['causal_reward'] = causal_reward
        info['calibrator_coeffs'] = self.calibrator.get_coefficients()
        
        return obs, causal_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and optionally clear calibrator."""
        # Note: We don't clear the calibrator on reset to maintain learned betas
        return self.env.reset(**kwargs)


if __name__ == "__main__":
    # Test the calibrator
    import matplotlib.pyplot as plt
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    
    # True confounder effects
    true_beta_market = 0.8
    true_beta_vix = -0.3
    
    # Simulate data
    market_returns = np.random.randn(n) * 0.02  # 2% daily vol
    vix_changes = np.random.randn(n) * 0.05  # 5% daily vol
    
    # True alpha (skill)
    alpha = np.random.randn(n) * 0.005  # Small skill signal
    
    # Observed rewards
    raw_rewards = (
        alpha + 
        true_beta_market * market_returns + 
        true_beta_vix * vix_changes
    )
    
    # Test calibrator
    calibrator = RewardCalibrator(lookback=100, update_frequency=10)
    
    causal_rewards = []
    for i in range(n):
        calibrator.add_observation(raw_rewards[i], market_returns[i], vix_changes[i])
        cr = calibrator.calibrate(raw_rewards[i], market_returns[i], vix_changes[i])
        causal_rewards.append(cr)
    
    print("Estimated coefficients:")
    print(calibrator.get_coefficients())
    print(f"\nTrue beta_market: {true_beta_market}, Estimated: {calibrator.beta_market:.4f}")
    print(f"True beta_vix: {true_beta_vix}, Estimated: {calibrator.beta_vix:.4f}")
    
    # Compare correlation with confounders
    print(f"\nCorrelation of raw rewards with market: {np.corrcoef(raw_rewards, market_returns)[0,1]:.4f}")
    print(f"Correlation of causal rewards with market: {np.corrcoef(causal_rewards[100:], market_returns[100:])[0,1]:.4f}")
