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
    
    Instead of regressing the agent's non-stationary PnL, we regress the underlying asset's return:
        asset_return = α + β × vix_change + ε
    
    The causal reward is then the agent's position multiplied by the residual (ε):
        causal_reward = position × (asset_return - β × vix_change)
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
        self.asset_returns_buffer = deque(maxlen=self.lookback)
        self.vix_changes_buffer = deque(maxlen=self.lookback)
        
        # Estimated coefficients
        self.beta_vix = 0.0
        self.alpha = 0.0
        
        # Step counter for update frequency
        self.step_count = 0
        self.fitted = False
        
    def add_observation(
        self, 
        asset_return: float, 
        vix_change: float
    ):
        """Add new observation to buffers."""
        self.asset_returns_buffer.append(asset_return)
        self.vix_changes_buffer.append(vix_change)
        
        self.step_count += 1
        
        # Check if we should re-fit
        if (self.step_count % self.update_frequency == 0 and 
            len(self.asset_returns_buffer) >= 20):  # Minimum samples
            self._fit()
    
    def _fit(self):
        """Fit linear regression to estimate VIX confounder effect on the asset."""
        asset_returns = np.array(self.asset_returns_buffer)
        vix_changes = np.array(self.vix_changes_buffer)
        
        # Handle NaN/Inf
        valid_mask = (
            np.isfinite(asset_returns) & 
            np.isfinite(vix_changes)
        )
        
        if valid_mask.sum() < 10:
            return  # Not enough valid data
        
        asset_returns = asset_returns[valid_mask]
        vix_changes = vix_changes[valid_mask]
        
        # Build design matrix [1, vix_change]
        X = np.column_stack([
            np.ones(len(asset_returns)),
            vix_changes
        ])
        
        # OLS: β = (X'X)^(-1) X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ asset_returns
            
            # Simple Ridge for stability
            betas = np.linalg.solve(XtX + 1e-6 * np.eye(2), Xty)
            
            self.alpha = betas[0]
            self.beta_vix = betas[1]
            self.fitted = True
            
        except np.linalg.LinAlgError:
            # If matrix is singular, keep previous values
            pass
    
    def calibrate(
        self, 
        position: float, 
        asset_return: float, 
        vix_change: float,
        trade_cost: float
    ) -> float:
        """
        Return calibrated (residualized) reward based on agent's position.
        
        causal_reward = position * (asset_return - β_vix × vix_change) - trade_cost
        """
        if not self.fitted:
            # Before fitting, return raw PnL
            return (position * asset_return) - trade_cost
        
        # Handle NaN in inputs
        if not np.isfinite(asset_return):
            asset_return = 0.0
        if not np.isfinite(vix_change):
            vix_change = 0.0
        
        # Calculate isolated alpha (asset movement independent of VIX)
        isolated_asset_return = asset_return - (self.beta_vix * vix_change)
        
        # The agent's causal reward is capturing this isolated alpha, minus costs
        causal_reward = (position * isolated_asset_return) - trade_cost
        
        return causal_reward
    
    def get_coefficients(self) -> dict:
        """Return current estimated coefficients."""
        return {
            'alpha': self.alpha,
            'beta_vix': self.beta_vix,
            'fitted': self.fitted,
            'n_samples': len(self.asset_returns_buffer)
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
        
        # Extract necessary components to rebuild the True Asset Return logic
        market_return = info.get('market_return', 0.0)
        vix_change = info.get('vix_change', 0.0)
        
        # Recover the agent's previous state
        position = self.env.unwrapped.position
        trade_cost = info.get('trade_cost', 0.0)  # We calculate this manually since the wrapper doesn't expose it perfectly
        
        # We need the previous position to figure out what the true trade cost was for this exact step
        # Since env.step already updated the position, we deduce the actual cost by looking at the raw reward difference
        if position != 0:
            asset_return = (reward + 0.001) / position if abs(position) > 0.01 else market_return # rough approximation of cost and return
        else:
            asset_return = market_return # If agent was flat, asset return is just market return

        # Fallback to the accurate market_return from the env for the asset (since SPY == market)
        actual_asset_return = market_return 
        
        # We also need the trade cost. Reward = position * market_return - trade_cost
        inferred_trade_cost = (position * actual_asset_return) - reward
        if inferred_trade_cost < 0: inferred_trade_cost = 0 # Sanity check

        # Add to calibrator's history (we only track the underlying asset against the confounder now)
        self.calibrator.add_observation(actual_asset_return, vix_change)
        
        # Get calibrated reward using the agent's position
        causal_reward = self.calibrator.calibrate(position, actual_asset_return, vix_change, inferred_trade_cost)
        
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
