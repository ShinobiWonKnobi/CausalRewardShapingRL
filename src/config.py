"""
Causal Reward Shaping for RL Trading - Configuration
"""
from dataclasses import dataclass, field
from typing import List
from datetime import date


@dataclass
class DataConfig:
    """Data fetching and processing configuration"""
    symbols: List[str] = field(default_factory=lambda: ["SPY", "^VIX"])
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    
    # Train/Val/Test splits (by date)
    train_end: str = "2020-12-31"
    val_end: str = "2022-12-31"
    # Test: 2023-01-01 to end_date
    
    cache_dir: str = "data/cache"


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Return horizons
    return_windows: List[int] = field(default_factory=lambda: [1, 5, 20])
    
    # Volatility
    volatility_window: int = 20
    
    # Technical indicators
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Normalization lookback for z-score
    normalize_window: int = 60


@dataclass
class EnvConfig:
    """Trading environment configuration"""
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    max_drawdown: float = 0.20  # 20% max drawdown terminates episode
    
    # Action space: position in [-1, 1]
    # -1 = full short, 0 = flat, 1 = full long


@dataclass
class CalibratorConfig:
    """Reward calibration configuration"""
    lookback: int = 60  # Days to estimate confounder effects
    confounders: List[str] = field(default_factory=lambda: ["market_return", "vix_change"])
    update_frequency: int = 20  # Re-estimate betas every N steps


@dataclass
class PPOConfig:
    """PPO training configuration"""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Training
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    
    # Logging
    log_dir: str = "logs"
    model_dir: str = "models"


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    calibrator: CalibratorConfig = field(default_factory=CalibratorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    
    seed: int = 42


# Global config instance
config = Config()
