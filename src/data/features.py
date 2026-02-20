"""
Feature Engineering - Technical indicators and transformations
"""
import numpy as np
import pandas as pd
from typing import Optional
import ta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config


class FeatureEngineer:
    """Computes features for the trading environment"""
    
    def __init__(self, feature_config=None):
        self.config = feature_config or config.features
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from aligned market data.
        
        Args:
            df: DataFrame with columns like SPY_Close, SPY_Volume, VIX_Close, etc.
            
        Returns:
            DataFrame with computed features, ready for environment
        """
        features = pd.DataFrame(index=df.index)
        
        # Get price series
        spy_close = df['SPY_Close'].squeeze()
        spy_high = df['SPY_High'].squeeze()
        spy_low = df['SPY_Low'].squeeze()
        spy_volume = df['SPY_Volume'].squeeze()
        vix_close = df['VIX_Close'].squeeze()
        
        # ===================
        # Returns (multiple horizons)
        # ===================
        for window in self.config.return_windows:
            features[f'return_{window}d'] = spy_close.pct_change(window)
        
        # Log returns for stationarity
        features['log_return'] = np.log(spy_close / spy_close.shift(1))
        
        # ===================
        # Volatility
        # ===================
        features['volatility'] = features['log_return'].rolling(
            self.config.volatility_window
        ).std() * np.sqrt(252)  # Annualized
        
        # ===================
        # Technical Indicators
        # ===================
        
        # RSI
        features['rsi'] = ta.momentum.RSIIndicator(
            close=spy_close, 
            window=self.config.rsi_window
        ).rsi()
        
        # MACD
        macd = ta.trend.MACD(
            close=spy_close,
            window_fast=self.config.macd_fast,
            window_slow=self.config.macd_slow,
            window_sign=self.config.macd_signal
        )
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands position (where is price relative to bands)
        bb = ta.volatility.BollingerBands(close=spy_close, window=20)
        features['bb_position'] = (spy_close - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-8
        )
        
        # ===================
        # VIX Features (Confounders)
        # ===================
        features['vix_level'] = vix_close
        features['vix_change'] = vix_close.pct_change()
        features['vix_sma_ratio'] = vix_close / vix_close.rolling(20).mean()
        
        # ===================
        # Market Return (Confounder for calibration)
        # ===================
        features['market_return'] = spy_close.pct_change()
        
        # ===================
        # Price for PnL calculation (not a feature, but needed)
        # ===================
        features['price'] = spy_close
        
        # ===================
        # Normalize features (z-score)
        # ===================
        features = self._normalize(features)
        
        # Drop NaN rows (from rolling calculations)
        features = features.dropna()
        
        return features
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling z-score normalization to features.
        Excludes 'price', 'market_return', 'vix_change' (needed raw for calibration).
        """
        exclude_cols = ['price', 'market_return', 'vix_change']
        
        for col in df.columns:
            if col not in exclude_cols:
                rolling_mean = df[col].rolling(self.config.normalize_window, min_periods=1).mean()
                rolling_std = df[col].rolling(self.config.normalize_window, min_periods=1).std()
                df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
    def get_feature_names(self) -> list:
        """Return list of feature names used as state (excluding non-features)"""
        return [
            'return_1d', 'return_5d', 'return_20d', 'log_return',
            'volatility', 'rsi', 'macd', 'macd_signal', 'macd_diff',
            'bb_position', 'vix_level', 'vix_sma_ratio'
        ]
    
    def get_confounder_names(self) -> list:
        """Return names of confounder variables for reward calibration"""
        return ['market_return', 'vix_change']


if __name__ == "__main__":
    from fetcher import DataFetcher
    
    # Test feature engineering
    fetcher = DataFetcher()
    raw_data = fetcher.get_aligned_data()
    
    engineer = FeatureEngineer()
    features = engineer.compute_features(raw_data)
    
    print(f"Feature shape: {features.shape}")
    print(f"Features: {list(features.columns)}")
    print(f"\nSample data:")
    print(features.tail())
