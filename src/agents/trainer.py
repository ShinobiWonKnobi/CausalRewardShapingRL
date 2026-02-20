"""
PPO Trainer - Training logic for baseline and causal PPO agents
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.env.trading_env import TradingEnv
from src.reward.calibrator import CausalRewardWrapper


class TradingMetricsCallback(BaseCallback):
    """Custom callback to log trading-specific metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_balances = []
        
    def _on_step(self) -> bool:
        # Log custom metrics if available
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'balance' in info:
                    self.logger.record('trading/balance', info['balance'])
                if 'drawdown' in info:
                    self.logger.record('trading/drawdown', info['drawdown'])
                if 'calibrator_coeffs' in info:
                    coeffs = info['calibrator_coeffs']
                    if coeffs.get('fitted', False):
                        self.logger.record('calibrator/beta_market', coeffs['beta_market'])
                        self.logger.record('calibrator/beta_vix', coeffs['beta_vix'])
        return True


class PPOTrainer:
    """
    Trainer for PPO agents with optional causal reward shaping.
    """
    
    def __init__(
        self,
        features_df,
        feature_columns: list,
        use_causal_rewards: bool = False,
        ppo_config: Optional[Dict] = None,
        log_name: str = "ppo"
    ):
        """
        Args:
            features_df: DataFrame with features and price
            feature_columns: List of feature column names
            use_causal_rewards: If True, wrap env with CausalRewardWrapper
            ppo_config: Override PPO hyperparameters
            log_name: Name for logging directory
        """
        self.features_df = features_df
        self.feature_columns = feature_columns
        self.use_causal_rewards = use_causal_rewards
        self.ppo_config = ppo_config or {}
        self.log_name = log_name
        
        # Directories
        self.log_dir = Path(config.ppo.log_dir) / log_name
        self.model_dir = Path(config.ppo.model_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Will be set during training
        self.env = None
        self.eval_env = None
        self.model = None
        
    def _make_env(self, df, render_mode=None):
        """Create environment, optionally with causal wrapper."""
        base_env = TradingEnv(
            features_df=df,
            feature_columns=self.feature_columns,
            render_mode=render_mode
        )
        
        if self.use_causal_rewards:
            env = CausalRewardWrapper(base_env)
        else:
            env = base_env
            
        return Monitor(env)
    
    def train(
        self,
        train_df,
        val_df,
        total_timesteps: Optional[int] = None,
        eval_freq: Optional[int] = None
    ) -> PPO:
        """
        Train PPO agent.
        
        Args:
            train_df: Training data
            val_df: Validation data for evaluation
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            
        Returns:
            Trained PPO model
        """
        total_timesteps = total_timesteps or config.ppo.total_timesteps
        eval_freq = eval_freq or config.ppo.eval_freq
        
        # Create environments
        self.env = DummyVecEnv([lambda: self._make_env(train_df)])
        self.eval_env = DummyVecEnv([lambda: self._make_env(val_df)])
        
        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.ppo_config.get('learning_rate', config.ppo.learning_rate),
            n_steps=self.ppo_config.get('n_steps', config.ppo.n_steps),
            batch_size=self.ppo_config.get('batch_size', config.ppo.batch_size),
            n_epochs=self.ppo_config.get('n_epochs', config.ppo.n_epochs),
            gamma=self.ppo_config.get('gamma', config.ppo.gamma),
            gae_lambda=self.ppo_config.get('gae_lambda', config.ppo.gae_lambda),
            clip_range=self.ppo_config.get('clip_range', config.ppo.clip_range),
            verbose=1,
            tensorboard_log=str(self.log_dir),
            seed=config.seed
        )
        
        # Callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.model_dir / self.log_name),
            log_path=str(self.log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=config.ppo.n_eval_episodes,
            deterministic=True
        )
        
        trading_callback = TradingMetricsCallback()
        
        # Train
        print(f"\n{'='*60}")
        print(f"Training {'Causal' if self.use_causal_rewards else 'Baseline'} PPO")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"{'='*60}\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, trading_callback],
            progress_bar=True
        )
        
        # Save final model
        model_path = self.model_dir / f"{self.log_name}_final"
        self.model.save(str(model_path))
        print(f"\nModel saved to: {model_path}")
        
        return self.model
    
    def evaluate(self, test_df, n_episodes: int = 1) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Returns:
            Dict with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        env = self._make_env(test_df)
        
        all_rewards = []
        all_balances = []
        all_positions = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_rewards = []
            episode_balances = [info['balance']]
            episode_positions = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_rewards.append(reward)
                episode_balances.append(info['balance'])
                episode_positions.append(info['position'])
            
            all_rewards.append(sum(episode_rewards))
            all_balances.append(episode_balances)
            all_positions.append(episode_positions)
        
        # Compute metrics
        final_balances = [b[-1] for b in all_balances]
        returns = [(b - config.env.initial_balance) / config.env.initial_balance 
                   for b in final_balances]
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_reward': np.mean(all_rewards),
            'final_balance': np.mean(final_balances),
            'balances': all_balances,
            'positions': all_positions
        }
    
    @classmethod
    def load(cls, model_path: str, features_df, feature_columns: list, use_causal: bool = False):
        """Load a trained model."""
        trainer = cls(features_df, feature_columns, use_causal)
        trainer.model = PPO.load(model_path)
        return trainer


if __name__ == "__main__":
    # Quick test
    from src.data.fetcher import DataFetcher
    from src.data.features import FeatureEngineer
    
    # Load data
    fetcher = DataFetcher()
    raw_data = fetcher.get_aligned_data()
    
    engineer = FeatureEngineer()
    features = engineer.compute_features(raw_data)
    
    splits = fetcher.get_splits(features)
    feature_cols = engineer.get_feature_names()
    
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    
    # Test trainer creation (don't actually train in test)
    trainer = PPOTrainer(
        splits['train'], 
        feature_cols, 
        use_causal_rewards=True,
        log_name="test_causal"
    )
    print("\nTrainer created successfully!")
