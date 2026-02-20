"""
Gradio Demo App - Interactive dashboard for Causal RL Trading
"""
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.fetcher import DataFetcher
from src.data.features import FeatureEngineer
from src.env.trading_env import TradingEnv
from src.reward.calibrator import CausalRewardWrapper
from src.evaluation.metrics import compute_metrics, regime_analysis

# Try to load trained models
try:
    from stable_baselines3 import PPO
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


class DemoApp:
    """Demo application state and logic."""
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.engineer = FeatureEngineer()
        self.data_loaded = False
        self.models_loaded = False
        
        self.baseline_model = None
        self.causal_model = None
        self.test_data = None
        self.feature_cols = None
        
    def load_data(self):
        """Load and process market data."""
        raw_data = self.fetcher.get_aligned_data()
        features = self.engineer.compute_features(raw_data)
        splits = self.fetcher.get_splits(features)
        
        self.test_data = splits['test']
        self.feature_cols = self.engineer.get_feature_names()
        self.data_loaded = True
        
        return f"✓ Data loaded: {len(self.test_data)} test days"
    
    def load_models(self):
        """Load trained models."""
        if not MODELS_AVAILABLE:
            return "⚠ stable-baselines3 not installed"
        
        baseline_path = Path("models/baseline_ppo_final.zip")
        causal_path = Path("models/causal_ppo_final.zip")
        
        msgs = []
        if baseline_path.exists():
            self.baseline_model = PPO.load(str(baseline_path))
            msgs.append("✓ Baseline model loaded")
        else:
            msgs.append(f"⚠ Baseline model not found at {baseline_path}")
            
        if causal_path.exists():
            self.causal_model = PPO.load(str(causal_path))
            msgs.append("✓ Causal model loaded")
        else:
            msgs.append(f"⚠ Causal model not found at {causal_path}")
        
        self.models_loaded = self.baseline_model is not None and self.causal_model is not None
        return "\n".join(msgs)
    
    def run_backtest(self, use_causal: bool = False):
        """Run backtest with selected model."""
        if not self.data_loaded:
            self.load_data()
        
        model = self.causal_model if use_causal else self.baseline_model
        if model is None:
            return None, "Model not loaded"
        
        # Create environment
        env = TradingEnv(self.test_data, self.feature_cols)
        if use_causal:
            env = CausalRewardWrapper(env)
        
        obs, info = env.reset()
        balances = [info['balance']]
        positions = []
        dates = self.test_data.index.tolist()
        
        done = False
        step = 0
        while not done and step < len(dates) - 1:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            balances.append(info['balance'])
            positions.append(info['position'])
            step += 1
        
        return balances, positions, dates[:len(balances)]
    
    def create_comparison_plot(self):
        """Create comparison plot of both agents."""
        if not self.models_loaded:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Models not loaded.\nRun 'python train.py' first.", 
                    ha='center', va='center', fontsize=14)
            return fig
        
        # Run both backtests
        baseline_result = self.run_backtest(use_causal=False)
        causal_result = self.run_backtest(use_causal=True)
        
        if baseline_result[0] is None or causal_result[0] is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "Error running backtest", ha='center', va='center')
            return fig
        
        baseline_balances, baseline_positions, dates = baseline_result
        causal_balances, causal_positions, _ = causal_result
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curves
        ax1 = axes[0]
        ax1.plot(dates[:len(baseline_balances)], baseline_balances, 
                 label='Baseline PPO', linewidth=2, alpha=0.8)
        ax1.plot(dates[:len(causal_balances)], causal_balances, 
                 label='Causal PPO', linewidth=2, alpha=0.8)
        ax1.axhline(y=config.env.initial_balance, color='gray', 
                    linestyle='--', label='Initial', alpha=0.5)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curves: Test Period Performance')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Position comparison
        ax2 = axes[1]
        ax2.fill_between(range(len(baseline_positions)), baseline_positions, 
                         alpha=0.3, label='Baseline')
        ax2.fill_between(range(len(causal_positions)), causal_positions, 
                         alpha=0.3, label='Causal')
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Position')
        ax2.set_title('Position History')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_metrics_table(self):
        """Get metrics comparison as DataFrame."""
        if not self.models_loaded:
            return pd.DataFrame({'Status': ['Models not loaded. Run train.py first.']})
        
        baseline_result = self.run_backtest(use_causal=False)
        causal_result = self.run_backtest(use_causal=True)
        
        if baseline_result[0] is None or causal_result[0] is None:
            return pd.DataFrame({'Status': ['Error running backtest']})
        
        baseline_metrics = compute_metrics(baseline_result[0])
        causal_metrics = compute_metrics(causal_result[0])
        
        data = {
            'Metric': ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Sortino Ratio'],
            'Baseline PPO': [
                f"{baseline_metrics['total_return']*100:.2f}%",
                f"{baseline_metrics['sharpe_ratio']:.2f}",
                f"{baseline_metrics['max_drawdown']*100:.2f}%",
                f"{baseline_metrics['win_rate']*100:.1f}%",
                f"{baseline_metrics['sortino_ratio']:.2f}"
            ],
            'Causal PPO': [
                f"{causal_metrics['total_return']*100:.2f}%",
                f"{causal_metrics['sharpe_ratio']:.2f}",
                f"{causal_metrics['max_drawdown']*100:.2f}%",
                f"{causal_metrics['win_rate']*100:.1f}%",
                f"{causal_metrics['sortino_ratio']:.2f}"
            ]
        }
        
        return pd.DataFrame(data)


def create_demo():
    """Create Gradio demo interface."""
    app = DemoApp()
    
    with gr.Blocks(title="Causal RL Trading Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 Causal Reward Shaping for RL Trading
        
        This demo compares two PPO agents:
        - **Baseline PPO**: Trained with raw profit/loss rewards
        - **Causal PPO**: Trained with confounder-adjusted rewards (VIX and market beta removed)
        
        The hypothesis is that the Causal PPO should be more robust across different market regimes.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Controls")
                load_btn = gr.Button("Load Data & Models", variant="primary")
                status = gr.Textbox(label="Status", lines=3, interactive=False)
                
                gr.Markdown("### 📈 Performance Metrics")
                metrics_table = gr.Dataframe(label="Comparison")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📉 Backtest Results")
                plot = gr.Plot(label="Equity Curves")
        
        def on_load():
            data_msg = app.load_data()
            model_msg = app.load_models()
            status_text = f"{data_msg}\n{model_msg}"
            
            fig = app.create_comparison_plot()
            metrics = app.get_metrics_table()
            
            return status_text, fig, metrics
        
        load_btn.click(
            fn=on_load,
            outputs=[status, plot, metrics_table]
        )
        
        gr.Markdown("""
        ---
        ### How It Works
        
        1. **Reward Calibration**: We use linear regression to estimate how much of the reward is due to:
           - Market return (β × SPY return)
           - VIX change (γ × VIX change)
        
        2. **Causal Reward** = Raw PnL - β × Market Return - γ × VIX Change
        
        3. This residualized reward represents pure "alpha" (trading skill), not market exposure.
        
        4. The agent trained on causal rewards should learn more robust strategies.
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)
