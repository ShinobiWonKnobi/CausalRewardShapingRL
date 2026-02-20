"""
Main Training Script - Train and compare baseline vs causal PPO
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.data.fetcher import DataFetcher
from src.data.features import FeatureEngineer
from src.agents.trainer import PPOTrainer
from src.evaluation.metrics import compute_metrics, regime_analysis, compute_vix_sensitivity, compare_agents


def main(args):
    """Main training pipeline."""
    
    print("=" * 60)
    print("Causal Reward Shaping for RL Trading")
    print("=" * 60)
    
    # =============================
    # 1. Load and prepare data
    # =============================
    print("\n[1/5] Loading data...")
    fetcher = DataFetcher()
    raw_data = fetcher.get_aligned_data(use_cache=True)
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
    
    # =============================
    # 2. Feature engineering
    # =============================
    print("\n[2/5] Computing features...")
    engineer = FeatureEngineer()
    features = engineer.compute_features(raw_data)
    print(f"  Features shape: {features.shape}")
    print(f"  Feature columns: {engineer.get_feature_names()}")
    
    # Split data
    splits = fetcher.get_splits(features)
    feature_cols = engineer.get_feature_names()
    
    print(f"\n  Train: {len(splits['train'])} days ({splits['train'].index[0].date()} to {splits['train'].index[-1].date()})")
    print(f"  Val:   {len(splits['val'])} days ({splits['val'].index[0].date()} to {splits['val'].index[-1].date()})")
    print(f"  Test:  {len(splits['test'])} days ({splits['test'].index[0].date()} to {splits['test'].index[-1].date()})")
    
    # =============================
    # Evaluated Seeds Logic
    # =============================
    print("\n[3/5 & 4/5] Training Baseline & Causal across 3 Seeds (Ablation)...")
    seeds = [42, 1337, 2026]
    baseline_returns_list = []
    causal_returns_list = []
    baseline_sharpes = []
    causal_sharpes = []
    
    b_balances_all = []
    c_balances_all = []
    b_positions_all = []
    c_positions_all = []
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        config.seed = seed
        
        baseline_trainer = PPOTrainer(features_df=splits['train'], feature_columns=feature_cols, use_causal_rewards=False, log_name=f"baseline_ppo_{seed}")
        baseline_trainer.train(train_df=splits['train'], val_df=splits['val'], total_timesteps=args.timesteps)
        
        causal_trainer = PPOTrainer(features_df=splits['train'], feature_columns=feature_cols, use_causal_rewards=True, log_name=f"causal_ppo_{seed}")
        causal_trainer.train(train_df=splits['train'], val_df=splits['val'], total_timesteps=args.timesteps)
        
        b_res = baseline_trainer.evaluate(splits['test'])
        c_res = causal_trainer.evaluate(splits['test'])
        
        b_met = compute_metrics(b_res['balances'][0])
        c_met = compute_metrics(c_res['balances'][0])
        
        baseline_returns_list.append(b_met['total_return'])
        causal_returns_list.append(c_met['total_return'])
        baseline_sharpes.append(b_met['sharpe_ratio'])
        causal_sharpes.append(c_met['sharpe_ratio'])
        
        b_balances_all.append(b_res['balances'][0])
        c_balances_all.append(c_res['balances'][0])
        b_positions_all.append(b_res['positions'][0])
        c_positions_all.append(c_res['positions'][0])

    # Get final VIX sensitivity from the last run's metrics for simplicity
    test_vix_changes = splits['test']['vix_change'].values[1:]
    
    baseline_returns_arr = np.diff(b_res['balances'][0]) / np.array(b_res['balances'][0][:-1])
    causal_returns_arr = np.diff(c_res['balances'][0]) / np.array(c_res['balances'][0][:-1])
    
    min_len = min(len(baseline_returns_arr), len(test_vix_changes))
    baseline_vix_sens = compute_vix_sensitivity(baseline_returns_arr[:min_len], test_vix_changes[:min_len])
    causal_vix_sens = compute_vix_sensitivity(causal_returns_arr[:min_len], test_vix_changes[:min_len])
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Averaged across 3 Seeds)")
    print("=" * 60)
    
    b_ret_mean, b_ret_std = np.mean(baseline_returns_list), np.std(baseline_returns_list)
    c_ret_mean, c_ret_std = np.mean(causal_returns_list), np.std(causal_returns_list)
    
    b_sharpe_mean, b_sharpe_std = np.mean(baseline_sharpes), np.std(baseline_sharpes)
    c_sharpe_mean, c_sharpe_std = np.mean(causal_sharpes), np.std(causal_sharpes)
    
    print(f"\nBaseline Return: {b_ret_mean*100:.2f}% ± {b_ret_std*100:.2f}%")
    print(f"Causal Return:   {c_ret_mean*100:.2f}% ± {c_ret_std*100:.2f}%")
    print(f"\nBaseline Sharpe: {b_sharpe_mean:.4f} ± {b_sharpe_std:.4f}")
    print(f"Causal Sharpe:   {c_sharpe_mean:.4f} ± {c_sharpe_std:.4f}")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Plot equity curves
    if not args.no_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curves
        ax1 = axes[0]
        # Plot individual seeds lightly
        for i in range(len(seeds)):
            ax1.plot(b_balances_all[i], color='tab:blue', alpha=0.2)
            ax1.plot(c_balances_all[i], color='tab:orange', alpha=0.2)
            
        # Plot mean
        ax1.plot(np.mean(b_balances_all, axis=0), label='Baseline PPO (Mean)', color='tab:blue', linewidth=2)
        ax1.plot(np.mean(c_balances_all, axis=0), label='Causal PPO (Mean)', color='tab:orange', linewidth=2)

        ax1.axhline(y=config.env.initial_balance, color='gray', linestyle='--', label='Initial')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curves: Baseline vs Causal PPO (3 Seeds)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Positions
        ax2 = axes[1]
        for i in range(len(seeds)):
            ax2.plot(b_positions_all[i], color='tab:blue', alpha=0.2)
            ax2.plot(c_positions_all[i], color='tab:orange', alpha=0.2)
            
        ax2.plot(np.mean(b_positions_all, axis=0), label='Baseline (Mean)', color='tab:blue', linewidth=2)
        ax2.plot(np.mean(c_positions_all, axis=0), label='Causal (Mean)', color='tab:orange', linewidth=2)
        
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Position')
        ax2.set_title('Position History (3 Seeds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "equity_curves.png", dpi=150)
        print(f"Plot saved to: {results_dir / 'equity_curves.png'}")
        
        if args.show_plot:
            plt.show()
    
    print("\n[DONE] Training complete!")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal RL Trading Agent")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100,000)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--show-plot", action="store_true",
                        help="Show plot interactively")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run with minimal timesteps")
    
    args = parser.parse_args()
    
    if args.test:
        args.timesteps = 1_000
        print("Running in TEST mode with 1,000 timesteps")
    
    main(args)
