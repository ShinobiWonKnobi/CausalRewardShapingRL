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
    # 3. Train Baseline PPO
    # =============================
    print("\n[3/5] Training Baseline PPO...")
    baseline_trainer = PPOTrainer(
        features_df=splits['train'],
        feature_columns=feature_cols,
        use_causal_rewards=False,
        log_name="baseline_ppo"
    )
    
    baseline_trainer.train(
        train_df=splits['train'],
        val_df=splits['val'],
        total_timesteps=args.timesteps
    )
    
    # =============================
    # 4. Train Causal PPO
    # =============================
    print("\n[4/5] Training Causal PPO...")
    causal_trainer = PPOTrainer(
        features_df=splits['train'],
        feature_columns=feature_cols,
        use_causal_rewards=True,
        log_name="causal_ppo"
    )
    
    causal_trainer.train(
        train_df=splits['train'],
        val_df=splits['val'],
        total_timesteps=args.timesteps
    )
    
    # =============================
    # 5. Evaluate on Test Set
    # =============================
    print("\n[5/5] Evaluating on test set...")
    
    baseline_results = baseline_trainer.evaluate(splits['test'])
    causal_results = causal_trainer.evaluate(splits['test'])
    
    # Compute full metrics
    baseline_metrics = compute_metrics(baseline_results['balances'][0])
    causal_metrics = compute_metrics(causal_results['balances'][0])
    
    # VIX sensitivity
    test_vix_changes = splits['test']['vix_change'].values[1:]
    
    baseline_returns = np.diff(baseline_results['balances'][0]) / np.array(baseline_results['balances'][0][:-1])
    causal_returns = np.diff(causal_results['balances'][0]) / np.array(causal_results['balances'][0][:-1])
    
    min_len = min(len(baseline_returns), len(test_vix_changes))
    baseline_vix_sens = compute_vix_sensitivity(baseline_returns[:min_len], test_vix_changes[:min_len])
    causal_vix_sens = compute_vix_sensitivity(causal_returns[:min_len], test_vix_changes[:min_len])
    
    # =============================
    # Results Summary
    # =============================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    comparison = compare_agents(baseline_metrics, causal_metrics)
    print("\n" + comparison.to_string(index=False))
    
    print("\n--- VIX Sensitivity ---")
    print(f"Baseline: Correlation = {baseline_vix_sens['correlation']:.4f}")
    print(f"Causal:   Correlation = {causal_vix_sens['correlation']:.4f}")
    print(f"Reduction: {abs(baseline_vix_sens['correlation']) - abs(causal_vix_sens['correlation']):.4f}")
    
    # =============================
    # Save results
    # =============================
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save metrics
    all_results = {
        'baseline': baseline_metrics,
        'causal': causal_metrics,
        'baseline_vix_sensitivity': baseline_vix_sens,
        'causal_vix_sensitivity': causal_vix_sens
    }
    
    results_df = pd.DataFrame({
        'metric': list(baseline_metrics.keys()) + ['vix_correlation'],
        'baseline': list(baseline_metrics.values()) + [baseline_vix_sens['correlation']],
        'causal': list(causal_metrics.values()) + [causal_vix_sens['correlation']]
    })
    results_df.to_csv(results_dir / "metrics.csv", index=False)
    print(f"\nResults saved to: {results_dir / 'metrics.csv'}")
    
    # Plot equity curves
    if not args.no_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curves
        ax1 = axes[0]
        ax1.plot(baseline_results['balances'][0], label='Baseline PPO', alpha=0.8)
        ax1.plot(causal_results['balances'][0], label='Causal PPO', alpha=0.8)
        ax1.axhline(y=config.env.initial_balance, color='gray', linestyle='--', label='Initial')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curves: Baseline vs Causal PPO')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Positions
        ax2 = axes[1]
        ax2.plot(baseline_results['positions'][0], label='Baseline', alpha=0.6)
        ax2.plot(causal_results['positions'][0], label='Causal', alpha=0.6)
        ax2.set_xlabel('Trading Days')
        ax2.set_ylabel('Position')
        ax2.set_title('Position History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "equity_curves.png", dpi=150)
        print(f"Plot saved to: {results_dir / 'equity_curves.png'}")
        
        if args.show_plot:
            plt.show()
    
    print("\n[DONE] Training complete!")
    return all_results


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
