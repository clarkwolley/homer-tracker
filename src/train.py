"""
Model training pipeline — retrains both models from collected data.

Usage:
    python -m src.train              # Train both models
    python -m src.train --homers     # HR model only
    python -m src.train --games      # Game winner model only
"""

import sys
import os
import time

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def train_hr_model():
    """Train the home run prediction model."""
    from src.data.history import load_game_data
    from src.models.hr_model import train_hr_model

    print("\n" + "=" * 60)
    print("💣 TRAINING: Home Run Prediction Model")
    print("=" * 60)

    game_log = load_game_data()
    print(f"\n📊 Training data: {len(game_log):,} batter-game rows")
    print(f"   Unique games: {game_log['game_pk'].nunique()}")
    print(f"   Unique players: {game_log['player_id'].nunique()}")
    print(f"   Date range: {game_log['game_date'].min()} → {game_log['game_date'].max()}")

    start = time.time()
    results = train_hr_model(game_log)
    elapsed = time.time() - start

    print(f"\n⏱️  Training took {elapsed:.1f}s")
    print(f"🏆 Best model: {results['best_model_name']}")
    print(f"   AUC: {results['results'][results['best_model_name']]['roc_auc']:.3f}")
    return results


def train_game_winner_model():
    """Train the game winner prediction model."""
    from src.models.game_model import train_game_model

    print("\n" + "=" * 60)
    print("🏆 TRAINING: Game Winner Model")
    print("=" * 60)

    results_path = os.path.join(DATA_DIR, "game_results.csv")
    if not os.path.exists(results_path):
        print("❌ No game_results.csv found! Run collection first.")
        return None

    game_results = pd.read_csv(results_path)
    print(f"\n📊 Training data: {len(game_results):,} games")
    print(f"   Date range: {game_results['game_date'].min()} → {game_results['game_date'].max()}")
    print(f"   Home win rate: {game_results['home_win'].mean()*100:.1f}%")

    start = time.time()
    results = train_game_model(game_results)
    elapsed = time.time() - start

    print(f"\n⏱️  Training took {elapsed:.1f}s")
    print(f"🏆 Best model: {results['best_model_name']}")
    print(f"   AUC: {results['results'][results['best_model_name']]['roc_auc']:.3f}")
    return results


def main():
    args = sys.argv[1:]

    print("⚾ HOMER TRACKER — Model Training Pipeline")
    print("=" * 60)

    do_hr = "--homers" in args or not args
    do_games = "--games" in args or not args

    if do_hr:
        train_hr_model()
    if do_games:
        train_game_winner_model()

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("   Run predictions with: python -m src.predictions.daily")
    print("=" * 60)


if __name__ == "__main__":
    main()
