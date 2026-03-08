"""
Recency-based sample weighting for model training.

We have 4+ seasons of data, but a game from April 2021 shouldn't
carry the same weight as one from last week. Player form, roster
composition, rule changes (pitch clock!), and park dimensions all
shift over time.

Strategy:
- The most recent RECENCY_FULL_WEIGHT_GAMES game-dates get weight 1.0
- Older games decay exponentially toward RECENCY_MIN_WEIGHT
- Old data still helps (prevents overfitting to a hot streak),
  but recent form dominates predictions

Visual:

  weight
  1.0 |████████████████████░░░░░░░░░░░░░░░░
  0.5 |                    ░░░░░░░░░░░░░░░░
  0.15|                              ░░░░░░
      +------------------------------------→ games ago
       recent (162)       older history
"""

import numpy as np
import pandas as pd

from src.config import (
    RECENCY_FULL_WEIGHT_GAMES,
    RECENCY_MIN_WEIGHT,
    RECENCY_DECAY_RATE,
)


def compute_recency_weights(
    dates: pd.Series,
    full_weight_games: int = RECENCY_FULL_WEIGHT_GAMES,
    min_weight: float = RECENCY_MIN_WEIGHT,
    decay_rate: float = RECENCY_DECAY_RATE,
) -> np.ndarray:
    """
    Compute per-sample weights based on game date recency.

    Args:
        dates: Series of date strings (YYYY-MM-DD) — one per sample.
        full_weight_games: Number of most recent unique game-dates
                           that get weight 1.0 (~162 = one full season).
        min_weight: Floor weight for the oldest data.
        decay_rate: Exponential decay rate for games older than the window.

    Returns:
        NumPy array of weights, same length as dates.
    """
    unique_dates = sorted(dates.unique(), reverse=True)
    total_dates = len(unique_dates)

    # Map each date to its recency rank (0 = most recent)
    date_to_rank = {d: i for i, d in enumerate(unique_dates)}
    ranks = dates.map(date_to_rank).values

    weights = np.ones(len(dates), dtype=float)

    # Games beyond the full-weight window decay exponentially
    beyond_window = ranks - full_weight_games
    decay_mask = beyond_window > 0

    weights[decay_mask] = np.exp(-decay_rate * beyond_window[decay_mask])
    weights = np.clip(weights, min_weight, 1.0)

    return weights


def print_weight_summary(dates: pd.Series, weights: np.ndarray) -> None:
    """Print a summary of how weights are distributed across the data."""
    unique_dates = sorted(dates.unique())
    total = len(weights)
    full_weight = (weights >= 0.99).sum()
    decayed = (weights < 0.99).sum()

    print(f"\n   \u2696\ufe0f  Recency Weighting Summary:")
    print(f"      Date range:     {unique_dates[0]} \u2192 {unique_dates[-1]}")
    print(f"      Unique dates:   {len(unique_dates)}")
    print(f"      Full weight:    {full_weight:,} samples ({full_weight/total*100:.0f}%)")
    print(f"      Decayed:        {decayed:,} samples ({decayed/total*100:.0f}%)")
    print(f"      Weight range:   {weights.min():.3f} \u2192 {weights.max():.3f}")
    print(f"      Mean weight:    {weights.mean():.3f}")

    # Show weight at key time horizons
    for months_ago, label in [(3, "~3 months"), (6, "~6 months"),
                               (12, "~1 year"), (24, "~2 years"),
                               (36, "~3 years")]:
        games_ago = months_ago * 27  # ~27 game-dates per month
        if games_ago < len(unique_dates):
            w = weights[dates == unique_dates[-(games_ago + 1)]]
            if len(w) > 0:
                print(f"      Weight {label} ago: {w.mean():.3f}")
