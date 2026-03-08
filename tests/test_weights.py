"""Tests for recency-based sample weighting."""

import numpy as np
import pandas as pd
import pytest

from src.models.weights import compute_recency_weights


@pytest.fixture
def multi_season_dates():
    """Simulate 4 seasons of game dates (~650 unique dates)."""
    dates = []
    for year in [2022, 2023, 2024, 2025]:
        for month in range(4, 10):  # Apr-Sep
            for day in range(1, 28):
                dates.append(f"{year}-{month:02d}-{day:02d}")
    # Multiple samples per date (like multiple batters per game)
    expanded = []
    for d in dates:
        expanded.extend([d] * 18)  # ~18 batters per game
    return pd.Series(expanded)


class TestComputeRecencyWeights:
    def test_output_length_matches_input(self, multi_season_dates):
        weights = compute_recency_weights(multi_season_dates)
        assert len(weights) == len(multi_season_dates)

    def test_most_recent_games_get_full_weight(self, multi_season_dates):
        weights = compute_recency_weights(multi_season_dates, full_weight_games=162)
        # The most recent date should get weight 1.0
        most_recent = multi_season_dates.max()
        recent_mask = multi_season_dates == most_recent
        assert (weights[recent_mask] == 1.0).all()

    def test_oldest_games_get_reduced_weight(self, multi_season_dates):
        weights = compute_recency_weights(multi_season_dates, full_weight_games=162)
        oldest = multi_season_dates.min()
        oldest_mask = multi_season_dates == oldest
        assert (weights[oldest_mask] < 1.0).all()

    def test_weights_never_below_min(self, multi_season_dates):
        min_weight = 0.15
        weights = compute_recency_weights(
            multi_season_dates, min_weight=min_weight
        )
        assert (weights >= min_weight).all()

    def test_weights_never_above_one(self, multi_season_dates):
        weights = compute_recency_weights(multi_season_dates)
        assert (weights <= 1.0).all()

    def test_weights_decrease_with_age(self, multi_season_dates):
        weights = compute_recency_weights(multi_season_dates, full_weight_games=162)
        dates_sorted = multi_season_dates.sort_values(ascending=False)
        unique_sorted = dates_sorted.unique()
        # Weight at game 200 should be less than at game 50
        if len(unique_sorted) > 200:
            w_50 = weights[multi_season_dates == unique_sorted[50]].mean()
            w_200 = weights[multi_season_dates == unique_sorted[200]].mean()
            assert w_50 >= w_200

    def test_small_dataset_all_full_weight(self):
        """If we have fewer dates than the window, everything gets weight 1.0."""
        dates = pd.Series(["2025-06-01"] * 10 + ["2025-06-02"] * 10)
        weights = compute_recency_weights(dates, full_weight_games=162)
        assert (weights == 1.0).all()

    def test_custom_decay_rate(self, multi_season_dates):
        """Higher decay rate = faster weight drop."""
        slow = compute_recency_weights(multi_season_dates, decay_rate=0.001)
        fast = compute_recency_weights(multi_season_dates, decay_rate=0.01)
        # Fast decay should have lower mean weight
        assert fast.mean() < slow.mean()

    def test_same_date_gets_same_weight(self, multi_season_dates):
        """All samples from the same game-date get identical weights."""
        weights = compute_recency_weights(multi_season_dates)
        for date in multi_season_dates.unique()[:5]:
            date_weights = weights[multi_season_dates == date]
            assert len(set(date_weights)) == 1, \
                f"Samples from {date} have different weights"
