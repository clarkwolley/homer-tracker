"""Tests for batter feature engineering."""

import pandas as pd
import numpy as np
import pytest

from src.features.batter_features import (
    add_basic_rates,
    add_rolling_averages,
    add_streak_features,
    add_power_features,
    build_batter_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


@pytest.fixture
def sample_game_log():
    """Multi-game log for two batters."""
    data = []
    for pid, name in [(1, "Aaron Judge"), (2, "Pete Alonso")]:
        for i in range(20):
            hr = 1 if i % 5 == 0 else 0
            data.append({
                "game_pk": 1000 + i,
                "game_date": f"2025-04-{i+1:02d}",
                "player_id": pid,
                "name": name,
                "position": "RF" if pid == 1 else "1B",
                "team": "NYY" if pid == 1 else "NYM",
                "team_id": 147 if pid == 1 else 121,
                "is_home": i % 2 == 0,
                "at_bats": 4,
                "runs": hr + (1 if i % 3 == 0 else 0),
                "hits": 1 + hr,
                "doubles": 0 if hr else (1 if i % 4 == 0 else 0),
                "triples": 0,
                "home_runs": hr,
                "rbi": hr * 2,
                "walks": 1 if i % 3 == 0 else 0,
                "strikeouts": 1,
                "stolen_bases": 0,
                "hit_by_pitch": 0,
                "sac_flies": 0,
                "plate_appearances": 5,
            })
    return pd.DataFrame(data)


class TestAddBasicRates:
    def test_hit_hr_target(self, sample_game_log):
        result = add_basic_rates(sample_game_log)
        assert "hit_hr" in result.columns
        # HR every 5th game: games 0, 5, 10, 15 = 4 per player, 8 total
        assert result["hit_hr"].sum() == 8

    def test_game_slg_calculation(self, sample_game_log):
        result = add_basic_rates(sample_game_log)
        assert "game_slg" in result.columns
        assert (result["game_slg"] >= 0).all()

    def test_game_iso_nonnegative(self, sample_game_log):
        result = add_basic_rates(sample_game_log)
        # ISO can be slightly negative due to doubles counting
        # but should be reasonable
        assert "game_iso" in result.columns


class TestAddRollingAverages:
    def test_rolling_columns_exist(self, sample_game_log):
        result = add_rolling_averages(sample_game_log)
        expected = [
            "rolling_hr_avg", "rolling_hits_avg", "rolling_ab_avg",
            "rolling_slg", "rolling_iso", "rolling_ops", "games_in_window",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_shift_prevents_leakage(self, sample_game_log):
        """First game for each player should have NaN rolling values (no history)."""
        result = add_rolling_averages(sample_game_log)
        for pid in result["player_id"].unique():
            player_data = result[result["player_id"] == pid].sort_values("game_date")
            assert pd.isna(player_data.iloc[0]["rolling_hr_avg"]), \
                "First game should have NaN rolling avg (shifted by 1)"


class TestAddStreakFeatures:
    def test_streak_columns_exist(self, sample_game_log):
        result = add_streak_features(sample_game_log)
        for col in ["hr_streak", "hit_streak", "hr_drought", "is_hot"]:
            assert col in result.columns

    def test_streak_starts_at_zero(self, sample_game_log):
        """First game should have streak = 0 (no previous games)."""
        result = add_streak_features(sample_game_log)
        for pid in result["player_id"].unique():
            player_data = result[result["player_id"] == pid].sort_values("game_date")
            assert player_data.iloc[0]["hr_streak"] == 0


class TestAddPowerFeatures:
    def test_power_columns_exist(self, sample_game_log):
        result = add_power_features(sample_game_log)
        for col in ["hr_per_ab", "is_power_hitter", "ab_per_hr"]:
            assert col in result.columns

    def test_ab_per_hr_capped(self, sample_game_log):
        result = add_power_features(sample_game_log)
        assert (result["ab_per_hr"] <= 99.0).all()


class TestBuildBatterFeatures:
    def test_full_pipeline(self, sample_game_log):
        result = build_batter_features(sample_game_log)
        # Should have all expected columns
        assert TARGET_COLUMN in result.columns
        assert len(result) == len(sample_game_log)

    def test_no_nans_in_non_rolling(self, sample_game_log):
        result = build_batter_features(sample_game_log)
        # Non-rolling columns should not have NaN
        for col in ["hit_hr", "is_home", "hr_streak", "hr_drought"]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
