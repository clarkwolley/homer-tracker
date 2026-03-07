"""Tests for configuration module."""

from src.config import (
    MLB_API_BASE,
    CURRENT_SEASON,
    ROLLING_WINDOW,
    PARK_HR_FACTORS,
    HIGH_POWER_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
)


class TestConfig:
    def test_api_base_url(self):
        assert MLB_API_BASE.startswith("https://")
        assert "statsapi.mlb.com" in MLB_API_BASE

    def test_season_is_reasonable(self):
        assert 2020 <= CURRENT_SEASON <= 2030

    def test_rolling_window_positive(self):
        assert ROLLING_WINDOW > 0

    def test_park_factors_all_30_teams(self):
        # MLB has 30 teams
        assert len(PARK_HR_FACTORS) == 30

    def test_park_factors_reasonable_range(self):
        for team, factor in PARK_HR_FACTORS.items():
            assert 0.7 <= factor <= 1.5, f"{team} park factor {factor} out of range"

    def test_coors_field_is_highest(self):
        assert PARK_HR_FACTORS["COL"] == max(PARK_HR_FACTORS.values())

    def test_power_threshold_is_fraction(self):
        assert 0 < HIGH_POWER_THRESHOLD < 1

    def test_test_size_valid(self):
        assert 0 < TEST_SIZE < 1

    def test_random_state_is_int(self):
        assert isinstance(RANDOM_STATE, int)
