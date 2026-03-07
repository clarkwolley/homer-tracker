"""Tests for pitcher feature engineering."""

import pytest

from src.features.pitcher_features import (
    calc_platoon_advantage,
    _default_pitcher_stats,
)


class TestPlatoonAdvantage:
    def test_opposite_hand(self):
        """Lefty batter vs righty pitcher = full advantage."""
        assert calc_platoon_advantage("L", "R") == 1.0

    def test_same_hand(self):
        """Righty batter vs righty pitcher = disadvantage."""
        assert calc_platoon_advantage("R", "R") == 0.5

    def test_switch_hitter(self):
        """Switch hitter always gets partial advantage."""
        assert calc_platoon_advantage("S", "R") == 0.75
        assert calc_platoon_advantage("S", "L") == 0.75

    def test_lefty_vs_lefty(self):
        assert calc_platoon_advantage("L", "L") == 0.5

    def test_righty_vs_lefty(self):
        assert calc_platoon_advantage("R", "L") == 1.0


class TestDefaultPitcherStats:
    def test_returns_league_averages(self):
        stats = _default_pitcher_stats()
        assert stats["era"] == 4.50
        assert stats["whip"] == 1.30
        assert stats["hr_per_9"] == 1.30
        assert stats["k_per_9"] == 8.0

    def test_returns_dict(self):
        stats = _default_pitcher_stats()
        assert isinstance(stats, dict)
        assert "pitcher_id" in stats
