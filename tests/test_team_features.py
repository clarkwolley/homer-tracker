"""Tests for team feature engineering."""

import pandas as pd
import pytest

from src.features.team_features import (
    build_team_strength,
    build_matchup_features,
    GAME_FEATURE_COLUMNS,
)


@pytest.fixture
def mock_standings():
    """Minimal standings DataFrame for two teams."""
    return pd.DataFrame([
        {
            "team": "NYY", "team_id": 147, "team_name": "New York Yankees",
            "games_played": 80, "wins": 50, "losses": 30,
            "win_pct": 0.625, "runs_scored": 400, "runs_allowed": 320,
            "run_diff": 80, "home_wins": 28, "home_losses": 12,
            "away_wins": 22, "away_losses": 18, "streak_code": "W3",
            "division": "ALE",
        },
        {
            "team": "BOS", "team_id": 111, "team_name": "Boston Red Sox",
            "games_played": 80, "wins": 42, "losses": 38,
            "win_pct": 0.525, "runs_scored": 360, "runs_allowed": 350,
            "run_diff": 10, "home_wins": 24, "home_losses": 16,
            "away_wins": 18, "away_losses": 22, "streak_code": "L2",
            "division": "ALE",
        },
    ])


class TestBuildTeamStrength:
    def test_adds_per_game_rates(self, mock_standings):
        result = build_team_strength(mock_standings)
        assert "runs_per_game" in result.columns
        assert "runs_allowed_pg" in result.columns
        assert "run_diff_pg" in result.columns

    def test_rates_are_correct(self, mock_standings):
        result = build_team_strength(mock_standings)
        nyy = result[result["team"] == "NYY"].iloc[0]
        assert nyy["runs_per_game"] == 400 / 80  # 5.0
        assert nyy["runs_allowed_pg"] == 320 / 80  # 4.0

    def test_park_factor_assigned(self, mock_standings):
        result = build_team_strength(mock_standings)
        nyy = result[result["team"] == "NYY"].iloc[0]
        bos = result[result["team"] == "BOS"].iloc[0]
        assert nyy["park_hr_factor"] == 1.12  # Yankee Stadium short porch
        assert bos["park_hr_factor"] == 1.10  # Fenway

    def test_win_pcts_bounded(self, mock_standings):
        result = build_team_strength(mock_standings)
        assert (result["home_win_pct"] >= 0).all()
        assert (result["home_win_pct"] <= 1).all()
        assert (result["away_win_pct"] >= 0).all()
        assert (result["away_win_pct"] <= 1).all()


class TestBuildMatchupFeatures:
    def test_produces_all_columns(self, mock_standings):
        strength = build_team_strength(mock_standings)
        features = build_matchup_features("NYY", "BOS", strength)
        for col in GAME_FEATURE_COLUMNS:
            assert col in features, f"Missing column: {col}"

    def test_win_pct_diff_direction(self, mock_standings):
        strength = build_team_strength(mock_standings)
        features = build_matchup_features("NYY", "BOS", strength)
        # NYY is better, so home-away diff should be positive
        assert features["win_pct_diff"] > 0

    def test_missing_team_returns_empty(self, mock_standings):
        strength = build_team_strength(mock_standings)
        features = build_matchup_features("NYY", "LAD", strength)
        assert features == {}

    def test_park_factor_from_home(self, mock_standings):
        strength = build_team_strength(mock_standings)
        features = build_matchup_features("NYY", "BOS", strength)
        assert features["park_hr_factor"] == 1.12  # NYY home park
