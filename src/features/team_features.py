"""
Team-level feature engineering for the game winner model.

MLB team features focus on:
- Run production and prevention rates
- Home/away splits
- Bullpen quality approximations
- Park factors for HR environment
"""

import pandas as pd
import numpy as np
from src.config import PARK_HR_FACTORS


def build_team_strength(standings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team strength metrics from standings.

    New columns: runs_per_game, runs_allowed_pg, run_diff_pg,
    home_win_pct, away_win_pct.
    """
    df = standings_df.copy()
    gp = df["games_played"].clip(lower=1)

    df["runs_per_game"] = df["runs_scored"] / gp
    df["runs_allowed_pg"] = df["runs_allowed"] / gp
    df["run_diff_pg"] = df["run_diff"] / gp

    home_games = (df["home_wins"] + df["home_losses"]).clip(lower=1)
    away_games = (df["away_wins"] + df["away_losses"]).clip(lower=1)

    df["home_win_pct"] = df["home_wins"] / home_games
    df["away_win_pct"] = df["away_wins"] / away_games

    # Park HR factor
    df["park_hr_factor"] = df["team"].map(PARK_HR_FACTORS).fillna(1.0)

    return df


def build_matchup_features(
    home_team: str,
    away_team: str,
    team_strength: pd.DataFrame,
) -> dict:
    """
    Build features for a specific home vs. away matchup.

    Returns dict of matchup features ready for model input.
    """
    home = team_strength[team_strength["team"] == home_team]
    away = team_strength[team_strength["team"] == away_team]

    if home.empty or away.empty:
        return {}

    home = home.iloc[0]
    away = away.iloc[0]

    return {
        # Home team stats
        "home_win_pct": home["win_pct"],
        "home_runs_pg": home["runs_per_game"],
        "home_runs_allowed_pg": home["runs_allowed_pg"],
        "home_run_diff_pg": home["run_diff_pg"],
        "home_home_win_pct": home["home_win_pct"],
        # Away team stats
        "away_win_pct": away["win_pct"],
        "away_runs_pg": away["runs_per_game"],
        "away_runs_allowed_pg": away["runs_allowed_pg"],
        "away_run_diff_pg": away["run_diff_pg"],
        "away_away_win_pct": away["away_win_pct"],
        # Matchup differentials
        "win_pct_diff": home["win_pct"] - away["win_pct"],
        "run_diff_pg_diff": home["run_diff_pg"] - away["run_diff_pg"],
        "offense_vs_defense": home["runs_per_game"] - away["runs_allowed_pg"],
        "defense_vs_offense": away["runs_per_game"] - home["runs_allowed_pg"],
        # Park factor
        "park_hr_factor": home.get("park_hr_factor", 1.0),
    }


GAME_FEATURE_COLUMNS = [
    "home_win_pct",
    "home_runs_pg",
    "home_runs_allowed_pg",
    "home_run_diff_pg",
    "home_home_win_pct",
    "away_win_pct",
    "away_runs_pg",
    "away_runs_allowed_pg",
    "away_run_diff_pg",
    "away_away_win_pct",
    "win_pct_diff",
    "run_diff_pg_diff",
    "offense_vs_defense",
    "defense_vs_offense",
    "park_hr_factor",
]
