"""
Batter-level feature engineering for the home run model.

Key features for HR prediction:
- Rolling HR rate, ISO (isolated power), SLG
- HR streak / drought detection
- Power hitter identification (AB per HR rate)
- Plate appearances (opportunity proxy)
- Batter handedness encoding

In baseball, ~3% of plate appearances result in a HR.
That's even more imbalanced than NHL goals (~15%), so
good features matter even more.
"""

import pandas as pd
import numpy as np
from src.config import ROLLING_WINDOW, STREAK_MIN_GAMES, HIGH_POWER_THRESHOLD


def add_basic_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-game rate features and the prediction target.

    New columns:
    - hit_hr (binary: did the batter homer? Our target!)
    - iso (isolated power: SLG - AVG, per-game estimate)
    """
    result = df.copy()
    result["hit_hr"] = (result["home_runs"] > 0).astype(int)

    # Per-game isolated power estimate
    ab = result["at_bats"].clip(lower=1)
    singles = result["hits"] - result["doubles"] - result["triples"] - result["home_runs"]
    singles = singles.clip(lower=0)
    total_bases = singles + (2 * result["doubles"]) + (3 * result["triples"]) + (4 * result["home_runs"])
    result["game_slg"] = total_bases / ab
    result["game_avg"] = result["hits"] / ab
    result["game_iso"] = result["game_slg"] - result["game_avg"]

    return result


def add_rolling_averages(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Calculate rolling averages over last N games per batter.

    New columns:
    - rolling_hr_avg: HR per game over window
    - rolling_hits_avg: Hits per game
    - rolling_ab_avg: At-bats per game (opportunity)
    - rolling_slg: Slugging over window
    - rolling_iso: Isolated power over window
    - rolling_ops: OPS over window (crude approximation)
    - games_in_window: Data points available in window
    """
    result = df.copy()
    result = result.sort_values(["player_id", "game_date"])

    if "game_slg" not in result.columns:
        result = add_basic_rates(result)

    rolling_cols = {
        "home_runs": "rolling_hr_avg",
        "hits": "rolling_hits_avg",
        "at_bats": "rolling_ab_avg",
        "game_slg": "rolling_slg",
        "game_iso": "rolling_iso",
        "walks": "rolling_walks_avg",
    }

    for raw_col, new_col in rolling_cols.items():
        result[new_col] = (
            result
            .groupby("player_id")[raw_col]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
        )

    # Rolling OPS approximation: (hits + walks) / (AB + walks) + SLG
    result["rolling_obp_approx"] = np.where(
        (result["rolling_ab_avg"] + result["rolling_walks_avg"]) > 0,
        (result["rolling_hits_avg"] + result["rolling_walks_avg"])
        / (result["rolling_ab_avg"] + result["rolling_walks_avg"]),
        0.0,
    )
    result["rolling_ops"] = result["rolling_obp_approx"] + result["rolling_slg"]

    result["games_in_window"] = (
        result
        .groupby("player_id")["home_runs"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).count())
    )

    return result


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect HR streaks and droughts.

    New columns:
    - hr_streak: Consecutive games with a HR (entering this game)
    - hit_streak: Consecutive games with a hit
    - hr_drought: Consecutive games WITHOUT a HR
    - is_hot: On a HR streak of 2+ games
    """
    result = df.copy()
    result = result.sort_values(["player_id", "game_date"])

    if "hit_hr" not in result.columns:
        result["hit_hr"] = (result["home_runs"] > 0).astype(int)

    def _calc_streak(series: pd.Series) -> pd.Series:
        streaks = []
        count = 0
        for val in series:
            count = count + 1 if val else 0
            streaks.append(count)
        return pd.Series(streaks, index=series.index)

    def _calc_drought(series: pd.Series) -> pd.Series:
        droughts = []
        count = 0
        for val in series:
            count = 0 if val else count + 1
            droughts.append(count)
        return pd.Series(droughts, index=series.index)

    # Shift by 1 — we want the streak ENTERING the game
    result["hr_streak"] = (
        result.groupby("player_id")["hit_hr"]
        .transform(lambda x: _calc_streak(x.astype(bool)).shift(1).fillna(0))
        .astype(int)
    )

    has_hit = (result["hits"] > 0)
    result["hit_streak"] = (
        result.groupby("player_id")["hits"]
        .transform(lambda x: _calc_streak(x > 0).shift(1).fillna(0))
        .astype(int)
    )

    result["hr_drought"] = (
        result.groupby("player_id")["hit_hr"]
        .transform(lambda x: _calc_drought(x.astype(bool)).shift(1).fillna(0))
        .astype(int)
    )

    result["is_hot"] = (result["hr_streak"] >= STREAK_MIN_GAMES).astype(int)

    return result


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add power-hitting identification features.

    New columns:
    - hr_per_ab: Season HR rate (HRs per at-bat)
    - is_power_hitter: Above-average HR rate
    - ab_per_hr: At-bats per HR (lower = more dangerous)
    """
    result = df.copy()

    # Rolling HR per AB rate
    rolling_hr = result.groupby("player_id")["home_runs"].transform(
        lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).sum()
    )
    rolling_ab = result.groupby("player_id")["at_bats"].transform(
        lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).sum()
    )

    result["hr_per_ab"] = np.where(rolling_ab > 0, rolling_hr / rolling_ab, 0.0)
    result["is_power_hitter"] = (result["hr_per_ab"] > HIGH_POWER_THRESHOLD).astype(int)

    result["ab_per_hr"] = np.where(
        rolling_hr > 0, rolling_ab / rolling_hr, 99.0
    )
    # Cap at 99 (never homered in window)
    result["ab_per_hr"] = result["ab_per_hr"].clip(upper=99.0)

    return result


def add_handedness_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode batter handedness as numeric features.

    New columns:
    - bats_left: 1 if left-handed batter
    - bats_right: 1 if right-handed batter
    - bats_switch: 1 if switch hitter
    """
    result = df.copy()
    bat_side = result.get("bat_side", pd.Series(["R"] * len(result)))
    result["bats_left"] = (bat_side == "L").astype(int)
    result["bats_right"] = (bat_side == "R").astype(int)
    result["bats_switch"] = (bat_side == "S").astype(int)
    return result


def build_batter_features(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline for batter-level HR prediction.

    Chains all individual feature functions into a single pipeline.
    """
    df = game_log.copy()
    df = add_basic_rates(df)
    df = add_rolling_averages(df)
    df = add_streak_features(df)
    df = add_power_features(df)
    df = add_handedness_encoding(df)
    df["is_home"] = df["is_home"].astype(int)
    return df


# The columns our model will use as inputs
FEATURE_COLUMNS = [
    # Rolling averages (recent form)
    "rolling_hr_avg",
    "rolling_hits_avg",
    "rolling_ab_avg",
    "rolling_slg",
    "rolling_iso",
    "rolling_ops",
    "games_in_window",
    # Streak features (momentum)
    "hr_streak",
    "hit_streak",
    "hr_drought",
    "is_hot",
    # Power profile
    "hr_per_ab",
    "is_power_hitter",
    "ab_per_hr",
    # Context
    "is_home",
    "bats_left",
    "bats_right",
    "bats_switch",
    # Pitcher matchup (injected at prediction time)
    "opp_pitcher_era",
    "opp_pitcher_whip",
    "opp_pitcher_hr9",
    "opp_pitcher_quality",
    "platoon_advantage",
    # Park factor (injected at prediction time)
    "park_hr_factor",
]

TARGET_COLUMN = "hit_hr"
