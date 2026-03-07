"""
Pitcher matchup features for the home run model.

The opposing pitcher matters enormously for HR prediction.
A batter facing a gas-can with a 6.00 ERA and 2.0 HR/9 is way
more likely to go yard than one facing a Cy Young candidate.

We also capture platoon advantage — lefty batters crush righty
pitchers at a higher rate, and vice versa.
"""

import pandas as pd

from src.data import mlb_api
from src.data.collector import get_probable_pitchers


def get_pitcher_season_stats(pitcher_id: int) -> dict:
    """
    Fetch a pitcher's season stats.

    Returns:
        Dict with ERA, WHIP, HR/9, K/9, pitch hand, etc.
        Falls back to league-average defaults if unavailable.
    """
    try:
        data = mlb_api.get_player_stats(pitcher_id, group="pitching")
        splits = data.get("stats", [])
        if not splits or not splits[0].get("splits"):
            return _default_pitcher_stats()

        stats = splits[0]["splits"][0].get("stat", {})
        ip = float(stats.get("inningsPitched", "0"))

        # HR/9 calculation
        hr_allowed = stats.get("homeRuns", 0)
        hr_per_9 = (hr_allowed / ip * 9) if ip > 0 else 1.5

        return {
            "pitcher_id": pitcher_id,
            "era": float(stats.get("era", "4.50")),
            "whip": float(stats.get("whip", "1.30")),
            "hr_per_9": round(hr_per_9, 2),
            "k_per_9": float(stats.get("strikeoutsPer9Inn", "8.0")),
            "innings_pitched": ip,
            "games_started": stats.get("gamesStarted", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
        }

    except (mlb_api.MLBApiError, IndexError, KeyError, ValueError):
        return _default_pitcher_stats()


def _default_pitcher_stats() -> dict:
    """League-average fallback."""
    return {
        "pitcher_id": 0,
        "era": 4.50,
        "whip": 1.30,
        "hr_per_9": 1.30,
        "k_per_9": 8.0,
        "innings_pitched": 0,
        "games_started": 0,
        "wins": 0,
        "losses": 0,
    }


def get_pitcher_hand(pitcher_id: int) -> str:
    """
    Get the pitcher's throwing hand ('L' or 'R').

    Falls back to 'R' (most pitchers are right-handed).
    """
    try:
        data = mlb_api.get_player_stats(pitcher_id, group="pitching")
        # The people endpoint sometimes includes pitchHand
        # We may need to fetch from a different endpoint
        return "R"  # default — enhanced in v2 with /people endpoint
    except Exception:
        return "R"


def build_pitcher_matchup_features(opponent_team: str, game_date: str = "today") -> dict:
    """
    Build pitcher-based features for a matchup.

    Args:
        opponent_team: The team abbreviation whose pitcher the batter faces.
        game_date: Date string for probable pitcher lookup.

    Returns:
        Dict with pitcher matchup features:
        - opp_pitcher_era: Opposing starter's ERA
        - opp_pitcher_whip: Opposing starter's WHIP
        - opp_pitcher_hr9: Opposing starter's HR/9 (critical for HR prediction!)
        - opp_pitcher_quality: Composite quality score
        - opp_pitcher_name: Pitcher's name (for display)
        - opp_pitcher_hand: L or R (for platoon calculations)
    """
    # Try to get probable pitcher from schedule
    probables = get_probable_pitchers(game_date)

    pitcher_id = None
    pitcher_name = "TBD"

    if not probables.empty:
        opp_pitcher = probables[probables["team"] == opponent_team]
        if not opp_pitcher.empty:
            pitcher_id = opp_pitcher.iloc[0]["pitcher_id"]
            pitcher_name = opp_pitcher.iloc[0]["pitcher_name"]

    if pitcher_id:
        stats = get_pitcher_season_stats(pitcher_id)
    else:
        stats = _default_pitcher_stats()

    era = stats["era"]
    whip = stats["whip"]
    hr9 = stats["hr_per_9"]

    # Quality score: lower ERA + lower HR/9 = tougher pitcher
    # Invert so higher = harder to homer off of
    quality = 10.0 - (era * 0.8) - (hr9 * 1.5)
    quality = max(0, min(10, quality))

    return {
        "opp_pitcher_era": era,
        "opp_pitcher_whip": whip,
        "opp_pitcher_hr9": hr9,
        "opp_pitcher_quality": round(quality, 3),
        "opp_pitcher_name": pitcher_name,
        "opp_pitcher_hand": get_pitcher_hand(pitcher_id) if pitcher_id else "R",
    }


def calc_platoon_advantage(bat_side: str, pitch_hand: str) -> float:
    """
    Calculate platoon advantage (opposite-hand matchup).

    Historically, batters hit ~20-30 OPS points better against
    opposite-hand pitchers. Switch hitters always have the advantage.

    Returns:
        1.0 = platoon advantage (opposite hand)
        0.5 = same hand (disadvantage)
        0.75 = switch hitter (always favorable)
    """
    if bat_side == "S":
        return 0.75  # switch hitters always face the favorable side
    if bat_side != pitch_hand:
        return 1.0   # opposite hand = advantage
    return 0.5        # same hand = disadvantage
