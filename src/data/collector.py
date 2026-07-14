"""
Data collector — transforms raw MLB API responses into clean DataFrames.

This is the ETL layer. The API client fetches raw JSON, this module
transforms it into tidy DataFrames for analysis and modeling.
"""

from datetime import date

import pandas as pd

from src.data import mlb_api


# --- Team ID mapping ---------------------------------------------------------
# The MLB API uses numeric team IDs. We need abbreviations for display.

_TEAM_MAP = None


def _get_team_map() -> dict[int, str]:
    """Lazy-load team ID → abbreviation mapping from the API."""
    global _TEAM_MAP
    if _TEAM_MAP is None:
        data = mlb_api.get_all_teams()
        _TEAM_MAP = {
            t["id"]: t["abbreviation"]
            for t in data.get("teams", [])
            if t.get("sport", {}).get("id") == 1  # MLB only
        }
    return _TEAM_MAP


def team_id_to_abbrev(team_id: int) -> str:
    """Convert MLB team ID to abbreviation (e.g., 147 → 'NYY')."""
    return _get_team_map().get(team_id, f"T{team_id}")


def team_abbrev_to_id(abbrev: str) -> int | None:
    """Convert team abbreviation to MLB team ID (e.g., 'NYY' → 147)."""
    mapping = _get_team_map()
    for tid, ab in mapping.items():
        if ab == abbrev:
            return tid
    return None


# --- Schedule ----------------------------------------------------------------


def get_todays_games() -> pd.DataFrame:
    """
    Get today's scheduled games as a clean DataFrame.

    The MLB API already filters by date at the request level, but we
    double-check against today's actual date as a safety net — same
    pattern as snipe-tracker where the NHL API returns a full week.

    Returns:
        DataFrame: game_pk, date, home_team, away_team, home_id, away_id,
                   game_state, home_score, away_score.
    """
    schedule = mlb_api.get_schedule()
    today_str = date.today().isoformat()  # 'YYYY-MM-DD'
    games = []

    for day in schedule.get("dates", []):
        if day["date"] != today_str:
            continue
        for game in day.get("games", []):
            home = game.get("teams", {}).get("home", {}).get("team", {})
            away = game.get("teams", {}).get("away", {}).get("team", {})

            games.append({
                "game_pk": game["gamePk"],
                "date": day["date"],
                "home_team": team_id_to_abbrev(home.get("id", 0)),
                "away_team": team_id_to_abbrev(away.get("id", 0)),
                "home_id": home.get("id", 0),
                "away_id": away.get("id", 0),
                "game_state": game.get("status", {}).get("abstractGameState", ""),
                "home_score": game.get("teams", {}).get("home", {}).get("score", 0),
                "away_score": game.get("teams", {}).get("away", {}).get("score", 0),
            })

    return pd.DataFrame(games)


# --- Standings ---------------------------------------------------------------


def get_standings_df() -> pd.DataFrame:
    """
    Get current standings as a DataFrame.

    Returns:
        DataFrame: team, team_id, team_name, games_played, wins, losses,
                   win_pct, runs_scored, runs_allowed, run_diff, etc.
    """
    data = mlb_api.get_standings()
    rows = []

    for division in data.get("records", []):
        for team in division.get("teamRecords", []):
            t = team.get("team", {})
            streak = team.get("streak", {})
            records = team.get("records", {}).get("splitRecords", [])

            # Extract home/away records from splits
            home_rec = next((r for r in records if r.get("type") == "home"), {})
            away_rec = next((r for r in records if r.get("type") == "away"), {})

            rows.append({
                "team": team_id_to_abbrev(t.get("id", 0)),
                "team_id": t.get("id", 0),
                "team_name": t.get("name", ""),
                "games_played": team.get("gamesPlayed", 0),
                "wins": team.get("wins", 0),
                "losses": team.get("losses", 0),
                "win_pct": float(team.get("winningPercentage", ".500")),
                "runs_scored": team.get("runsScored", 0),
                "runs_allowed": team.get("runsAllowed", 0),
                "run_diff": team.get("runDifferential", 0),
                "home_wins": home_rec.get("wins", 0),
                "home_losses": home_rec.get("losses", 0),
                "away_wins": away_rec.get("wins", 0),
                "away_losses": away_rec.get("losses", 0),
                "streak_code": f"{streak.get('streakType', '')[0]}{streak.get('streakNumber', 0)}"
                if streak.get("streakType") else "",
                "division": division.get("division", {}).get("abbreviation", ""),
            })

    return pd.DataFrame(rows)


# --- Boxscore / Player Stats ------------------------------------------------


def get_game_batter_stats(game_pk: int) -> pd.DataFrame:
    """
    Get per-batter stats from a single game's boxscore.

    Returns:
        DataFrame with one row per batter per game:
        game_pk, game_date, player_id, name, position, team, team_id,
        is_home, at_bats, runs, hits, home_runs, rbi, walks, strikeouts,
        avg, obp, slg, ops, stolen_bases, hit_by_pitch, sac_flies.
    """
    box = mlb_api.get_boxscore(game_pk)
    feed = mlb_api.get_game_feed(game_pk)

    game_date = feed.get("gameData", {}).get("datetime", {}).get("officialDate", "")

    rows = []
    teams_data = box.get("teams", {})

    for side, is_home in [("away", False), ("home", True)]:
        team_data = teams_data.get(side, {})
        team_info = team_data.get("team", {})
        team_id = team_info.get("id", 0)
        team_abbrev = team_id_to_abbrev(team_id)
        batters = team_data.get("batters", [])
        players = team_data.get("players", {})

        for batter_id in batters:
            player_key = f"ID{batter_id}"
            player = players.get(player_key, {})
            person = player.get("person", {})
            position = player.get("position", {}).get("abbreviation", "")
            batting = player.get("stats", {}).get("batting", {})

            # Skip pitchers who didn't bat meaningfully
            if not batting or batting.get("atBats", 0) == 0:
                continue

            rows.append({
                "game_pk": game_pk,
                "game_date": game_date,
                "player_id": person.get("id", batter_id),
                "name": person.get("fullName", f"Player {batter_id}"),
                "position": position,
                "team": team_abbrev,
                "team_id": team_id,
                "is_home": is_home,
                "at_bats": batting.get("atBats", 0),
                "runs": batting.get("runs", 0),
                "hits": batting.get("hits", 0),
                "doubles": batting.get("doubles", 0),
                "triples": batting.get("triples", 0),
                "home_runs": batting.get("homeRuns", 0),
                "rbi": batting.get("rbi", 0),
                "walks": batting.get("baseOnBalls", 0),
                "strikeouts": batting.get("strikeOuts", 0),
                "stolen_bases": batting.get("stolenBases", 0),
                "hit_by_pitch": batting.get("hitByPitch", 0),
                "sac_flies": batting.get("sacFlies", 0),
                "plate_appearances": batting.get("plateAppearances", 0)
                or (batting.get("atBats", 0) + batting.get("baseOnBalls", 0)
                    + batting.get("hitByPitch", 0) + batting.get("sacFlies", 0)),
            })

    return pd.DataFrame(rows)


def _parse_ip(ip_str) -> float:
    """Convert MLB inningsPitched string ('2.1' = 2⅓ IP) to decimal innings."""
    try:
        parts = str(ip_str).split(".")
        return round(int(parts[0]) + (int(parts[1]) if len(parts) > 1 else 0) / 3, 3)
    except (ValueError, IndexError):
        return 0.0


def get_game_pitcher_stats(game_pk: int) -> pd.DataFrame:
    """
    Get per-pitcher stats from a single game's boxscore.

    Returns one row per pitcher appearance with:
    game_pk, game_date, player_id, name, team, team_id, is_home,
    is_starter, innings_pitched, pitches_thrown, batters_faced,
    earned_runs, strikeouts, walks, games_finished.
    """
    box = mlb_api.get_boxscore(game_pk)   # served from cache if batter stats already fetched
    feed = mlb_api.get_game_feed(game_pk)

    game_date = feed.get("gameData", {}).get("datetime", {}).get("officialDate", "")

    rows = []
    teams_data = box.get("teams", {})

    for side, is_home in [("away", False), ("home", True)]:
        team_data = teams_data.get(side, {})
        team_info = team_data.get("team", {})
        team_id = team_info.get("id", 0)
        team_abbrev = team_id_to_abbrev(team_id)
        pitchers = team_data.get("pitchers", [])
        players = team_data.get("players", {})

        for i, pitcher_id in enumerate(pitchers):
            player = players.get(f"ID{pitcher_id}", {})
            person = player.get("person", {})
            pitching = player.get("stats", {}).get("pitching", {})

            if not pitching or pitching.get("battersFaced", 0) == 0:
                continue

            rows.append({
                "game_pk": game_pk,
                "game_date": game_date,
                "player_id": person.get("id", pitcher_id),
                "name": person.get("fullName", f"Pitcher {pitcher_id}"),
                "team": team_abbrev,
                "team_id": team_id,
                "is_home": is_home,
                "is_starter": int(i == 0),
                "innings_pitched": _parse_ip(pitching.get("inningsPitched", "0.0")),
                "pitches_thrown": pitching.get("numberOfPitches", 0),
                "batters_faced": pitching.get("battersFaced", 0),
                "earned_runs": pitching.get("earnedRuns", 0),
                "strikeouts": pitching.get("strikeOuts", 0),
                "walks": pitching.get("baseOnBalls", 0),
                "games_finished": pitching.get("gamesFinished", 0),
            })

    return pd.DataFrame(rows)


def get_team_batters(team_id: int) -> pd.DataFrame:
    """Get season batting stats for all batters on a team's active roster."""
    roster = mlb_api.get_team_roster(team_id)
    team_abbrev = team_id_to_abbrev(team_id)
    rows = []

    for entry in roster.get("roster", []):
        person = entry.get("person", {})
        position = entry.get("position", {}).get("abbreviation", "")
        if position == "P":
            continue

        player_id = person.get("id", 0)
        player_name = person.get("fullName", "")
        bat_side = person.get("batSide", {}).get("code", "R")

        # Try current season stats first, fallback to defaults
        stats = None
        try:
            stats_data = mlb_api.get_player_stats(player_id, group="hitting")
            splits = stats_data.get("stats", [])
            if splits and splits[0].get("splits"):
                stat_entry = splits[0]["splits"][0].get("stat", {})
                if stat_entry and stat_entry.get("gamesPlayed", 0) > 0:
                    stats = stat_entry
        except (mlb_api.MLBApiError, IndexError, KeyError, AttributeError):
            pass

        # Early season: use sensible defaults
        if stats is None:
            stats = {
                "gamesPlayed": 0, "atBats": 3, "hits": 1, "homeRuns": 0,
                "rbi": 0, "baseOnBalls": 0, "strikeOuts": 0, "stolenBases": 0,
                "avg": ".333", "obp": ".400", "slg": ".400", "ops": ".800",
                "plateAppearances": 3,
            }

        rows.append({
            "player_id": player_id, "name": player_name, "position": position,
            "team": team_abbrev, "team_id": team_id,
            "games_played": stats.get("gamesPlayed", 0),
            "at_bats": stats.get("atBats", 0), "hits": stats.get("hits", 0),
            "home_runs": stats.get("homeRuns", 0), "rbi": stats.get("rbi", 0),
            "walks": stats.get("baseOnBalls", 0),
            "strikeouts": stats.get("strikeOuts", 0),
            "stolen_bases": stats.get("stolenBases", 0),
            "avg": float(stats.get("avg", ".000")),
            "obp": float(stats.get("obp", ".000")),
            "slg": float(stats.get("slg", ".000")),
            "ops": float(stats.get("ops", ".000")),
            "plate_appearances": stats.get("plateAppearances", 0),
            "bat_side": bat_side,
        })

    return pd.DataFrame(rows)



def get_probable_pitchers(date: str = "today") -> pd.DataFrame:
    """
    Get probable starting pitchers for games on a date.

    Returns:
        DataFrame: game_pk, team, team_id, pitcher_id, pitcher_name, is_home.
    """
    schedule = mlb_api.get_schedule(date)
    rows = []

    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            game_pk = game["gamePk"]

            for side, is_home in [("away", False), ("home", True)]:
                pitcher = (
                    game.get("teams", {})
                    .get(side, {})
                    .get("probablePitcher", {})
                )
                team = game.get("teams", {}).get(side, {}).get("team", {})

                if pitcher:
                    rows.append({
                        "game_pk": game_pk,
                        "team": team_id_to_abbrev(team.get("id", 0)),
                        "team_id": team.get("id", 0),
                        "pitcher_id": pitcher.get("id", 0),
                        "pitcher_name": pitcher.get("fullName", "TBD"),
                        "is_home": is_home,
                    })

    return pd.DataFrame(rows)


def get_back_to_back_status(team_id: int, game_date: str) -> dict:
    """
    Check if a team played yesterday (day game after night game, etc.).

    In MLB this is less about B2B and more about series/travel fatigue.

    Returns:
        Dict with days_rest and is_travel_day info.
    """
    from datetime import datetime, timedelta

    try:
        schedule = mlb_api.get_team_schedule(team_id)
    except mlb_api.MLBApiError:
        return {"days_rest": 1, "is_travel_day": False}

    completed = []
    for day in schedule.get("dates", []):
        for game in day.get("games", []):
            state = game.get("status", {}).get("abstractGameState", "")
            if state == "Final" and day["date"] < game_date:
                completed.append(day["date"])

    if not completed:
        return {"days_rest": 3, "is_travel_day": False}

    last_date = max(completed)
    last = datetime.strptime(last_date, "%Y-%m-%d")
    current = datetime.strptime(game_date, "%Y-%m-%d")
    days_rest = (current - last).days

    return {
        "days_rest": days_rest,
        "is_travel_day": days_rest == 1,  # played yesterday
    }
