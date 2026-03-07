"""
MLB Stats API client.

Talks to the free MLB Stats API (statsapi.mlb.com) to fetch schedules,
standings, rosters, boxscores, and player stats. No API key needed.

This module is a pure data access layer — fetch and return dicts.
No transformation, no analysis, no opinions.

Rate limiting strategy (learned the hard way from our NHL project):
- Proactive: minimum delay between requests
- Reactive: exponential backoff + jitter on 429 responses
- Caching: TTL-based in-memory cache for repeated endpoints
"""

import random
import time
import threading
from typing import Any

import requests
from src.config import MLB_API_BASE


DEFAULT_TIMEOUT = 15  # seconds (MLB API can be sluggish)


class MLBApiError(Exception):
    """Raised when an MLB API request fails."""


# --- Rate limiting -----------------------------------------------------------

MAX_RETRIES = 5
BASE_BACKOFF = 10
MAX_BACKOFF = 120
MIN_REQUEST_GAP = 0.35

_last_request_time = 0.0
_rate_lock = threading.Lock()


def _throttle() -> None:
    """Enforce minimum gap between requests."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < MIN_REQUEST_GAP:
            time.sleep(MIN_REQUEST_GAP - elapsed)
        _last_request_time = time.monotonic()


# --- Response cache ----------------------------------------------------------

DEFAULT_CACHE_TTL = 300  # 5 minutes

_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _cache_get(key: str) -> dict | None:
    """Return cached response if still fresh."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.monotonic() - entry["ts"] < entry["ttl"]:
            return entry["data"]
        _cache.pop(key, None)
        return None


def _cache_set(key: str, data: dict, ttl: float = DEFAULT_CACHE_TTL) -> None:
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.monotonic(), "ttl": ttl}


def clear_cache() -> None:
    """Clear the in-memory response cache."""
    with _cache_lock:
        _cache.clear()


# --- Core request function ---------------------------------------------------


def _get(url: str, params: dict | None = None, cache_ttl: float = DEFAULT_CACHE_TTL) -> dict:
    """
    Make a GET request to the MLB Stats API.

    Args:
        url: Full URL or path relative to MLB_API_BASE.
        params: Optional query parameters.
        cache_ttl: Cache TTL in seconds. 0 to skip caching.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        MLBApiError: If the request fails after all retries.
    """
    if url.startswith("/"):
        url = f"{MLB_API_BASE}{url}"

    # Build a cache key from URL + sorted params
    cache_key = url
    if params:
        sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_key = f"{url}?{sorted_params}"

    if cache_ttl > 0:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    for attempt in range(MAX_RETRIES + 1):
        _throttle()
        try:
            response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if cache_ttl > 0:
                _cache_set(cache_key, data, cache_ttl)

            return data

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < MAX_RETRIES:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    wait = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                    wait += random.uniform(0, wait * 0.25)

                print(
                    f"  ⏳ Rate limited, retrying in {wait:.0f}s... "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(wait)
                continue
            raise MLBApiError(f"MLB API request failed: {url} — {e}") from e
        except requests.RequestException as e:
            raise MLBApiError(f"MLB API request failed: {url} — {e}") from e


# --- Public API functions ----------------------------------------------------


def get_schedule(date: str = "today", season: int | None = None) -> dict:
    """
    Fetch the game schedule for a date.

    Args:
        date: Date string 'YYYY-MM-DD' or 'today'.
        season: Optional season year override.

    Returns:
        Schedule dict with 'dates' list, each containing 'games'.
    """
    params = {"sportId": 1, "gameType": "R"}
    if date == "today":
        from datetime import datetime
        date = datetime.now().strftime("%Y-%m-%d")
    params["date"] = date
    if season:
        params["season"] = season
    return _get("/schedule", params=params, cache_ttl=300)


def get_standings(season: int | None = None) -> dict:
    """
    Fetch current league standings.

    Returns:
        Standings dict with 'records' list — one entry per division,
        each containing 'teamRecords'.
    """
    from src.config import CURRENT_SEASON
    params = {
        "leagueId": "103,104",  # AL + NL
        "season": season or CURRENT_SEASON,
        "standingsTypes": "regularSeason",
    }
    return _get("/standings", params=params, cache_ttl=600)


def get_team_roster(team_id: int, season: int | None = None) -> dict:
    """
    Fetch a team's active roster.

    Args:
        team_id: MLB team ID (e.g., 147 for Yankees).

    Returns:
        Dict with 'roster' list of player entries.
    """
    from src.config import CURRENT_SEASON
    params = {"rosterType": "active", "season": season or CURRENT_SEASON}
    return _get(f"/teams/{team_id}/roster", params=params, cache_ttl=600)


def get_boxscore(game_pk: int) -> dict:
    """
    Fetch the boxscore for a specific game.

    Args:
        game_pk: MLB game PK (e.g., 745623).

    Returns:
        Full boxscore with team totals and individual player stats.
    """
    return _get(f"/game/{game_pk}/boxscore", cache_ttl=3600)


def get_game_feed(game_pk: int) -> dict:
    """
    Fetch the full live feed for a game (includes linescore + decisions).

    Args:
        game_pk: MLB game PK.

    Returns:
        Complete game feed with gameData, liveData, etc.
    """
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    return _get(url, cache_ttl=3600)


def get_player_stats(player_id: int, season: int | None = None, group: str = "hitting") -> dict:
    """
    Fetch a player's season stats.

    Args:
        player_id: MLB player ID.
        season: Season year.
        group: Stat group ('hitting' or 'pitching').

    Returns:
        Dict with player info and 'stats' list.
    """
    from src.config import CURRENT_SEASON
    params = {
        "stats": "season",
        "season": season or CURRENT_SEASON,
        "group": group,
    }
    return _get(f"/people/{player_id}/stats", params=params, cache_ttl=600)


def get_player_game_log(player_id: int, season: int | None = None, group: str = "hitting") -> dict:
    """
    Fetch a player's game-by-game log for a season.

    Args:
        player_id: MLB player ID.
        season: Season year.
        group: 'hitting' or 'pitching'.

    Returns:
        Dict with 'stats' list containing 'splits' (per-game stats).
    """
    from src.config import CURRENT_SEASON
    params = {
        "stats": "gameLog",
        "season": season or CURRENT_SEASON,
        "group": group,
    }
    return _get(f"/people/{player_id}/stats", params=params, cache_ttl=600)


def get_team_schedule(team_id: int, season: int | None = None) -> dict:
    """
    Fetch a team's full season schedule.

    Args:
        team_id: MLB team ID.
        season: Season year.

    Returns:
        Schedule dict with all games for the season.
    """
    from src.config import CURRENT_SEASON
    params = {
        "teamId": team_id,
        "season": season or CURRENT_SEASON,
        "sportId": 1,
        "gameType": "R",  # regular season only
    }
    return _get("/schedule", params=params, cache_ttl=600)


def get_all_teams() -> dict:
    """
    Fetch all MLB teams.

    Returns:
        Dict with 'teams' list — id, name, abbreviation, etc.
    """
    params = {"sportId": 1}
    return _get("/teams", params=params, cache_ttl=3600)
