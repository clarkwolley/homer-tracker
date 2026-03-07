"""
Bulk data collector — pulls full seasons of game data.

Uses team schedules to discover all game IDs, then fetches boxscores
only for games we don't already have. Checkpoint saves every N games.

Usage:
    python -m src.data.collect_bulk                 # Current season
    python -m src.data.collect_bulk --season 2024   # Specific season
"""

import os
import sys
import time

import pandas as pd

from src.data import mlb_api
from src.data.collector import (
    get_game_batter_stats,
    get_standings_df,
    team_id_to_abbrev,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
GAME_LOG_FILE = os.path.join(DATA_DIR, "game_log.csv")
GAME_RESULTS_FILE = os.path.join(DATA_DIR, "game_results.csv")
CHECKPOINT_INTERVAL = 50


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_existing_game_pks() -> set:
    """Load game PKs we've already collected."""
    if not os.path.exists(GAME_LOG_FILE):
        return set()
    df = pd.read_csv(GAME_LOG_FILE, usecols=["game_pk"])
    return set(df["game_pk"].unique())


def discover_season_games(season: int) -> pd.DataFrame:
    """
    Discover all completed regular-season games for a season.

    Makes ~30 API calls (one per team schedule) and deduplicates.
    Extracts game results from schedule data — no extra calls needed!

    Returns:
        DataFrame with one row per unique completed game.
    """
    print(f"\n🔍 Discovering games for {season} season...")

    standings = get_standings_df()
    team_ids = standings["team_id"].unique().tolist()
    print(f"   Found {len(team_ids)} teams")

    all_games = {}
    for i, team_id in enumerate(team_ids, 1):
        team_abbrev = team_id_to_abbrev(team_id)
        try:
            data = mlb_api.get_team_schedule(team_id, season)

            for day in data.get("dates", []):
                for game in day.get("games", []):
                    gpk = game["gamePk"]
                    if gpk in all_games:
                        continue

                    state = game.get("status", {}).get("abstractGameState", "")
                    if state != "Final":
                        continue

                    home = game.get("teams", {}).get("home", {})
                    away = game.get("teams", {}).get("away", {})
                    home_score = home.get("score", 0)
                    away_score = away.get("score", 0)

                    all_games[gpk] = {
                        "game_pk": gpk,
                        "game_date": day["date"],
                        "home_team": team_id_to_abbrev(
                            home.get("team", {}).get("id", 0)
                        ),
                        "away_team": team_id_to_abbrev(
                            away.get("team", {}).get("id", 0)
                        ),
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_win": int(home_score > away_score),
                        "total_runs": home_score + away_score,
                    }

            print(f"   [{i:2d}/{len(team_ids)}] {team_abbrev}: schedule fetched")
            time.sleep(0.3)

        except mlb_api.MLBApiError as e:
            print(f"   [{i:2d}/{len(team_ids)}] ⚠️  {team_abbrev}: {e}")

    df = pd.DataFrame(all_games.values())
    if not df.empty:
        df = df.sort_values("game_date").reset_index(drop=True)
    print(f"   ✅ Found {len(df)} unique completed regular-season games")
    return df


def collect_boxscores(
    game_pks: list[int],
    delay: float = 0.4,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Fetch batter-level boxscore data for a list of games.

    Features progress updates, checkpoint saves, and graceful error handling.
    """
    total = len(game_pks)
    if total == 0:
        print("   No games to fetch!")
        return pd.DataFrame()

    est_minutes = (total * delay) / 60
    print(f"\n📦 Fetching {total} boxscores (~{est_minutes:.1f} min at {delay}s/call)...")

    all_frames = []
    failed = []
    start_time = time.time()

    for i, gpk in enumerate(game_pks, 1):
        try:
            df = get_game_batter_stats(gpk)
            all_frames.append(df)
        except Exception as e:
            failed.append(gpk)
            if len(failed) <= 5:
                print(f"   ⚠️  Game {gpk}: {e}")

        if i % 25 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(
                f"   [{i:4d}/{total}] "
                f"{i/total*100:5.1f}% | "
                f"{rate:.1f} games/sec | "
                f"~{remaining/60:.1f} min remaining"
            )

        if checkpoint_path and i % CHECKPOINT_INTERVAL == 0 and all_frames:
            _save_checkpoint(all_frames, checkpoint_path)

        time.sleep(delay)

    if failed:
        print(f"   ⚠️  Failed to fetch {len(failed)} games")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"   ✅ Collected {len(combined):,} batter-game rows from {total - len(failed)} games")
    return combined


def _save_checkpoint(frames: list, path: str):
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)


def save_game_log(new_data: pd.DataFrame):
    """Merge new batter-game data with existing game_log.csv."""
    _ensure_data_dir()

    if os.path.exists(GAME_LOG_FILE):
        existing = pd.read_csv(GAME_LOG_FILE)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["game_pk", "player_id"], keep="last"
        )
    else:
        combined = new_data

    combined = combined.sort_values("game_date").reset_index(drop=True)
    combined.to_csv(GAME_LOG_FILE, index=False)
    print(f"💾 Game log: {len(combined):,} rows ({combined['game_pk'].nunique()} games)")


def save_game_results(results_df: pd.DataFrame):
    """Merge new game results with existing game_results.csv."""
    _ensure_data_dir()

    if os.path.exists(GAME_RESULTS_FILE):
        existing = pd.read_csv(GAME_RESULTS_FILE)
        combined = pd.concat([existing, results_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["game_pk"], keep="last")
    else:
        combined = results_df

    combined = combined.sort_values("game_date").reset_index(drop=True)
    combined.to_csv(GAME_RESULTS_FILE, index=False)
    print(f"💾 Game results: {len(combined):,} games")


def collect_season(season: int, delay: float = 0.4):
    """Full collection pipeline for one season."""
    results_df = discover_season_games(season)
    if results_df.empty:
        print("No games found for this season!")
        return

    save_game_results(results_df)

    existing_pks = _load_existing_game_pks()
    new_pks = [
        gpk for gpk in results_df["game_pk"].tolist()
        if gpk not in existing_pks
    ]
    print(f"\n   Already have: {len(existing_pks)} games")
    print(f"   New to fetch: {len(new_pks)} games")

    if not new_pks:
        print("   Nothing new to collect! 🎉")
        return

    checkpoint_path = os.path.join(DATA_DIR, f"_checkpoint_{season}.csv")
    new_data = collect_boxscores(new_pks, delay=delay, checkpoint_path=checkpoint_path)

    if new_data.empty:
        return

    save_game_log(new_data)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("🧹 Cleaned up checkpoint file")


def collect_all(seasons: list[int] | None = None, delay: float = 0.4):
    """Collect data for multiple seasons."""
    if seasons is None:
        from src.config import CURRENT_SEASON
        seasons = [CURRENT_SEASON - 1, CURRENT_SEASON]

    print("⚾ HOMER TRACKER — Bulk Data Collection")
    print("=" * 50)
    print(f"   Seasons: {', '.join(str(s) for s in seasons)}")
    print(f"   API delay: {delay}s per call")

    start = time.time()
    for season in seasons:
        collect_season(season, delay=delay)

    elapsed = time.time() - start
    print(f"\n🎉 Done! Total time: {elapsed/60:.1f} minutes")

    if os.path.exists(GAME_LOG_FILE):
        gl = pd.read_csv(GAME_LOG_FILE)
        print(f"\n📊 Final dataset:")
        print(f"   Batter-game rows: {len(gl):,}")
        print(f"   Unique games:     {gl['game_pk'].nunique():,}")
        print(f"   Unique players:   {gl['player_id'].nunique():,}")
        print(f"   Date range:       {gl['game_date'].min()} → {gl['game_date'].max()}")


def main():
    """CLI entry point."""
    from src.config import CURRENT_SEASON
    seasons = None
    delay = 0.4

    args = sys.argv[1:]
    if "--season" in args:
        idx = args.index("--season")
        if idx + 1 < len(args):
            seasons = [int(args[idx + 1])]
    if "--fast" in args:
        delay = 0.25
    if "--slow" in args:
        delay = 0.6

    collect_all(seasons=seasons, delay=delay)


if __name__ == "__main__":
    main()
