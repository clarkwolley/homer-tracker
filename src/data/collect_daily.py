"""
Daily incremental data collector.

Instead of re-pulling entire seasons, this fetches ONLY games
from a specific date (default: yesterday). Designed to run
automatically every morning to keep the dataset fresh.

The full pipeline:
1. Fetch schedule for target date → discover completed game PKs
2. Skip any games already in our game_log.csv
3. Fetch boxscores for new games only
4. Append to game_log.csv and game_results.csv

Typical daily run: ~15 games × 2 API calls = ~30 calls = ~15 seconds.
Compare that to bulk collection: ~2,400 games × 2 = hours. 🐶

Usage:
    python -m src.data.collect_daily                 # Yesterday's games
    python -m src.data.collect_daily --date 2025-06-15   # Specific date
    python -m src.data.collect_daily --catchup 7     # Last 7 days
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from src.data import mlb_api
from src.data.collector import get_game_batter_stats, team_id_to_abbrev
from src.data.collect_bulk import (
    save_game_log,
    save_game_results,
    _load_existing_game_pks,
    DATA_DIR,
    _ensure_data_dir,
)


def discover_games_for_date(date_str: str) -> tuple[pd.DataFrame, list[int]]:
    """
    Discover completed games for a single date.

    Returns:
        Tuple of (game_results_df, list_of_game_pks)
    """
    schedule = mlb_api.get_schedule(date_str)
    games = {}

    for day in schedule.get("dates", []):
        if day["date"] != date_str:
            continue
        for game in day.get("games", []):
            gpk = game["gamePk"]
            state = game.get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue

            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_score = home.get("score", 0)
            away_score = away.get("score", 0)

            games[gpk] = {
                "game_pk": gpk,
                "game_date": date_str,
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

    results_df = pd.DataFrame(games.values())
    game_pks = list(games.keys())
    return results_df, game_pks


def collect_date(date_str: str, delay: float = 0.35) -> dict:
    """
    Collect all boxscore data for a single date.

    Returns:
        Dict with stats about what was collected.
    """
    print(f"\n📅 Collecting games for {date_str}...")

    results_df, game_pks = discover_games_for_date(date_str)
    if not game_pks:
        print(f"   No completed games for {date_str}")
        return {"date": date_str, "found": 0, "new": 0, "collected": 0}

    existing = _load_existing_game_pks()
    new_pks = [gpk for gpk in game_pks if gpk not in existing]

    print(f"   Found {len(game_pks)} completed games, {len(new_pks)} new")

    if not new_pks:
        print(f"   Already have all games for {date_str} ✅")
        # Still save results in case they were missing
        if not results_df.empty:
            save_game_results(results_df)
        return {"date": date_str, "found": len(game_pks), "new": 0, "collected": 0}

    # Fetch boxscores for new games
    all_frames = []
    failed = []

    for i, gpk in enumerate(new_pks, 1):
        try:
            df = get_game_batter_stats(gpk)
            all_frames.append(df)
        except Exception as e:
            failed.append(gpk)
            print(f"   ⚠️  Game {gpk}: {e}")

        if i % 5 == 0 or i == len(new_pks):
            print(f"   [{i}/{len(new_pks)}] boxscores fetched")

        time.sleep(delay)

    # Save results
    if not results_df.empty:
        save_game_results(results_df)

    if all_frames:
        new_data = pd.concat(all_frames, ignore_index=True)
        save_game_log(new_data)
        print(f"   ✅ Collected {len(new_data)} batter-game rows from {len(all_frames)} games")
    else:
        new_data = pd.DataFrame()

    if failed:
        print(f"   ⚠️  Failed: {len(failed)} games")

    return {
        "date": date_str,
        "found": len(game_pks),
        "new": len(new_pks),
        "collected": len(all_frames),
        "failed": len(failed),
        "batter_rows": len(new_data),
    }


def collect_yesterday() -> dict:
    """Collect yesterday's games. The most common daily operation."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return collect_date(yesterday)


def collect_catchup(days: int = 7) -> list[dict]:
    """
    Collect games from the last N days.

    Useful after a gap (vacation, server downtime, etc.).
    """
    print(f"\n🔄 Catching up on the last {days} days...")
    results = []
    for i in range(days, 0, -1):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        stats = collect_date(date_str)
        results.append(stats)

    total_new = sum(r.get("collected", 0) for r in results)
    total_rows = sum(r.get("batter_rows", 0) for r in results)
    print(f"\n🎉 Catchup complete: {total_new} new games, {total_rows} batter rows")
    return results


def games_since_last_train() -> int:
    """Count how many new games have been collected since the last model training."""
    import joblib

    model_meta_path = os.path.join(DATA_DIR, "..", "models", "hr_model_meta.pkl")
    game_log_path = os.path.join(DATA_DIR, "game_log.csv")

    if not os.path.exists(model_meta_path) or not os.path.exists(game_log_path):
        return 999  # No model trained yet — trigger retraining

    # Get model file modification time
    model_mtime = os.path.getmtime(model_meta_path)
    model_date = datetime.fromtimestamp(model_mtime).strftime("%Y-%m-%d")

    # Count games in log after model training date
    game_log = pd.read_csv(game_log_path, usecols=["game_pk", "game_date"])
    new_games = game_log[game_log["game_date"] > model_date]["game_pk"].nunique()

    return new_games


def days_since_last_train() -> int:
    """Days since the model was last trained."""
    model_meta_path = os.path.join(DATA_DIR, "..", "models", "hr_model_meta.pkl")

    if not os.path.exists(model_meta_path):
        return 999

    model_mtime = os.path.getmtime(model_meta_path)
    model_date = datetime.fromtimestamp(model_mtime)
    return (datetime.now() - model_date).days


def main():
    """CLI entry point."""
    args = sys.argv[1:]

    if "--date" in args:
        idx = args.index("--date")
        if idx + 1 < len(args):
            collect_date(args[idx + 1])
        else:
            print("Usage: --date YYYY-MM-DD")
    elif "--catchup" in args:
        idx = args.index("--catchup")
        days = int(args[idx + 1]) if idx + 1 < len(args) else 7
        collect_catchup(days)
    else:
        collect_yesterday()


if __name__ == "__main__":
    main()
