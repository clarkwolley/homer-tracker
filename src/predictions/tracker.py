"""
Prediction tracker — saves picks and grades them against actual results.
"""

import os
from datetime import datetime

import pandas as pd
import numpy as np

from src.data import mlb_api
from src.data.collector import get_game_batter_stats


TRACKER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
PICKS_FILE = os.path.join(TRACKER_DIR, "picks_ledger.csv")
GRADED_FILE = os.path.join(TRACKER_DIR, "graded_ledger.csv")


def save_predictions(pred_df: pd.DataFrame):
    """Save today's predictions to the running ledger."""
    save_cols = [
        "player_id", "name", "team", "opponent", "position",
        "is_home", "hr_probability", "rolling_hr_avg",
        "rolling_ops", "season_hr", "season_gp",
        "hr_streak", "hr_drought", "is_hot", "is_power_hitter",
        "opp_pitcher_era", "opp_pitcher_hr9", "opp_pitcher_name",
        "park_hr_factor", "platoon_advantage",
    ]
    available_cols = [c for c in save_cols if c in pred_df.columns]
    picks = pred_df[available_cols].copy()
    picks["prediction_date"] = datetime.now().strftime("%Y-%m-%d")
    picks["predicted_at"] = datetime.now().isoformat()

    if os.path.exists(PICKS_FILE):
        existing = pd.read_csv(PICKS_FILE)
        existing = existing[existing["prediction_date"] != picks["prediction_date"].iloc[0]]
        combined = pd.concat([existing, picks], ignore_index=True)
    else:
        combined = picks

    combined.to_csv(PICKS_FILE, index=False)
    print(f"💾 Saved {len(picks)} predictions to ledger ({len(combined)} total rows)")


def grade_predictions(date_str: str) -> pd.DataFrame:
    """Grade predictions for a specific date against actual HR results."""
    if not os.path.exists(PICKS_FILE):
        print("No predictions to grade!")
        return pd.DataFrame()

    ledger = pd.read_csv(PICKS_FILE)
    day_picks = ledger[ledger["prediction_date"] == date_str].copy()

    if day_picks.empty:
        print(f"No predictions found for {date_str}")
        return pd.DataFrame()

    print(f"📊 Fetching actual results for {date_str}...")
    schedule = mlb_api.get_schedule(date_str)
    game_pks = []
    for day in schedule.get("dates", []):
        if day["date"] != date_str:
            continue
        for game in day.get("games", []):
            state = game.get("status", {}).get("abstractGameState", "")
            if state == "Final":
                game_pks.append(game["gamePk"])

    if not game_pks:
        print(f"No completed games for {date_str}")
        return pd.DataFrame()

    actual_frames = []
    for gpk in game_pks:
        try:
            gdf = get_game_batter_stats(gpk)
            actual_frames.append(gdf)
        except Exception as e:
            print(f"  ⚠️  Failed game {gpk}: {e}")

    if not actual_frames:
        return pd.DataFrame()

    actuals = pd.concat(actual_frames, ignore_index=True)
    actuals["actual_hr"] = actuals["home_runs"]
    actuals["actual_hit_hr"] = (actuals["home_runs"] > 0).astype(int)

    graded = day_picks.merge(
        actuals[["player_id", "actual_hr", "actual_hit_hr"]],
        on="player_id", how="left",
    )

    graded["played"] = graded["actual_hr"].notna().astype(int)
    graded["actual_hr"] = graded["actual_hr"].fillna(0).astype(int)
    graded["actual_hit_hr"] = graded["actual_hit_hr"].fillna(0).astype(int)

    # Threshold for "predicted HR" — top picks
    graded["predicted_hr"] = (graded["hr_probability"] >= 0.10).astype(int)
    graded["correct"] = (graded["predicted_hr"] == graded["actual_hit_hr"]).astype(int)
    graded["hit"] = ((graded["predicted_hr"] == 1) & (graded["actual_hit_hr"] == 1)).astype(int)

    return graded


def save_graded(graded: pd.DataFrame):
    """Append graded results to the graded ledger."""
    if graded.empty:
        return

    if os.path.exists(GRADED_FILE):
        existing = pd.read_csv(GRADED_FILE)
        date = graded["prediction_date"].iloc[0]
        existing = existing[existing["prediction_date"] != date]
        combined = pd.concat([existing, graded], ignore_index=True)
    else:
        combined = graded

    combined.to_csv(GRADED_FILE, index=False)
    print(f"💾 Saved graded results ({len(combined)} total rows)")


def print_scorecard(graded: pd.DataFrame):
    """Pretty-print grading results."""
    if graded.empty:
        return

    date = graded["prediction_date"].iloc[0]
    played = graded[graded["played"] == 1]

    print(f"\n{'='*65}")
    print(f"📋 SCORECARD — {date}")
    print(f"{'='*65}")

    total = len(played)
    actual_hr = int(played["actual_hit_hr"].sum())
    predicted_hr = int(played["predicted_hr"].sum())
    hits = int(played["hit"].sum())

    print(f"  Batters tracked:     {total}")
    print(f"  Actually homered:    {actual_hr}")
    print(f"  We predicted HR:     {predicted_hr}")
    print(f"  Hits (predicted + HR): {hits}/{predicted_hr} "
          f"({hits/max(predicted_hr,1)*100:.1f}% precision)")

    # Top hits
    print(f"\n  ✅ TOP HITS:")
    top_hits = played[played["actual_hit_hr"] == 1].nlargest(10, "hr_probability")
    for _, r in top_hits.iterrows():
        print(f"    ✅ {r['name']} ({r['team']}) — "
              f"{r['hr_probability']*100:.0f}% predicted, {int(r['actual_hr'])} HR(s)")

    print(f"{'='*65}")


def run_grading(date_str: str):
    """Full grading pipeline."""
    graded = grade_predictions(date_str)
    if not graded.empty:
        print_scorecard(graded)
        save_graded(graded)
    return graded


def lifetime_stats():
    """Print running stats across all graded predictions."""
    if not os.path.exists(GRADED_FILE):
        print("No graded predictions yet!")
        return

    df = pd.read_csv(GRADED_FILE)
    played = df[df["played"] == 1]
    dates = played["prediction_date"].nunique()

    print(f"\n{'='*65}")
    print(f"📈 LIFETIME MODEL PERFORMANCE")
    print(f"{'='*65}")
    print(f"  Days tracked:    {dates}")
    print(f"  Total picks:     {len(played)}")
    print(f"  Actual HRs:      {int(played['actual_hit_hr'].sum())}")

    predicted = played[played["predicted_hr"] == 1]
    if len(predicted) > 0:
        precision = predicted["actual_hit_hr"].mean() * 100
        print(f"  Predicted HRs:   {len(predicted)}")
        print(f"  Hit rate:        {precision:.1f}%")

    brier = ((played["hr_probability"] - played["actual_hit_hr"]) ** 2).mean()
    print(f"  Brier score:     {brier:.4f}")
    print(f"{'='*65}")
