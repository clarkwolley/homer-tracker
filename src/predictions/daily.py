"""
Daily home run and game winner predictions.

Full daily flow:
1. Grade yesterday's picks (scorecard + save to graded ledger)
2. Pull tonight's schedule and build features for every batter
3. Run the trained model and rank players by HR probability
4. Predict game winners
5. Save picks and generate HTML report
"""

import pandas as pd
import numpy as np
from tabulate import tabulate

from src.data import mlb_api
from src.data.collector import (
    get_todays_games,
    get_team_batters,
    team_abbrev_to_id,
)
from src.data.history import load_game_data
from src.features.batter_features import build_batter_features, FEATURE_COLUMNS
from src.features.pitcher_features import (
    build_pitcher_matchup_features,
    calc_platoon_advantage,
)
from src.models.hr_model import load_hr_model, predict_hr_probability
from src.config import PARK_HR_FACTORS


def _get_teams_playing_today() -> list[dict]:
    """Get all teams playing in today's games."""
    schedule = get_todays_games()
    if schedule.empty:
        return []

    today = schedule[schedule["date"] == schedule["date"].min()]
    game_date = today["date"].iloc[0] if not today.empty else ""
    teams = []
    for _, game in today.iterrows():
        teams.append({
            "team": game["home_team"], "team_id": game["home_id"],
            "is_home": True, "opponent": game["away_team"],
            "opp_id": game["away_id"], "game_pk": game["game_pk"],
            "game_date": game_date,
        })
        teams.append({
            "team": game["away_team"], "team_id": game["away_id"],
            "is_home": False, "opponent": game["home_team"],
            "opp_id": game["home_id"], "game_pk": game["game_pk"],
            "game_date": game_date,
        })
    return teams


def _build_prediction_features(teams: list[dict], game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features for all batters on tonight's teams.

    Strategy:
    1. Get season stats for each team's roster
    2. Look up each batter's recent game log (rolling averages + streaks)
    3. Inject pitcher matchup features
    4. Inject park HR factor
    """
    featured = build_batter_features(game_log)

    # Get latest rolling features per player
    latest = (
        featured
        .sort_values("game_date")
        .groupby("player_id")
        .last()
        .reset_index()
    )

    # Pre-fetch pitcher matchup data per opponent
    opponents = set(t["opponent"] for t in teams)
    pitcher_cache = {}
    game_date = teams[0]["game_date"] if teams else "today"
    for opp in opponents:
        try:
            pitcher_cache[opp] = build_pitcher_matchup_features(opp, game_date)
        except Exception as e:
            print(f"  ⚠️  Could not fetch pitcher for {opp}: {e}")
            pitcher_cache[opp] = {
                "opp_pitcher_era": 4.50, "opp_pitcher_whip": 1.30,
                "opp_pitcher_hr9": 1.30, "opp_pitcher_quality": 5.0,
                "opp_pitcher_name": "TBD", "opp_pitcher_hand": "R",
            }

    all_predictions = []

    for team_info in teams:
        team = team_info["team"]
        team_id = team_info["team_id"]
        is_home = team_info["is_home"]
        opponent = team_info["opponent"]
        pitcher_feats = pitcher_cache.get(opponent, {})

        # Park factor — use home team's park
        home_team = team if is_home else opponent
        park_factor = PARK_HR_FACTORS.get(home_team, 1.0)

        try:
            roster = get_team_batters(team_id)
        except Exception as e:
            print(f"  ⚠️  Could not fetch roster for {team}: {e}")
            continue

        for _, player in roster.iterrows():
            pid = player["player_id"]
            player_history = latest[latest["player_id"] == pid]

            if player_history.empty:
                # No game log — use season averages as fallback
                gp = max(player["games_played"], 1)
                ab = max(player["at_bats"], 1)
                row = {
                    "player_id": pid,
                    "name": player["name"],
                    "team": team,
                    "opponent": opponent,
                    "position": player["position"],
                    "is_home": int(is_home),
                    "rolling_hr_avg": player["home_runs"] / gp,
                    "rolling_hits_avg": player["hits"] / gp,
                    "rolling_ab_avg": player["at_bats"] / gp,
                    "rolling_slg": player["slg"],
                    "rolling_iso": player["slg"] - player["avg"],
                    "rolling_ops": player["ops"],
                    "games_in_window": min(gp, 15),
                    "hr_streak": 0,
                    "hit_streak": 0,
                    "hr_drought": 0,
                    "is_hot": 0,
                    "hr_per_ab": player["home_runs"] / ab,
                    "is_power_hitter": int(player["home_runs"] / ab > 0.06),
                    "ab_per_hr": ab / max(player["home_runs"], 1),
                    "bats_left": int(player.get("bat_side", "R") == "L"),
                    "bats_right": int(player.get("bat_side", "R") == "R"),
                    "bats_switch": int(player.get("bat_side", "R") == "S"),
                    "season_hr": player["home_runs"],
                    "season_gp": player["games_played"],
                    "season_avg": player["avg"],
                    "season_ops": player["ops"],
                }
            else:
                h = player_history.iloc[0]
                row = {
                    "player_id": pid,
                    "name": player["name"],
                    "team": team,
                    "opponent": opponent,
                    "position": player["position"],
                    "is_home": int(is_home),
                    "rolling_hr_avg": h.get("rolling_hr_avg", 0),
                    "rolling_hits_avg": h.get("rolling_hits_avg", 0),
                    "rolling_ab_avg": h.get("rolling_ab_avg", 0),
                    "rolling_slg": h.get("rolling_slg", 0),
                    "rolling_iso": h.get("rolling_iso", 0),
                    "rolling_ops": h.get("rolling_ops", 0),
                    "games_in_window": h.get("games_in_window", 1),
                    "hr_streak": int(h.get("hr_streak", 0)),
                    "hit_streak": int(h.get("hit_streak", 0)),
                    "hr_drought": int(h.get("hr_drought", 0)),
                    "is_hot": int(h.get("is_hot", 0)),
                    "hr_per_ab": h.get("hr_per_ab", 0),
                    "is_power_hitter": int(h.get("is_power_hitter", 0)),
                    "ab_per_hr": h.get("ab_per_hr", 99),
                    "bats_left": int(h.get("bats_left", 0)),
                    "bats_right": int(h.get("bats_right", 0)),
                    "bats_switch": int(h.get("bats_switch", 0)),
                    "season_hr": player["home_runs"],
                    "season_gp": player["games_played"],
                    "season_avg": player["avg"],
                    "season_ops": player["ops"],
                }

            # Inject pitcher matchup
            row["opp_pitcher_era"] = pitcher_feats.get("opp_pitcher_era", 4.50)
            row["opp_pitcher_whip"] = pitcher_feats.get("opp_pitcher_whip", 1.30)
            row["opp_pitcher_hr9"] = pitcher_feats.get("opp_pitcher_hr9", 1.30)
            row["opp_pitcher_quality"] = pitcher_feats.get("opp_pitcher_quality", 5.0)
            row["opp_pitcher_name"] = pitcher_feats.get("opp_pitcher_name", "TBD")

            # Platoon advantage
            bat_side = "L" if row["bats_left"] else ("S" if row["bats_switch"] else "R")
            pitch_hand = pitcher_feats.get("opp_pitcher_hand", "R")
            row["platoon_advantage"] = calc_platoon_advantage(bat_side, pitch_hand)

            # Park factor
            row["park_hr_factor"] = park_factor

            all_predictions.append(row)

    return pd.DataFrame(all_predictions)


def predict_tonight() -> pd.DataFrame:
    """Generate HR predictions for tonight's games."""
    print("⚾ HOMER TRACKER — Tonight's Home Run Predictions")
    print("=" * 55)

    teams = _get_teams_playing_today()
    if not teams:
        print("No games scheduled today!")
        return pd.DataFrame()

    team_abbrevs = sorted(set(t["team"] for t in teams))
    print(f"\n📅 Teams playing: {', '.join(team_abbrevs)}")

    print("🤖 Loading model...")
    model, scaler, meta = load_hr_model()
    print(f"   Model type: {meta['model_type']}")
    print(f"   Training AUC: {meta['metrics']['roc_auc']:.3f}")

    print("⚙️  Building features...")
    game_log = load_game_data()
    pred_df = _build_prediction_features(teams, game_log)
    print(f"   {len(pred_df)} batters across {len(team_abbrevs)} teams")

    pred_df[FEATURE_COLUMNS] = pred_df[FEATURE_COLUMNS].fillna(0)

    print("🎯 Running predictions...\n")
    pred_df["hr_probability"] = predict_hr_probability(model, scaler, pred_df)
    pred_df = pred_df.sort_values("hr_probability", ascending=False)

    return pred_df


def print_top_picks(pred_df: pd.DataFrame, top_n: int = 25):
    """Pretty-print the top predicted HR hitters."""
    if pred_df.empty:
        print("No predictions available.")
        return

    print(f"\n💣 TOP {top_n} MOST LIKELY HOME RUN HITTERS TONIGHT")
    print("=" * 75)

    display = pred_df.head(top_n).copy()
    display["prob_%"] = (display["hr_probability"] * 100).round(1)
    display["matchup"] = display.apply(
        lambda r: f"{'vs' if r['is_home'] else '@'} {r['opponent']}", axis=1
    )
    display["streak"] = display.apply(
        lambda r: f"🔥{int(r.get('hr_streak', 0))}" if r.get("is_hot", 0)
        else (f"❄️{int(r.get('hr_drought', 0))}" if r.get("hr_drought", 0) >= 10 else ""),
        axis=1,
    )

    cols = ["name", "team", "position", "matchup", "prob_%", "streak",
            "season_hr", "season_ops", "opp_pitcher_name", "park_hr_factor"]
    headers = ["Player", "Team", "Pos", "Matchup", "HR%", "Streak",
               "Season HR", "OPS", "vs Pitcher", "Park"]

    print(tabulate(
        display[cols].values, headers=headers, tablefmt="simple",
        floatfmt=(".0f", ".0f", ".0f", ".0f", ".1f", ".0f", ".0f", ".3f", ".0f", ".2f"),
    ))


def predict_game_winners() -> pd.DataFrame:
    """Predict winners for tonight's games."""
    from src.models.game_model import load_game_model, predict_game_winner
    from src.data.collector import get_standings_df
    from src.features.team_features import build_team_strength

    schedule = get_todays_games()
    if schedule.empty:
        return pd.DataFrame()

    today = schedule[schedule["date"] == schedule["date"].min()]

    try:
        model, scaler, meta = load_game_model()
    except FileNotFoundError:
        print("⚠️  No game winner model trained yet.")
        return pd.DataFrame()

    standings = get_standings_df()
    strength = build_team_strength(standings)

    rows = []
    for _, game in today.iterrows():
        prob = predict_game_winner(game["home_team"], game["away_team"], model, scaler)
        winner = game["home_team"] if prob > 0.5 else game["away_team"]
        confidence = max(prob, 1 - prob) * 100

        rows.append({
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_win_prob": round(prob * 100, 1),
            "away_win_prob": round((1 - prob) * 100, 1),
            "predicted_winner": winner,
            "confidence": round(confidence, 1),
        })

    return pd.DataFrame(rows)


def print_game_picks(game_df: pd.DataFrame):
    """Pretty-print game winner predictions."""
    if game_df.empty:
        return

    print(f"\n\n🏆 GAME WINNER PREDICTIONS")
    print("=" * 70)

    for _, g in game_df.iterrows():
        conf = g["confidence"]
        conf_bar = "🟢" if conf >= 60 else "🟡" if conf >= 55 else "⚪"
        venue = "🏠" if g["home_win_prob"] > 50 else "✈️"

        print(f"  {g['away_team']} @ {g['home_team']}")
        print(f"    → {conf_bar} {venue} {g['predicted_winner']} ({conf:.1f}%)")
        print()


def grade_yesterday():
    """Grade yesterday's predictions and print the scorecard."""
    from datetime import datetime, timedelta
    from src.predictions.tracker import grade_predictions, save_graded, print_scorecard

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n📊 Grading yesterday's picks ({yesterday})...")

    graded = grade_predictions(yesterday)
    if graded.empty:
        print("   No predictions to grade for yesterday.")
        return None

    print_scorecard(graded)
    save_graded(graded)
    return graded


def run():
    """Main entry point — grades yesterday, then predicts tonight."""
    # Step 1: How'd we do yesterday?
    grade_yesterday()

    # Step 2: Tonight's predictions
    pred_df = predict_tonight()
    if not pred_df.empty:
        print_top_picks(pred_df)

    game_df = predict_game_winners()
    if not game_df.empty:
        print_game_picks(game_df)

    if not pred_df.empty:
        from src.predictions.tracker import save_predictions
        save_predictions(pred_df)

        from src.predictions.report import generate_html_report
        report_path = generate_html_report(pred_df, game_df=game_df)
        print(f"\n🌐 Open in browser: file://{report_path}")

    return pred_df


if __name__ == "__main__":
    run()
