"""
Game Winner Prediction Model.

Binary classification: "Will the home team win?"
Baseline to beat: always pick home team = ~54% accuracy (MLB).

Uses recency weighting so recent season results carry more influence
than games from 3+ years ago.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import TEST_SIZE, RANDOM_STATE
from src.data.collector import get_standings_df
from src.features.team_features import (
    build_team_strength,
    build_matchup_features,
    GAME_FEATURE_COLUMNS,
)
from src.models.evaluate import evaluate_model, print_evaluation
from src.models.weights import compute_recency_weights, print_weight_summary


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def prepare_game_training_data(game_results: pd.DataFrame) -> tuple:
    """
    Build features for historical games using current standings.

    Returns:
        Tuple of (X, y, sample_weights)
    """
    standings = get_standings_df()
    strength = build_team_strength(standings)

    rows = []
    targets = []
    game_dates = []

    for _, game in game_results.iterrows():
        features = build_matchup_features(
            game["home_team"], game["away_team"], strength
        )
        if not features:
            continue
        rows.append(features)
        targets.append(game["home_win"])
        game_dates.append(game["game_date"])

    X = pd.DataFrame(rows)[GAME_FEATURE_COLUMNS]
    y = pd.Series(targets, name="home_win")
    dates = pd.Series(game_dates, name="game_date")

    sample_weights = compute_recency_weights(dates)

    return X, y, sample_weights, dates


def train_game_model(game_results: pd.DataFrame) -> dict:
    """Train and evaluate game winner models with recency weighting."""
    print("🏗️  Preparing game features...")
    X, y, sample_weights, dates = prepare_game_training_data(game_results)
    print(f"   {len(X)} games, {len(GAME_FEATURE_COLUMNS)} features")
    print(f"   Home win rate: {y.mean()*100:.1f}%")

    print_weight_summary(dates, sample_weights)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    print("\n🔵 Training Logistic Regression...")
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr.fit(X_train_scaled, y_train, sample_weight=w_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_probs, threshold=0.5)
    print_evaluation(lr_metrics, "Logistic Regression (Game Winner)")
    results["logistic_regression"] = lr_metrics

    # Gradient Boosting
    print("\n🟢 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train, sample_weight=w_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_metrics = evaluate_model(y_test, gb_probs, threshold=0.5)
    print_evaluation(gb_metrics, "Gradient Boosting (Game Winner)")
    results["gradient_boosting"] = gb_metrics

    # Pick winner
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = lr if best_name == "logistic_regression" else gb
    best_scaler = scaler if best_name == "logistic_regression" else None

    print(f"\n🏆 Winner: {best_name} (AUC: {results[best_name]['roc_auc']:.3f})")

    _ensure_model_dir()
    joblib.dump(best_model, os.path.join(MODEL_DIR, "game_model.pkl"))
    if best_scaler is not None:
        joblib.dump(best_scaler, os.path.join(MODEL_DIR, "game_scaler.pkl"))
    joblib.dump(
        {"model_type": best_name, "metrics": results[best_name],
         "feature_columns": GAME_FEATURE_COLUMNS,
         "needs_scaling": best_scaler is not None},
        os.path.join(MODEL_DIR, "game_model_meta.pkl"),
    )
    print("💾 Model saved!")

    return {"best_model_name": best_name, "results": results,
            "model": best_model, "scaler": best_scaler}


def load_game_model() -> tuple:
    """Load a trained game winner model."""
    model_path = os.path.join(MODEL_DIR, "game_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained game model. Run: python -m src.train")

    model = joblib.load(model_path)
    meta = joblib.load(os.path.join(MODEL_DIR, "game_model_meta.pkl"))
    needs_scaling = meta.get("needs_scaling", False)
    scaler_path = os.path.join(MODEL_DIR, "game_scaler.pkl")
    scaler = joblib.load(scaler_path) if needs_scaling and os.path.exists(scaler_path) else None
    return model, scaler, meta


def predict_game_winner(home_team: str, away_team: str, model=None, scaler=None) -> float:
    """Predict probability that the home team wins."""
    if model is None:
        model, scaler, _ = load_game_model()

    standings = get_standings_df()
    strength = build_team_strength(standings)
    features = build_matchup_features(home_team, away_team, strength)

    if not features:
        return 0.5

    X = pd.DataFrame([features])[GAME_FEATURE_COLUMNS]
    if scaler is not None:
        X = scaler.transform(X)

    return model.predict_proba(X)[0, 1]
