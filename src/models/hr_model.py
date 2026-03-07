"""
Home Run Prediction Model.

Binary classification: "Will this batter hit a HR tonight?"
~3% positive rate, so class balancing is critical.

We train Logistic Regression (baseline) and Gradient Boosting (advanced),
compare them, and save the winner.
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
from src.features.batter_features import (
    build_batter_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.models.evaluate import evaluate_model, print_evaluation


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def _inject_training_pitcher_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pitcher matchup features to training data.

    Uses league-average defaults for historical games since we can't
    retroactively look up who started each game from the boxscore alone.
    The model learns that these features have signal even with approximate
    values — the real juice comes at prediction time with actual starters.
    """
    result = df.copy()

    # Default pitcher features for training (league averages)
    result["opp_pitcher_era"] = 4.50
    result["opp_pitcher_whip"] = 1.30
    result["opp_pitcher_hr9"] = 1.30
    result["opp_pitcher_quality"] = 5.0
    result["platoon_advantage"] = 0.75  # average

    # Park factor from config
    from src.config import PARK_HR_FACTORS

    # For home games, use the batter's team park factor
    # For away games, use the opponent's park factor
    # Since we don't always have opponent in training data, use 1.0 default
    result["park_hr_factor"] = result["team"].map(
        lambda t: PARK_HR_FACTORS.get(t, 1.0)
    )

    return result


def prepare_training_data(game_log: pd.DataFrame) -> tuple:
    """
    Prepare features and target from raw game log.

    Returns:
        Tuple of (X, y, full_df)
    """
    df = build_batter_features(game_log)
    df = _inject_training_pitcher_features(df)

    # Drop rows with missing features (first appearances)
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df.dropna(subset=available_features)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y, df


def train_hr_model(game_log: pd.DataFrame) -> dict:
    """
    Train and evaluate HR prediction models.

    Pipeline:
    1. Build features from raw game log
    2. Split train/test
    3. Train LR (baseline) and GB (advanced)
    4. Evaluate and save the best
    """
    print("🏗️  Preparing features...")
    X, y, df = prepare_training_data(game_log)
    print(f"   {len(X)} samples, {len(FEATURE_COLUMNS)} features")
    print(f"   HR rate: {y.mean()*100:.1f}% of batter-games")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Logistic Regression ---
    print("\n🔵 Training Logistic Regression (baseline)...")
    lr = LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        max_iter=1000,
    )
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_probs, threshold=0.10)
    print_evaluation(lr_metrics, "Logistic Regression")
    results["logistic_regression"] = lr_metrics

    print("\n   Feature weights:")
    for feat, coef in sorted(
        zip(FEATURE_COLUMNS, lr.coef_[0]),
        key=lambda x: abs(x[1]), reverse=True,
    ):
        direction = "↑" if coef > 0 else "↓"
        print(f"     {direction} {feat}: {coef:+.3f}")

    # --- Gradient Boosting ---
    print("\n🟢 Training Gradient Boosting (advanced)...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    gb_metrics = evaluate_model(y_test, gb_probs, threshold=0.10)
    print_evaluation(gb_metrics, "Gradient Boosting")
    results["gradient_boosting"] = gb_metrics

    print("\n   Feature importance:")
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, gb.feature_importances_),
        key=lambda x: x[1], reverse=True,
    ):
        bar = "█" * int(imp * 50)
        print(f"     {feat:25s} {imp:.3f} {bar}")

    # --- Pick winner ---
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = lr if best_name == "logistic_regression" else gb
    best_scaler = scaler if best_name == "logistic_regression" else None

    print(f"\n🏆 Winner: {best_name} (AUC: {results[best_name]['roc_auc']:.3f})")

    _ensure_model_dir()
    joblib.dump(best_model, os.path.join(MODEL_DIR, "hr_model.pkl"))
    if best_scaler is not None:
        joblib.dump(best_scaler, os.path.join(MODEL_DIR, "hr_scaler.pkl"))
    joblib.dump(
        {"model_type": best_name, "metrics": results[best_name],
         "feature_columns": FEATURE_COLUMNS, "needs_scaling": best_scaler is not None},
        os.path.join(MODEL_DIR, "hr_model_meta.pkl"),
    )
    print(f"💾 Model saved!")

    return {"best_model_name": best_name, "results": results,
            "model": best_model, "scaler": best_scaler}


def load_hr_model() -> tuple:
    """Load a trained HR model. Returns (model, scaler_or_None, metadata)."""
    model_path = os.path.join(MODEL_DIR, "hr_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained HR model. Run: python -m src.train")

    model = joblib.load(model_path)
    meta = joblib.load(os.path.join(MODEL_DIR, "hr_model_meta.pkl"))
    needs_scaling = meta.get("needs_scaling", False)
    scaler_path = os.path.join(MODEL_DIR, "hr_scaler.pkl")
    scaler = joblib.load(scaler_path) if needs_scaling and os.path.exists(scaler_path) else None

    return model, scaler, meta


def predict_hr_probability(model, scaler, features: pd.DataFrame) -> np.ndarray:
    """Predict HR probability for batters. Returns array of probabilities."""
    X = features[FEATURE_COLUMNS].copy()
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]
