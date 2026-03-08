"""
Smart model retraining — only retrains when it makes sense.

Triggers:
1. Too many new games since last training (RETRAIN_INTERVAL_GAMES)
2. Too many days since last training (RETRAIN_INTERVAL_DAYS)
3. No model exists at all

This avoids wasting time retraining daily when nothing has changed,
but also prevents the model from going stale mid-season.
"""

import os
import logging

from src.config import RETRAIN_INTERVAL_GAMES, RETRAIN_INTERVAL_DAYS
from src.data.collect_daily import games_since_last_train, days_since_last_train


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")


def should_retrain() -> tuple[bool, str]:
    """
    Check if models should be retrained.

    Returns:
        Tuple of (should_retrain: bool, reason: str)
    """
    hr_model = os.path.join(MODEL_DIR, "hr_model.pkl")

    if not os.path.exists(hr_model):
        return True, "no model exists"

    new_games = games_since_last_train()
    if new_games >= RETRAIN_INTERVAL_GAMES:
        return True, f"{new_games} new games since last training (threshold: {RETRAIN_INTERVAL_GAMES})"

    days = days_since_last_train()
    if days >= RETRAIN_INTERVAL_DAYS:
        return True, f"{days} days since last training (threshold: {RETRAIN_INTERVAL_DAYS})"

    return False, f"model is fresh ({new_games} new games, {days} days old)"


def retrain_if_needed(log: logging.Logger | None = None) -> bool:
    """
    Retrain models if the trigger conditions are met.

    Returns:
        True if retraining was performed.
    """
    do_retrain, reason = should_retrain()

    msg = f"🔄 Retrain check: {reason}"
    if log:
        log.info(msg)
    else:
        print(msg)

    if not do_retrain:
        return False

    msg = "🏗️  Retraining models..."
    if log:
        log.info(msg)
    else:
        print(msg)

    try:
        from src.train import train_hr_model, train_game_model

        hr_result = train_hr_model()
        if hr_result:
            auc = hr_result["results"][hr_result["best_model_name"]]["roc_auc"]
            msg = f"   ✅ HR model retrained (AUC: {auc:.3f})"
            if log:
                log.info(msg)
            else:
                print(msg)

        game_result = train_game_model()
        if game_result:
            auc = game_result["results"][game_result["best_model_name"]]["roc_auc"]
            msg = f"   ✅ Game model retrained (AUC: {auc:.3f})"
            if log:
                log.info(msg)
            else:
                print(msg)

        return True

    except Exception as e:
        msg = f"   ❌ Retraining failed: {e}"
        if log:
            log.error(msg, exc_info=True)
        else:
            print(msg)
        return False
