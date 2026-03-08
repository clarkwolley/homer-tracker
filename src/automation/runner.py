"""
Daily automation runner for Homer Tracker.

The full daily pipeline (runs every morning):
1. COLLECT  — fetch yesterday's boxscores (incremental, ~15 seconds)
2. GRADE    — score yesterday's predictions vs actual HR results
3. RETRAIN  — retrain models if stale (every ~75 games or 7 days)
4. PREDICT  — generate tonight's HR picks + game winner predictions
5. NOTIFY   — deliver report via email + Telegram

Usage:
    python -m src.automation.runner              # Full daily pipeline
    python -m src.automation.runner --predict    # Predictions only
    python -m src.automation.runner --grade      # Grade yesterday only
    python -m src.automation.runner --collect    # Collect data only
    python -m src.automation.runner --retrain    # Force retrain
    python -m src.automation.runner --catchup 7  # Collect last N days
    python -m src.automation.runner --notify     # Re-send latest report
    python -m src.automation.runner --status     # Check system status
"""

import os
import sys
import glob
import logging
from datetime import datetime, timedelta


LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")


def _setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("homer-tracker")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(LOG_DIR, "runner.log"))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)

    return logger


# --- Pipeline steps ----------------------------------------------------------


def step_collect(log: logging.Logger, catchup_days: int = 1) -> None:
    """Collect recent game data (default: yesterday only)."""
    from src.data.collect_daily import collect_yesterday, collect_catchup

    if catchup_days > 1:
        log.info(f"📦 Collecting last {catchup_days} days of games...")
        collect_catchup(catchup_days)
    else:
        log.info("📦 Collecting yesterday's games...")
        stats = collect_yesterday()
        log.info(
            f"   Found {stats.get('found', 0)} games, "
            f"{stats.get('new', 0)} new, "
            f"{stats.get('collected', 0)} collected"
        )


def step_grade(log: logging.Logger) -> None:
    """Grade yesterday's predictions against actual results."""
    from src.predictions.tracker import grade_predictions, save_graded, print_scorecard
    from src.notifications.telegram_sender import send_grade

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    log.info(f"📊 Grading yesterday's predictions ({yesterday})...")

    graded = grade_predictions(yesterday)
    if graded.empty:
        log.info("   No predictions to grade for yesterday.")
        return

    print_scorecard(graded)
    save_graded(graded)

    try:
        send_grade(graded)
    except Exception as e:
        log.warning(f"   Telegram grade notification failed: {e}")


def step_retrain(log: logging.Logger, force: bool = False) -> None:
    """Retrain models if needed (or forced)."""
    from src.automation.retrain import retrain_if_needed, should_retrain

    if force:
        log.info("🏗️  Forced retraining...")
        from src.train import train_hr_model, train_game_model
        train_hr_model()
        train_game_model()
        log.info("   ✅ Retraining complete")
    else:
        retrain_if_needed(log)


def step_predict(log: logging.Logger) -> tuple:
    """Generate tonight's predictions."""
    from src.predictions.daily import (
        predict_tonight, print_top_picks,
        predict_game_winners, print_game_picks,
    )
    from src.predictions.tracker import save_predictions
    from src.predictions.report import generate_html_report

    log.info("⚾ Generating today's predictions...")

    pred_df = predict_tonight()
    if pred_df.empty:
        log.info("   No games scheduled today.")
        return None, None, None

    print_top_picks(pred_df)

    game_df = predict_game_winners()
    if not game_df.empty:
        print_game_picks(game_df)

    save_predictions(pred_df)
    report_path = generate_html_report(pred_df, game_df=game_df)
    log.info(f"   Report saved: {report_path}")

    return pred_df, game_df, report_path


def step_notify(log: logging.Logger, pred_df=None, report_path=None) -> None:
    """Deliver predictions via email + Telegram."""
    from src.notifications.email_sender import send_report
    from src.notifications.telegram_sender import send_picks
    from src.notifications.settings import is_email_configured, is_telegram_configured

    if pred_df is None or (hasattr(pred_df, "empty") and pred_df.empty):
        log.info("   No predictions to deliver.")
        return

    if report_path is None:
        report_path = _find_latest_report()

    log.info("📬 Delivering results...")

    if is_email_configured():
        try:
            send_report(report_path)
        except Exception as e:
            log.error(f"   Email failed: {e}")
    else:
        log.info("   ⏭️  Email not configured.")

    if is_telegram_configured():
        try:
            send_picks(pred_df)
        except Exception as e:
            log.error(f"   Telegram failed: {e}")
    else:
        log.info("   ⏭️  Telegram not configured.")


# --- Utilities ---------------------------------------------------------------


def _find_latest_report() -> str | None:
    pattern = os.path.join(REPORT_DIR, "picks_*.html")
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def check_status() -> None:
    """Print full system status."""
    from src.notifications.settings import is_email_configured, is_telegram_configured
    from src.data.collect_daily import games_since_last_train, days_since_last_train
    from src.config import RETRAIN_INTERVAL_GAMES, RETRAIN_INTERVAL_DAYS

    print("\n🐶 Homer Tracker — System Status")
    print("=" * 55)

    # Notification config
    print(f"  📧 Email:      {'✅ configured' if is_email_configured() else '❌ not configured'}")
    print(f"  💬 Telegram:   {'✅ configured' if is_telegram_configured() else '❌ not configured'}")

    # Models
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hr_model.pkl")
    game_model = os.path.join(os.path.dirname(__file__), "..", "..", "models", "game_model.pkl")
    print(f"  🤖 HR Model:   {'✅ trained' if os.path.exists(model_path) else '❌ not trained'}")
    print(f"  🏆 Game Model: {'✅ trained' if os.path.exists(game_model) else '❌ not trained'}")

    # Data
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "game_log.csv")
    if os.path.exists(data_path):
        import pandas as pd
        gl = pd.read_csv(data_path, usecols=["game_pk", "game_date"])
        n_games = gl["game_pk"].nunique()
        date_range = f"{gl['game_date'].min()} → {gl['game_date'].max()}"
        print(f"  📊 Data:       ✅ {n_games:,} games ({date_range})")
    else:
        print(f"  📊 Data:       ❌ no data")

    # Freshness
    if os.path.exists(model_path):
        new_games = games_since_last_train()
        days_old = days_since_last_train()
        fresh = new_games < RETRAIN_INTERVAL_GAMES and days_old < RETRAIN_INTERVAL_DAYS
        icon = "✅" if fresh else "🟡"
        print(f"  🔄 Freshness:  {icon} {new_games} new games, {days_old}d since training")
        print(f"                 (retrain at {RETRAIN_INTERVAL_GAMES} games or {RETRAIN_INTERVAL_DAYS} days)")

    # Latest report
    latest = _find_latest_report()
    print(f"  📄 Latest:     {os.path.basename(latest) if latest else 'no reports yet'}")

    print("=" * 55)
    if not is_email_configured() or not is_telegram_configured():
        print("\n  💡 Run: python -m src.automation.setup\n")


# --- Main entrypoint ---------------------------------------------------------


def run(mode: str = "full", catchup_days: int = 1) -> None:
    """Execute the daily pipeline."""
    if mode == "status":
        check_status()
        return

    log = _setup_logging()
    log.info(f"{'=' * 55}")
    log.info(f"⚾ Homer Tracker Runner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"   Mode: {mode}")

    try:
        # Step 1: Collect new data
        if mode in ("full", "collect", "catchup"):
            step_collect(log, catchup_days=catchup_days)

        # Step 2: Grade yesterday
        if mode in ("full", "grade"):
            step_grade(log)

        # Step 3: Retrain if needed
        if mode in ("full", "retrain"):
            force = mode == "retrain"
            step_retrain(log, force=force)

        # Step 4: Predict tonight
        pred_df = None
        report_path = None
        if mode in ("full", "predict"):
            pred_df, _, report_path = step_predict(log)

        # Step 5: Notify
        if mode in ("full", "predict", "notify"):
            if mode == "notify":
                from src.predictions.tracker import PICKS_FILE
                import pandas as pd
                if os.path.exists(PICKS_FILE):
                    ledger = pd.read_csv(PICKS_FILE)
                    today = datetime.now().strftime("%Y-%m-%d")
                    pred_df = ledger[ledger["prediction_date"] == today]
                report_path = _find_latest_report()

            step_notify(log, pred_df, report_path)

        log.info("✅ Done!")

    except Exception as e:
        log.error(f"❌ Runner failed: {e}", exc_info=True)
        raise


def main():
    mode = "full"
    catchup_days = 1

    args = sys.argv[1:]

    if "--status" in args:
        mode = "status"
    elif "--predict" in args:
        mode = "predict"
    elif "--grade" in args:
        mode = "grade"
    elif "--collect" in args:
        mode = "collect"
    elif "--retrain" in args:
        mode = "retrain"
    elif "--notify" in args:
        mode = "notify"
    elif "--catchup" in args:
        mode = "catchup"
        idx = args.index("--catchup")
        if idx + 1 < len(args):
            catchup_days = int(args[idx + 1])
        else:
            catchup_days = 7

    run(mode, catchup_days=catchup_days)


if __name__ == "__main__":
    main()
