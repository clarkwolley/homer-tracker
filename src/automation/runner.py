"""
Daily automation runner for Homer Tracker.

Usage:
    python -m src.automation.runner              # Full daily run
    python -m src.automation.runner --predict    # Predictions only
    python -m src.automation.runner --grade      # Grade yesterday only
    python -m src.automation.runner --notify     # Re-send latest report
    python -m src.automation.runner --status     # Check config status
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

    fh = logging.FileHandler(os.path.join(LOG_DIR, "runner.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

    return logger


def step_grade_yesterday(log: logging.Logger) -> None:
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


def step_predict_today(log: logging.Logger) -> tuple:
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


def step_notify(log, pred_df=None, report_path=None) -> None:
    from src.notifications.email_sender import send_report
    from src.notifications.telegram_sender import send_picks
    from src.notifications.settings import is_email_configured, is_telegram_configured

    if pred_df is None or (hasattr(pred_df, 'empty') and pred_df.empty):
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


def _find_latest_report() -> str | None:
    pattern = os.path.join(REPORT_DIR, "picks_*.html")
    files = sorted(glob.glob(pattern), reverse=True)
    return files[0] if files else None


def check_status() -> None:
    from src.notifications.settings import is_email_configured, is_telegram_configured

    print("\n🐶 Homer Tracker — System Status")
    print("=" * 50)

    print(f"  📧 Email:    {'✅ configured' if is_email_configured() else '❌ not configured'}")
    print(f"  💬 Telegram: {'✅ configured' if is_telegram_configured() else '❌ not configured'}")

    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hr_model.pkl")
    print(f"  🤖 HR Model: {'✅ trained' if os.path.exists(model_path) else '❌ not trained'}")

    game_model = os.path.join(os.path.dirname(__file__), "..", "..", "models", "game_model.pkl")
    print(f"  🏆 Game Model: {'✅ trained' if os.path.exists(game_model) else '❌ not trained'}")

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "game_log.csv")
    print(f"  📊 Data:     {'✅ collected' if os.path.exists(data_path) else '❌ no data'}")

    latest = _find_latest_report()
    print(f"  📄 Latest:   {os.path.basename(latest) if latest else 'no reports yet'}")

    print("=" * 50)
    if not is_email_configured() or not is_telegram_configured():
        print("\n  💡 Run: python -m src.automation.setup\n")


def run(mode: str = "full") -> None:
    if mode == "status":
        check_status()
        return

    log = _setup_logging()
    log.info(f"{'=' * 50}")
    log.info(f"⚾ Homer Tracker Runner — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info(f"   Mode: {mode}")

    try:
        if mode in ("full", "grade"):
            step_grade_yesterday(log)

        pred_df = None
        report_path = None

        if mode in ("full", "predict"):
            pred_df, _, report_path = step_predict_today(log)

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
    if len(sys.argv) > 1:
        arg = sys.argv[1].lstrip("-")
        if arg in ("predict", "grade", "notify", "status"):
            mode = arg
    run(mode)


if __name__ == "__main__":
    main()
