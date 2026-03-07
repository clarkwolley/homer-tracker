"""
Telegram delivery for Homer Tracker picks.
Sends a compact HR picks summary via Telegram Bot API.
"""

import requests
import pandas as pd

from src.notifications.settings import load_settings


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 4096


def _format_picks_message(pred_df: pd.DataFrame, top_n: int = 15) -> str:
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    top = pred_df.head(top_n)

    lines = [
        f"⚾ *Homer Tracker* — {today}",
        f"_{len(pred_df)} batters analyzed_",
        "",
        "💣 *Top HR Picks*",
        "```",
        f"{'#':>2} {'Player':<20} {'Matchup':>7} {'HR%':>5} {'Streak'}",
        "-" * 45,
    ]

    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = row["name"][:20]
        prob = f"{row['hr_probability'] * 100:.0f}%"
        matchup = f"{'vs' if row['is_home'] else '@'}{row['opponent']}"
        streak = ""
        if row.get("is_hot", 0):
            streak = f"🔥{int(row.get('hr_streak', 0))}"
        elif row.get("hr_drought", 0) >= 10:
            streak = f"❄️{int(row.get('hr_drought', 0))}"
        lines.append(f"{i:>2} {name:<20} {matchup:>7} {prob:>5} {streak}")

    lines.append("```")
    lines.append("")
    lines.append("_Full report sent via email_ 📧")
    return "\n".join(lines)


def _format_grade_message(graded: pd.DataFrame) -> str:
    played = graded[graded["played"] == 1]
    date = graded["prediction_date"].iloc[0]

    total = len(played)
    actual = int(played["actual_hit_hr"].sum())
    predicted = int(played["predicted_hr"].sum())
    hits = int(played["hit"].sum())
    precision = hits / max(predicted, 1) * 100

    lines = [
        f"📊 *HR Scorecard* — {date}",
        "",
        f"Batters tracked: {total}",
        f"Actually homered: {actual}",
        f"Predicted HR: {predicted}",
        f"Hits: {hits}/{predicted} ({precision:.0f}% precision)",
    ]

    top_hits = played[played["actual_hit_hr"] == 1].nlargest(5, "hr_probability")
    if not top_hits.empty:
        lines.append("")
        lines.append("✅ *Top Hits*")
        for _, r in top_hits.iterrows():
            lines.append(f"  {r['name']} ({r['team']}) — "
                         f"{r['hr_probability']*100:.0f}% → {int(r['actual_hr'])} HR")

    return "\n".join(lines)


def send_picks(pred_df: pd.DataFrame, top_n: int = 15) -> bool:
    settings = load_settings()
    token = settings.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = settings.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        print("  ⚠️  Telegram not configured.")
        return False

    message = _format_picks_message(pred_df, top_n)
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH - 20] + "\n\n_(truncated)_"

    return _send_message(token, chat_id, message)


def send_grade(graded: pd.DataFrame) -> bool:
    settings = load_settings()
    token = settings.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = settings.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        return False

    return _send_message(token, chat_id, _format_grade_message(graded))


def _send_message(token: str, chat_id: str, text: str) -> bool:
    url = TELEGRAM_API.format(token=token)
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown",
               "disable_web_page_preview": True}

    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print("  ✅ Telegram message sent")
        return True
    except requests.exceptions.HTTPError:
        # Fallback to plain text on markdown parse errors
        payload["parse_mode"] = None
        try:
            resp2 = requests.post(url, json=payload, timeout=15)
            resp2.raise_for_status()
            print("  ✅ Telegram message sent (plain text)")
            return True
        except Exception:
            pass
        print(f"  ❌ Telegram failed")
        return False
    except Exception as e:
        print(f"  ❌ Telegram failed: {e}")
        return False
