"""
Settings loader for notification credentials.
Reads from .env file in project root. Falls back to env vars.

Priority: .env file wins over shell env vars.
This lets each project have its own bot/credentials even if
the shell has globals set (e.g., snipe-tracker's bot).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_settings() -> dict[str, str]:
    """Load notification settings. .env file takes priority over env vars."""
    settings: dict[str, str] = {}

    # First: pick up shell env vars as defaults
    env_keys = [
        "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
        "EMAIL_RECIPIENT", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    ]
    for key in env_keys:
        env_val = os.environ.get(key)
        if env_val:
            settings[key] = env_val

    # Second: .env file OVERRIDES env vars (project-specific wins)
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                settings[key.strip()] = value.strip().strip("'\"")

    return settings


def is_email_configured() -> bool:
    s = load_settings()
    return all(s.get(k) for k in ["SMTP_USER", "SMTP_PASSWORD", "EMAIL_RECIPIENT"])


def is_telegram_configured() -> bool:
    s = load_settings()
    return all(s.get(k) for k in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"])
