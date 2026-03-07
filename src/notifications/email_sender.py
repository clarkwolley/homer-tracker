"""
Email delivery for Homer Tracker reports.
Sends the full HTML prediction report via SMTP.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from src.notifications.settings import load_settings


def _build_message(sender: str, recipient: str, subject: str, html_path: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    plain = f"{subject}\n\nYour daily Homer Tracker report is attached as HTML. ⚾"
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(Path(html_path).read_text(), "html"))
    return msg


def send_report(html_path: str, subject: str | None = None) -> bool:
    """Send the HTML report via email."""
    settings = load_settings()

    smtp_host = settings.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(settings.get("SMTP_PORT", "587"))
    smtp_user = settings.get("SMTP_USER", "")
    smtp_password = settings.get("SMTP_PASSWORD", "")
    recipient = settings.get("EMAIL_RECIPIENT", "")

    if not all([smtp_user, smtp_password, recipient]):
        print("  ⚠️  Email not configured.")
        return False

    if not os.path.exists(html_path):
        print(f"  ⚠️  Report not found: {html_path}")
        return False

    if subject is None:
        from datetime import datetime
        subject = f"⚾ Homer Tracker Picks — {datetime.now().strftime('%Y-%m-%d')}"

    msg = _build_message(smtp_user, recipient, subject, html_path)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print(f"  ✅ Email sent to {recipient}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("  ❌ Email auth failed. Check SMTP_USER and SMTP_PASSWORD.")
        return False
    except Exception as e:
        print(f"  ❌ Email failed: {e}")
        return False
