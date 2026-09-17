from __future__ import annotations

import json
import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


EMAIL_PATTERN = re.compile(r"^[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}$")


def send_email_tool(to: str, subject: str, body: str) -> dict:
    if not EMAIL_PATTERN.match(to or ""):
        return {"status": "error", "error": "Invalid recipient email address."}
    if not subject or not subject.strip():
        return {"status": "error", "error": "Subject is required."}
    if not body or not body.strip():
        return {"status": "error", "error": "Body is required."}

    mode = os.getenv("EMAIL_MODE", "mock").lower()
    if mode == "smtp":
        return _send_smtp(to, subject, body)
    return _send_mock(to, subject, body)


def _send_mock(to: str, subject: str, body: str) -> dict:
    Path("sent_emails").mkdir(exist_ok=True)
    payload = {"status": "sent", "mode": "mock", "recipient": to, "subject": subject, "body": body}
    path = Path("sent_emails") / f"email_{len(list(Path('sent_emails').glob('email_*.json'))) + 1:03d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "path": str(path)}


def _send_smtp(to: str, subject: str, body: str) -> dict:
    required = ["SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD", "EMAIL_FROM"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        return {"status": "error", "error": f"Missing SMTP configuration: {', '.join(missing)}"}
    message = EmailMessage()
    message["From"] = os.environ["EMAIL_FROM"]
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            if os.getenv("SMTP_USE_TLS", "true").lower() == "true":
                smtp.starttls()
            smtp.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
            smtp.send_message(message)
    except Exception as exc:
        return {"status": "error", "error": f"SMTP send failed: {exc}"}
    return {"status": "sent", "mode": "smtp", "recipient": to}


try:
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("meeting-email")

    @app.tool()
    def send_email(to: str, subject: str, body: str) -> dict:
        """Validate and send an email through the configured email mechanism."""
        return send_email_tool(to=to, subject=subject, body=body)
except Exception:
    app = None


if __name__ == "__main__":
    if app is None:
        raise SystemExit("MCP SDK is not installed. Install requirements.txt first.")
    app.run()
