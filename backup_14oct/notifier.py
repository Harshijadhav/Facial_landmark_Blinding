"""
Delivers one-time codes.

Email: sent through SMTP when these environment variables are set (otherwise demo mode):
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM (optional)
SMS: no SMS provider is connected, so phone codes always run in demo mode.

In demo mode the caller shows the code on screen. That is only acceptable for a demo,
because a real system never reveals the code to whoever is asking for it.
"""
import os
import smtplib
import ssl
from email.message import EmailMessage

from database import RESET_CODE_MINUTES


def smtp_configured() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return (name[:1] + "*" * max(len(name) - 1, 1)) + "@" + domain


def mask_phone(phone: str) -> str:
    return "*" * max(len(phone) - 4, 0) + phone[-4:]


def _send_email(to_addr: str, username: str, code: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = "Your password reset code"
    msg["From"] = os.environ.get("SMTP_FROM", os.environ["SMTP_USER"])
    msg["To"] = to_addr
    msg.set_content(
        f"Hello {username},\n\n"
        f"Your one-time password reset code is {code}.\n"
        f"It expires in {RESET_CODE_MINUTES} minutes. If you did not ask for it, ignore this email.\n"
    )
    port = int(os.environ.get("SMTP_PORT", "587"))
    with smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=15) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(msg)


def deliver_code(channel: str, email: str, phone: str, username: str, code: str):
    """Returns (really_delivered, note)."""
    if channel == "Email" and email and smtp_configured():
        try:
            _send_email(email, username, code)
            return True, f"A code was sent to {mask_email(email)}."
        except Exception as exc:
            return False, f"The email could not be sent ({type(exc).__name__}), so demo mode is used."
    what = "email" if channel == "Email" else "SMS"
    return False, f"Demo mode: no {what} service is connected, so the code is shown here instead of being sent." 