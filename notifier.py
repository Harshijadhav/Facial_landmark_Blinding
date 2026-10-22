"""
Delivers one-time codes.

Email is sent by whichever of these is set up (tried in this order):
  1. Brevo web API (works even where SMTP ports are blocked):  BREVO_API_KEY, BREVO_SENDER
  2. SMTP with STARTTLS on port 587 and then SSL on port 465:   SMTP_HOST, SMTP_USER, SMTP_PASSWORD
     (optional: SMTP_PORT, SMTP_FROM)
Put the settings in a file named .env next to app.py (see .env.example) or in the terminal.

If email is set up but sending fails, the code is NOT shown on screen (it would defeat the point),
unless ALLOW_DEMO_FALLBACK=1. If email is not set up at all, or for phone numbers (no SMS provider
is connected), the app runs in demo mode and shows the code on screen.
"""
import json
import os
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage

from database import RESET_CODE_MINUTES


def _load_dotenv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _smtp_password():
    return os.environ.get("SMTP_PASSWORD", "").replace(" ", "")   # Gmail shows app passwords in groups


def smtp_configured() -> bool:
    return bool(os.environ.get("SMTP_HOST") and os.environ.get("SMTP_USER") and _smtp_password())


def brevo_configured() -> bool:
    return bool(os.environ.get("BREVO_API_KEY") and os.environ.get("BREVO_SENDER"))


def email_configured() -> bool:
    return smtp_configured() or brevo_configured()


def allow_demo_fallback() -> bool:
    return os.environ.get("ALLOW_DEMO_FALLBACK") == "1"


def mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return (name[:1] + "*" * max(len(name) - 1, 1)) + "@" + domain


def mask_phone(phone: str) -> str:
    return "*" * max(len(phone) - 4, 0) + phone[-4:]


def _message(to_addr: str, username: str, code: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "Your password reset code"
    msg["From"] = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or os.environ.get("BREVO_SENDER", "")
    msg["To"] = to_addr
    msg.set_content(
        f"Hello {username},\n\n"
        f"Your one-time password reset code is {code}.\n"
        f"It expires in {RESET_CODE_MINUTES} minutes. If you did not ask for it, ignore this email.\n"
    )
    return msg


def _send_brevo(to_addr, username, code):
    body = json.dumps({
        "sender": {"name": "Landmark Blinding Studio", "email": os.environ["BREVO_SENDER"]},
        "to": [{"email": to_addr}],
        "subject": "Your password reset code",
        "textContent": _message(to_addr, username, code).get_content(),
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email", data=body, method="POST",
        headers={"api-key": os.environ["BREVO_API_KEY"], "content-type": "application/json",
                 "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"HTTP {resp.status}")


def _send_smtp(to_addr, username, code, port):
    host, user, password = os.environ["SMTP_HOST"], os.environ["SMTP_USER"], _smtp_password()
    msg = _message(to_addr, username, code)
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=15, context=context) as server:
            server.login(user, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls(context=context)
            server.login(user, password)
            server.send_message(msg)


def send_email(to_addr: str, username: str, code: str):
    """Tries every configured route. Returns (ok, detail). detail names the failures, never a password."""
    errors = []
    if brevo_configured():
        try:
            _send_brevo(to_addr, username, code)
            return True, "sent through the Brevo API"
        except Exception as exc:
            errors.append(f"Brevo API: {type(exc).__name__} {str(exc)[:80]}")
    if smtp_configured():
        ports = [int(os.environ["SMTP_PORT"])] if os.environ.get("SMTP_PORT") else [587, 465]
        if ports == [587]:
            ports.append(465)
        for port in ports:
            try:
                _send_smtp(to_addr, username, code, port)
                return True, f"sent through SMTP port {port}"
            except Exception as exc:
                errors.append(f"SMTP {port}: {type(exc).__name__} {str(exc)[:80]}")
    return False, " | ".join(errors) or "email is not set up"


def deliver_code(channel: str, email: str, phone: str, username: str, code: str) -> dict:
    """Returns {"delivered": bool, "note": str, "show_code": bool}."""
    if channel == "Email" and email:
        if email_configured():
            ok, detail = send_email(email, username, code)
            if ok:
                return {"delivered": True, "note": f"A code was sent to {mask_email(email)}.", "show_code": False}
            return {"delivered": False, "show_code": allow_demo_fallback(),
                    "note": f"The email could not be sent ({detail}). Run test_email.py to find the cause."}
        return {"delivered": False, "show_code": True,
                "note": "Demo mode: email is not set up, so the code is shown here instead of being sent."}
    return {"delivered": False, "show_code": True,
            "note": "Demo mode: no SMS service is connected, so the code is shown here instead of being sent."}
