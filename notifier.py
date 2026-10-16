"""
Delivers one-time codes.

Email goes out through SMTP when these settings exist (otherwise the app uses demo mode):
    SMTP_HOST, SMTP_USER, SMTP_PASSWORD   (required)
    SMTP_PORT                             (optional, default 465)
    SMTP_FROM                             (optional, defaults to SMTP_USER)

Where the settings come from:
    1. Real environment variables (for example $env:SMTP_HOST in PowerShell), or
    2. A file named .env next to this file, one KEY=value per line.
Real environment variables win over .env, so a stale value in your terminal
overrides the file. Restart Streamlit after changing either one.

Sending is tolerant of blocked networks: the configured port is tried first and, if the
connection itself fails (disconnect, timeout, SSL problem), the other standard mode is tried
automatically (465 uses SSL, 587 uses STARTTLS). A wrong password is reported immediately
and never retried.

SMS: no SMS provider is connected, so phone codes always run in demo mode.

In demo mode the caller shows the code on screen. That is only acceptable for a demo,
because a real system never reveals the code to whoever is asking for it.

Self-test (sends a real test email, prints what happened at each step):
    python notifier.py                 # sends to SMTP_USER
    python notifier.py someone@x.com   # sends to another address
"""
import logging
import os
import smtplib
import socket
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path

try:
    from database import RESET_CODE_MINUTES
except ImportError:  # lets `python notifier.py` run on its own
    RESET_CODE_MINUTES = 10

log = logging.getLogger("notifier")

ENV_FILE = Path(__file__).resolve().parent / ".env"

# Errors that mean "the connection itself failed", so trying the other mode may help.
CONNECTION_ERRORS = (
    smtplib.SMTPServerDisconnected,
    smtplib.SMTPConnectError,
    ConnectionError,
    TimeoutError,
    socket.timeout,
    ssl.SSLError,
)


class SendError(Exception):
    """Raised when every attempt failed. `kind` is 'auth', 'network' or 'other'."""

    def __init__(self, kind: str, detail: str):
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def _load_env_file(path: Path = None) -> bool:
    """Reads KEY=value lines from .env without needing python-dotenv.
    Existing environment variables are never overwritten. Returns True if a file was read."""
    path = path or ENV_FILE
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def _settings() -> dict:
    """Reads and cleans the SMTP settings (called fresh each time, so edits are picked up)."""
    _load_env_file()
    user = os.environ.get("SMTP_USER", "").strip()
    try:
        port = int(os.environ.get("SMTP_PORT", "465").strip() or "465")
    except ValueError:
        port = 465
    return {
        "host": os.environ.get("SMTP_HOST", "").strip(),
        "user": user,
        # Google shows app passwords as four groups of four; spaces are not part of it.
        "password": os.environ.get("SMTP_PASSWORD", "").strip().replace(" ", ""),
        "from_addr": os.environ.get("SMTP_FROM", "").strip() or user,
        "port": port,
    }


def smtp_configured() -> bool:
    s = _settings()
    return bool(s["host"] and s["user"] and s["password"])


def mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return (name[:1] + "*" * max(len(name) - 1, 1)) + "@" + domain


def mask_phone(phone: str) -> str:
    return "*" * max(len(phone) - 4, 0) + phone[-4:]


def _attempt_plan(port: int):
    """Ordered (mode, port) pairs to try. Only the two standard ports get a fallback."""
    if port == 465:
        return [("ssl", 465), ("starttls", 587)]
    if port == 587:
        return [("starttls", 587), ("ssl", 465)]
    return [("starttls", port)]


def _deliver(settings: dict, msg: EmailMessage, mode: str, port: int) -> None:
    context = ssl.create_default_context()
    if mode == "ssl":
        with smtplib.SMTP_SSL(settings["host"], port, context=context, timeout=15) as server:
            server.login(settings["user"], settings["password"])
            server.send_message(msg)
    else:
        with smtplib.SMTP(settings["host"], port, timeout=15) as server:
            server.starttls(context=context)
            server.login(settings["user"], settings["password"])
            server.send_message(msg)


def _send_email(to_addr: str, username: str, code: str) -> str:
    """Sends the code. Returns a label of the method that worked, e.g. 'SSL on port 465'.
    Raises SendError if nothing worked."""
    s = _settings()
    if not (s["host"] and s["user"] and s["password"]):
        raise SendError("other", "SMTP_HOST, SMTP_USER or SMTP_PASSWORD is not set")

    msg = EmailMessage()
    msg["Subject"] = "Your password reset code"
    msg["From"] = s["from_addr"]
    msg["To"] = to_addr
    msg.set_content(
        f"Hello {username},\n\n"
        f"Your one-time password reset code is {code}.\n"
        f"It expires in {RESET_CODE_MINUTES} minutes. If you did not ask for it, ignore this email.\n"
    )

    failures = []
    for mode, port in _attempt_plan(s["port"]):
        label = f"{'SSL' if mode == 'ssl' else 'STARTTLS'} on port {port}"
        try:
            _deliver(s, msg, mode, port)
            log.info("Reset email sent using %s", label)
            return label
        except smtplib.SMTPAuthenticationError as exc:
            # Wrong credentials will not be fixed by another port, so stop here.
            detail = f"{label}: {type(exc).__name__}: {exc}"
            log.warning("Email login rejected: %s", detail)
            raise SendError("auth", detail) from exc
        except CONNECTION_ERRORS as exc:
            detail = f"{label}: {type(exc).__name__}: {exc}"
            log.warning("Email attempt failed, trying next option. %s", detail)
            failures.append(detail)
        except Exception as exc:  # recipient refused, message rejected, and so on
            detail = f"{label}: {type(exc).__name__}: {exc}"
            log.warning("Email attempt failed: %s", detail)
            raise SendError("other", detail) from exc

    raise SendError("network", " | ".join(failures))


def _friendly_reason(exc: Exception) -> str:
    if isinstance(exc, SendError):
        if exc.kind == "auth":
            return "SMTPAuthenticationError - the login was rejected; Gmail needs a 16-character App Password"
        if exc.kind == "network":
            return "connection failed on every port - the network or antivirus may be blocking email"
        return exc.detail
    return type(exc).__name__


def deliver_code(channel: str, email: str, phone: str, username: str, code: str):
    """Returns (really_delivered, note)."""
    if channel == "Email" and email and smtp_configured():
        try:
            _send_email(email, username, code)
            return True, f"A code was sent to {mask_email(email)}."
        except Exception as exc:
            return False, f"The email could not be sent ({_friendly_reason(exc)}), so demo mode is used."
    what = "email" if channel == "Email" else "SMS"
    return False, f"Demo mode: no {what} service is connected, so the code is shown here instead of being sent."


def _self_test() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    had_file = _load_env_file()
    s = _settings()
    print(".env file      :", f"read from {ENV_FILE}" if had_file else f"not found at {ENV_FILE} (optional)")
    print("SMTP_HOST      :", s["host"] or "NOT SET")
    print("SMTP_PORT      :", s["port"])
    print("SMTP_USER      :", s["user"] or "NOT SET")
    print("Password length:", len(s["password"]), "(a Gmail app password is 16)")
    if not smtp_configured():
        print("\nSet SMTP_HOST, SMTP_USER and SMTP_PASSWORD (terminal or .env), then run again.")
        return 1
    to_addr = sys.argv[1] if len(sys.argv) > 1 else s["user"]
    print(f"\nSending a test email to {to_addr} ...")
    try:
        how = _send_email(to_addr, "tester", "123456")
    except SendError as exc:
        print("\nFAILED:", _friendly_reason(exc))
        print("Details:", exc.detail)
        return 1
    print(f"\nSUCCESS using {how}. Check the inbox (and spam).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
