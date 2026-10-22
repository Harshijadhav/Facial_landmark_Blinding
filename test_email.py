"""
Finds out exactly why email is not working.   Run:   python test_email.py someone@example.com
Steps: settings present, internet name lookup, connection to the mail server, login, real test mail.
"""
import os
import smtplib
import socket
import ssl
import sys

import notifier


def line(ok, text):
    print(("  OK   " if ok else "  FAIL ") + text)


def main():
    to = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SMTP_USER", "")
    print("1. Settings")
    host, user = os.environ.get("SMTP_HOST", ""), os.environ.get("SMTP_USER", "")
    line(bool(host), f"SMTP_HOST = {host or '(missing)'}")
    line(bool(user), f"SMTP_USER = {user or '(missing)'}")
    pwd = notifier._smtp_password()
    line(bool(pwd), f"SMTP_PASSWORD present, {len(pwd)} characters (a Gmail app password has exactly 16)")
    if notifier.brevo_configured():
        line(True, "Brevo API settings found")
    if not notifier.email_configured():
        print("\nSet the values in a file called .env next to app.py (copy .env.example) and run this again.")
        return

    if host:
        print("2. Internet and mail server")
        try:
            socket.gethostbyname(host)
            line(True, f"found {host}")
        except OSError as exc:
            line(False, f"cannot find {host} ({exc}). No internet, or DNS is blocked.")
            return
        for port in (587, 465):
            try:
                socket.create_connection((host, port), timeout=8).close()
                line(True, f"can reach {host}:{port}")
            except OSError as exc:
                line(False, f"cannot reach {host}:{port} ({type(exc).__name__}). The network blocks it: try a mobile hotspot.")

        print("3. Login")
        try:
            with smtplib.SMTP(host, 587, timeout=15) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(user, pwd)
            line(True, "login on port 587 works")
        except smtplib.SMTPAuthenticationError as exc:
            line(False, f"login refused ({exc.smtp_code}). Use a Google APP password (2-Step Verification must be on), "
                        "not your normal password, and check SMTP_USER is the same Gmail.")
        except Exception as exc:
            line(False, f"port 587: {type(exc).__name__} {str(exc)[:100]}")

    print("4. Real test mail to", to or "(no address given)")
    if to:
        ok, detail = notifier.send_email(to, "tester", "123456")
        line(ok, detail)
        if ok:
            print("\nEmail works. Check the inbox and the spam folder.")


if __name__ == "__main__":
    main()
