import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import string
from datetime import datetime, timedelta

DB_FILE = "users.db"
PBKDF2_ITERATIONS = 200_000

# ------------------------------------------------------------------
# ACCOUNT RULES  (change them here; the registration form and the checks follow)
# ------------------------------------------------------------------
USERNAME_MIN_LEN = 6
USERNAME_SPECIALS = "._-"
USERNAME_REQUIRE_SPECIAL = True   # at least one of the special characters
USERNAME_ALLOW_DIGITS = False     # numbers not allowed in usernames

_rule_parts = [f"At least {USERNAME_MIN_LEN} characters", "letters and the symbols " + " ".join(USERNAME_SPECIALS)]
if USERNAME_REQUIRE_SPECIAL:
    _rule_parts.append("at least one symbol")
if not USERNAME_ALLOW_DIGITS:
    _rule_parts.append("no numbers")
USERNAME_RULE_TEXT = ", ".join(_rule_parts) + "."

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")
PHONE_PATTERN = re.compile(r"^[6-9][0-9]{9}$")   # 10-digit Indian mobile number

RESET_CODE_MINUTES = 10
RESET_MAX_ATTEMPTS = 5
RESET_RESEND_SECONDS = 60

# ------------------------------------------------------------------
# PASSWORD POLICY
# ------------------------------------------------------------------
PASSWORD_RULES = [
    ("At least 8 characters", lambda p: len(p) >= 8),
    ("One uppercase letter (A-Z)", lambda p: bool(re.search(r"[A-Z]", p))),
    ("One lowercase letter (a-z)", lambda p: bool(re.search(r"[a-z]", p))),
    ("One digit (0-9)", lambda p: bool(re.search(r"[0-9]", p))),
    ("One special symbol (!@#$%^&*)", lambda p: bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", p))),
]


def password_checks(password: str) -> list[tuple[str, bool]]:
    """Per-rule pass/fail list, used for the checklist in the UI."""
    return [(label, fn(password)) for label, fn in PASSWORD_RULES]


def validate_password_policy(password: str) -> tuple[bool, str]:
    """Enforces standard security policy (NIST SP 800-63B standard rules)."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)."
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one numeric digit (0-9)."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character (!@#$%^&*)."
    return True, "Valid"


# ------------------------------------------------------------------
# FIELD VALIDATION
# ------------------------------------------------------------------
def valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email.strip()))


def valid_phone(phone: str) -> bool:
    return bool(PHONE_PATTERN.match(phone.strip()))


def valid_full_name(name: str) -> bool:
    n = " ".join(name.split())
    return (2 <= len(n) <= 60 and sum(c.isalpha() for c in n) >= 2
            and all(c.isalpha() or c in " .'-" for c in n))


def validate_username(username: str) -> tuple[bool, str]:
    u = username.strip()
    if len(u) < USERNAME_MIN_LEN:
        return False, f"Username must be at least {USERNAME_MIN_LEN} characters."
    if not USERNAME_ALLOW_DIGITS and any(c.isdigit() for c in u):
        return False, "Username cannot contain numbers."
    allowed = set(string.ascii_letters + USERNAME_SPECIALS + (string.digits if USERNAME_ALLOW_DIGITS else ""))
    if any(c not in allowed for c in u):
        return False, "Username can only use letters and these symbols: " + " ".join(USERNAME_SPECIALS)
    if USERNAME_REQUIRE_SPECIAL and not any(c in USERNAME_SPECIALS for c in u):
        return False, "Username must contain at least one special character (" + " ".join(USERNAME_SPECIALS) + ")."
    return True, "Valid"


def registration_errors(full_name, email, phone, username, password, confirm, accepted):
    """Every problem with a registration form, in the order the fields appear."""
    errors = []
    if not valid_full_name(full_name):
        errors.append("Enter your full name.")
    if not valid_email(email):
        errors.append("Enter a valid email id.")
    if not valid_phone(phone):
        errors.append("Enter a valid 10-digit phone number.")
    ok, msg = validate_username(username)
    if not ok:
        errors.append(msg)
    ok, _ = validate_password_policy(password)
    if not ok:
        errors.append("Password does not meet the requirements.")
    elif password != confirm:
        errors.append("Passwords do not match.")
    if not accepted:
        errors.append("Accept the Terms of Use to continue.")
    return errors


# ------------------------------------------------------------------
# DATABASE SETUP
# ------------------------------------------------------------------
def get_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            full_name TEXT,
            email TEXT,
            phone TEXT,
            terms_accepted_at TEXT,
            terms_version TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            image_hash TEXT NOT NULL,
            psnr REAL NOT NULL,
            ssim REAL NOT NULL,
            defense_status TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            reset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            code_hash TEXT NOT NULL,
            code_salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    """)

    # Migration: add the new profile columns to an older users table
    existing = {row["name"] for row in cursor.execute("PRAGMA table_info(users)")}
    for column in ("full_name", "email", "phone", "terms_accepted_at", "terms_version"):
        if column not in existing:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT")

    conn.commit()
    conn.close()


# ------------------------------------------------------------------
# HASHING (PBKDF2-HMAC-SHA256, with fallback for old SHA-256 accounts)
# ------------------------------------------------------------------
def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2${PBKDF2_ITERATIONS}${digest}", salt


def verify_password(password, stored_hash, salt):
    if stored_hash.startswith("pbkdf2$"):
        _, iterations, expected = stored_hash.split("$")
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
        ).hex()
    else:  # legacy account created with plain salted SHA-256
        expected = stored_hash
        actual = hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
    return hmac.compare_digest(actual, expected)


def _now_text():
    return datetime.now().isoformat(sep=" ", timespec="seconds")


# ------------------------------------------------------------------
# ACCOUNT OPERATIONS
# ------------------------------------------------------------------
def register_user(full_name, email, phone, username, password, accepted_terms,
                  terms_version, role="Researcher"):
    """Returns (success, message). All rules are enforced here, not only in the form."""
    errors = registration_errors(full_name, email, phone, username, password, password, accepted_terms)
    if errors:
        return False, errors[0]

    username = username.strip()
    email = email.strip().lower()
    phone = phone.strip()
    full_name = " ".join(full_name.split())

    conn = get_connection()
    cursor = conn.cursor()
    try:
        if cursor.execute("SELECT 1 FROM users WHERE lower(username) = lower(?)", (username,)).fetchone():
            return False, "Username already registered."
        if cursor.execute("SELECT 1 FROM users WHERE lower(email) = ?", (email,)).fetchone():
            return False, "This email is already registered."
        if cursor.execute("SELECT 1 FROM users WHERE phone = ?", (phone,)).fetchone():
            return False, "This phone number is already registered."

        pwd_hash, salt = hash_password(password)
        now = _now_text()
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, salt, role, created_at,
                               full_name, email, phone, terms_accepted_at, terms_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (username, pwd_hash, salt, role, now, full_name, email, phone, now, terms_version),
        )
        conn.commit()
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already registered."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    user = cursor.fetchone()

    if user is None:
        conn.close()
        return False, None

    if verify_password(password, user["password_hash"], user["salt"]):
        # Silently upgrade legacy SHA-256 hashes to PBKDF2 on successful login
        if not user["password_hash"].startswith("pbkdf2$"):
            new_hash, new_salt = hash_password(password)
            cursor.execute(
                "UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?",
                (new_hash, new_salt, user["user_id"]),
            )
            conn.commit()
        conn.close()
        return True, dict(user)

    conn.close()
    return False, None


def accept_terms(user_id, terms_version):
    """Record that an existing user accepted the terms. Returns the refreshed user dict."""
    conn = get_connection()
    conn.execute("UPDATE users SET terms_accepted_at = ?, terms_version = ? WHERE user_id = ?",
                 (_now_text(), terms_version, user_id))
    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(user)


# ------------------------------------------------------------------
# PASSWORD RESET BY ONE-TIME CODE
# ------------------------------------------------------------------
def _find_user(cursor, identifier):
    ident = identifier.strip()
    return cursor.execute(
        "SELECT * FROM users WHERE username = ? OR lower(email) = lower(?)", (ident, ident)
    ).fetchone()


def create_reset_code(identifier):
    """
    Creates a 6-digit one-time code for the account matching a username or email.

    Returns None if there is no usable account, {"wait": seconds} if a code was requested
    too recently, or {"code", "email", "phone", "username"} for the caller to deliver.
    The code itself is stored only as a salted PBKDF2 hash and expires after RESET_CODE_MINUTES.
    """
    conn = get_connection()
    cursor = conn.cursor()
    user = _find_user(cursor, identifier)
    if user is None or not (user["email"] or user["phone"]):
        conn.close()
        return None

    last = cursor.execute(
        "SELECT created_at FROM password_resets WHERE user_id = ? ORDER BY reset_id DESC LIMIT 1",
        (user["user_id"],),
    ).fetchone()
    if last:
        age = (datetime.now() - datetime.fromisoformat(last["created_at"])).total_seconds()
        if age < RESET_RESEND_SECONDS:
            conn.close()
            return {"wait": int(RESET_RESEND_SECONDS - age) + 1}

    code = f"{secrets.randbelow(1_000_000):06d}"
    code_hash, code_salt = hash_password(code)
    cursor.execute("DELETE FROM password_resets WHERE user_id = ?", (user["user_id"],))
    cursor.execute(
        "INSERT INTO password_resets (user_id, code_hash, code_salt, created_at, expires_at, attempts) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (user["user_id"], code_hash, code_salt, _now_text(),
         (datetime.now() + timedelta(minutes=RESET_CODE_MINUTES)).isoformat(sep=" ", timespec="seconds")),
    )
    conn.commit()
    result = {"code": code, "email": user["email"], "phone": user["phone"], "username": user["username"]}
    conn.close()
    return result


def verify_and_reset(identifier, code, new_password):
    """Checks the code (expiry, attempts, hash) and sets the new password. Returns (ok, message)."""
    generic = "Invalid or expired code. Request a new one."
    conn = get_connection()
    cursor = conn.cursor()
    user = _find_user(cursor, identifier)
    if user is None:
        conn.close()
        return False, generic

    row = cursor.execute(
        "SELECT * FROM password_resets WHERE user_id = ? ORDER BY reset_id DESC LIMIT 1",
        (user["user_id"],),
    ).fetchone()
    if row is None or datetime.now() > datetime.fromisoformat(row["expires_at"]) \
            or row["attempts"] >= RESET_MAX_ATTEMPTS:
        conn.close()
        return False, generic

    if not verify_password(code.strip(), row["code_hash"], row["code_salt"]):
        attempts = row["attempts"] + 1
        if attempts >= RESET_MAX_ATTEMPTS:
            cursor.execute("DELETE FROM password_resets WHERE reset_id = ?", (row["reset_id"],))
        else:
            cursor.execute("UPDATE password_resets SET attempts = ? WHERE reset_id = ?",
                           (attempts, row["reset_id"]))
        conn.commit()
        conn.close()
        return False, generic

    # The code is right; a weak new password does not burn it
    ok, msg = validate_password_policy(new_password)
    if not ok:
        conn.close()
        return False, msg

    new_hash, new_salt = hash_password(new_password)
    cursor.execute("UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?",
                   (new_hash, new_salt, user["user_id"]))
    cursor.execute("DELETE FROM password_resets WHERE user_id = ?", (user["user_id"],))
    conn.commit()
    conn.close()
    return True, "Password reset successfully."


# ------------------------------------------------------------------
# AUDIT TRAIL
# ------------------------------------------------------------------
def log_audit_trail(user_id, image_hash, psnr, ssim, status):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO audit_logs (user_id, timestamp, image_hash, psnr, ssim, defense_status)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (user_id, _now_text(), image_hash, psnr, ssim, status),
    )
    conn.commit()
    conn.close()


def get_user_audit_history(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, image_hash, psnr, ssim, defense_status
        FROM audit_logs
        WHERE user_id = ?
        ORDER BY timestamp DESC LIMIT 5
    """,
        (user_id,),
    )
    logs = cursor.fetchall()
    conn.close()
    return logs


init_db()
