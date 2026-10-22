import time

import streamlit as st

from database import (
    USERNAME_RULE_TEXT,
    accept_terms,
    authenticate_user,
    create_reset_code,
    password_checks,
    register_user,
    registration_errors,
    verify_and_reset,
    RESET_CODE_MINUTES,
)
from notifier import deliver_code
from terms import TERMS_TEXT, TERMS_VERSION

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 30
POLICY_TEXT = (
    "Password standards: minimum 8 characters, 1 uppercase, 1 lowercase, "
    "1 digit and 1 special symbol."
)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _init_state():
    defaults = {
        "auth_view": "briefing",
        "login_attempts": 0,
        "locked_until": 0.0,
        "pending_user": None,
        "reset_stage": "request",
        "reset_identifier": "",
        "reset_demo": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _go(view: str):
    st.session_state.auth_view = view
    st.rerun()


def _centered():
    _, mid, _ = st.columns([1, 1.6, 1])
    return mid


def _checklist(password: str, unmet: str = "❌"):
    lines = [f"{'✅' if ok else unmet} {label}" for label, ok in password_checks(password)]
    st.markdown("  \n".join(lines))


@st.dialog("Please fix the following")
def _error_dialog(errors, password=""):
    for message in errors:
        st.markdown(f"- {message}")
    if any(message.startswith("Password does not meet") for message in errors):
        st.markdown("**Password requirements**")
        _checklist(password)
    if st.button("Got it", type="primary", width="stretch"):
        st.rerun()


def _terms_box():
    with st.container(height=240, border=True):
        st.markdown(TERMS_TEXT)


def _finish_login(user):
    st.session_state.user = user
    st.session_state.defense_cache = None
    st.session_state.last_processed_file = None
    st.session_state.login_attempts = 0
    st.session_state.pending_user = None
    st.session_state.auth_view = "briefing"
    st.rerun()


# ------------------------------------------------------------------
# views
# ------------------------------------------------------------------
def _briefing():
    # ---- Hero ----
    st.markdown(
        """<div class="hero">
<div class="hero-title">🛡️ Proactive Biometric Defense Console</div>
<div class="hero-sub">Add invisible protection to your photo on your own device, before you upload it anywhere.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # ---- How deepfake tools find a face (a real sequence, so numbered) ----
    st.markdown('<div class="section-title">How deepfake tools lock onto a face</div>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    steps = [
        ("1", "Find the landmarks",
         "Tools such as InsightFace, SimSwap and LivePortrait first locate 5 to 468 "
         "points on the eyes, nose and mouth."),
        ("2", "Align the face",
         "Those points are used to warp the face into a standard 112×112 frame "
         "using an affine matrix."),
        ("3", "Swap the identity",
         "Only after alignment can the generator replace or re-animate the face. "
         "If alignment fails, the swap fails."),
    ]
    for col, (num, title, body) in zip((s1, s2, s3), steps):
        with col:
            st.markdown(
                f"""<div class="step-card">
<div class="step-num">{num}</div>
<div class="step-title">{title}</div>
<div class="step-body">{body}</div>
</div>""",
                unsafe_allow_html=True,
            )

    # ---- What this tool does ----
    st.markdown('<div class="section-title">What this console does</div>',
                unsafe_allow_html=True)
    st.markdown(
        """<div class="shield-callout">
<b>It disrupts step 1 and leaves your photo looking the same.</b>
<ul>
<li><b>Hidden pattern:</b> a faint high-frequency pattern is added around the face. It is built to survive social-media JPEG compression.</li>
<li><b>Face still detected:</b> the photo still counts as a face (confidence change ≈ 0), but the landmark points drift out of place.</li>
<li><b>Quality limits:</b> PSNR ≥ 38 dB and SSIM ≥ 0.96, so the change is hard to see.</li>
<li><b>Private:</b> everything runs locally in memory. No image leaves this computer.</li>
</ul>
</div>""",
        unsafe_allow_html=True,
    )

    # ---- If you are targeted ----
    st.markdown('<div class="section-title">If you are targeted by a deepfake</div>',
                unsafe_allow_html=True)
    h1, h2 = st.columns([1, 1.4])
    with h1:
        st.markdown(
            """<div class="help-card">
<div class="help-number">1930</div>
<div class="help-label">National cyber crime helpline</div>
<a href="https://cybercrime.gov.in" target="_blank">cybercrime.gov.in</a>
</div>""",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            """<div class="alert-callout">
<b>Act fast.</b> Under India's IT (Intermediary Guidelines and Digital Media Ethics Code) Rules, platforms must act on complaints about non-consensual synthetic media within 24 hours.
<ul>
<li>Save the post URL, the date and time, and screenshots before reporting.</li>
<li>Keep the original image file. Its metadata helps prove which version is real.</li>
</ul>
</div>""",
            unsafe_allow_html=True,
        )

    # ---- Call to action ----
    st.write("")
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("Proceed to login", type="primary", width="stretch"):
            _go("login")


def _login():
    with _centered():
        st.markdown("## 🔐 Access Defense Studio")
        st.caption("Sign in to start a session with local inference and audit logging.")

        flash = st.session_state.pop("flash", None)
        if flash:
            st.success(flash)

        remaining = int(st.session_state.locked_until - time.time())
        locked = remaining > 0

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Login", type="primary", width="stretch", disabled=locked
            )

        if locked:
            st.warning(f"Too many failed attempts. Try again in {remaining} seconds.")

        if submitted:
            if not username or not password:
                st.warning("Enter both your username and password.")
            else:
                ok, user = authenticate_user(username, password)
                if ok:
                    if not user.get("terms_accepted_at"):
                        st.session_state.pending_user = user
                        _go("terms_gate")
                    _finish_login(user)
                else:
                    st.session_state.login_attempts += 1
                    left = MAX_ATTEMPTS - st.session_state.login_attempts
                    if left <= 0:
                        st.session_state.locked_until = time.time() + LOCKOUT_SECONDS
                        st.session_state.login_attempts = 0
                        st.rerun()
                    st.error("Invalid username or password.")
                    st.info(f"{POLICY_TEXT}  \nAttempts left: {left}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Forgot password?", width="stretch"):
                st.session_state.reset_stage = "request"
                _go("forgot")
        with c2:
            if st.button("Create account", width="stretch"):
                _go("register")
        if st.button("Back to overview", type="tertiary"):
            _go("briefing")


def _terms_gate():
    pending = st.session_state.get("pending_user")
    if not pending:
        _go("login")
    with _centered():
        st.markdown("## Terms of Use")
        st.caption("Please read and accept the terms before you use the app.")
        _terms_box()
        accepted = st.checkbox("I have read and accept the Terms of Use and Acceptable Use Policy :red[*]",
                               key="gate_accept")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Accept and continue", type="primary", width="stretch", disabled=not accepted):
                _finish_login(accept_terms(pending["user_id"], TERMS_VERSION))
        with c2:
            if st.button("Decline", width="stretch"):
                st.session_state.pending_user = None
                _go("login")


def _register():
    with _centered():
        st.markdown("## 📝 Create account")
        st.caption("Fields marked :red[*] are required.")

        full_name = st.text_input("Full name :red[*]", key="r_name", placeholder="Your full name")
        email = st.text_input("Email :red[*]", key="r_email", placeholder="name@example.com")
        phone = st.text_input("Phone number :red[*]", key="r_phone", placeholder="XXXXXXXXXX", max_chars=10)
        username = st.text_input("Username :red[*]", key="r_user", placeholder="for example: asha_rao")
        st.caption(USERNAME_RULE_TEXT)
        password = st.text_input("Password :red[*]", type="password", key="r_pass",
                                 placeholder="8+ characters: A-Z, a-z, 0-9 and a symbol")
        confirm = st.text_input("Confirm password :red[*]", type="password", key="r_confirm",
                                placeholder="Type the password again")

        st.markdown("**Terms of Use and Acceptable Use Policy**")
        _terms_box()
        accepted = st.checkbox("I have read and accept the Terms of Use and Acceptable Use Policy :red[*]",
                               key="r_accept")

        if st.button("Create account", type="primary", width="stretch", disabled=not accepted):
            errors = registration_errors(full_name, email, phone, username, password, confirm, accepted)
            if errors:
                _error_dialog(errors, password)
            else:
                ok, msg = register_user(full_name, email, phone, username, password, accepted, TERMS_VERSION)
                if ok:
                    st.session_state.flash = "Account created. Please log in."
                    _go("login")
                else:
                    st.error(msg)

        if st.button("Back to login", type="tertiary"):
            _go("login")


def _forgot():
    with _centered():
        st.markdown("## 🔑 Reset password")
        stage = st.session_state.reset_stage

        if stage == "request":
            st.caption("Enter your username or email. A 6-digit code is sent to your registered email or phone.")
            identifier = st.text_input("Username or email", key="f_ident")
            channel = st.radio("Send the code to", ["Email", "Phone number"], horizontal=True, key="f_channel")
            if st.button("Send code", type="primary", width="stretch"):
                if not identifier.strip():
                    st.warning("Enter your username or email.")
                else:
                    res = create_reset_code(identifier)
                    if res and "wait" in res:
                        st.warning(f"Please wait {res['wait']} seconds before asking for another code.")
                    else:
                        st.session_state.reset_demo = None
                        send_failed = None
                        if res and "code" in res:
                            info = deliver_code(channel, res["email"], res["phone"],
                                                res["username"], res["code"])
                            if not info["delivered"]:
                                if info["show_code"]:
                                    st.session_state.reset_demo = {"code": res["code"], "note": info["note"]}
                                else:
                                    send_failed = info["note"]
                        if send_failed:
                            st.error(send_failed)
                        else:
                            st.session_state.reset_identifier = identifier.strip()
                            st.session_state.reset_stage = "verify"
                            st.rerun()
        else:
            st.info(f"If an account matches, a 6-digit code has been sent to its registered "
                    f"{'email or phone' }. It expires in {RESET_CODE_MINUTES} minutes.")
            demo = st.session_state.reset_demo
            if demo:
                st.warning(f"{demo['note']}  \nYour code: **{demo['code']}**")
            code = st.text_input("6-digit code", key="f_code", max_chars=6, placeholder="XXXXXX")
            new_pwd = st.text_input("New password", type="password", key="f_new",
                                    placeholder="8+ characters: A-Z, a-z, 0-9 and a symbol")
            confirm = st.text_input("Confirm new password", type="password", key="f_confirm",
                                    placeholder="Type the password again")

            if st.button("Reset password", type="primary", width="stretch"):
                errors = []
                if not (code.strip().isdigit() and len(code.strip()) == 6):
                    errors.append("Enter the 6-digit code.")
                if not all(ok for _, ok in password_checks(new_pwd)):
                    errors.append("Password does not meet the requirements.")
                elif new_pwd != confirm:
                    errors.append("Passwords do not match.")
                if errors:
                    _error_dialog(errors, new_pwd)
                else:
                    ok, msg = verify_and_reset(st.session_state.reset_identifier, code, new_pwd)
                    if ok:
                        st.session_state.flash = "Password updated. Please log in."
                        st.session_state.reset_stage = "request"
                        st.session_state.reset_demo = None
                        _go("login")
                    else:
                        st.error(msg)
            if st.button("Send a new code", type="tertiary"):
                st.session_state.reset_stage = "request"
                st.rerun()

        if st.button("Back to login", type="tertiary"):
            st.session_state.reset_stage = "request"
            _go("login")


# ------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------
def render_auth_gateway():
    _init_state()
    views = {
        "briefing": _briefing,
        "login": _login,
        "terms_gate": _terms_gate,
        "register": _register,
        "forgot": _forgot,
    }
    views[st.session_state.auth_view]()
