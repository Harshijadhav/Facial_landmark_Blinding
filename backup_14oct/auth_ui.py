import streamlit as st

from database import (
    authenticate_user,
    create_reset_code,
    password_checks,
    register_user,
    registration_errors,
    validate_username,
    verify_and_reset,
    USERNAME_RULE_TEXT,
)
from notifier import deliver_code
from terms import TERMS_TEXT, TERMS_VERSION


def _login_box():
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Enter both your username and your password.")
            return
        ok, user = authenticate_user(username, password)
        if ok:
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Username or password not recognized.")


def _register_box():
    with st.form("reg_form", clear_on_submit=False):
        name = st.text_input("Full name")
        email = st.text_input("Email id")
        phone = st.text_input("Phone number (10-digit Indian mobile)")
        username = st.text_input("Username", help=USERNAME_RULE_TEXT)
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm password", type="password")

        st.caption("Password requirements:")
        for label, met in password_checks(password):
            color = "#2ea043" if met else "#8b949e"
            tick = "✔" if met else "○"
            st.markdown(f"<span style='color:{color}; font-size:12px;'>{tick} {label}</span>", unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:12px;">Terms of Use</div>', unsafe_allow_html=True)
        with st.container(border=True, height=140):
            st.markdown(TERMS_TEXT)

        accepted = st.checkbox("I accept the Terms of Use and Acceptable Use Policy")
        submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)

    if submitted:
        errors = registration_errors(name, email, phone, username, password, confirm, accepted)
        if errors:
            for err in errors:
                st.error(err)
            return

        ok, msg = register_user(
            name, email, phone, username, password,
            accepted_terms=accepted,
            terms_version=TERMS_VERSION,
        )
        if ok:
            st.success("Account created! You can sign in now.")
        else:
            st.error(msg)


def _reset_box():
    st.caption("Enter your registered username or email id to receive a one-time reset code.")

    if "reset_stage" not in st.session_state:
        st.session_state.reset_stage = "request"

    if st.session_state.reset_stage == "request":
        with st.form("request_code_form"):
            ident = st.text_input("Username or Email id")
            channel = st.radio("Send code via", ["Email", "SMS (Demo Mode)"], horizontal=True)
            submitted = st.form_submit_button("Send code", type="primary", use_container_width=True)

        if submitted:
            if not ident.strip():
                st.error("Please enter your username or email.")
                return
            res = create_reset_code(ident)
            if res is None:
                st.error("No account found with that username or email.")
            elif "wait" in res:
                st.warning(f"Please wait {res['wait']} seconds before requesting another code.")
            else:
                delivered, note = deliver_code(channel.split()[0], res["email"], res["phone"], res["username"], res["code"])
                st.session_state.reset_ident = ident
                st.session_state.reset_code = res["code"]
                st.session_state.reset_note = note
                st.session_state.reset_stage = "verify"
                st.rerun()

    elif st.session_state.reset_stage == "verify":
        st.info(st.session_state.get("reset_note", "Code sent."))
        if "reset_code" in st.session_state:
            st.code(f"Demo Mode Code: {st.session_state.reset_code}")

        with st.form("verify_code_form"):
            code = st.text_input("6-digit code")
            new_pwd = st.text_input("New password", type="password")
            confirm_pwd = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Reset password", type="primary", use_container_width=True)

        if submitted:
            if new_pwd != confirm_pwd:
                st.error("Passwords do not match.")
                return
            ok, msg = verify_and_reset(st.session_state.reset_ident, code, new_pwd)
            if ok:
                st.success("Password changed successfully! You can sign in now.")
                st.session_state.reset_stage = "request"
                st.session_state.pop("reset_ident", None)
                st.session_state.pop("reset_code", None)
                st.session_state.pop("reset_note", None)
            else:
                st.error(msg)

        if st.button("Cancel", type="secondary"):
            st.session_state.reset_stage = "request"
            st.session_state.pop("reset_ident", None)
            st.session_state.pop("reset_code", None)
            st.session_state.pop("reset_note", None)
            st.rerun()


def render_auth_gateway():
    st.markdown(
        """<div class="hero">
<div class="hero-title">🛡️ Proactive Biometric Landmark Blinding Console</div>
<div class="hero-sub">Protecting portraits from deepfake harvesting by disrupting facial landmark alignment.</div>
</div>""",
        unsafe_allow_html=True,
    )

    action = st.radio(
        "Select Action",
        ["Sign in", "Create account", "Forgot password"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if action == "Sign in":
        _login_box()
    elif action == "Create account":
        _register_box()
    elif action == "Forgot password":
        _reset_box() 