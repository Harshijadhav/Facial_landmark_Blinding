import hashlib
import os

import cv2
import numpy as np
import streamlit as st

# Database, auth and vision modules
from database import find_audit_entry, get_user_audit_history, init_db, log_audit_trail
from auth_ui import render_auth_gateway
from src.detector import FaceDetector
from src.landmarks import LandmarkDetector
from src.pipeline import process_image
from src.evaluation import evaluate_images, paired_tests, summarize_conditions, summarize_robustness

# Initialize SQLite database schema
init_db()

st.set_page_config(
    page_title="Proactive Biometric Defense Console",
    page_icon="🛡️",
    layout="wide",
)

# Load External CSS
if os.path.exists("style.css"):
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Cache Vision Models to eliminate reload latency
@st.cache_resource
def load_vision_engines():
    return FaceDetector(), LandmarkDetector()


face_detector, landmark_detector = load_vision_engines()

SIZE_OPTIONS = {"Fast (1024 px)": 1024, "Balanced (1280 px)": 1280, "Full resolution": None}
MODE_OPTIONS = {
    "Anchor-targeted (recommended)": "anchors",
    "Auto-tuned for this photo (search)": "optimized",
}
DISPLAY_NAMES = {
    "Ours": "Our perturbation",
    "JPEG only": "JPEG only (control)",
    "Random noise": "Random noise, same energy (control)",
}
SUMMARY_SHORT = {
    "Landmarks lost (%)": "Lost %",
    "Mean key-point error (% eye dist)": "Mean error %",
    "Median key-point error (% eye dist)": "Median error %",
    "Mean alignment shift (px)": "Shift px",
    "Disrupted (%)": "Disrupted %",
    "Mean PSNR (dB)": "PSNR dB",
    "Mean PSNR in face region (dB)": "Face PSNR dB",
    "Mean SSIM": "SSIM",
}
LEVEL_LABEL = {"DISRUPTED": "🛡️ Disrupted", "PARTIAL": "🟠 Partial", "MINIMAL": "⚪ Minimal",
               "IMMUNIZED": "🛡️ Immunized (old label)"}


# ------------------------------------------------
# small helpers
# ------------------------------------------------
def badge(ok: bool, good_text: str, bad_text: str) -> str:
    cls = "badge-ok" if ok else "badge-bad"
    return f'<span class="badge {cls}">{good_text if ok else bad_text}</span>'


def panel_header(title: str, sub: str) -> None:
    st.markdown(
        f'<div class="panel-title">{title}</div><div class="panel-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def message_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="empty-card"><div class="empty-title">{title}</div>'
        f'<div class="empty-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def pct(x, digits=2):
    return "lost" if x is None else f"{x * 100:.{digits}f}%"


def num(x, digits=2, unit=""):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}{unit}"


def dot_for_level(level: str) -> str:
    return {"DISRUPTED": ":green[●]", "PARTIAL": ":orange[●]", "MINIMAL": ":red[●]"}[level]


def level_banner(cache) -> None:
    """Banner text is computed from the measurements, never fixed."""
    ours = cache["ours"]
    ctrl = cache["conditions"].get("JPEG only")
    level = cache["level"]

    if ours["nme_5pt"] is None:
        cls, text = "banner-ok", "<b>Disrupted.</b> No face landmarks could be found on the shielded image."
    else:
        p = ours["nme_5pt"] * 100
        if level == "DISRUPTED":
            cls = "banner-ok"
            text = (f"<b>Disrupted.</b> The 5 alignment points moved by {p:.1f}% of the eye distance, "
                    "enough to throw off face alignment.")
        elif level == "PARTIAL":
            cls = "banner-warn"
            text = (f"<b>Partial effect.</b> The alignment points moved by {p:.2f}% of the eye distance. "
                    "Alignment is slightly off, but a swap tool would probably still work.")
        else:
            cls = "banner-bad"
            ctrl_txt = ""
            if ctrl and ctrl["nme_5pt"] is not None:
                ctrl_txt = f" JPEG compression alone moved them {ctrl['nme_5pt'] * 100:.2f}%."
            text = (f"<b>Minimal effect.</b> The alignment points moved by only {p:.2f}% of the eye "
                    f"distance, close to normal detector noise.{ctrl_txt} Try a higher strength.")
    st.markdown(f'<div class="status-banner {cls}">{text}</div>', unsafe_allow_html=True)


# ------------------------------------------------
# tab renderers
# ------------------------------------------------
def render_empty():
    message_card("Upload a portrait to begin",
                 "Choose a JPG or PNG in the sidebar and set the sliders. The defense runs automatically "
                 "after the first upload. After that, change the settings and press Run to update.")


def render_no_face():
    message_card("No face found",
                 "No face landmarks were detected in this image. Try a clear, front-facing portrait.")


def render_results(cache, stale):
    if stale:
        st.markdown(
            '<div class="status-banner banner-warn"><b>Settings changed.</b> The results below use the '
            "previous settings. Click Run blinding defense to update them.</div>",
            unsafe_allow_html=True,
        )
    level_banner(cache)

    ours, conds, imgs = cache["ours"], cache["conditions"], cache["images"]
    how = "chosen automatically" if cache["auto_strength"] else "set by you"
    if cache["auto_strength"] and cache["calibration_reached"] is False:
        q_cls = "banner-warn"
        q_text = ("<b>Quality target cannot be reached on this photo.</b> Even the weakest pattern leaves the "
                  f"exported file at PSNR {ours['psnr']:.2f} dB or SSIM {ours['ssim']:.4f}, below the targets "
                  "(38 dB, 0.96).")
    elif cache["quality_ok"]:
        q_cls = "banner-ok"
        q_text = (f"<b>Quality target met.</b> Exported file: PSNR {ours['psnr']:.2f} dB (target 38) and "
                  f"SSIM {ours['ssim']:.4f} (target 0.96). Strength used: {cache['strength_used'] * 100:.1f}% ({how}). "
                  f"After a platform re-compresses it: PSNR {ours['psnr_platform']:.2f} dB, SSIM {ours['ssim_platform']:.4f}.")
    else:
        q_cls = "banner-warn"
        q_text = (f"<b>Quality target not met.</b> Exported file: PSNR {ours['psnr']:.2f} dB, SSIM {ours['ssim']:.4f}. "
                  "Lower the strength or turn on automatic strength.")
    st.markdown(f'<div class="status-banner {q_cls}">{q_text}</div>', unsafe_allow_html=True)
    ctrl = conds.get("JPEG only")

    m1, m2, m3 = st.columns(3)
    m1.metric("Landmarks found", f"{ours['landmarks_found']}",
              delta=f"{ours['landmarks_found'] - cache['orig_landmarks']:+d} vs original", delta_color="off")
    delta_txt = None
    if ours["nme_5pt"] is not None and ctrl and ctrl["nme_5pt"] is not None:
        delta_txt = f"{(ours['nme_5pt'] - ctrl['nme_5pt']) * 100:+.2f} points vs JPEG only"
    m2.metric("Key-point error (%)", pct(ours["nme_5pt"]), delta=delta_txt, delta_color="off",
              help="Average distance the 5 alignment points moved, divided by the distance between the eyes.")
    m3.metric("Alignment shift", num(ours["canonical_shift_px"], 2, " px"),
              help="How far the face would land from the right place inside the 112x112 frame deepfake tools use.")

    st.markdown('<div class="section-title">Compared with controls</div>', unsafe_allow_html=True)
    st.dataframe(
        [
            {
                "Condition": DISPLAY_NAMES[name],
                "Key-point error (% eye dist)": "lost" if c["nme_5pt"] is None else f"{c['nme_5pt'] * 100:.2f}",
                "Alignment shift (px)": num(c["canonical_shift_px"]),
                "PSNR export (dB)": "identical" if np.isinf(c["psnr"]) else num(c["psnr"]),
                "PSNR after platform (dB)": num(c["psnr_platform"]),
                "Result": LEVEL_LABEL[c["level"]],
            }
            for name, c in conds.items()
        ],
        hide_index=True, width="stretch",
    )
    st.caption("If the perturbation is not clearly above both controls, the pattern is doing nothing special.")

    hist = cache.get("opt_history")
    if hist:
        st.markdown('<div class="section-title">Search progress</div>', unsafe_allow_html=True)
        st.line_chart({"Best key-point error found so far (% of eye distance)": [100.0 * v for v in hist]})
        st.caption(f"The search tried {len(hist)} steps and kept the best pattern, at the same energy and the same "
                   "quality as the fixed pattern. A rising line means it found a pattern that moves the key points more.")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            panel_header("Original", f"Face mesh found: {cache['orig_landmarks']} landmarks")
            st.image(imgs["orig_annotated"], width="stretch")
    with c2:
        with st.container(border=True):
            panel_header("Shielded", f"After JPEG quality {cache['params'][2]}. Detected landmarks in red.")
            st.image(imgs["prot_view"], width="stretch")

    st.download_button(
        "Download protected image (.png)",
        cache["encoded_buf"].tobytes(),
        f"protected_{cache['crypto_hash'][:8]}.png",
        "image/png",
        type="primary",
    )

    with st.expander("Run details"):
        w, h = cache["size"]
        st.caption(f"Processed at {w} x {h} px in {cache['timings']['Total']:.1f} s.")
        st.dataframe([{"Stage": k, "Seconds": f"{v:.2f}"} for k, v in cache["timings"].items()],
                     hide_index=True, width="stretch")


def render_mask(cache):
    imgs, ours = cache["images"], cache["ours"]
    p1, p2, p3 = st.columns(3)
    with p1:
        with st.container(border=True):
            panel_header("Target region", "Face area that receives the pattern")
            st.image(imgs["mask_overlay"], width="stretch")
    with p2:
        with st.container(border=True):
            panel_header("Feathered mask", "Soft edges blend the pattern in")
            st.image(imgs["mask"], clamp=True, width="stretch")
    with p3:
        with st.container(border=True):
            panel_header("Amplified residual", "Difference from the original, boosted to be visible")
            st.image(imgs["diff_map"], width="stretch")

    st.markdown('<div class="section-title">Image quality</div>', unsafe_allow_html=True)
    det = cache["det_metrics"] or {}
    conf_v = float(det.get("confidence_drop", 0.0))
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("PSNR, whole image", num(ours["psnr"], 2, " dB"))
        st.markdown(badge(ours["psnr"] >= 38.0, "Meets standard (≥ 38.0 dB)", "Below standard (≥ 38.0 dB)"),
                    unsafe_allow_html=True)
    with b2:
        st.metric("PSNR, face only", num(ours["psnr_masked"], 2, " dB"),
                  help="Whole-image PSNR is flattered by the untouched background. This measures only where the pattern is.")
    with b3:
        st.metric("SSIM", f"{ours['ssim']:.4f}")
        st.markdown(badge(ours["ssim"] >= 0.96, "Meets standard (≥ 0.9600)", "Below standard (≥ 0.9600)"),
                    unsafe_allow_html=True)
    with b4:
        st.metric("Detector confidence", f"{conf_v:.2f}")
        st.markdown(badge(conf_v <= 0.05, "Face still detected", "Detector affected"), unsafe_allow_html=True)
    st.caption("Fidelity is measured on the file you download. After a platform re-compresses it "
               f"(JPEG {cache['params'][2]}): PSNR {ours['psnr_platform']:.2f} dB, SSIM {ours['ssim_platform']:.4f}.")


def render_stress(cache):
    imgs, ours = cache["images"], cache["ours"]
    st.caption("Deepfake tools such as InsightFace and SimSwap find 5 key points (eyes, nose, mouth corners) and "
               "fit a transform to a fixed 112 x 112 template. The same fit is computed here on both images.")

    a1, a2, a3 = st.columns(3)
    with a1:
        with st.container(border=True):
            panel_header("Original photo", "Reference alignment")
            st.image(imgs["aligned_orig"], width="stretch")
            st.markdown('<div class="shield-callout">Reference: the alignment a swap tool would compute '
                        "from the clean image.</div>", unsafe_allow_html=True)
    with a2:
        with st.container(border=True):
            panel_header("Shielded photo", "Alignment from the shielded image's own landmarks")
            st.image(imgs["aligned_prot"], width="stretch")
            level_banner(cache)
    with a3:
        with st.container(border=True):
            panel_header("Difference", "Between the two aligned crops, boosted to be visible")
            st.image(imgs["aligned_diff"], width="stretch")

    st.markdown('<div class="section-title">Measured alignment change</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Alignment shift", num(ours["canonical_shift_px"], 2, " px"),
              help="Average distance the face would land from the correct place inside the 112 x 112 frame.")
    k2.metric("Rotation error", num(ours["angle_err_deg"], 2, " °"))
    k3.metric("Scale change", num(ours["scale_ratio"], 3, " x"))
    k4.metric("Crop similarity (SSIM)", num(cache["aligned_ssim"], 3))

    st.markdown('<div class="section-title">Where the key points moved</div>', unsafe_allow_html=True)
    st.image(imgs["shift_view"], width="stretch")
    st.caption("Green dots: the 5 alignment points on the original photo. Red dots: the same points found on the "
               f"shielded photo, each shift magnified {cache.get('shift_magnification', 10)} times so it can be seen. "
               "A line joins each pair.")


def render_audit():
    st.caption("Your last 5 runs, stored in the local users.db file with a SHA-256 fingerprint of each output.")
    logs = get_user_audit_history(st.session_state.user["user_id"])
    if logs:
        rows = [
            {
                "Time": str(log["timestamp"])[:19],
                "Fingerprint (SHA-256)": log["image_hash"],
                "PSNR (dB)": f"{log['psnr']:.2f}",
                "SSIM": f"{log['ssim']:.4f}",
                "Result": LEVEL_LABEL.get(log["defense_status"], log["defense_status"].title()),
            }
            for log in logs
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        message_card("No runs recorded yet", "Every defense run is logged here automatically.")

    st.markdown('<div class="section-title">Verify a file</div>', unsafe_allow_html=True)
    st.caption("Choose a protected image you downloaded earlier. The app recomputes its SHA-256 fingerprint "
               "and looks for it in your audit log.")
    checked = st.file_uploader("File to verify", key="verify_file", label_visibility="collapsed")
    if checked is not None:
        digest = hashlib.sha256(checked.getvalue()).hexdigest().upper()
        entry = find_audit_entry(st.session_state.user["user_id"], digest[:16])
        if entry:
            st.success(f"Match. This exact file was produced by your account on {str(entry['timestamp'])[:19]} "
                       f"(PSNR {entry['psnr']:.2f} dB, SSIM {entry['ssim']:.4f}, result "
                       f"{LEVEL_LABEL.get(entry['defense_status'], entry['defense_status'])}). Fingerprint {digest[:16]}.")
        else:
            st.error("No match. This file was not produced by your account's runs, or it has been changed since "
                     "(even re-saving or re-compressing changes the fingerprint).")


def _read_image_bytes(data: bytes):
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


ALLOWED_EXT = (".jpg", ".jpeg", ".png")


def check_upload(uploaded):
    """Returns (problem, image). problem is (title, message) or None."""
    name = uploaded.name
    ext = os.path.splitext(name)[1].lower() or "no extension"
    if ext not in ALLOWED_EXT:
        return ("This is not a JPG or PNG photo",
                f"'{name}' is a {ext} file. Please choose a JPG or PNG picture of a face."), None
    img = _read_image_bytes(uploaded.getvalue())
    if img is None:
        return ("The photo can't be opened",
                f"'{name}' looks damaged, or it is not a real image. Please try another JPG or PNG."), None
    if min(img.shape[:2]) < 100:
        return ("The photo is too small",
                f"The shortest side is only {min(img.shape[:2])} px. Please use a photo at least 100 px wide and high, "
                "with the face clearly visible."), None
    return None, img


@st.dialog("Please check your photo")
def _upload_dialog(title, message):
    st.markdown(f"**{title}**")
    st.write(message)
    if st.button("Choose another photo", type="primary", width="stretch"):
        st.session_state.uploader_key += 1
        st.session_state.defense_cache = None
        st.session_state.last_processed_file = None
        st.rerun()


def render_evaluation(strength, jpeg_quality, max_side, mode, opt_steps=25, auto=False, *_):
    from src.alignment import FAIL_NME, PARTIAL_NME

    st.caption("Runs the defense on many photos and compares it with two controls: JPEG only and random noise.")
    res = st.session_state.get("eval_result")

    # ---------------- 1. Setup (folds away once there are results) ----------------
    with st.expander("Setup: choose photos and run", expanded=res is None):
        files = st.file_uploader("Add test images", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="eval_files")
        folder = st.text_input("Or read images from a folder (leave empty to skip)", value="")
        o1, o2 = st.columns(2)
        max_n = o1.number_input("Maximum number of images", min_value=1, max_value=500, value=30)
        with_rob = o2.checkbox("Include robustness tests (slower)", value=True)
        st.caption(f"Uses the sidebar settings: strength {int(round(strength * 100))}%, "
                   f"JPEG quality {jpeg_quality}, size limit {max_side or 'none'}, pattern: {mode}.")
        run = st.button("Run evaluation", type="primary")

    if run:
        items = []
        for f in files or []:
            img = _read_image_bytes(f.getvalue())
            if img is not None:
                items.append((f.name, img))
        if folder.strip() and os.path.isdir(folder.strip()):
            for name in sorted(os.listdir(folder.strip())):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    with open(os.path.join(folder.strip(), name), "rb") as fh:
                        img = _read_image_bytes(fh.read())
                    if img is not None:
                        items.append((name, img))
        items = items[: int(max_n)]

        if not items:
            st.warning("No images to evaluate. Add images or give a valid folder.")
        else:
            bar = st.progress(0.0, text="Starting")

            def on_progress(i, n, name):
                bar.progress(min(i / n, 1.0) if n else 1.0, text=f"Processing {name} ({min(i + 1, n)} of {n})")

            per_image, robust, skipped = evaluate_images(
                items, landmark_detector, face_detector, strength, jpeg_quality,
                max_side=max_side, robustness=with_rob, progress=on_progress, mode=mode,
                opt_iterations=opt_steps, auto_strength=auto)
            bar.empty()
            st.session_state.eval_result = {
                "per_image": per_image, "robust": robust, "skipped": skipped, "n": len(items),
                "params": (int(round(strength * 100)), jpeg_quality, max_side, mode),
            }

    res = st.session_state.get("eval_result")
    if not res:
        return

    p = res["params"]
    used = res["n"] - len(res["skipped"])
    summary = summarize_conditions(res["per_image"])
    if summary.empty:
        st.warning("No usable images in this run.")
        return
    paired = paired_tests(res["per_image"])

    # ---------------- 2. One-line verdict + three big numbers ----------------
    ours = summary[summary["Condition"] == "Ours"].iloc[0]
    ours_err = float(ours["Mean key-point error (% eye dist)"])

    if not np.isfinite(ours_err):
        reach = "No landmarks were found on the shielded images."
    elif ours_err >= FAIL_NME * 100:
        reach = f"The mean error ({ours_err:.2f}%) reaches the Disrupted line ({FAIL_NME * 100:g}%)."
    elif ours_err >= PARTIAL_NME * 100:
        reach = f"The mean error ({ours_err:.2f}%) reaches the Partial line ({PARTIAL_NME * 100:g}%)."
    else:
        reach = f"The mean error ({ours_err:.2f}%) stays below the Partial line ({PARTIAL_NME * 100:g}%)."

    if paired.empty:
        cls = "banner-warn"
        text = "<b>Not enough photos for a statistical test.</b> Use at least 3, and ideally 50 or more."
    else:
        higher = sum(str(v).startswith("Ours is higher") for v in paired["Verdict"])
        if higher == len(paired):
            cls, text = "banner-ok", f"<b>Better than both controls.</b> {reach}"
        else:
            cls = "banner-bad"
            text = (f"<b>Not clearly better than both controls.</b> {reach} "
                    "The effect cannot be told apart from ordinary JPEG and detector noise at these settings.")
    st.markdown(f'<div class="status-banner {cls}">{text}</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Photos used", f"{used} of {res['n']}")
    k2.metric("Our key-point error", num(ours_err, 2, "%"), help="Share of the eye distance the 5 alignment points moved.")
    k3.metric("Images disrupted", f"{float(ours['Disrupted (%)']):.0f}%")
    k4.metric("Mean PSNR", num(float(ours["Mean PSNR (dB)"]), 2, " dB"), help="Quality target is 38 dB or higher.")
    st.caption(f"Strength {p[0]}%, JPEG {p[1]}, pattern {p[3]}."
               + (f" Skipped (no face found): {', '.join(res['skipped'])}." if res["skipped"] else ""))

    # ---------------- 3. Results in small tabs ----------------
    t_sum, t_stat, t_rob, t_img = st.tabs(["Summary", "Statistics", "After post-processing", "Per-image"])

    with t_sum:
        col = "Mean key-point error (% eye dist)"
        try:
            import altair as alt
            import pandas as pd

            data = summary[["Condition", col]].rename(columns={col: "Error"})
            top = max(FAIL_NME * 100 * 1.15, float(data["Error"].max()) * 1.1)
            bars = alt.Chart(data).mark_bar().encode(
                y=alt.Y("Condition:N", sort=None, title=None),
                x=alt.X("Error:Q", title="Mean key-point error (% of eye distance)",
                        scale=alt.Scale(domain=[0, top])))
            lines = pd.DataFrame({
                "Threshold": [f"Partial effect: {PARTIAL_NME * 100:g}%", f"Disrupted: {FAIL_NME * 100:g}%"],
                "x": [PARTIAL_NME * 100, FAIL_NME * 100]})
            rules = alt.Chart(lines).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
                x="x:Q", color=alt.Color("Threshold:N", title=None))
            try:
                st.altair_chart(bars + rules, width="stretch")
            except TypeError:
                st.altair_chart(bars + rules, use_container_width=True)
        except Exception:
            st.bar_chart(summary.set_index("Condition")[[col]], horizontal=True)

        short_cols = ["Condition", col, "Disrupted (%)", "Mean PSNR (dB)"]
        st.dataframe(summary[short_cols].rename(columns=SUMMARY_SHORT), hide_index=True, width="stretch")
        with st.expander("More columns"):
            st.dataframe(summary.rename(columns=SUMMARY_SHORT), hide_index=True, width="stretch")

    with t_stat:
        if paired.empty:
            st.info("The statistical test needs at least 3 usable photos.")
        else:
            st.dataframe(paired, hide_index=True, width="stretch")
            st.caption("A p-value below 0.05 means the difference is unlikely to be chance.")

    with t_rob:
        if res["robust"].empty:
            st.info("Robustness tests were switched off or gave no data.")
        else:
            st.dataframe(summarize_robustness(res["robust"]), hide_index=True, width="stretch")

    with t_img:
        st.dataframe(res["per_image"], hide_index=True, width="stretch")

    # ---------------- 4. One help box and one download area ----------------
    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Key-point error** is how far the 5 alignment points (eyes, nose, mouth corners) moved, "
            "as a percentage of the distance between the eyes. Higher means more disruption.\n"
            "- The **dashed lines** are where this app calls a result Partial or Disrupted.\n"
            "- The defense only counts if it is clearly above **both** controls and reaches these lines.\n"
            "- **Alignment shift** is measured inside the 112 x 112 frame that deepfake tools use."
        )

    d1, d2 = st.columns(2)
    d1.download_button("Download per-image results (.csv)",
                       res["per_image"].to_csv(index=False).encode("utf-8"),
                       "evaluation_per_image.csv", "text/csv")
    d2.download_button("Download summary (.csv)",
                       summary.to_csv(index=False).encode("utf-8"),
                       "evaluation_summary.csv", "text/csv")


# ------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------
for key, default in (("user", None), ("defense_cache", None),
                     ("last_processed_file", None), ("eval_result", None)):
    if key not in st.session_state:
        st.session_state[key] = default
if "logged_hashes" not in st.session_state:
    st.session_state.logged_hashes = set()

# ==============================================================================
# SECTION 1: BRIEFING + LOGIN / REGISTER / FORGOT PASSWORD (see auth_ui.py)
# ==============================================================================
if st.session_state.user is None:
    render_auth_gateway()
    st.stop()

# ==============================================================================
# SECTION 2: THE RESEARCH & DEFENSE STUDIO
# ==============================================================================
username = st.session_state.user["username"]

# ---- Sidebar: Session ----
st.sidebar.markdown('<div class="side-title">Session</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"Signed in as `{username}`")
st.sidebar.caption("Runs locally in memory. No image leaves this computer.")
if st.sidebar.button("Sign out", width="stretch"):
    st.session_state.user = None
    st.session_state.defense_cache = None
    st.session_state.last_processed_file = None
    st.session_state.auth_view = "login"
    st.rerun()

# ---- Sidebar: Parameters ----
st.sidebar.markdown('<div class="side-title">Defense settings</div>', unsafe_allow_html=True)
auto_strength = st.sidebar.toggle(
    "Set the strength automatically", value=True,
    help="On: the app picks the strongest pattern that still keeps the exported photo at PSNR 38 dB or better "
         "and SSIM 0.96 or better. Off: you choose the strength yourself.",
)
strength = 0.08
if not auto_strength:
    strength = st.sidebar.slider(
        "Perturbation budget (strength %)", 1, 30, 8,
        help="How strong the hidden pattern is. Higher moves the key points more but lowers photo quality.",
    ) / 100.0
jpeg_quality = st.sidebar.slider(
    "Platform re-compression (JPEG quality)", 40, 95, 75,
    help="The quality a social platform re-compresses your photo to after you post it. The landmark tests run on this re-compressed copy; the photo you download is saved losslessly as PNG.",
)
size_label = st.sidebar.selectbox(
    "Working image size", list(SIZE_OPTIONS.keys()), index=1,
    help="Photos larger than this (longest side, in pixels) are shrunk before processing, which makes runs faster. The protected image you download has this size too. Choose Full resolution to keep the original size (slower).",
)
max_side = SIZE_OPTIONS[size_label]
mode_label = st.sidebar.selectbox(
    "Pattern type", list(MODE_OPTIONS.keys()), index=0,
    help="Colour pattern: the original pattern, which shifts the colour channels slightly. Brightness only: the same pattern added to brightness, as in the synopsis. Compare them in the Evaluation tab.",
)
mode = MODE_OPTIONS[mode_label]
opt_steps = 25
if mode == "optimized":
    opt_steps = st.sidebar.slider(
        "Search steps", 5, 60, 25,
        help="How many rounds the search runs. More steps can find a stronger pattern but take longer "
             "(about a second or two per step).")

# ---- Sidebar: Upload ----
st.sidebar.markdown('<div class="side-title">Portrait</div>', unsafe_allow_html=True)
st.session_state.setdefault("uploader_key", 0)
uploaded_file = st.sidebar.file_uploader(
    "Upload a portrait (JPG or PNG)", label_visibility="collapsed",
    key=f"portrait_{st.session_state.uploader_key}",
)

# ---- Page header ----
st.markdown(
    """<div class="hero">
<div class="hero-title">🛡️ Proactive Biometric Landmark Blinding Studio</div>
<div class="hero-sub">Protect a portrait from deepfake tools by disturbing the face points they depend on.</div>
</div>""",
    unsafe_allow_html=True,
)

cache = None
stale = False
upload_problem = None

if uploaded_file is None:
    st.session_state.defense_cache = None
    st.session_state.last_processed_file = None
else:
    problem, image = check_upload(uploaded_file)
    if problem:
        upload_problem = problem
        st.session_state.defense_cache = None
        st.session_state.last_processed_file = None
    else:
        # Invalidate the cache when a different image is uploaded
        file_sig = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_processed_file != file_sig:
            st.session_state.defense_cache = None
            st.session_state.last_processed_file = file_sig

        trigger_defense = st.sidebar.button("Run blinding defense", type="primary", width="stretch")
        params = (file_sig, round(float(strength), 3), int(jpeg_quality), max_side, mode, opt_steps, auto_strength)

        # Heavy computation runs once (first upload) or when the Run button is pressed
        if trigger_defense or st.session_state.defense_cache is None:
            with st.spinner("Running the defense and measuring the result..."):
                result = process_image(image, landmark_detector, face_detector, strength, jpeg_quality,
                                       max_side=max_side, mode=mode, opt_iterations=opt_steps,
                                       auto_strength=auto_strength)
            result["params"] = params
            st.session_state.defense_cache = result

            # One audit row per distinct output
            if result["face_found"] and result["crypto_hash"] not in st.session_state.logged_hashes:
                log_audit_trail(
                    user_id=st.session_state.user["user_id"],
                    image_hash=result["crypto_hash"][:16],
                    psnr=float(result["ours"]["psnr"]),
                    ssim=float(result["ours"]["ssim"]),
                    status=result["level"],
                )
                st.session_state.logged_hashes.add(result["crypto_hash"])

        cache = st.session_state.defense_cache
        stale = cache["params"] != params

if upload_problem is None and cache is not None and not cache.get("face_found", True):
    upload_problem = ("No face found",
                      "The app could not find a face in this photo. Use a clear, front-facing photo of one person, "
                      "in good light, with the face filling a good part of the picture.")
if upload_problem:
    _upload_dialog(*upload_problem)

has_result = cache is not None and cache.get("face_found", False)

# ---- Navigation ----
if has_result:
    quality_ok = cache["ours"]["psnr"] >= 38.0 and cache["ours"]["ssim"] >= 0.96
    lvl_dot = dot_for_level(cache["level"])
    q_dot = ":green[●]" if quality_ok else ":orange[●]"
    labels = [
        f":material/shield: Results {lvl_dot}",
        f":material/layers: Mask and residual {q_dot}",
        f":material/science: Stress test {lvl_dot}",
        ":material/history: Audit history",
        ":material/bar_chart: Evaluation",
    ]
else:
    labels = [":material/shield: Results", ":material/layers: Mask and residual",
              ":material/science: Stress test", ":material/history: Audit history",
              ":material/bar_chart: Evaluation"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(labels)


def render_or_placeholder(fn, *args):
    if has_result:
        fn(*args)
    elif cache is not None:
        render_no_face()
    else:
        render_empty()


with tab1:
    render_or_placeholder(render_results, cache, stale)
with tab2:
    render_or_placeholder(render_mask, cache)
with tab3:
    render_or_placeholder(render_stress, cache)
with tab4:
    render_audit()
with tab5:
    render_evaluation(strength, jpeg_quality, max_side, mode, opt_steps, auto_strength)
