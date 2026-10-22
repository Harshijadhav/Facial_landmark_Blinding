"""
Batch evaluation: run the same pipeline on many images and compare the defense with
two controls (JPEG only, random noise of equal energy), then stress it with
post-processing a platform might apply (re-compression, resize, blur).
"""

import cv2
import numpy as np 
import pandas as pd

from src.alignment import measure_alignment
from src.pipeline import CONDITION_ORDER, first_face, jpeg_roundtrip, process_image

TRANSFORMS = {
    "JPEG Q90": lambda img: jpeg_roundtrip(img, 90)[0],
    "JPEG Q75": lambda img: jpeg_roundtrip(img, 75)[0],
    "JPEG Q50": lambda img: jpeg_roundtrip(img, 50)[0],
    "Resize 50%": lambda img: cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
    "Blur 5x5": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
}

_COLUMNS = ["image", "condition", "found", "landmarks_found", "nme_5pt", "nme_all",
            "canonical_shift_px", "angle_err_deg", "psnr", "ssim", "psnr_masked", "level"]


def _detect(landmark_detector, bgr):
    return first_face(landmark_detector.detect_landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def evaluate_images(items, landmark_detector, face_detector, strength, jpeg_quality,
                    max_side=None, robustness=True, progress=None, mode="rgb", opt_iterations=25):
    """
    items: list of (name, BGR image).
    Returns (per_image_df, robustness_df, skipped_names).
    """
    rows, rob_rows, skipped = [], [], []

    for i, (name, bgr) in enumerate(items):
        if progress:
            progress(i, len(items), name)

        res = process_image(bgr, landmark_detector, face_detector, strength, jpeg_quality,
                            max_side=max_side, with_baselines=True, with_visuals=False, mode=mode,
                            opt_iterations=opt_iterations)
        if not res["face_found"]:
            skipped.append(name)
            continue

        for cond in CONDITION_ORDER:
            m = res["conditions"][cond]
            rows.append({"image": name, "condition": cond, **{k: m[k] for k in _COLUMNS[2:]}})

        if robustness:
            arrays = res["arrays"]
            w, h = res["size"]
            for t_name, fn in TRANSFORMS.items():
                for cond_name, source in (("Ours", arrays["raw_protected"]), ("Original (control)", arrays["image"])):
                    face = _detect(landmark_detector, fn(source))
                    m = measure_alignment(arrays["ref_face"], face, w, h)
                    rob_rows.append({
                        "image": name, "transform": t_name, "condition": cond_name,
                        "found": m["found"], "nme_5pt": m["nme_5pt"], "level": m["level"],
                    })

    if progress:
        progress(len(items), len(items), "done")

    per_image = pd.DataFrame(rows, columns=_COLUMNS)
    robust = pd.DataFrame(rob_rows, columns=["image", "transform", "condition", "found", "nme_5pt", "level"])
    return per_image, robust, skipped


def summarize_conditions(df):
    if df.empty:
        return pd.DataFrame()
    out = []
    for cond in CONDITION_ORDER:
        d = df[df["condition"] == cond]
        if d.empty:
            continue
        nme = d["nme_5pt"].dropna()
        out.append({
            "Condition": cond,
            "Images": int(len(d)),
            "Landmarks lost (%)": round(100.0 * float((d["landmarks_found"] == 0).mean()), 1),
            "Mean key-point error (% eye dist)": round(100.0 * float(nme.mean()), 2) if len(nme) else np.nan,
            "Median key-point error (% eye dist)": round(100.0 * float(nme.median()), 2) if len(nme) else np.nan,
            "Mean alignment shift (px)": round(float(d["canonical_shift_px"].mean()), 2),
            "Disrupted (%)": round(100.0 * float((d["level"] == "DISRUPTED").mean()), 1),
            "Mean PSNR (dB)": round(float(d["psnr"].replace([np.inf], np.nan).mean()), 2),
            "Mean PSNR in face region (dB)": round(float(d["psnr_masked"].replace([np.inf], np.nan).mean()), 2),
            "Mean SSIM": round(float(d["ssim"].mean()), 4),
        })
    return pd.DataFrame(out)


def summarize_robustness(rob):
    if rob.empty:
        return pd.DataFrame()
    r = rob.copy()
    r["nme_pct"] = 100.0 * r["nme_5pt"]
    table = r.pivot_table(index="transform", columns="condition", values="nme_pct", aggfunc="mean")
    lost = r.assign(lost=(~r["found"]).astype(float)).pivot_table(
        index="transform", columns="condition", values="lost", aggfunc="mean") * 100.0
    table.columns = [f"{c.replace(' (control)', '')} error %" for c in table.columns]
    lost.columns = [f"{c.replace(' (control)', '')} lost %" for c in lost.columns]
    return table.join(lost).round(2).reset_index().rename(columns={"transform": "Post-processing"})


def paired_tests(df):
    """
    Is our key-point error larger than each control, image by image?
    Uses the paired Wilcoxon signed-rank test (no normality assumption).
    """
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot(index="image", columns="condition", values="nme_5pt")
    out = []
    for control in ("JPEG only", "Random noise"):
        if "Ours" not in wide or control not in wide:
            continue
        pair = wide[["Ours", control]].dropna()
        if len(pair) < 3:
            continue
        diff = (pair["Ours"] - pair[control]).to_numpy()
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(pair["Ours"], pair[control]).pvalue)
        except Exception:
            p = float("nan")
        if np.isnan(p):
            verdict = "Not enough variation to test"
        elif p < 0.05 and diff.mean() > 0:
            verdict = "Ours is higher (p < 0.05)"
        elif p < 0.05:
            verdict = "Ours is lower (p < 0.05)"
        else:
            verdict = "No clear difference"
        out.append({
            "Comparison": f"Ours vs {control}",
            "Mean difference (points of % eye dist)": round(100.0 * float(diff.mean()), 3),
            "Ours larger in": f"{int((diff > 0).sum())} of {len(diff)} images",
            "p-value (Wilcoxon)": round(p, 4) if not np.isnan(p) else np.nan,
            "Verdict": verdict,
        })
    return pd.DataFrame(out)
