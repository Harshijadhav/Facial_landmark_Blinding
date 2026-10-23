"""
One pipeline, used by the app AND the batch evaluation, so what you show is what you measure.

Per image it runs the defense, then measures the result against two controls:
  - "JPEG only":     the original, re-encoded at the same JPEG quality (no perturbation)
  - "Random noise":  Gaussian noise of the same energy in the same face region, then the same JPEG
Without the controls you cannot tell whether a landmark shift comes from the perturbation
or from ordinary JPEG / detector jitter.
"""

import hashlib
import time

import cv2
import numpy as np

from src.alignment import five_points, landmarks_to_array, measure_alignment, warp_aligned
from src.mask import create_landmark_mask
from src.metrics import (
    calculate_psnr,
    calculate_psnr_masked,
    calculate_ssim,
    evaluate_detection_drop,
    generate_diff_visualization,
)
from src.optimizer import optimize_anchor_pattern
from src.perturbation import anchor_mask, apply_anchor_perturbation, apply_perturbation

CONDITION_ORDER = ["Ours", "JPEG only", "Random noise"]

# The photo the user exports is saved losslessly (PNG). Fidelity (PSNR, SSIM) is judged on that file,
# so it measures only what the pattern changed. A platform then re-compresses it as JPEG at jpeg_quality;
# the landmark tests run on that second version.
TARGET_PSNR = 38.0
TARGET_SSIM = 0.96


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def limit_size(image, max_side):
    """Downscale so the longest side is at most max_side (None = keep full resolution)."""
    if not max_side:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    s = max_side / longest
    return cv2.resize(image, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def jpeg_roundtrip(image, quality):
    """Encode to JPEG and decode again. Returns (decoded_image, encoded_buffer)."""
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), buf


def first_face(result):
    if result is not None and result.multi_face_landmarks:
        return result.multi_face_landmarks[0]
    return None


def _effective_alpha(mask):
    """Same softening that apply_perturbation applies to the mask."""
    alpha = mask.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    alpha = cv2.dilate(alpha, kernel, iterations=1)
    return cv2.GaussianBlur(alpha, (25, 25), 0)


def matched_noise_baseline(image, mask, delta, seed=0):
    """Gaussian noise in the same region with the same mean energy as the real perturbation."""
    rng = np.random.default_rng(seed)
    alpha = _effective_alpha(mask)[:, :, None]
    raw = rng.standard_normal(image.shape).astype(np.float32) * alpha
    target = float(np.mean(delta.astype(np.float64) ** 2))
    current = float(np.mean(raw.astype(np.float64) ** 2))
    scale = np.sqrt(target / current) if current > 0 else 0.0
    return np.clip(image.astype(np.float32) + raw * scale, 0, 255).astype(np.uint8)


def _quality(original, exported, platform, mask):
    return {
        "psnr": calculate_psnr(original, exported),
        "ssim": calculate_ssim(original, exported),
        "psnr_masked": calculate_psnr_masked(original, exported, mask),
        "psnr_platform": calculate_psnr(original, platform),
        "ssim_platform": calculate_ssim(original, platform),
    }


def calibrate_strength(image, mask, ref_face, mode, target_psnr=TARGET_PSNR, target_ssim=TARGET_SSIM,
                       lo=0.01, hi=0.40):
    """
    Largest strength whose EXPORTED file still meets the quality target (PSNR and SSIM).
    Returns (strength, target_reached). The pattern's energy sets the quality, so the fixed anchor
    pattern is used to find the strength even when the searched pattern will be used later.
    """
    h, w = image.shape[:2]
    fixed_mode = "anchors" if mode in ("anchors", "optimized") else mode
    anchors = None
    if fixed_mode == "anchors":
        all_pts = landmarks_to_array(ref_face, w, h)
        anchors = list(five_points(all_pts)) + [all_pts[133], all_pts[362]]

    def exported(s):
        if fixed_mode == "anchors":
            _, ref = apply_perturbation(image, mask, strength=s, mode="luminance")
            raw, _ = apply_anchor_perturbation(image, anchors, float(np.mean(ref.astype(np.float64) ** 2)))
        else:
            raw, _ = apply_perturbation(image, mask, strength=s, mode=fixed_mode)
        return raw

    if calculate_psnr(image, exported(lo)) < target_psnr:
        return lo, False
    if calculate_psnr(image, exported(hi)) >= target_psnr:
        best = hi
    else:
        a, b = lo, hi
        for _ in range(9):
            mid = (a + b) / 2.0
            a, b = (mid, b) if calculate_psnr(image, exported(mid)) >= target_psnr else (a, mid)
        best = a
    for _ in range(8):                      # SSIM is the slower check, so it is applied last
        if calculate_ssim(image, exported(best)) >= target_ssim:
            return best, True
        best *= 0.9
        if best < lo:
            return lo, False
    return best, False


def _detect(landmark_detector, bgr):
    return first_face(landmark_detector.detect_landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def _annotate_original(rgb, result_face):
    import mediapipe as mp  # imported lazily so the numeric core can be tested without it

    canvas = rgb.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        image=canvas,
        landmark_list=result_face,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    return canvas


def _shift_view(rgb, ref_face, cond_face, magnification=10):
    """Green = original 5 key points, red = key points on the shielded photo (shift magnified)."""
    h, w = rgb.shape[:2]
    scale = max(1.0, 700.0 / max(h, w))
    canvas = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    ref5 = five_points(landmarks_to_array(ref_face, w, h)) * scale
    if cond_face is None:
        return canvas
    cond5 = five_points(landmarks_to_array(cond_face, w, h)) * scale
    for a, b in zip(ref5, cond5):
        tip = a + (b - a) * magnification
        cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in tip), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(int(v) for v in a), 5, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(int(v) for v in tip), 5, (255, 60, 60), -1, cv2.LINE_AA)
    return canvas


def _aligned_view(image, M, size=240):
    """Aligned 112x112 crop shown at `size` px (RGB). Black canvas if alignment was impossible."""
    if M is None:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(canvas, "NO LANDMARKS", (28, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 90, 90), 2)
        return canvas, None
    crop = warp_aligned(image, M, 112)
    shown = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), crop


# ------------------------------------------------------------------
# main entry point
# ------------------------------------------------------------------
def process_image(image, landmark_detector, face_detector, strength, jpeg_quality,
                  max_side=None, with_baselines=True, with_visuals=True, seed=0, mode="rgb", opt_iterations=25, auto_strength=False):
    clock = time.perf_counter
    t_start = clock()
    timings = {}

    image = limit_size(image, max_side)
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    t = clock()
    ref_face = first_face(landmark_detector.detect_landmarks(rgb))
    timings["Landmarks on original"] = clock() - t
    if ref_face is None:
        return {"face_found": False, "size": (w, h)}

    mask, mask_overlay = create_landmark_mask(image, ref_face)

    calibrated_ok = None
    if auto_strength:
        t_cal = clock()
        strength, calibrated_ok = calibrate_strength(image, mask, ref_face, mode)
        timings["Strength calibration"] = clock() - t_cal

    t = clock()
    opt_history = None
    if mode in ("anchors", "optimized"):
        # same energy as the face-wide brightness pattern, but only on 7 anchor points
        _, ref_delta = apply_perturbation(image, mask, strength=strength, mode="luminance")
        all_pts = landmarks_to_array(ref_face, w, h)
        anchors = list(five_points(all_pts)) + [all_pts[133], all_pts[362]]
        mask = anchor_mask(image.shape, anchors)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        red = np.zeros_like(image, dtype=np.float32)
        red[:, :, 2] = 255.0
        mask_overlay = (image.astype(np.float32) * (1.0 - 0.35 * alpha) + red * 0.35 * alpha).astype(np.uint8)
        target_energy = float(np.mean(ref_delta.astype(np.float64) ** 2))
        if mode == "anchors":
            raw_protected, delta = apply_anchor_perturbation(image, anchors, target_energy)
        else:
            t_opt = clock()
            delta_opt, opt_history = optimize_anchor_pattern(
                image, ref_face, lambda bgr: _detect(landmark_detector, bgr),
                mask.astype(np.float32) / 255.0, target_energy, jpeg_quality,
                iterations=opt_iterations, seed=seed)
            raw_protected = np.clip(image.astype(np.float32) + delta_opt, 0, 255).astype(np.uint8)
            delta = raw_protected.astype(np.float32) - image.astype(np.float32)
            timings["Optimization search"] = clock() - t_opt
    else:
        raw_protected, delta = apply_perturbation(image, mask, strength=strength, mode=mode)
    export_img = raw_protected
    _, buf = cv2.imencode(".png", export_img)
    protected, _ = jpeg_roundtrip(export_img, jpeg_quality)       # what a platform keeps
    timings["Perturbation + export"] = clock() - t

    t = clock()
    prot_face = _detect(landmark_detector, protected)
    timings["Landmarks on shielded"] = clock() - t

    candidates = {"Ours": (protected, prot_face, export_img)}

    if with_baselines:
        t = clock()
        jpeg_exp = image.copy()
        jpeg_img, _ = jpeg_roundtrip(jpeg_exp, jpeg_quality)
        candidates["JPEG only"] = (jpeg_img, _detect(landmark_detector, jpeg_img), jpeg_exp)
        noisy = matched_noise_baseline(image, mask, delta, seed=seed)
        noisy_exp = noisy
        noisy_img, _ = jpeg_roundtrip(noisy_exp, jpeg_quality)
        candidates["Random noise"] = (noisy_img, _detect(landmark_detector, noisy_img), noisy_exp)
        timings["Control conditions"] = clock() - t

    t = clock()
    conditions = {}
    for name, (img, face, exp) in candidates.items():
        conditions[name] = {**measure_alignment(ref_face, face, w, h), **_quality(image, exp, img, mask)}
    timings["Alignment + quality metrics"] = clock() - t

    det_metrics = None
    if face_detector is not None:
        det_metrics = evaluate_detection_drop(face_detector.detector, image, export_img)

    ours = conditions["Ours"]
    result = {
        "face_found": True,
        "size": (w, h),
        "orig_landmarks": len(ref_face.landmark),
        "conditions": conditions,
        "ours": ours,
        "level": ours["level"],
        "det_metrics": det_metrics,
        "encoded_buf": buf,
        "crypto_hash": hashlib.sha256(buf.tobytes()).hexdigest().upper(),
        "opt_history": opt_history,
        "strength_used": float(strength),
        "auto_strength": bool(auto_strength),
        "calibration_reached": calibrated_ok,
        "quality_ok": bool(ours["psnr"] >= TARGET_PSNR and ours["ssim"] >= TARGET_SSIM),
        "arrays": {"image": image, "raw_protected": raw_protected, "protected": protected,
                   "mask": mask, "ref_face": ref_face},
    }

    if with_visuals:
        t = clock()
        prot_rgb = cv2.cvtColor(protected, cv2.COLOR_BGR2RGB)
        if prot_face is not None:
            radius = max(1, int(round(min(h, w) / 500)))
            prot_view = prot_rgb.copy()
            for p in prot_face.landmark:
                cv2.circle(prot_view, (int(p.x * w), int(p.y * h)), radius, (255, 60, 60), -1)
        else:
            prot_view = prot_rgb

        a_orig, crop_orig = _aligned_view(image, ours["M_ref"])
        a_prot, crop_prot = _aligned_view(protected, ours["M_cond"])
        if crop_orig is not None and crop_prot is not None:
            from skimage.metrics import structural_similarity as _ssim
            aligned_ssim = float(_ssim(crop_orig, crop_prot, channel_axis=2, data_range=255))
            diff = cv2.absdiff(crop_orig, crop_prot).astype(np.float32)
            diff = np.clip(diff * (255.0 / max(float(diff.max()), 1.0)), 0, 255).astype(np.uint8)
            a_diff = cv2.cvtColor(cv2.resize(diff, (240, 240), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        else:
            aligned_ssim = None
            a_diff = np.zeros((240, 240, 3), dtype=np.uint8)

        result["aligned_ssim"] = aligned_ssim
        result["images"] = {
            "orig_annotated": _annotate_original(rgb, ref_face),
            "prot_view": prot_view,
            "mask_overlay": cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB),
            "mask": mask,
            "diff_map": cv2.cvtColor(generate_diff_visualization(image, export_img), cv2.COLOR_BGR2RGB),
            "aligned_orig": a_orig,
            "aligned_prot": a_prot,
            "aligned_diff": a_diff,
            "shift_view": _shift_view(rgb, ref_face, prot_face, 10),
        }
        result["shift_magnification"] = 10
        timings["Visuals"] = clock() - t

    timings["Total"] = clock() - t_start
    result["timings"] = timings
    return result
