"""
Per-photo optimization of the anchor pattern (black-box, query-based).

The fixed patterns in perturbation.py are not tuned to any model. Here the pattern is searched
for one photo at a time: a small brightness field is placed on the anchor windows, rescaled to the
SAME energy budget as the fixed pattern, passed through JPEG, and scored by how far MediaPipe's
5 alignment points move. A natural-evolution-strategy search (antithetic Gaussian samples) improves
the field step by step. No gradients are used, only repeated calls to the real detector.

Same energy, same quality limit: any gain over the fixed anchor pattern comes from the pattern's
shape, not from extra noise.
"""
import cv2
import numpy as np

from src.alignment import IDX_EYE_OUTER_A, IDX_EYE_OUTER_B, five_points, landmarks_to_array

GRID = 24            # the field is GRID x GRID, enlarged smoothly to the photo size
LOST_REWARD = 0.20   # a photo whose face is lost counts as 20% error (above the 8% failure line)


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2))) or 1.0


def make_delta(z, window, target_energy):
    """Brightness change from field z, confined to the anchor windows, with fixed mean energy."""
    h, w = window.shape
    field = cv2.resize(z.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * window
    delta = np.repeat(field[:, :, None], 3, axis=2)
    energy = float(np.mean(delta.astype(np.float64) ** 2))
    if energy <= 0:
        return np.zeros_like(delta)
    return delta * np.float32(np.sqrt(target_energy / energy))


def _jpeg(image, quality):
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def optimize_anchor_pattern(image, ref_face, detect, window, target_energy, jpeg_quality,
                            iterations=25, pairs=8, sigma=0.5, step=0.6, seed=0, on_step=None,
                            export_quality=None):
    """
    image: BGR uint8. ref_face: landmarks of the clean image. detect(bgr) -> landmarks or None.
    window: float32 mask 0..1 (anchor windows). Returns (delta, history), where history[i] is the
    best key-point error (fraction of eye distance) found after step i.
    """
    h, w = image.shape[:2]
    all_ref = landmarks_to_array(ref_face, w, h)
    ref5 = five_points(all_ref)
    eye_dist = float(np.linalg.norm(all_ref[IDX_EYE_OUTER_A] - all_ref[IDX_EYE_OUTER_B])) or 1.0
    base = image.astype(np.float32)

    def score(z):
        delta = make_delta(z, window, target_energy)
        exported = np.clip(base + delta, 0, 255).astype(np.uint8)
        if export_quality:
            exported = _jpeg(exported, export_quality)
        candidate = _jpeg(exported, jpeg_quality)
        face = detect(candidate)
        if face is None:
            return LOST_REWARD
        pts5 = five_points(landmarks_to_array(face, w, h))
        return float(np.linalg.norm(ref5 - pts5, axis=1).mean() / eye_dist)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((GRID, GRID))
    z /= _rms(z)
    best_f, best_z = -1.0, z.copy()
    history = []

    for it in range(iterations):
        f0 = score(z)
        if f0 > best_f:
            best_f, best_z = f0, z.copy()
        grad = np.zeros_like(z)
        for _ in range(pairs):
            e = rng.standard_normal(z.shape)
            f_plus, f_minus = score(z + sigma * e), score(z - sigma * e)
            grad += (f_plus - f_minus) * e
            for cand, fc in ((z + sigma * e, f_plus), (z - sigma * e, f_minus)):
                if fc > best_f:
                    best_f, best_z = fc, cand.copy()
        history.append(best_f)
        if on_step:
            on_step(it + 1, iterations, best_f)
        g = _rms(grad)
        if np.any(grad):
            z = z + step * grad / g
            z /= _rms(z)

    return make_delta(best_z, window, target_energy), history
