"""
Real (measured) face-alignment analysis.

Deepfake tools such as InsightFace/SimSwap align a face by fitting a similarity
transform from 5 detected key points to a fixed 112x112 template. This module
does exactly that on the original and on the protected image and measures how
far the two alignments disagree. Nothing here is simulated.
"""

import cv2
import numpy as np

# Standard 5-point template (ArcFace / InsightFace, 112x112):
# image-left eye, image-right eye, nose tip, image-left mouth corner, image-right mouth corner
CANONICAL_5PT = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)

# MediaPipe Face Mesh indices
IDX_EYE_OUTER_A, IDX_EYE_OUTER_B = 33, 263     # outer eye corners (used for eye distance)
IDX_IRIS_A, IDX_IRIS_B = 468, 473              # iris centres (only present with refine_landmarks=True)
IDX_NOSE, IDX_MOUTH_A, IDX_MOUTH_B = 1, 61, 291

# Thresholds on the 5-point error, as a fraction of the outer eye distance.
# 0.08 is the usual "failure" threshold in landmark benchmarks (e.g. 300W).
# 0.02 is a noise-floor choice made for this project; change it if you can justify another value.
FAIL_NME = 0.08
PARTIAL_NME = 0.02


def landmarks_to_array(face, width, height):
    """MediaPipe normalized landmarks -> (N, 2) pixel coordinates."""
    return np.array([(p.x * width, p.y * height) for p in face.landmark], dtype=np.float64)


def five_points(all_pts):
    """Pick the 5 alignment points (eyes, nose, mouth corners) from the full landmark array."""
    if len(all_pts) > IDX_IRIS_B:
        eyes = sorted([all_pts[IDX_IRIS_A], all_pts[IDX_IRIS_B]], key=lambda p: p[0])
    else:
        eyes = sorted([all_pts[IDX_EYE_OUTER_A], all_pts[IDX_EYE_OUTER_B]], key=lambda p: p[0])
    mouth = sorted([all_pts[IDX_MOUTH_A], all_pts[IDX_MOUTH_B]], key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], all_pts[IDX_NOSE], mouth[0], mouth[1]], dtype=np.float64)


def fit_similarity(src, dst):
    """
    Least-squares similarity transform (rotation + uniform scale + translation), Umeyama 1991.
    Returns (M, scale, angle_radians) where M is a 2x3 matrix with dst ~ M @ [src; 1].
    """
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = (sc ** 2).sum() / n
    cov = dc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1.0
    R = U @ D @ Vt
    scale = float((S * np.diag(D)).sum() / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * (R @ mu_s)
    M = np.hstack([scale * R, t[:, None]])
    angle = float(np.arctan2(R[1, 0], R[0, 0]))
    return M, scale, angle


def apply_transform(M, pts):
    return pts @ M[:, :2].T + M[:, 2]


def disruption_level(nme_5pt):
    """None means no landmarks were found at all."""
    if nme_5pt is None or nme_5pt >= FAIL_NME:
        return "DISRUPTED"
    if nme_5pt >= PARTIAL_NME:
        return "PARTIAL"
    return "MINIMAL"


def measure_alignment(ref_face, cond_face, width, height):
    """
    Compare the landmarks of a condition image (protected, JPEG-only, noise, ...)
    against the landmarks found on the original image.

    ref_face / cond_face: MediaPipe landmark lists (or None if no face was found).
    width, height: size of the ORIGINAL image (landmarks are normalized, so this also
    works when the condition image was resized).
    """
    ref_all = landmarks_to_array(ref_face, width, height)
    ref5 = five_points(ref_all)
    M_ref, s_ref, a_ref = fit_similarity(ref5, CANONICAL_5PT)
    eye_dist = float(np.linalg.norm(ref_all[IDX_EYE_OUTER_A] - ref_all[IDX_EYE_OUTER_B])) or 1.0

    if cond_face is None:
        return {
            "found": False, "landmarks_found": 0,
            "nme_5pt": None, "nme_all": None, "pixel_drift": None,
            "canonical_shift_px": None, "angle_err_deg": None, "scale_ratio": None,
            "level": "DISRUPTED", "M_ref": M_ref, "M_cond": None,
        }

    cond_all = landmarks_to_array(cond_face, width, height)
    cond5 = five_points(cond_all)
    M_c, s_c, a_c = fit_similarity(cond5, CANONICAL_5PT)

    n = min(len(ref_all), len(cond_all))
    per_point = np.linalg.norm(ref_all[:n] - cond_all[:n], axis=1)
    nme_all = float(per_point.mean() / eye_dist)
    nme_5 = float(np.linalg.norm(ref5 - cond5, axis=1).mean() / eye_dist)

    # Where would the *true* face points land in the 112x112 frame if the tool trusted the
    # protected image's alignment, compared with the alignment from the clean image?
    shift = float(np.linalg.norm(apply_transform(M_c, ref5) - apply_transform(M_ref, ref5), axis=1).mean())
    angle_err = float(abs((np.degrees(a_c - a_ref) + 180.0) % 360.0 - 180.0))

    return {
        "found": True,
        "landmarks_found": len(cond_face.landmark),
        "nme_5pt": nme_5,
        "nme_all": nme_all,
        "pixel_drift": float(per_point.mean()),
        "canonical_shift_px": shift,
        "angle_err_deg": angle_err,
        "scale_ratio": float(s_c / s_ref) if s_ref else None,
        "level": disruption_level(nme_5),
        "M_ref": M_ref,
        "M_cond": M_c,
    }


def warp_aligned(image, M, size=112):
    """Warp an image into the canonical frame using a 2x3 matrix from fit_similarity."""
    return cv2.warpAffine(
        image, M.astype(np.float32), (size, size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
