import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(original, protected):
    if np.array_equal(original, protected):
        return float("inf")
    return float(psnr(original, protected, data_range=255))


def calculate_psnr_masked(original, protected, mask, threshold=26):
    """
    PSNR computed only inside the perturbed region (feathered mask above `threshold`/255).
    Whole-image PSNR is flattered by the untouched background; this one is not.
    """
    region = mask > threshold
    if not np.any(region):
        return float("inf")
    diff = original.astype(np.float64)[region] - protected.astype(np.float64)[region]
    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def calculate_ssim(original, protected):
    return float(ssim(original, protected, channel_axis=2, data_range=255))


def calculate_mae(original, protected):
    diff = np.abs(original.astype(np.float32) - protected.astype(np.float32))
    return float(np.mean(diff))


def generate_diff_visualization(original, protected):
    diff = np.abs(original.astype(np.float32) - protected.astype(np.float32))
    diff_gray = np.max(diff, axis=2)
    max_val = np.max(diff_gray)
    if max_val > 0:
        diff_scaled = (diff_gray / max_val) * 255.0
    else:
        diff_scaled = diff_gray
    return cv2.applyColorMap(diff_scaled.astype(np.uint8), cv2.COLORMAP_JET)


def evaluate_detection_drop(detector, original, protected):
    """
    Measures face detector confidence drop post-protection.
    """
    def get_best_confidence(image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        if not result or not result.detections:
            return 0.0, 0
        scores = [d.score[0] for d in result.detections]
        return float(max(scores)), len(result.detections)

    orig_conf, orig_faces = get_best_confidence(original)
    prot_conf, prot_faces = get_best_confidence(protected)

    return {
        "original_confidence": orig_conf,
        "protected_confidence": prot_conf,
        "confidence_drop": max(0.0, orig_conf - prot_conf),
        "original_faces": orig_faces,
        "protected_faces": prot_faces
    }


def evaluate_landmark_detection(landmark_detector, original, protected, face_detector=None, confidence_thresh=0.75):
    """
    LEGACY helper, no longer used by the app (see src/alignment.py and src/pipeline.py).

    Kept for reference. The "is_blinded" rule here (drift >= 0.50 px) is an arbitrary
    threshold, and "protected_faces" now reports the real number of faces found instead of
    forcing it to 0 when the rule fires.
    """
    h, w = original.shape[:2]
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    protected_rgb = cv2.cvtColor(protected, cv2.COLOR_BGR2RGB)

    orig_conf = 1.0
    prot_conf = 1.0
    if face_detector is not None:
        det_orig = face_detector.process(original_rgb)
        if det_orig and det_orig.detections:
            orig_conf = float(max([d.score[0] for d in det_orig.detections]))

        det_prot = face_detector.process(protected_rgb)
        if det_prot and det_prot.detections:
            prot_conf = float(max([d.score[0] for d in det_prot.detections]))
        else:
            prot_conf = 0.0

    original_result = landmark_detector.detect_landmarks(original_rgb)
    protected_result = landmark_detector.detect_landmarks(protected_rgb)

    orig_detected = bool(original_result and original_result.multi_face_landmarks)
    raw_prot_detected = bool(protected_result and protected_result.multi_face_landmarks)

    norm_drift = 0.0
    pixel_drift = 0.0

    if orig_detected and raw_prot_detected:
        orig_pts = original_result.multi_face_landmarks[0].landmark
        prot_pts = protected_result.multi_face_landmarks[0].landmark
        pts_count = min(len(orig_pts), len(prot_pts))

        distances = []
        px_distances = []
        for i in range(pts_count):
            dx = orig_pts[i].x - prot_pts[i].x
            dy = orig_pts[i].y - prot_pts[i].y
            distances.append(np.hypot(dx, dy))
            px_distances.append(np.hypot(dx * w, dy * h))

        if distances:
            norm_drift = float(np.mean(distances))
            pixel_drift = float(np.mean(px_distances))

    conf_drop = orig_conf - prot_conf
    is_blinded = (
        (not raw_prot_detected) or
        (pixel_drift >= 0.50) or
        (prot_conf < confidence_thresh) or
        (conf_drop >= 0.05)
    )

    orig_faces = len(original_result.multi_face_landmarks) if orig_detected else 0
    prot_faces = len(protected_result.multi_face_landmarks) if raw_prot_detected else 0

    return {
        "original_faces": orig_faces,
        "protected_faces": prot_faces,
        "landmark_position_change": norm_drift,
        "pixel_drift": pixel_drift,
        "is_blinded": is_blinded
    }


def evaluate_protection(original, protected):
    return {
        "psnr": calculate_psnr(original, protected),
        "ssim": calculate_ssim(original, protected),
        "mae": calculate_mae(original, protected)
    }
