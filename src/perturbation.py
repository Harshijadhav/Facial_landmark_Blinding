import cv2
import numpy as np


def apply_perturbation(image, mask, strength=0.15, seed=None, mode="rgb"):
    """
    Applies a structured periodic pattern inside the feathered face mask.

    mode="rgb"        : original pattern, colour channels shifted with opposite signs.
    mode="luminance"  : the same pattern added to brightness only (Y of YCrCb), as in the
                        synopsis. JPEG keeps brightness detail better than colour detail.

    This is NOT an optimized adversarial attack: the pattern is fixed, not tuned against a model.
    """
    if mode not in ("rgb", "luminance"):
        raise ValueError(f"unknown mode: {mode}")
    if seed is not None:
        np.random.seed(seed)

    h, w, _ = image.shape
    img_float = image.astype(np.float32)

    # 1. Expand and soften the mask to cover detection anchor zones
    if mask.dtype != np.float32:
        alpha = mask.astype(np.float32) / 255.0
    else:
        alpha = mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    alpha = cv2.dilate(alpha, kernel, iterations=1)
    alpha = cv2.GaussianBlur(alpha, (25, 25), 0)

    # 2. Form multi-frequency spatial wave
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    wave1 = np.sin(x * 0.12 + y * 0.08)
    wave2 = np.cos(x * 0.20 - y * 0.15)
    wave_low = (wave1 + wave2) / 2.0

    block_size = max(4, int(min(h, w) * 0.015))
    grid = ((x // block_size) % 2 == (y // block_size) % 2).astype(np.float32) * 2.0 - 1.0
    grid = cv2.GaussianBlur(grid, (5, 5), 0)

    signal = 0.55 * wave_low + 0.45 * grid

    # 3. Scale by the strength budget
    epsilon = strength * 255.0 * 0.35

    if mode == "luminance":
        # Adding the same value to B, G and R changes only Y; Cr and Cb stay untouched.
        # 0.87 matches the RMS energy of the colour version, so the same strength means the same energy.
        eps_y = epsilon * 0.87
        delta = np.repeat((signal * alpha * eps_y)[:, :, None], 3, axis=2)
    else:
        delta = np.zeros_like(img_float)
        delta[:, :, 0] = signal * alpha * epsilon           # Blue
        delta[:, :, 1] = -signal * alpha * (epsilon * 0.5)  # Green
        delta[:, :, 2] = -signal * alpha * epsilon          # Red

    protected_float = np.clip(img_float + delta, 0.0, 255.0)
    protected = protected_float.astype(np.uint8)

    actual_delta = protected_float - img_float
    return protected, actual_delta


# ----------------------------------------------------------------------
# Anchor-targeted pattern
# ----------------------------------------------------------------------
def anchor_mask(shape, anchors):
    """Feathered mask made of one Gaussian window per anchor point. Returns uint8 0..255."""
    h, w = shape[:2]
    radius = max(6.0, 0.04 * min(h, w))
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    for cx, cy in anchors:
        win = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * (0.8 * radius) ** 2))
        mask = np.maximum(mask, win)
    return (mask * 255.0).astype(np.uint8)


def apply_anchor_perturbation(image, anchors, target_energy):
    """
    Puts the whole energy budget on the alignment anchors instead of spreading it over the face.

    Around each anchor: oriented stripes whose sign flips across a line through the anchor
    (a local gradient inversion), added to brightness only. The result is scaled so that its mean
    squared change equals `target_energy` (taken from the face-wide pattern at the same strength),
    which makes the comparison with the face-wide pattern energy-matched.
    """
    h, w = image.shape[:2]
    radius = max(6.0, 0.04 * min(h, w))
    k = 0.95  # radians per pixel, a period of about 6.6 px
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    signal = np.zeros((h, w), dtype=np.float32)
    for i, (cx, cy) in enumerate(anchors):
        dx, dy = x - cx, y - cy
        win = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * (0.8 * radius) ** 2))
        theta = (np.pi / 4.0) * (i % 4)
        stripes = np.sin(k * (dx * np.cos(theta) + dy * np.sin(theta)))
        flip = np.sign(dx * np.cos(theta + np.pi / 2) + dy * np.sin(theta + np.pi / 2) + 1e-6)
        signal += win * stripes * flip

    delta = np.repeat(signal[:, :, None], 3, axis=2)
    energy = float(np.mean(delta.astype(np.float64) ** 2))
    scale = np.sqrt(target_energy / energy) if energy > 0 else 0.0

    img_float = image.astype(np.float32)
    protected_float = np.clip(img_float + delta * scale, 0.0, 255.0)
    return protected_float.astype(np.uint8), protected_float - img_float
