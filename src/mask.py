import cv2
import numpy as np


# Key facial landmark indices for alignment anchors
LANDMARK_INDICES = [
    # Left eyebrow
    70, 63, 105, 66, 107,

    # Right eyebrow
    336, 296, 334, 293, 300,

    # Left eye
    33, 133, 159, 145,

    # Right eye
    362, 263, 386, 374,

    # Nose & bridge
    1, 2, 98, 327,

    # Mouth & lips
    61, 291, 13, 14,

    # Chin & jaw base
    152
]


def create_landmark_mask(image, face_landmarks):
    """
    Creates an adaptive, localized mask around key facial landmarks
    with smooth blending boundaries for imperceptible perturbation injection.
    """
    height, width = image.shape[:2]

    # Dynamically scale patch radius relative to image dimensions
    base_dim = min(height, width)
    patch_radius = max(8, int(base_dim * 0.025))
    kernel_size = max(9, int(base_dim * 0.035))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Initialize empty 8-bit mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Plot landmark centers
    for index in LANDMARK_INDICES:
        landmark = face_landmarks.landmark[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(mask, (x, y), patch_radius, 255, -1)

    # Dilate slightly to bridge adjacent landmark anchors
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Gaussian blur for feathering to eliminate visible seams
    blur_ksize = kernel_size * 2 + 1
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    # Vectorized fast overlay for Streamlit visualization
    alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    red_layer = np.zeros_like(image, dtype=np.float32)
    red_layer[:, :, 2] = 255.0  # BGR Red channel

    img_float = image.astype(np.float32)
    overlay = (img_float * (1.0 - 0.35 * alpha) + red_layer * (0.35 * alpha)).astype(np.uint8)

    return mask, overlay 