"""Frozen STEP06 preprocessing used by the deployment demo.

This is an ultrasound *fan* heuristic, NOT anatomical kidney segmentation.
The function matches STEP06 / original STEP16 inference:
    raw grayscale -> detect_roi -> centered square containing 80% of
    the smaller ROI-bounding-box dimension -> mask outside fan -> 224x224.

Important: the saved MobileNetV3Large contains a Rescaling(1/127.5,
-1) layer. Feed **float32 pixels in [0, 255]**, not [0, 1].
Do not apply CLAHE or a second external rescaling.
"""

import cv2
import numpy as np

ROI_THRESH = 12
MIN_ROI_FRAC = 0.12
FULLFRAME_FRAC = 0.92
MARGIN_FRAC = 0.06


def _margin_mask(h: int, w: int, frac: float = MARGIN_FRAC) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    dy, dx = int(round(h * frac)), int(round(w * frac))
    mask[dy:h - dy, dx:w - dx] = 1
    return mask


def detect_roi(gray: np.ndarray) -> tuple[np.ndarray, str]:
    """Return binary fan mask and heuristic status ('fan', 'fullframe', 'fallback')."""
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected grayscale uint8 image.")
    h, w = gray.shape
    if h < 32 or w < 32:
        raise ValueError("Image must be at least 32×32 pixels.")
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, ROI_THRESH, 255, cv2.THRESH_BINARY)
    k = max(3, int(round(min(h, w) * 0.03)))
    k += int(k % 2 == 0)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return _margin_mask(h, w), "fallback"
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (lab == j).astype(np.uint8)
    if comp.mean() < MIN_ROI_FRAC:
        return _margin_mask(h, w), "fallback"
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if mask.mean() >= FULLFRAME_FRAC:
        return mask * _margin_mask(h, w), "fullframe"
    return mask, "fan"


def preprocess_for_prediction(gray: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the exact 224×224 grayscale uint8 input and fan ROI status."""
    mask, status = detect_roi(gray)
    x, y, w, h = cv2.boundingRect(mask)
    side = max(8, min(min(gray.shape), int(round(min(w, h) * 0.80))))
    left = int(np.clip(round(x + w / 2 - side / 2), 0, gray.shape[1] - side))
    top = int(np.clip(round(y + h / 2 - side / 2), 0, gray.shape[0] - side))
    patch = gray[top:top + side, left:left + side].copy()
    patch[mask[top:top + side, left:left + side] == 0] = 0
    resized = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA)
    return resized, status


def model_batch(image_u8: np.ndarray) -> np.ndarray:
    """(224,224) uint8 -> (1,224,224,3) float32 on 0–255 input scale."""
    if image_u8.shape != (224, 224) or image_u8.dtype != np.uint8:
        raise ValueError("Input must be preprocessed uint8 224×224.")
    return np.repeat(image_u8[None, :, :, None].astype(np.float32), 3, axis=-1)
