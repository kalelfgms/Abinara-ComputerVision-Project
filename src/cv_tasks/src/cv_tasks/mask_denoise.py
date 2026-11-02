"""Task 4: remove speckle noise from a noisy mask video (rotating white donut
with white dots on black) while keeping the object and its real hole."""

import cv2
import numpy as np

from .common import put_label

REL_KEEP = 0.15          # keep blobs >= 15% of the largest blob
MIN_AREA_RATIO = 0.002   # ... and >= 0.2% of the frame
HOLE_REL = 0.01          # background holes < 1% of the object area are noise
SMOOTH_K = 7             # median blur size (odd)


def keep_large_components(binary, rel_keep, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    thresh = max(min_area, rel_keep * areas.max())
    lut = np.zeros(n, np.uint8)
    lut[1:] = (areas >= thresh).astype(np.uint8) * 255
    return lut[labels], int((areas >= thresh).sum())


def fill_small_holes(mask, hole_rel):
    """Fill small background islands enclosed by the object."""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), connectivity=4)
    if n <= 1:
        return mask
    max_hole = hole_rel * cv2.countNonZero(mask)
    lut = np.zeros(n, np.uint8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and area < max_hole:
            lut[i] = 255
    return cv2.bitwise_or(mask, lut[labels])


def clean(frame, proc_width=0):
    """Returns (clean binary mask at the frame's resolution, number of objects)."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    small = gray
    if proc_width and w > proc_width:
        small = cv2.resize(gray, (proc_width, int(h * proc_width / w)), interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(small, (3, 3), 0)
    otsu, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu < 60:  # Otsu failed (almost empty frame)
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    mask, n_obj = keep_large_components(binary, REL_KEEP, MIN_AREA_RATIO * binary.size)
    mask = fill_small_holes(mask, HOLE_REL)
    mask = cv2.medianBlur(mask, SMOOTH_K | 1)

    if mask.shape != gray.shape:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask, n_obj


def _fit_width(img, width):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def process(frame, proc_width=0, view_width=640):
    """Side-by-side: noisy input | cleaned mask."""
    mask, n_obj = clean(frame, proc_width)
    left = _fit_width(frame, view_width)
    right = _fit_width(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), view_width)
    put_label(left, "Input (noisy)", org=(10, 30), scale=0.8, color=(0, 255, 255))
    put_label(right, f"Cleaned | objects {n_obj}", org=(10, 30), scale=0.8, color=(0, 255, 0))
    return np.hstack([left, right])
