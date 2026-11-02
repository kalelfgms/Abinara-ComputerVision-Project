"""Task 1: detect triangle, square, pentagon, hexagon and circle on a
background whose color changes over time.

The background color is estimated every frame (dominant color), so a fixed
threshold is never needed. Everything that differs from it is a shape."""

import cv2
import numpy as np

DIFF_THRESH = 40          # min color distance from the background
MIN_AREA_RATIO = 0.002    # ignore blobs smaller than this fraction of the frame
CIRCLE_FILL_RATIO = 0.90  # area / min-enclosing-circle area (circle ~ 1.0)
EPS_FACTOR = 0.03         # approxPolyDP epsilon as a fraction of the perimeter

COLORS = {
    "Segitiga": (0, 255, 255),
    "Persegi": (0, 255, 0),
    "Pentagon": (255, 0, 255),
    "Heksagon": (255, 128, 0),
    "Lingkaran": (0, 0, 255),
}


def estimate_background(frame):
    """Dominant color: most frequent bin of a coarse 3D histogram."""
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    pix = small.reshape(-1, 3)
    q = (pix >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    mode_key = np.bincount(key).argmax()
    return pix[key == mode_key].mean(axis=0)


def segment_shapes(frame):
    bg = estimate_background(frame)
    diff = np.abs(frame.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
    mask = (diff > DIFF_THRESH).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def classify(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0 or area == 0:
        return None

    # a circle fills ~all of its enclosing circle (hexagon .83, pentagon .76,
    # square .64, triangle .41)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    fill_ratio = area / (np.pi * radius * radius)
    circularity = 4 * np.pi * area / (peri * peri)
    if fill_ratio > CIRCLE_FILL_RATIO and circularity > 0.88:
        return "Lingkaran"

    n = len(cv2.approxPolyDP(contour, EPS_FACTOR * peri, True))
    if n == 3:
        return "Segitiga"
    if n == 4:
        return "Persegi"
    if n == 5:
        return "Pentagon"
    if n == 6:
        return "Heksagon"
    if n > 6:
        return "Lingkaran" if fill_ratio > 0.85 else None
    return None


def detect(frame):
    """Returns ([(label, contour), ...], mask)."""
    h, w = frame.shape[:2]
    mask = segment_shapes(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_RATIO * h * w
    results = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        label = classify(c)
        if label is not None:
            results.append((label, c))
    return results, mask


def draw(frame, results):
    scale = max(0.6, frame.shape[1] / 1600)
    for label, c in results:
        color = COLORS[label]
        cv2.drawContours(frame, [c], -1, color, 3)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        m = cv2.moments(c)
        cx = int(m["m10"] / m["m00"]) if m["m00"] else x + w // 2
        cy = int(m["m01"] / m["m00"]) if m["m00"] else y + h // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6), (cx + tw // 2 + 4, cy + 6), (0, 0, 0), -1)
        cv2.putText(frame, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2, cv2.LINE_AA)
    return frame
