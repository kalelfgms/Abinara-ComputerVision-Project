"""Task 3: detect every object (falling diamonds) on a black background,
without classifying them."""

import cv2
import numpy as np

THRESH_MARGIN = 30        # threshold = background median + margin
MIN_THRESH = 25           # never go below this (compression noise)
MIN_AREA_RATIO = 0.00001  # smallest blob as a fraction of the frame
MIN_AREA_PX = 20          # ... and in pixels (full-res equivalent)
CLOSE_RATIO = 0.006       # closing kernel size relative to frame width
BOX_COLOR = (0, 255, 0)


def detect(frame, proc_width=960):
    """Returns ([(x, y, w, h), ...] in full-res coordinates, mask)."""
    h, w = frame.shape[:2]
    scale = 1.0
    if proc_width and w > proc_width:
        scale = proc_width / w
        small = cv2.resize(frame, (proc_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame

    # brightest channel keeps the dark-green facets visible
    v = cv2.GaussianBlur(small.max(axis=2), (3, 3), 0)
    t = max(MIN_THRESH, float(np.median(v)) + THRESH_MARGIN)
    _, mask = cv2.threshold(v, t, 255, cv2.THRESH_BINARY)

    k = max(3, int(small.shape[1] * CLOSE_RATIO) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(MIN_AREA_PX * scale * scale, MIN_AREA_RATIO * small.shape[0] * small.shape[1])
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)))
    return boxes, mask


def draw(frame, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
    return frame
