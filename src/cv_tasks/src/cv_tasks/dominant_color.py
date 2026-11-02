"""Task 2: find every unique color in a frame (counted per pixel) and segment
the most frequent one.

Colors are quantized into a 3D histogram, then perceptually similar bins are
merged (Lab distance) so anti-aliasing / compression noise does not split one
flat color into many "unique" colors."""

import cv2
import numpy as np

from .common import put_label

BITS = 4                 # bits per channel -> 16 levels, 4096 bins
MERGE_TOL = 14.0         # Lab distance under which two bins are the same color
MIN_FRAC = 0.002         # a color is "unique" if it covers >= 0.2% of the pixels
MAX_SEED_BINS = 256      # most frequent bins used to seed the clusters
TOP_K = 8                # colors shown in the palette panel

LEVELS = 1 << BITS
SHIFT = 8 - BITS
NBINS = LEVELS ** 3


def _bin_tables():
    idx = np.arange(NBINS)
    b = (idx >> (2 * BITS)) & (LEVELS - 1)
    g = (idx >> BITS) & (LEVELS - 1)
    r = idx & (LEVELS - 1)
    step = 1 << SHIFT
    centers = np.stack([b, g, r], axis=1) * step + step // 2
    bgr = centers.astype(np.float32).reshape(-1, 1, 3) / 255.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return centers.astype(np.float32), lab


BIN_BGR, BIN_LAB = _bin_tables()


def pixel_keys(img):
    q = img >> SHIFT
    return ((q[..., 0].astype(np.int32) << (2 * BITS))
            | (q[..., 1].astype(np.int32) << BITS)
            | q[..., 2].astype(np.int32))


def analyze(img):
    """Returns (keys, lut, clusters).

    lut[bin] -> cluster id (-1 if far from every cluster)
    clusters: list of dicts sorted by pixel count: id, count, frac, bgr
    """
    keys = pixel_keys(img)
    counts = np.bincount(keys.ravel(), minlength=NBINS)
    total = keys.size

    # greedy merge, most frequent bins first
    order = np.argsort(counts)[::-1][:MAX_SEED_BINS]
    order = order[counts[order] > 0]
    rep_bins, rep_lab = [], np.empty((0, 3), np.float32)
    for b in order:
        if len(rep_bins):
            if np.linalg.norm(rep_lab - BIN_LAB[b], axis=1).min() < MERGE_TOL:
                continue
        rep_bins.append(b)
        rep_lab = np.vstack([rep_lab, BIN_LAB[b]])

    # every bin -> nearest representative (or -1 if too far)
    dist = np.linalg.norm(BIN_LAB[:, None, :] - rep_lab[None, :, :], axis=2)
    lut = np.where(dist.min(axis=1) < MERGE_TOL * 1.5, dist.argmin(axis=1), -1).astype(np.int16)

    # per-cluster pixel totals (vectorized)
    k = len(rep_bins)
    nz = np.nonzero(counts)[0]
    cid = lut[nz]
    valid = cid >= 0
    nz, cid = nz[valid], cid[valid]
    w = counts[nz].astype(np.float64)
    cluster_counts = np.bincount(cid, weights=w, minlength=k)
    cluster_bgr = np.stack([np.bincount(cid, weights=w * BIN_BGR[nz, ch], minlength=k)
                            for ch in range(3)], axis=1)

    clusters = []
    for c in np.argsort(cluster_counts)[::-1]:
        if cluster_counts[c] / total < MIN_FRAC:
            continue
        clusters.append({
            "id": int(c),
            "count": int(cluster_counts[c]),
            "frac": float(cluster_counts[c] / total),
            "bgr": tuple(int(v) for v in cluster_bgr[c] / cluster_counts[c]),
        })
    return keys, lut, clusters


def segment(keys, lut, cluster_id):
    mask = (lut[keys] == cluster_id).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def _text_color(bgr):
    return (0, 0, 0) if (0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]) > 140 else (255, 255, 255)


def process(frame, proc_width=640):
    """Full task for one frame. Returns a composite image:
    [dominant color highlighted | dominant color only] + palette panel."""
    h, w = frame.shape[:2]
    if proc_width and w > proc_width:
        s = proc_width / w
        # NEAREST: do not invent blended colors while downscaling
        small = cv2.resize(frame, (proc_width, int(h * s)), interpolation=cv2.INTER_NEAREST)
    else:
        small = frame

    keys, lut, clusters = analyze(small)
    if clusters:
        mask = segment(keys, lut, clusters[0]["id"])
    else:
        mask = np.zeros(small.shape[:2], np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    highlight = np.where(mask[..., None] > 0, frame, (frame * 0.25).astype(np.uint8))
    seg = np.zeros_like(frame)
    if clusters:
        seg[mask > 0] = clusters[0]["bgr"]
    view = np.hstack([highlight, seg])

    pw = view.shape[1]
    panel_h = max(60, h // 8)
    panel = np.full((panel_h, pw, 3), 25, np.uint8)
    sw = pw // TOP_K
    scale = max(0.5, pw / 3200)
    for i, c in enumerate(clusters[:TOP_K]):
        x0 = i * sw
        cv2.rectangle(panel, (x0 + 2, 2), (x0 + sw - 2, panel_h - 2), c["bgr"], -1)
        label = f"{c['frac'] * 100:.1f}%" + ("  DOM" if i == 0 else "")
        cv2.putText(panel, label, (x0 + 10, panel_h // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, _text_color(c["bgr"]), 2, cv2.LINE_AA)

    out = np.vstack([view, panel])
    return put_label(out, f"unique colors: {len(clusters)}", org=(15, 35))
