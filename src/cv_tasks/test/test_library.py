#!/usr/bin/env python3
"""Sanity tests for the pure-OpenCV library (no ROS needed):
    cd src/cv_tasks && python3 -m unittest discover -s test -v
"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cv_tasks import diamond_detection, dominant_color, hollow_knight, mask_denoise, shape_detection  # noqa: E402


def regular_polygon(img, center, radius, n, color, rot=-np.pi / 2):
    pts = np.array([[center[0] + radius * np.cos(rot + 2 * np.pi * i / n),
                     center[1] + radius * np.sin(rot + 2 * np.pi * i / n)] for i in range(n)], np.int32)
    cv2.fillPoly(img, [pts], color)


class TestShapes(unittest.TestCase):
    def test_all_five_shapes_on_any_background(self):
        for bg in [(180, 20, 80), (30, 160, 40), (40, 40, 200)]:
            img = np.full((720, 1280, 3), bg, np.uint8)
            fg = (200, 205, 195)
            cv2.circle(img, (150, 150), 90, fg, -1)
            cv2.rectangle(img, (500, 60), (680, 240), fg, -1)
            regular_polygon(img, (1000, 150), 100, 3, fg)
            regular_polygon(img, (300, 520), 100, 5, fg)
            regular_polygon(img, (800, 520), 100, 6, fg, rot=0)
            labels, _ = zip(*shape_detection.detect(img)[0])
            self.assertEqual(sorted(labels),
                             sorted(["Lingkaran", "Persegi", "Segitiga", "Pentagon", "Heksagon"]), bg)


class TestDominantColor(unittest.TestCase):
    def test_counts_pixels_and_picks_largest(self):
        img = np.zeros((360, 640, 3), np.uint8)
        img[:] = (200, 180, 40)                 # 80% sky-ish
        img[:, 512:] = (40, 160, 60)            # 20% green
        cv2.circle(img, (100, 100), 30, (30, 30, 230), -1)   # small red
        noise = np.random.default_rng(0).integers(-4, 5, img.shape)
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        keys, lut, clusters = dominant_color.analyze(img)
        self.assertEqual(len(clusters), 3)
        self.assertGreater(clusters[0]["frac"], 0.6)
        mask = dominant_color.segment(keys, lut, clusters[0]["id"])
        self.assertGreater(mask[:, :500].mean() / 255, 0.9)
        self.assertLess(mask[:, 520:].mean() / 255, 0.05)
        out = dominant_color.process(img, 0)
        self.assertEqual(out.dtype, np.uint8)


class TestDiamonds(unittest.TestCase):
    def test_finds_blobs_including_dark_interior(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        for cx, cy, r in [(300, 300, 60), (800, 400, 90), (1000, 150, 12)]:
            pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
            cv2.fillPoly(img, [pts], (35, 70, 30))                    # dark green body
            cv2.polylines(img, [pts], True, (200, 220, 230), 2)       # bright edge
        boxes, _ = diamond_detection.detect(img, 960)
        self.assertEqual(len(boxes), 3)


class TestDenoise(unittest.TestCase):
    def test_removes_speckles_and_keeps_donut_hole(self):
        mask = np.zeros((720, 1280), np.uint8)
        cv2.circle(mask, (640, 360), 200, 255, -1)
        cv2.circle(mask, (640, 360), 70, 0, -1)
        rng = np.random.default_rng(1)
        for _ in range(600):
            x, y = int(rng.integers(0, 1280)), int(rng.integers(0, 720))
            mask[y:y + 2, x:x + 2] = int(rng.integers(120, 255))
        mask[600, :] = 255                                            # thin line artifact
        frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        clean, n = mask_denoise.clean(frame)
        self.assertEqual(n, 1)
        ideal = np.zeros_like(mask)
        cv2.circle(ideal, (640, 360), 200, 255, -1)
        cv2.circle(ideal, (640, 360), 70, 0, -1)
        wrong = np.count_nonzero(clean != ideal) / ideal.size
        self.assertLess(wrong, 0.01)
        self.assertEqual(clean[360, 640], 0)                          # hole preserved


class TestYoloDecode(unittest.TestCase):
    def test_letterbox_and_layouts(self):
        canvas, r, left, top = hollow_knight.letterbox(np.zeros((720, 1280, 3), np.uint8), 640)
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertAlmostEqual(r, 0.5)
        self.assertEqual((left, top), (0, 140))
        # v8 layout (4+nc, N)
        v8 = np.zeros((5, 100), np.float32)
        v8[:, 7] = [100, 200, 20, 30, 0.9]
        b, c, k = hollow_knight.decode_yolo_output(v8, 0.4)
        self.assertEqual((len(c), int(k[0])), (1, 0))
        # v5 layout (N, 5+nc)
        v5 = np.zeros((300, 6), np.float32)
        v5[7] = [100, 200, 20, 30, 0.9, 0.8]
        b, c, k = hollow_knight.decode_yolo_output(v5, 0.4)
        self.assertEqual(len(c), 1)
        self.assertAlmostEqual(float(c[0]), 0.72, places=4)


if __name__ == "__main__":
    unittest.main()
