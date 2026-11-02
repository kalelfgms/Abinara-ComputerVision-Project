"""Small helpers shared by the nodes (no ROS imports)."""

import time

import cv2


class FpsMeter:
    """Smoothed FPS + overall summary."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.fps = 0.0
        self.n = 0
        self.t_prev = None
        self.t_start = None

    def tick(self):
        t = time.time()
        if self.t_prev is None:
            self.t_start = t
        else:
            inst = 1.0 / max(t - self.t_prev, 1e-6)
            self.fps = inst if self.n == 1 else self.alpha * self.fps + (1 - self.alpha) * inst
        self.t_prev = t
        self.n += 1
        return self.fps

    def summary(self):
        if self.n < 2:
            return "fewer than 2 frames processed"
        elapsed = self.t_prev - self.t_start
        return (f"{self.n} frames, {elapsed:.1f} s from first to last frame, "
                f"{(self.n - 1) / max(elapsed, 1e-6):.1f} FPS average")


def put_label(img, text, org=(15, 35), scale=1.0, color=(255, 255, 255)):
    """Text with a dark outline so it is readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return img
