"""Task 5: find the Hollow Knight character in a video with a YOLO model.

Supported model files:
  .onnx -> pure OpenCV (cv2.dnn). YOLOv5 (N, boxes, 5+nc) and
           YOLOv8/YOLO11 (N, 4+nc, boxes) output layouts are detected automatically.
  .pt   -> needs `pip install ultralytics`.
"""

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)


def letterbox(img, size):
    """Resize keeping aspect ratio and pad to size x size.
    Returns (canvas, scale, left_pad, top_pad)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def decode_yolo_output(out, conf_thresh):
    """out: 2D raw network output. Returns (boxes_xywh_center, conf, cls)."""
    if out.shape[0] < out.shape[1]:          # YOLOv8 / v11: (4+nc, N)
        out = out.T
        boxes, scores = out[:, :4], out[:, 4:]
    else:                                    # YOLOv5: (N, 5+nc)
        boxes, scores = out[:, :4], out[:, 5:] * out[:, 4:5]
    cls = scores.argmax(axis=1)
    conf = scores[np.arange(len(cls)), cls]
    keep = conf >= conf_thresh
    return boxes[keep], conf[keep], cls[keep]


class OnnxDetector:
    """YOLO ONNX inference with cv2.dnn. imgsz must equal the export size."""

    def __init__(self, path, imgsz=640, conf=0.4, iou=0.5, names=("hollow_knight",), cuda=False):
        self.net = cv2.dnn.readNetFromONNX(path)
        if cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.names = {i: n for i, n in enumerate(names)}

    def __call__(self, frame):
        img, r, left, top = letterbox(frame, self.imgsz)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0]

        boxes, conf, cls = decode_yolo_output(out, self.conf)
        if len(conf) == 0:
            return []

        cx, cy, bw, bh = boxes.T
        x, y = (cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r
        bw, bh = bw / r, bh / r
        xywh = np.stack([x, y, bw, bh], axis=1)

        idx = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.conf, self.iou)
        dets = []
        for i in np.array(idx).flatten():
            x0, y0, w0, h0 = xywh[i]
            dets.append((int(x0), int(y0), int(x0 + w0), int(y0 + h0), float(conf[i]), int(cls[i])))
        return dets


class UltralyticsDetector:
    """YOLO .pt inference with ultralytics."""

    def __init__(self, path, imgsz=640, conf=0.4, iou=0.5, cuda=False):
        from ultralytics import YOLO
        self.model = YOLO(path)
        self.names = self.model.names
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = 0 if cuda else "cpu"

    def __call__(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            dets.append((x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
        return dets


def load_detector(path, imgsz=640, conf=0.4, iou=0.5, names=("hollow_knight",), cuda=False):
    if path.lower().endswith(".onnx"):
        return OnnxDetector(path, imgsz, conf, iou, names, cuda)
    return UltralyticsDetector(path, imgsz, conf, iou, cuda)


def draw(frame, dets, names):
    for (x1, y1, x2, y2, c, k) in dets:
        label = f"{names.get(k, k)} {c:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame
