"""Task 5 - Cari karakter Hollow Knight di video memakai model YOLO.

Model sudah dilatih (best.onnx atau best.pt). OpenCV bertugas menyiapkan
gambar, menjalankan model (cv2.dnn), dan menggambar hasilnya.

Catatan: model ONNX punya ukuran input yang tetap, jadi imgsz saat menjalankan
harus sama dengan imgsz saat export."""

import cv2
import numpy as np

from .common import put_label

BOX_COLOR = (0, 255, 0)


def letterbox(img, size):
    """Mengubah gambar jadi persegi size x size TANPA merusak rasio.

    Cara kerja: gambar diperkecil sampai sisi terpanjangnya = size, lalu sisa
    ruang diisi abu-abu. Mengembalikan (gambar, skala, padding kiri, padding atas),
    karena angka-angka itu dibutuhkan untuk mengembalikan koordinat ke frame asli."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top


def decode_yolo_output(out, conf_thresh):
    """Membaca output mentah YOLOv8/YOLO11 yang bentuknya (4 + jumlah_kelas, N).

    Setiap kolom = satu kotak kandidat: 4 angka pertama = kotak (cx, cy, lebar,
    tinggi), sisanya = skor tiap kelas. Diambil kelas dengan skor tertinggi, lalu
    kandidat dengan skor di bawah conf_thresh dibuang."""
    out = out.T                                    # jadi (N, 4 + jumlah_kelas)
    boxes, scores = out[:, :4], out[:, 4:]
    cls, conf = scores.argmax(axis=1), scores.max(axis=1)
    keep = conf >= conf_thresh
    return boxes[keep], conf[keep], cls[keep]


class OnnxDetector:
    """Menjalankan model .onnx dengan OpenCV (cv2.dnn), tanpa library tambahan."""

    def __init__(self, path, imgsz=640, conf=0.4, iou=0.5, names=("hollow_knight",), cuda=False):
        self.net = cv2.dnn.readNetFromONNX(path)
        if cuda:   # hanya jalan kalau OpenCV dibuild dengan CUDA; kalau tidak, pakai CPU
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.names = dict(enumerate(names))

    def __call__(self, frame):
        """Mengembalikan [(x1, y1, x2, y2, confidence, kelas), ...] di koordinat frame asli."""
        # 1. siapkan gambar: letterbox -> blob (skala 0-1, urutan warna RGB)
        img, scale, left, top = letterbox(frame, self.imgsz)
        self.net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True))

        # 2. jalankan model
        try:
            out = self.net.forward()[0]
        except cv2.error as e:
            raise RuntimeError(
                f"OpenCV gagal menjalankan model pada imgsz={self.imgsz}. Model ONNX punya ukuran "
                f"input tetap: pakai imgsz yang sama dengan saat export, atau export ulang dengan "
                f"`yolo export model=best.pt format=onnx imgsz={self.imgsz}`.\nError asli: {e}") from None

        # 3. baca hasilnya, lalu kembalikan koordinat dari gambar letterbox ke frame asli
        boxes, conf, cls = decode_yolo_output(out, self.conf)
        if len(conf) == 0:
            return []
        cx, cy, bw, bh = boxes.T
        xywh = np.stack([(cx - bw / 2 - left) / scale, (cy - bh / 2 - top) / scale,
                         bw / scale, bh / scale], axis=1)      # (x kiri, y atas, lebar, tinggi)

        # 4. NMS: kalau beberapa kotak menumpuk di objek yang sama, simpan yang terbaik
        keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self.conf, self.iou)).flatten()
        return [(int(xywh[i, 0]), int(xywh[i, 1]), int(xywh[i, 0] + xywh[i, 2]),
                 int(xywh[i, 1] + xywh[i, 3]), float(conf[i]), int(cls[i])) for i in keep]


class UltralyticsDetector:
    """Alternatif untuk file .pt (butuh `pip install ultralytics`). Bisa memakai GPU."""

    def __init__(self, path, imgsz=640, conf=0.4, iou=0.5, cuda=False):
        from ultralytics import YOLO
        self.model = YOLO(path)
        self.names = self.model.names
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = 0 if cuda else "cpu"

    def __call__(self, frame):
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                    device=self.device, verbose=False)[0]
        return [(*(int(v) for v in b.xyxy[0].tolist()), float(b.conf[0]), int(b.cls[0]))
                for b in result.boxes]


def load_detector(path, imgsz=640, conf=0.4, iou=0.5, names=("hollow_knight",), cuda=False):
    """Memilih detektor dari ekstensi file: .onnx -> OpenCV, selain itu -> ultralytics."""
    if path.lower().endswith(".onnx"):
        return OnnxDetector(path, imgsz, conf, iou, names, cuda)
    return UltralyticsDetector(path, imgsz, conf, iou, cuda)


def draw(frame, detections, names):
    """Menggambar kotak dan label (nama kelas + confidence) tiap deteksi."""
    for x1, y1, x2, y2, conf, cls in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        put_label(frame, f"{names.get(cls, cls)} {conf:.2f}", (x1, max(y1 - 8, 20)), 0.6, BOX_COLOR)
    return frame