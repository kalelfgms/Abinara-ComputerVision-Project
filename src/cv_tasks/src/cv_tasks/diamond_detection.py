"""Task 3 - Deteksi semua objek (berlian jatuh) di background hitam, tanpa
mengklasifikasikan jenis objeknya.

Ide: background hitam, jadi objek = apa pun yang lebih terang dari hitam."""

import cv2
import numpy as np

THRESH_MARGIN = 30   # threshold = tingkat kecerahan background + angka ini
MIN_THRESH = 25      # threshold tidak boleh lebih rendah (hindari noise kompresi)
MIN_AREA_PX = 20     # objek lebih kecil dari ini (dalam piksel) dibuang
CLOSE_RATIO = 0.006  # ukuran penyambung bagian berlian, relatif terhadap lebar frame
BOX_COLOR = (0, 255, 0)


def detect(frame, proc_width=960):
    """Mengembalikan ([(x, y, lebar, tinggi), ...], mask) dalam koordinat frame asli.

    Cara kerja:
    1. Frame dikecilkan agar cepat.
    2. Kecerahan tiap piksel = nilai kanal warna tertinggi (supaya bagian hijau
       gelap berlian tetap terlihat, tidak hanya bagian putihnya).
    3. Threshold sedikit di atas kecerahan background -> objek jadi putih.
    4. Morphology close menyambung rangka, sisi, dan bagian gelap satu berlian
       menjadi satu blob.
    5. Cari kontur luar tiap blob (bagian dalam yang gelap ikut terhitung),
       buang yang terlalu kecil, lalu ambil kotak pembatasnya."""
    scale = min(1.0, proc_width / frame.shape[1]) if proc_width else 1.0
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame

    brightness = cv2.GaussianBlur(small.max(axis=2), (3, 3), 0)
    thresh = max(MIN_THRESH, float(np.median(brightness)) + THRESH_MARGIN)
    _, mask = cv2.threshold(brightness, thresh, 255, cv2.THRESH_BINARY)

    k = max(3, int(small.shape[1] * CLOSE_RATIO) | 1)           # ukuran ganjil
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) >= MIN_AREA_PX * scale * scale:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append(tuple(int(v / scale) for v in (x, y, w, h)))   # kembalikan ke ukuran asli
    return boxes, mask


def draw(frame, boxes):
    """Menggambar kotak hijau di sekeliling tiap objek."""
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
    return frame