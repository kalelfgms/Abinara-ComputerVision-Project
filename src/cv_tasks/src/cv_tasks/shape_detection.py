"""Task 1 - Deteksi bentuk: segitiga, persegi, pentagon, heksagon, lingkaran.

Ide: warna background berubah-ubah, jadi di setiap frame kita cari dulu warna
background-nya (warna yang paling banyak muncul). Semua yang warnanya beda
dari background dianggap bentuk."""

import cv2
import numpy as np

from .common import put_label

DIFF_THRESH = 40        # selisih warna minimum dari background agar dianggap bentuk
MIN_AREA_RATIO = 0.002  # bentuk lebih kecil dari 0.2% luas frame dibuang (noise)
CIRCLE_FILL = 0.90      # luas bentuk / luas lingkaran pembungkus; lingkaran ~ 1.0
POLY_NAMES = {3: "Segitiga", 4: "Persegi", 5: "Pentagon", 6: "Heksagon"}
COLORS = {"Segitiga": (0, 255, 255), "Persegi": (0, 255, 0), "Pentagon": (255, 0, 255),
          "Heksagon": (255, 128, 0), "Lingkaran": (0, 0, 255)}


def estimate_background(frame):
    """Mencari warna background = warna yang paling sering muncul di frame.

    Cara kerja: frame dikecilkan (supaya cepat), warna dikelompokkan per 16 level
    (supaya noise kecil tidak memecah satu warna menjadi banyak), lalu dicari
    kelompok warna dengan anggota terbanyak."""
    pix = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    q = (pix // 16).astype(np.int32)                  # kelompok warna tiap piksel
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]      # satu nomor unik per kelompok
    most_common = np.bincount(key).argmax()           # kelompok dengan piksel terbanyak
    return pix[key == most_common].mean(axis=0)       # warna rata-rata kelompok itu


def segment_shapes(frame):
    """Membuat mask hitam-putih: putih = piksel yang berbeda dari background.

    Cara kerja: hitung selisih tiap piksel terhadap warna background, piksel yang
    selisihnya besar jadi putih. Lalu mask dirapikan dengan morphology."""
    bg = estimate_background(frame)
    diff = np.abs(frame.astype(np.float32) - bg).max(axis=2)   # selisih terbesar di 3 kanal
    mask = (diff > DIFF_THRESH).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)      # buang titik kecil
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)     # tutup lubang kecil
    return mask


def classify(contour):
    """Menebak nama bentuk dari satu kontur.

    Cara kerja:
    1. Lingkaran hampir memenuhi lingkaran pembungkusnya (rasio ~1.0), sedangkan
       heksagon hanya ~0.83, pentagon ~0.76, persegi ~0.64.
    2. Selain itu, kontur disederhanakan jadi poligon (approxPolyDP) lalu dihitung
       jumlah sudutnya: 3 = segitiga, 4 = persegi, 5 = pentagon, 6 = heksagon.
       Lebih dari 6 sudut berarti sisinya hampir melengkung, jadi dianggap lingkaran."""
    area = cv2.contourArea(contour)
    (_, _), radius = cv2.minEnclosingCircle(contour)
    if area == 0 or radius == 0:
        return None
    if area / (np.pi * radius * radius) > CIRCLE_FILL:
        return "Lingkaran"

    perimeter = cv2.arcLength(contour, True)
    corners = len(cv2.approxPolyDP(contour, 0.03 * perimeter, True))
    if corners > 6:
        return "Lingkaran"
    return POLY_NAMES.get(corners)


def detect(frame):
    """Fungsi utama task: mengembalikan ([(nama, kontur), ...], mask).

    Cara kerja: buat mask -> cari kontur tiap blob putih -> buang yang terlalu
    kecil -> klasifikasi tiap kontur."""
    mask = segment_shapes(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_RATIO * frame.shape[0] * frame.shape[1]
    results = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            name = classify(c)
            if name:
                results.append((name, c))
    return results, mask


def draw(frame, results):
    """Menggambar garis tepi tiap bentuk dan menulis namanya di atasnya."""
    for name, contour in results:
        color = COLORS[name]
        cv2.drawContours(frame, [contour], -1, color, 3)
        x, y, _, _ = cv2.boundingRect(contour)
        put_label(frame, name, (x, max(y - 10, 25)), scale=max(0.6, frame.shape[1] / 1600), color=color)
    return frame