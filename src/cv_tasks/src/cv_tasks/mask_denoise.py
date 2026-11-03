"""Task 4 - Hapus noise pada video mask (donat putih berputar dengan titik-titik
putih di sekitarnya).

Ide: objek asli itu besar dan hanya satu, sedangkan noise berupa titik atau
garis yang kecil. Jadi cukup simpan bagian putih yang besar."""

import cv2
import numpy as np

from .common import put_label

THRESH = 127           # piksel lebih terang dari ini dianggap putih
REL_KEEP = 0.15        # simpan bagian putih yang luasnya >= 15% dari yang terbesar
MIN_AREA_RATIO = 0.002 # ... dan minimal 0.2% luas frame
SMOOTH_K = 7           # ukuran median blur (harus ganjil) untuk menghaluskan tepi


def keep_large_components(binary):
    """Hanya menyimpan bagian putih yang besar. Mengembalikan (mask, jumlah objek).

    Cara kerja: connectedComponentsWithStats memberi nomor tiap bagian putih yang
    terhubung beserta luasnya. Bagian yang luasnya di bawah batas dibuang,
    sehingga bintik dan garis tipis hilang tapi donat tetap utuh. Lubang donat
    ikut aman, karena lubang itu bagian hitam, bukan putih."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:                                              # tidak ada apa pun yang putih
        return np.zeros_like(binary), 0
    areas = stats[1:, cv2.CC_STAT_AREA]                     # luas tiap bagian (tanpa background)
    min_area = max(MIN_AREA_RATIO * binary.size, REL_KEEP * areas.max())
    keep = np.zeros(n, np.uint8)
    keep[1:] = (areas >= min_area) * 255                    # 255 = simpan, 0 = buang
    return keep[labels], int((areas >= min_area).sum())


def clean(frame, proc_width=0):
    """Fungsi utama task. Mengembalikan (mask bersih seukuran frame, jumlah objek).

    Cara kerja: grayscale -> blur ringan -> threshold jadi hitam-putih ->
    buang bagian putih yang kecil -> median blur untuk menghaluskan tepi."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if proc_width and w > proc_width:                       # opsional: kecilkan biar lebih cepat
        gray = cv2.resize(gray, (proc_width, int(h * proc_width / w)), interpolation=cv2.INTER_AREA)

    _, binary = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), THRESH, 255, cv2.THRESH_BINARY)
    mask, n_objects = keep_large_components(binary)
    mask = cv2.medianBlur(mask, SMOOTH_K)

    if mask.shape[:2] != (h, w):                            # besarkan kembali ke ukuran frame
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask, n_objects


def process(frame, proc_width=0, view_width=640):
    """Menampilkan input yang noisy (kiri) dan hasil bersih (kanan) berdampingan."""
    mask, n_objects = clean(frame, proc_width)

    def fit(img):                                           # samakan lebar tampilan
        return cv2.resize(img, (view_width, int(img.shape[0] * view_width / img.shape[1])))

    left, right = fit(frame), fit(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    put_label(left, "Input (noisy)", (10, 30), 0.8, (0, 255, 255))
    put_label(right, f"Hasil | objek: {n_objects}", (10, 30), 0.8, (0, 255, 0))
    return np.hstack([left, right])