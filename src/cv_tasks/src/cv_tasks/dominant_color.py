"""Task 2 - Cari semua warna unik (dihitung per piksel) dan segmentasi warna
yang paling banyak muncul.

Masalah: video punya noise kompresi dan tepi yang halus, jadi satu warna "rata"
bisa terpecah jadi banyak warna yang nyaris sama. Solusi: warna dikelompokkan,
lalu kelompok yang mirip (jarak di ruang warna Lab) digabung."""

import cv2
import numpy as np

from .common import put_label

NBINS = 4096         # tiap kanal B,G,R dibagi 16 level -> 16 x 16 x 16 kelompok warna
MERGE_TOL = 14.0     # dua kelompok dengan jarak Lab di bawah ini dianggap satu warna
MIN_FRAC = 0.002     # warna dianggap "unik" kalau luasnya >= 0.2% frame
MAX_SEEDS = 256      # jumlah kelompok terbanyak yang dipakai sebagai calon warna wakil
TOP_K = 8            # jumlah warna yang ditampilkan di palet


def _make_bin_tables():
    """Menyiapkan (sekali di awal) warna BGR dan warna Lab dari tiap kelompok."""
    idx = np.arange(NBINS)
    b, g, r = idx >> 8, (idx >> 4) & 15, idx & 15           # level tiap kanal
    bgr = np.stack([b, g, r], axis=1) * 16 + 8              # warna tengah kelompok (0-255)
    lab = cv2.cvtColor(bgr.astype(np.float32).reshape(-1, 1, 3) / 255, cv2.COLOR_BGR2LAB)
    return bgr.astype(np.float32), lab.reshape(-1, 3)


BIN_BGR, BIN_LAB = _make_bin_tables()


def pixel_keys(img):
    """Mengubah tiap piksel menjadi nomor kelompok warnanya (0 sampai 4095)."""
    q = (img >> 4).astype(np.int32)
    return (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]


def analyze(img):
    """Menghitung warna unik di sebuah gambar.

    Mengembalikan (keys, lut, clusters):
      keys     = nomor kelompok tiap piksel
      lut      = tabel: nomor kelompok -> nomor warna (-1 kalau tidak masuk mana pun)
      clusters = daftar warna unik, dari yang terbanyak: id, jumlah piksel, persen, warna
    """
    keys = pixel_keys(img)
    counts = np.bincount(keys.ravel(), minlength=NBINS)     # jumlah piksel tiap kelompok

    # 1. Pilih warna wakil: mulai dari kelompok terbanyak, hanya ambil kelompok
    #    yang cukup berbeda dari wakil-wakil sebelumnya.
    reps, rep_lab = [], np.zeros((MAX_SEEDS, 3), np.float32)
    for b in np.argsort(counts)[::-1][:MAX_SEEDS]:
        if counts[b] == 0:
            break
        if reps and np.linalg.norm(rep_lab[:len(reps)] - BIN_LAB[b], axis=1).min() < MERGE_TOL:
            continue                                         # terlalu mirip, gabung saja
        rep_lab[len(reps)] = BIN_LAB[b]
        reps.append(b)
    rep_lab = rep_lab[:len(reps)]

    # 2. Tiap kelompok warna masuk ke wakil terdekatnya (kalau cukup dekat).
    dist = np.linalg.norm(BIN_LAB[:, None, :] - rep_lab[None, :, :], axis=2)
    lut = np.where(dist.min(axis=1) < MERGE_TOL * 1.5, dist.argmin(axis=1), -1).astype(np.int16)

    # 3. Jumlahkan piksel dan rata-ratakan warna untuk tiap wakil.
    used = np.nonzero(counts)[0]
    owner = lut[used]
    used, owner = used[owner >= 0], owner[owner >= 0]
    weight = counts[used].astype(np.float64)
    n = np.bincount(owner, weights=weight, minlength=len(reps))
    bgr = np.stack([np.bincount(owner, weights=weight * BIN_BGR[used, c], minlength=len(reps))
                    for c in range(3)], axis=1)

    clusters = [{"id": int(c), "count": int(n[c]), "frac": float(n[c] / keys.size),
                 "bgr": tuple(int(v) for v in bgr[c] / n[c])}
                for c in np.argsort(n)[::-1] if n[c] / keys.size >= MIN_FRAC]
    return keys, lut, clusters


def segment(keys, lut, cluster_id):
    """Mask putih untuk semua piksel yang termasuk satu warna (cluster_id)."""
    mask = (lut[keys] == cluster_id).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # buang titik kecil


def process(frame, proc_width=640):
    """Fungsi utama task. Mengembalikan gambar berisi:
    kiri = frame dengan warna dominan disorot, kanan = hanya warna dominan,
    bawah = palet warna terbanyak beserta persentasenya."""
    h, w = frame.shape[:2]
    small = frame
    if proc_width and w > proc_width:    # analisis di resolusi kecil biar real-time
        # NEAREST supaya tidak muncul warna campuran baru saat dikecilkan
        small = cv2.resize(frame, (proc_width, int(h * proc_width / w)), interpolation=cv2.INTER_NEAREST)

    keys, lut, clusters = analyze(small)
    if not clusters:
        return frame
    dominant = clusters[0]

    # mask dibuat di resolusi kecil, lalu dibesarkan kembali ke ukuran frame
    mask = cv2.resize(segment(keys, lut, dominant["id"]), (w, h), interpolation=cv2.INTER_NEAREST)
    highlight = np.where(mask[..., None] > 0, frame, frame // 4)   # bagian lain digelapkan
    only_color = np.zeros_like(frame)
    only_color[mask > 0] = dominant["bgr"]
    view = np.hstack([highlight, only_color])

    # palet: satu kotak warna per warna unik, lengkap dengan persentase piksel
    panel = np.full((60, view.shape[1], 3), 25, np.uint8)
    box_w = view.shape[1] // TOP_K
    for i, c in enumerate(clusters[:TOP_K]):
        cv2.rectangle(panel, (i * box_w + 2, 2), ((i + 1) * box_w - 2, 58), c["bgr"], -1)
        put_label(panel, ("DOM " if i == 0 else "") + f"{c['frac'] * 100:.1f}%", (i * box_w + 8, 38), 0.6)

    return put_label(np.vstack([view, panel]), f"warna unik: {len(clusters)}")