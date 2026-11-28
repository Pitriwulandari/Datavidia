"""Konfigurasi global pipeline Task-2 (Datavidia 11.0).

Semua path bersifat relatif terhadap root proyek agar notebook dapat dijalankan
ulang tanpa perubahan kode.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "cache"
MODELS = ROOT / "models"
SUBMISSIONS = ROOT / "submissions"
NOTEBOOKS = ROOT / "notebooks"

PHENOMENA_DIR = DATA_RAW / "Fenomena"
SAMPLE_SUBMISSION = DATA_RAW / "sample_submission.csv"

THUMB_DIR = CACHE / "thumbs"          # cache gambar 256px (RGB, .npy per split)
FEATURE_DIR = CACHE / "features"      # embedding & matriks fitur
FIGURE_DIR = CACHE / "figures"        # montage / plot diagnostik
for _d in (DATA_PROCESSED, CACHE, MODELS, SUBMISSIONS, THUMB_DIR, FEATURE_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA_PROCESSED / "manifest.csv"
CHANGE_STATS = DATA_PROCESSED / "change_stats.csv"

# ---------------------------------------------------------------- label space
# Urutan kelas mengikuti halaman kompetisi (dipakai konsisten di seluruh kode).
CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# ---------------------------------------------------------------- image params
THUMB_SIZE = 256        # resolusi untuk statistik perubahan & montage
IMG_SIZE = 224          # resolusi input encoder
JPEG_QUALITY = 92

# Encoder pure-vision (tanpa prompt teks) - lihat catatan di README/write-up.
ENCODERS = {
    "dinov2_s": "vit_small_patch14_dinov2.lvd142m",   # DINOv2 ViT-S/14, self-supervised
    "effb0": "mobilenetv3_large_100.ra_in1k",     # ImageNet-1k, ringan (alternatif)
}

# ---------------------------------------------------------------- fitur
PCA_DIMS = {"dinov2_s": 192, "effb0": 192}
HANDCRAFTED_WEIGHT = 0.35   # bobot blok statistik perubahan pada matriks fitur
CONSENSUS_RUNS = 24         # jumlah run clustering untuk matriks co-association

# ---------------------------------------------------------------- misc
SEED = 42
N_JOBS = 2                  # mesin 2 core; ubah bila hardware berbeda
BATCH_SIZE = 16
