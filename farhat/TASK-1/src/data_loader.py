"""Pemuatan, validasi, dan penyimpanan data kompetisi Task 1.

Seluruh path dihitung dari lokasi paket ini (``src/`` -> root proyek), sehingga
notebook dapat dijalankan dari folder mana pun tanpa bergantung pada ``cwd``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

# ---------------------------------------------------------------- lokasi proyek
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EXTERNAL_DIR = BASE_DIR / "data" / "external"
SUBMISSION_DIR = BASE_DIR / "submissions"

TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
SAMPLE_SUBMISSION = RAW_DIR / "sample_submission.csv"

for _d in (PROCESSED_DIR, EXTERNAL_DIR, SUBMISSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- konfigurasi
TARGET = "status_penanganan"
ID_COL = "shipment_id"
CLASSES = [
    "normal",
    "konfirmasi_alamat",
    "tertunda_operasional",
    "gagal_antar_awal",
    "retur_pengirim",
    "rusak_perjalanan",
    "klaim_hilang",
]
TABLES = [
    "shipments",
    "scan_events",
    "route_segments",
    "invoices",
    "customers",
    "couriers",
    "hubs",
    "service_packages",
    "promotions",
]
SPLITS = ("train", "test")


# --------------------------------------------------------------------- pembacaan
def path_of(split: str, table: str) -> Path:
    """Path CSV mentah untuk ``split`` ("train"/"test") dan nama ``table``.

    Nama tabel boleh ditulis dengan atau tanpa akhiran ``.csv``.
    """
    base = TRAIN_DIR if split == "train" else TEST_DIR
    name = table[:-4] if table.lower().endswith(".csv") else table
    return base / f"{name}.csv"


def read_table(split: str, table: str, **kwargs) -> pd.DataFrame:
    """Baca satu tabel mentah sebagai pandas DataFrame."""
    return pd.read_csv(path_of(split, table), **kwargs)


def scan_table(split: str, table: str) -> pl.LazyFrame:
    """Baca satu tabel mentah secara lazy (polars) untuk agregasi hemat memori."""
    return pl.scan_csv(path_of(split, table), infer_schema_length=5000)


def read_shipments(split: str, **kwargs) -> pd.DataFrame:
    """Tabel utama (satu baris per pengiriman)."""
    return read_table(split, "shipments", **kwargs)


def table_overview() -> pd.DataFrame:
    """Ringkasan jumlah baris, kolom, dan ukuran setiap CSV pada train & test."""
    rows = []
    for table in TABLES:
        for split in SPLITS:
            path = path_of(split, table)
            if not path.exists():
                continue
            lf = scan_table(split, table)
            rows.append(
                {
                    "table": table,
                    "split": split,
                    "rows": lf.select(pl.len()).collect().item(),
                    "cols": len(lf.collect_schema().names()),
                    "size_mb": round(path.stat().st_size / 1e6, 2),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ data hasil olahan
def processed_path(name: str) -> Path:
    """Path artefak di ``data/processed`` (parquet, model, metrik)."""
    if not name.lower().endswith(".parquet"):
        name = f"{name}.parquet"
    return PROCESSED_DIR / name


def read_processed(name: str, **kwargs) -> pd.DataFrame:
    """Baca artefak parquet dari ``data/processed``."""
    return pd.read_parquet(processed_path(name), **kwargs)


def write_processed(df: pd.DataFrame, name: str) -> Path:
    """Simpan DataFrame ke ``data/processed/<name>.parquet``."""
    out = processed_path(f"{name}.parquet")
    df.to_parquet(out, index=False)
    return out
