# Pemeriksaan struktur dataset Task-1 
from pathlib import Path
import pandas as pd

BASE = Path("data/raw/task-1-datavidia-11-0/Task-1-Datavidia-11.0")
TABEL = ["couriers", "customers", "hubs", "invoices", "promotions",
         "route_segments", "scan_events", "service_packages", "shipments"]


def hitung_baris(path):
    """Hitung jumlah baris (tanpa header) dengan membaca biner, lebih cepat untuk file besar."""
    total = 0
    with open(path, "rb") as f:
        while True:
            blok = f.read(1024 * 1024 * 16)
            if not blok:
                break
            total += blok.count(b"\n")
    return total - 1


for split in ["train", "test"]:
    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")
    for nama in TABEL:
        path = BASE / split / f"{nama}.csv"
        if not path.exists():
            print(f"[TIDAK ADA] {path}")
            continue
        df = pd.read_csv(path, nrows=5)
        print(f"\n--- {nama} | baris: {hitung_baris(path)} | kolom: {df.shape[1]}")
        print(list(df.columns))
        print(df.dtypes.to_string())
        print(df.head(2).to_string())

# Cek kolom yang hanya ada di train (kandidat target)
print("\n" + "=" * 20 + " KOLOM HANYA DI TRAIN " + "=" * 20)
for nama in TABEL:
    p_tr, p_te = BASE / "train" / f"{nama}.csv", BASE / "test" / f"{nama}.csv"
    if p_tr.exists() and p_te.exists():
        selisih = set(pd.read_csv(p_tr, nrows=1).columns) - set(pd.read_csv(p_te, nrows=1).columns)
        if selisih:
            print(f"[KANDIDAT TARGET] {nama}: {selisih}")

print("\nSelesai.")