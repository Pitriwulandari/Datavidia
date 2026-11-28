"""Task-2 v2: analisis fisik per klaster untuk bantu audit visual.

Menggunakan fitur phys_v2.npz (sudah de-augmented) untuk menghitung statistik
per klaster: luminance, edge, saturation, ExG, dll. → memberi clue kelas.
Output: print ringkasan + simpan cluster_features_v2.csv untuk inspeksi.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

FEATURES = np.load(C.FEATURE_DIR / "phys_v2.npz")
X = FEATURES["features"]  # (3180, 74)
names = FEATURES["feature_names"].tolist()

CLUSTERS = np.load(C.FEATURE_DIR / "clusters_v2.npz")
labels = CLUSTERS["labels"]
n_com = labels.max() + 1

ids = pd.read_csv(C.SAMPLE_SUBMISSION)["Id"].tolist()

# Indeks fitur kunci (delta = post - pre)
name_to_idx = {n: i for i, n in enumerate(names)}
key_names = [
    "lum_mean_delta", "lum_std_delta", "dark_frac_delta", "vdark_frac_delta",
    "sob_mean_delta", "lap_var_delta", "canny_dens_delta",
    "exg_std_pre", "exg_std_post", "exg_std_delta",
    "sat_mean_pre", "sat_mean_post", "sat_mean_delta",
    "low_sat_pre", "low_sat_post", "dark_low_sat_pre", "dark_low_sat_post",
    "is_gray_pre", "is_gray_post",
    "blk_frac_pre", "blk_frac_post",
]
idx = [name_to_idx[k] for k in key_names if k in name_to_idx]
key_names = [k for k in key_names if k in name_to_idx]

print(f"Klaster: {n_com}, Fitur kunci: {len(key_names)}")
print()

rows = []
for c in range(n_com):
    mask = labels == c
    n = mask.sum()
    if n == 0:
        continue
    Xc = X[mask]
    stats = {"cluster": c, "n": int(n)}
    for k, i in zip(key_names, idx):
        stats[f"{k}_mean"] = float(Xc[:, i].mean())
        stats[f"{k}_std"] = float(Xc[:, i].std())
    rows.append(stats)

df = pd.DataFrame(rows)
df.to_csv(C.FEATURE_DIR / "cluster_features_v2.csv", index=False)
print(df[["cluster", "n"] + [c for c in df.columns if c.endswith("_mean") and "delta" in c]].to_string(index=False))
print()
print("Tersimpan:", C.FEATURE_DIR / "cluster_features_v2.csv")

# Heuristik sederhana
print("\n=== HEURISTIK KELAS (berdasarkan fitur dominan) ===")
for _, r in df.iterrows():
    c, n = int(r["cluster"]), int(r["n"])
    # Earthquake: banyak debris, edge naik, lum_std naik
    # Fire: ExG turun (vegetasi hilang), sat_mean turun, dark_low_sat naik
    # Hurricane: flood → dark_frac naik, canny turun (awan), blk_frac tinggi?
    # Tsunami: air → low_sat tinggi, dark_low_sat tinggi, ExG rendah
    # Volcano: abu → is_gray tinggi, lum_mean tinggi, sat_mean rendah
    # Explosion: kecil, edge naik drastis, area hitam (crater)
    
    iso_gray = r.get("is_gray_pre_mean", 0) > 0.5 or r.get("is_gray_post_mean", 0) > 0.5
    high_dark = r.get("dark_frac_delta_mean", 0) > 0.1
    high_sat_drop = r.get("sat_mean_delta_mean", 0) < -0.05
    high_exg_drop = r.get("exg_std_delta_mean", 0) < -0.02
    high_edge = r.get("sob_mean_delta_mean", 0) > 5.0
    high_lum = r.get("lum_mean_delta_mean", 0) > 10.0
    high_blk = r.get("blk_frac_pre_mean", 0) > 0.15 or r.get("blk_frac_post_mean", 0) > 0.15
    
    hints = []
    if iso_gray: hints += ["Volcano(abu)"]
    if high_exg_drop and high_sat_drop: hints += ["Fire"]
    if high_dark and r.get("low_sat_post_mean", 0) > 0.3: hints += ["Tsunami"]
    if high_dark and r.get("canny_dens_delta_mean", 0) < -0.01: hints += ["Hurricane"]
    if high_edge and n < 50: hints += ["Explosion"]
    if high_lum and not iso_gray: hints += ["Earthquake/debris"]
    if not hints: hints += ["?"]
    
    print(f"  C{c:02d} (n={n:3d}): {', '.join(hints)}  | lum_d={r.get('lum_mean_delta_mean',0):+.1f} edge_d={r.get('sob_mean_delta_mean',0):+.1f} sat_d={r.get('sat_mean_delta_mean',0):+.3f} exg_d={r.get('exg_std_delta_mean',0):+.3f} dark_d={r.get('dark_frac_delta_mean',0):+.3f} gray={iso_gray}")