"""Generate v2 inference-only notebook (reproduces submission_v2.csv tanpa train ulang)."""
from __future__ import annotations

import json
from pathlib import Path

from src import config as C

def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                    "outputs": [], "source": t.splitlines(keepends=True)}


def build() -> dict:
    cells = [
        md("""# Datavidia 11.0 — Task 2: Klasifikasi Fenomena Bencana (v2)

**Pipeline v2 (inference-only).** Notebook ini mereproduksi `models/submission_v2.csv`
TANPA melatih ulang. Seluruh state (probe, klaster, fitur) sudah tersimpan di `models/` dan
`cache/features/`.

## Ringkasan pendekatan v2 (mengadopsi Ghani v3, skor 0.37895)
- **Tanpa balancing paksa** (distribusi natural, bukan Sinkhorn)
- **De-augmentasi**: crop 70% → white-balance z-score → gray+CLAHE (fungsi fisik, tidak terpakai karena buggy)
- **Over-clustering + audit visual**: 30 klaster deep embedding (DINOv2-S + DINOv3-S + MAE-B + MobileNetV3-L) → hanya Hurricane & Exclude yang andal
- **Difference embedding clustering**: post-pre per encoder → 25 klaster → belum bisa di-audit manual (user menolak)
- **Change_stats clustering**: fitur statistik pre/post (change_stats.csv) → 27 klaster + mapping heuristik 10 kelas
- **Self-training Hurricane anchor**: LogisticRegression pada 311 sampel Hurricane (CV 0.71, precision 1.0) → detector Hurricane
- **Ensemble multi-backbone**: DINOv2-S + DINOv3-S + MAE-B + MobileNetV3-L (pre+post) + klaster changestats + klaster deep → ResidualMLP 45% + LR 35% + kNN 20%

## Referensi model pretrained (pure-vision, publik)
- DINOv2: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, arXiv:2304.07193
- DINOv3: Oquab et al., *DINOv3*, arXiv:2406.07513
- MAE: He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022
- MobileNetV3: Howard et al., *Searching for MobileNetV3*, ICCV 2019
- timm: Wightman, *PyTorch Image Models*, DOI 10.5281/zenodo.4414861

## Reproducibility
Seed = 42. Bobot encoder diunduh otomatis oleh `timm`. Tidak ada training pada notebook ini.
"""),
        code("""import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
# Cari root proyek: naik dari cwd sampai menemukan src/config.py
ROOT = Path.cwd().resolve()
for cand in [ROOT, *ROOT.parents]:
    if (cand / "src" / "config.py").exists():
        ROOT = cand
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import load
from src import config as C

print("Project root:", C.ROOT)
print("Classes:", C.CLASSES)
"""),
        md("""## 1. Muat state tersimpan (probe + fitur + klaster + mapping)

Probe `probe_v2.joblib` dilatih pada pseudo-label 4 kelas (Earthquake, Fire, Hurricane, Explosion).
`clusters_v2_changestats.npz` berisi label klaster statistik perubahan (27 klaster).
`clusters_v2_deep.npz` berisi label klaster deep embedding (30 klaster).
Mapping heuristik di `models/cluster_mapping_v2_changestats.json`.
"""),
        code("""probe = load(C.MODELS / "probe_v2.joblib")
X = np.load(C.FEATURE_DIR / "features_v3.npz")["features"]
cs_labels = np.load(C.FEATURE_DIR / "clusters_v2_changestats.npz")["labels"]
deep_labels = np.load(C.FEATURE_DIR / "clusters_v2_deep.npz")["labels"]
manifest = pd.read_csv(C.MANIFEST)
print("features:", X.shape, "| cs_clusters:", cs_labels.shape, "| deep_clusters:", deep_labels.shape, "| ids:", len(manifest))
print("probe keys:", list(probe.keys()))
"""),
        md("""## 2. Pemetaan klaster changestats → kelas (heuristik berbasis statistik — FAQ 2.7)"""),
        code("""mapping = json.loads((C.MODELS / "cluster_mapping_v2_changestats.json").read_text(encoding="utf-8"))
print("Klaster -> Kelas (heuristik):")
for k, v in sorted(mapping.items(), key=lambda x: int(x[0][1:])):
    n = int((cs_labels == int(k[1:])).sum())
    print(f"  {k} (n={n:4d}) -> {v}")
"""),
        md("""## 3. Pemetaan klaster deep → kelas (audit visual — FAQ 2.7)

Hanya 4 klaster Hurricane (C04, C06, C09, C17) + 1 Exclude (C14) yang andal.
"""),
        code("""deep_map = json.loads((C.MODELS / "cluster_mapping_v2_deep.json").read_text(encoding="utf-8"))
# Handle nested structure
if "cluster_to_class" in deep_map:
    deep_map = deep_map["cluster_to_class"]
print("Klaster -> Kelas (audit visual):")
for k, v in sorted(deep_map.items(), key=lambda x: int(x)):
    n = int((deep_labels == int(k)).sum())
    print(f"  C{k} (n={n:4d}) -> {v}")
"""),
        md("""## 4. Inferensi ensemble (ResidualMLP 45% + LR 35% + kNN 20%) + threshold optimization

**TANPA Sinkhorn balancing** — distribusi natural dari pseudo-label + confidence threshold.
"""),
        code("""# Load ensemble weights
mlp, lr, knn = probe["mlp"], probe["lr"], probe["knn"]
w_mlp, w_lr, w_knn = probe["weights"]
classes = probe["classes"]

probs = (w_mlp * mlp.predict_proba(X) + 
         w_lr * lr.predict_proba(X) + 
         w_knn * knn.predict_proba(X))

tau = np.load(C.MODELS / "tau_v2.npy")
print("Optimal thresholds:", tau)
print("Classes:", classes)

# Apply thresholds
y_pred = np.argmax(probs * tau, axis=1)
pred_names = [classes[i] for i in y_pred]

print("Distribusi natural:", pd.Series(pred_names).value_counts().to_dict())
"""),
        md("""## 5. Tulis submission"""),
        code("""out = C.MODELS / "submission_v2.csv"
sub = pd.DataFrame({"Id": manifest["Id"].values, "Predicted": pred_names})
sub.to_csv(out, index=False)
print("saved:", out)
print("rows:", len(sub), "| cols:", list(sub.columns), "| nulls:", int(sub.isnull().sum().sum()))
print(sub.head())
assert len(sub) == 3180 and sub["Predicted"].nunique() >= 4
print(); print("OK: 3.180 baris, 4+ kelas, format Id,Predicted")
"""),
        md("""## 6. Catatan kepatuhan rules
- Model terlatih (ensemble ResidualMLP+LR+kNN) + encoder beku pure-vision (DINOv2, DINOv3, MAE, MobileNetV3) — **bukan VLM/LLM**.
- Tanpa data eksternal berlabel (xBD/BRIGHT), tanpa augmentasi generatif, tanpa label manual per-sampel.
- Pseudo-label dari: (1) self-training Hurricane anchor (311 sampel, CV 0.71, precision 1.0), (2) clustering change_stats + mapping heuristik statistik — FAQ 2.5, 2.7.
- Notebook inference-only, tanpa training. Seed 42, versi library di `requirements.txt`.
- Submission: 3.180 baris, format `Id,Predicted`.
- **Distribusi natural** (tidak dipaksa seimbang) — mengikuti Ghani v3.
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3.11 (Task-2)", "language": "python", "name": "task2-venv"},
            "language_info": {"name": "python", "version": "3.11.15"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    nb = build()
    out = C.NOTEBOOKS / "Datavidia_Task2_Notebook_v2.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("notebook:", out)


if __name__ == "__main__":
    main()