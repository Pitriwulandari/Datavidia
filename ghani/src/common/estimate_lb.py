"""Estimasi sebaran skor public LB (30% test ~ 108.7k baris) dari prediksi OOF.

Menarik ribuan subset acak berukuran 108.7k dari OOF 1.24 jt baris dan menghitung Macro F1 tiap subset,
supaya kita tahu rentang wajar skor LB dan peluang melewati skor tertentu.
Catatan: ini perkiraan optimistis tipis (bobot kelas dituning di OOF yang sama) dan memakai prob-blend,
bukan hybrid stacker final.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm

TARGET = float(os.environ.get("TARGET", 0.62045))

y_raw = pd.read_csv("Task-1-Datavidia-11.0/train/shipments.csv", usecols=["status_penanganan"])["status_penanganan"]
classes = sorted(y_raw.unique())
y = y_raw.map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
K = len(classes)
meta = json.load(open("ghani/models/v6/meta_v6_ensemble.json"))
pw = meta["blend_info"]["prob_weights"]

files = {
    "GLMM": ("ghani/cache/v6/glmm_oof_proba.npy", False),
    "LGB_res": ("ghani/cache/v6/lgb_res_oof_logits.npy", True),
    "LGB_full": ("ghani/cache/v6/lgb_full_oof_logits.npy", True),
    "XGBoost": ("ghani/cache/v6/xgb_oof_logits.npy", True),
    "ResNet": ("ghani/cache/v6/nn_oof_logits.npy", True),
    "RGResNet": ("ghani/cache/v6/nn2_oof_logits.npy", True),
}
p = np.zeros((len(y), K), dtype=np.float32)
for name, w in pw.items():
    if w <= 0 or name not in files:
        continue
    path, is_logit = files[name]
    a = np.load(path).astype(np.float32)
    p += w * (glmm.softmax(a) if is_logit else a)
p /= max(sum(v for k, v in pw.items() if k in files), 1e-9)

w_cls, f1_all = glmm.optimize_macro_f1_weights(p, y, n_restarts=2)
pred = (p * w_cls).argmax(1)
print(f"Macro F1 seluruh OOF (bobot dituning di OOF): {f1_all:.5f}")

rng = np.random.RandomState(0)
n_sub = int(0.30 * 362245)
scores = []
for _ in range(2000):
    idx = rng.choice(len(y), n_sub, replace=False)
    scores.append(glmm.fast_macro_f1(y[idx], pred[idx], K))
scores = np.array(scores)
print(f"Subset {n_sub} baris (mirip public LB): mean={scores.mean():.5f} std={scores.std():.5f}")
print(f"  persentil 5/50/95: {np.percentile(scores, 5):.5f} / {np.percentile(scores, 50):.5f} / {np.percentile(scores, 95):.5f}")
print(f"  peluang > {TARGET}: {(scores > TARGET).mean() * 100:.1f}%")
for t in [0.60979, 0.61695, 0.62045, 0.6225]:
    print(f"  peluang > {t}: {(scores > t).mean() * 100:.1f}%")
