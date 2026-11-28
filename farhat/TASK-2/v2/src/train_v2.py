"""Task-2 v2 Part 4: training probe ensemble + pseudo-label.

Menggabungkan:
- Embedding deep: dinov2_s, dinov3_s, mae_b, mobilenetv3_l (dari embed_v3 --encoders ...)
- Fisik: phys_v2.npz (74d de-augmented)
- Klaster: 28 komunitas dari cluster_v2.npz
- Mapping: audit visual manual (cluster_mapping_v2.json)

Output: models/probe_v2.joblib, models/cv_report_v2.json, submissions/submission_v2.csv
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src import config as C
from src.utils import get_logger, Timer

log = get_logger("train_v2")

PROBE_OUT = C.MODELS / "probe_v2.joblib"
CV_OUT = C.MODELS / "cv_report_v2.json"
SUB_OUT = C.SUBMISSIONS / "submission_v2.csv"

CLASSES = ["Earthquake", "Fire", "Hurricane", "Tsunami", "Volcano", "Explosion"]
N_CLASSES = len(CLASSES)


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def load_embeddings() -> np.ndarray:
    """Gabungkan semua embedding deep + fisik."""
    parts = []

    # 1. DINOv2-S (features_v3.npz)
    f1 = np.load(C.FEATURE_DIR / "features_v3.npz")
    e1 = f1["features"]
    log.info("dinov2_s: %s", e1.shape)
    parts.append(l2norm(e1))

    # 2. DINOv3-S (emb_dinov3_s_224.npz)
    f2 = np.load(C.FEATURE_DIR / "emb_dinov3_s_224.npz")
    e2 = np.hstack([f2["pre"], f2["post"]])
    log.info("dinov3_s: %s", e2.shape)
    parts.append(l2norm(e2))

    # 3. MAE-Base (emb_mae_b_224.npz) - from embed_v3
    try:
        f3 = np.load(C.FEATURE_DIR / "emb_mae_b_224.npz")
        e3 = np.hstack([f3["pre"], f3["post"]])
        log.info("mae_b: %s", e3.shape)
        parts.append(l2norm(e3))
    except FileNotFoundError:
        log.warning("emb_mae_b_224.npz belum ada (embed_v3 belum selesai)")

    # 4. MobileNetV3-Large (emb_mobilenetv3_l_224.npz) - from embed_v3
    try:
        f4 = np.load(C.FEATURE_DIR / "emb_mobilenetv3_l_224.npz")
        e4 = np.hstack([f4["pre"], f4["post"]])
        log.info("mobilenetv3_l: %s", e4.shape)
        parts.append(l2norm(e4))
    except FileNotFoundError:
        log.warning("emb_mobilenetv3_l_224.npz belum ada (embed_v3 belum selesai)")

    # 5. Fisik v2 (74d)
    f5 = np.load(C.FEATURE_DIR / "phys_v2.npz")
    e5 = f5["features"]
    scaler = StandardScaler()
    e5 = scaler.fit_transform(e5)
    log.info("phys_v2: %s", e5.shape)
    parts.append(l2norm(e5))

    E = np.hstack(parts)
    log.info("TOTAL concat: %s", E.shape)
    return E


def load_pseudo_labels() -> np.ndarray:
    """Baca pseudo-label dari cluster_v2 + mapping_v2."""
    cl = np.load(C.FEATURE_DIR / "clusters_v2.npz")
    labels = cl["labels"]
    n_com = labels.max() + 1

    mapping = json.load(open(C.MODELS / "cluster_mapping_v2.json"))
    cluster_to_class = mapping.get("cluster_to_class", {})

    pseudo = np.full(len(labels), -1, dtype=int)
    for c in range(n_com):
        cls_name = cluster_to_class.get(str(c))
        if cls_name and cls_name in CLASSES:
            pseudo[labels == c] = CLASSES.index(cls_name)

    unmapped = (pseudo == -1).sum()
    if unmapped > 0:
        log.warning("%d sampel tidak termapping (cluster tanpa mapping)", unmapped)

    return pseudo


def train_probe(X: np.ndarray, y: np.ndarray) -> dict:
    """Train ensemble: LogisticRegression + kNN + soft voting."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_scores, knn_scores = [], []
    lr_models, knn_models = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        # Standardize per fold
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)

        # LogisticRegression
        lr = LogisticRegression(C=8.0, penalty="l2", solver="lbfgs", max_iter=2000,
                                class_weight="balanced", random_state=42, n_jobs=1)
        lr.fit(Xtr_s, ytr)
        lr_models.append((lr, scaler))
        yp_lr = lr.predict(Xva_s)
        lr_scores.append(f1_score(yva, yp_lr, average="macro"))

        # kNN (cosine)
        knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine", n_jobs=2)
        knn.fit(Xtr_s, ytr)
        knn_models.append((knn, scaler))
        yp_knn = knn.predict(Xva_s)
        knn_scores.append(f1_score(yva, yp_knn, average="macro"))

        log.info("fold %d: LR=%.4f kNN=%.4f", fold + 1, lr_scores[-1], knn_scores[-1])

    # Ensemble weights (from Ghani: 45% ResMLP, 35% LR, 20% kNN -> simplified)
    log.info("CV LR macro-F1: %.4f ± %.4f", np.mean(lr_scores), np.std(lr_scores))
    log.info("CV kNN macro-F1: %.4f ± %.4f", np.mean(knn_scores), np.std(knn_scores))

    # Retrain full
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LogisticRegression(C=8.0, penalty="l2", solver="lbfgs", max_iter=2000,
                            class_weight="balanced", random_state=42, n_jobs=1)
    lr.fit(Xs, y)

    knn = KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine", n_jobs=2)
    knn.fit(Xs, y)

    probe = {
        "lr": lr, "knn": knn, "scaler": scaler,
        "lr_weight": 0.65, "knn_weight": 0.35,
        "classes": CLASSES, "n_features": X.shape[1],
        "cv_lr_macro": float(np.mean(lr_scores)), "cv_knn_macro": float(np.mean(knn_scores)),
    }
    return probe


def predict_probe(probe: dict, X: np.ndarray) -> np.ndarray:
    scaler = probe["scaler"]
    Xs = scaler.transform(X)
    p_lr = probe["lr"].predict_proba(Xs)
    p_knn = probe["knn"].predict_proba(Xs)
    p_ens = probe["lr_weight"] * p_lr + probe["knn_weight"] * p_knn
    return p_ens.argmax(axis=1)


def main():
    with Timer("train_v2", log):
        # Cek apakah mapping sudah diisi
        mapping = json.load(open(C.MODELS / "cluster_mapping_v2.json"))
        if not mapping.get("cluster_to_class"):
            log.error("cluster_mapping_v2.json KOSONG - isi manual dulu dari audit montage!")
            return

        X = load_embeddings()
        y = load_pseudo_labels()

        # Hanya train pada sampel yang termapping
        mask = y != -1
        X_train, y_train = X[mask], y[mask]
        log.info("training samples: %d/%d", len(X_train), len(X))

        probe = train_probe(X_train, y_train)

        # Full prediction
        y_pred = predict_probe(probe, X)

        # Submission
        ids = pd.read_csv(C.SAMPLE_SUBMISSION)["Id"].tolist()
        sub = pd.DataFrame({"Id": ids, "Phenomenon": [CLASSES[i] for i in y_pred]})
        sub.to_csv(SUB_OUT, index=False)
        log.info("submission -> %s", SUB_OUT)

        # Distribusi
        vc = sub["Phenomenon"].value_counts()
        for k, v in vc.items():
            log.info("  %s: %d (%.1f%%)", k, v, v / len(sub) * 100)

        # Save probe
        dump(probe, PROBE_OUT)
        log.info("probe -> %s", PROBE_OUT)

        # CV report
        cv_report = {
            "cv_folds": 5,
            "lr_macro_f1": probe["cv_lr_macro"],
            "knn_macro_f1": probe["cv_knn_macro"],
            "ensemble_weights": {"lr": probe["lr_weight"], "knn": probe["knn_weight"]},
            "n_train_samples": int(len(X_train)),
            "n_features": int(X.shape[1]),
        }
        with open(CV_OUT, "w") as f:
            json.dump(cv_report, f, indent=2)
        log.info("cv report -> %s", CV_OUT)


if __name__ == "__main__":
    main()