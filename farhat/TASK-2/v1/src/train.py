"""Tahap 5: pelatih *probe* supervisi-lemah di atas pseudo-label klaster.

Klaster dari tahap unsupervised diberi nama fenomena melalui berkas
models/cluster_mapping.json. Probe (regresi logistik + kNN) lalu dilatih
menggunakan pseudo-label tersebut dan dievaluasi dengan 5-fold CV. Catatan:
metrik CV mengukur konsistensi terhadap pseudo-label, BUKAN akurasi ground truth.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from src import config as C
from src.utils import get_logger, load_json, save_json, set_seed

log = get_logger("train")


def make_model() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")


def make_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=25, weights="distance", metric="cosine")


def expand_proba(model, X: np.ndarray, n_classes: int = C.N_CLASSES) -> np.ndarray:
    """Petakan probabilitas model ke ruang 6 kelas (kelas yang absen -> 0)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def oof_evaluate(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = C.SEED) -> dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_lr = np.zeros((len(y), C.N_CLASSES))
    oof_knn = np.zeros((len(y), C.N_CLASSES))
    for tr, va in skf.split(X, y):
        lr = make_model().fit(X[tr], y[tr])
        knn = make_knn().fit(X[tr], y[tr])
        oof_lr[va] = expand_proba(lr, X[va])
        oof_knn[va] = expand_proba(knn, X[va])
    blend = 0.7 * oof_lr + 0.3 * oof_knn
    pred = blend.argmax(1)
    return {
        "cv_folds": folds,
        "macro_f1_vs_pseudo_labels": float(f1_score(y, pred, average="macro")),
        "accuracy_vs_pseudo_labels": float(accuracy_score(y, pred)),
        "logistic_macro_f1": float(f1_score(y, oof_lr.argmax(1), average="macro")),
        "knn_macro_f1": float(f1_score(y, oof_knn.argmax(1), average="macro")),
        "oof_probs": blend,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="main")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    set_seed(C.SEED)

    mapping = load_json(C.MODELS / "cluster_mapping.json")["cluster_to_class"]
    d = np.load(C.FEATURE_DIR / f"clusters_{args.clusters}.npz", allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    X, lab = d["X"], d["labels"]
    y = np.array([C.CLASS_TO_IDX[mapping[str(c)]] for c in lab], dtype=int)

    rep = oof_evaluate(X, y)
    probs = rep.pop("oof_probs")
    np.save(C.FEATURE_DIR / f"oof_probs_{args.tag}.npy", probs)

    lr = make_model().fit(X, y)
    knn = make_knn().fit(X, y)
    bundle = {"lr": lr, "knn": knn, "classes": C.CLASSES, "mapping": mapping,
              "features": list(d["blocks"]), "seed": C.SEED}
    path = C.MODELS / f"probe_{args.tag}.joblib"
    joblib.dump(bundle, path)
    save_json(C.MODELS / f"cv_report_{args.tag}.json", rep)
    log.info("probe tersimpan: %s", path)
    log.info("CV (vs pseudo-label): %s", rep)


if __name__ == "__main__":
    main()
