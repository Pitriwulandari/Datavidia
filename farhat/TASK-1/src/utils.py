"""Metrik evaluasi dan fungsi pembantu."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Menyamakan seed numpy & random agar hasil reproducible."""
    np.random.seed(seed)
    random.seed(seed)


def macro_f1(y_true, y_pred) -> float:
    """Metrik resmi kompetisi: macro F1-score."""
    return float(f1_score(y_true, y_pred, average="macro"))


def evaluate(y_true, y_pred, proba=None, classes=None) -> dict:
    """Hitung macro/weighted F1, akurasi, dan (opsional) multi-logloss."""
    out = {
        "macro_f1": round(macro_f1(y_true, y_pred), 5),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 5),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
    }
    if proba is not None:
        out["log_loss"] = round(float(log_loss(y_true, proba)), 5)
    if classes is not None:
        out["per_class_f1"] = {
            c: round(float(v), 4)
            for c, v in zip(
                classes, f1_score(y_true, y_pred, average=None, labels=range(len(classes)))
            )
        }
    return out


def print_report(y_true, y_pred, classes) -> None:
    """Tampilkan classification report (presisi/recall/F1 per kelas)."""
    print(
        classification_report(
            y_true, y_pred, target_names=list(classes), digits=4, zero_division=0
        )
    )


def save_json(obj: dict, path) -> Path:
    """Simpan dictionary ke JSON (dipakai untuk metrik & metadata model)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    return path
