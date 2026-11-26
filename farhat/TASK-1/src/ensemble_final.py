"""
Final Ensemble: weighted average of Stage-1 + Stage-2 logits/probabilities.
Mendukung output 5-Fold (run_stage1.py) maupun single split (run_stage1_simple.py).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these exist: {paths}")


def load_outputs():
    """Ambil pasangan (fit) logits Stage-1/Stage-2 + true label, mendukung dua mode."""
    s1 = _first_existing("models/stage1_oof_logits.npy", "models/stage1_val_logits.npy")
    s1_true = _first_existing("models/stage1_oof_true.npy", "models/stage1_val_true.npy")
    s2 = _first_existing("models/stage2_fit_logits.npy", "models/stage2_val_logits.npy")
    return (
        np.load(s1), np.load(s2), np.load(s1_true),
        np.load("models/stage1_test_logits.npy"),
        np.load("models/stage2_test_logits.npy"),
    )


def search_weights(fit_1, fit_2, true, grid_step=0.01):
    best_w, best_f1 = 0.5, -1.0
    for w in np.arange(0.0, 1.0 + grid_step, grid_step):
        f1 = macro_f1(true, np.argmax(w * fit_1 + (1 - w) * fit_2, axis=1))
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


def main():
    log("Loading Stage-1 and Stage-2 outputs...")
    fit_1, fit_2, true, test_1, test_2 = load_outputs()
    log(f"Fit shapes: Stage-1 {fit_1.shape}, Stage-2 {fit_2.shape}")

    f1_1 = macro_f1(true, np.argmax(fit_1, axis=1))
    f1_2 = macro_f1(true, np.argmax(fit_2, axis=1))
    log(f"Stage-1 Macro F1: {f1_1:.6f}")
    log(f"Stage-2 Macro F1: {f1_2:.6f}")

    log("Searching ensemble weights (logit space)...")
    w_logit, f1_logit = search_weights(fit_1, fit_2, true)
    log(f"  logit  -> w={w_logit:.2f}, F1={f1_logit:.6f}")

    log("Searching ensemble weights (prob space)...")
    p1, p2 = softmax(fit_1, axis=1), softmax(fit_2, axis=1)
    w_prob, f1_prob = search_weights(p1, p2, true)
    log(f"  prob   -> w={w_prob:.2f}, F1={f1_prob:.6f}")

    if f1_prob > f1_logit:
        log("Using probability-space ensemble")
        best_w, best_f1, space = w_prob, f1_prob, "prob"
        fit_ens = best_w * p1 + (1 - best_w) * p2
        test_ens = best_w * softmax(test_1, axis=1) + (1 - best_w) * softmax(test_2, axis=1)
    else:
        log("Using logit-space ensemble")
        best_w, best_f1, space = w_logit, f1_logit, "logit"
        fit_ens = best_w * fit_1 + (1 - best_w) * fit_2
        test_ens = best_w * test_1 + (1 - best_w) * test_2

    fit_pred = np.argmax(fit_ens, axis=1)
    test_pred = np.argmax(test_ens, axis=1)
    log(f"Final Macro F1: {best_f1:.6f} (w_stage1={best_w:.2f}, {space})")

    # Per-class F1 (benar: average=None atas seluruh prediksi)
    n_classes = fit_1.shape[1]
    per_class = f1_score(true, fit_pred, average=None, labels=list(range(n_classes)))
    for c in range(n_classes):
        log(f"  Class {c}: F1={per_class[c]:.4f} (n={int((true == c).sum())})")

    os.makedirs("models", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    np.save("models/ensemble_fit_logits.npy", fit_ens)
    np.save("models/ensemble_test_logits.npy", test_ens)
    np.save("models/ensemble_test_pred.npy", test_pred)
    np.save("models/ensemble_weight.npy", np.array([best_w], dtype=np.float32))

    # Submission
    meta_path = "models/meta_stage1.json"
    if os.path.exists(meta_path):
        class_names = json.load(open(meta_path))["classes"]
    else:
        class_names = [
            "gagal_antar_awal", "klaim_hilang", "konfirmasi_alamat", "normal",
            "retur_pengirim", "rusak_perjalanan", "tertunda_operasional",
        ]

    test_df = pd.read_parquet("data/processed/feat_test_entity.parquet")
    submission = pd.DataFrame({
        "shipment_id": test_df["shipment_id"].values,
        "status_penanganan": [class_names[p] for p in test_pred],
    })
    submission.to_csv("submissions/submission_ensemble.csv", index=False)
    log(f"SAVED: submissions/submission_ensemble.csv ({len(submission)} rows)")


if __name__ == "__main__":
    main()
