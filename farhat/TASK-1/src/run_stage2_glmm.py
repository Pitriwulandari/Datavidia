"""
Stage-2 GLMM (Group Logistic Mixed Model) for TASK-1-exp.
Calibrates Stage-1 logits using entity-level random effects (hub, courier, customer).
Mendukung dua sumber Stage-1:
  - 5-Fold (run_stage1.py)              -> models/stage1_oof_logits.npy (OOF seluruh train)
  - Single split (run_stage1_simple.py) -> models/stage1_val_logits.npy (hold-out 10%)
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


def fit_glmm_per_class(logits, true, groups, n_classes, entity_name, reg=1.0):
    """Fit GLMM: logit = stage1_logit + u_entity where u ~ N(0, sigma^2).
    Empirical Bayes: u_entity = (sum residuals) / (n_entity + reg)."""
    corrections = {}
    groups_s = pd.Series(np.asarray(groups)).astype(str)
    probs = softmax(logits, axis=1)

    for c in range(n_classes):
        y_c = (true == c).astype(float)
        resid = y_c - probs[:, c]

        entity_resid = pd.Series(resid).groupby(groups_s.values).sum()
        entity_count = groups_s.value_counts()

        u = entity_resid / (entity_count + reg)
        corrections[c] = u.to_dict()

    return corrections


def apply_glmm(logits, groups, corrections, n_classes):
    """Apply GLMM corrections to logits (logit_c += u_c per entity)."""
    corrected = logits.copy()
    for c in range(n_classes):
        u_map = corrections[c]
        u_vals = np.array([u_map.get(g, 0.0) for g in groups])
        corrected[:, c] += u_vals
    return corrected


def load_stage1():
    """Return (fit_logits, fit_true, test_logits, mode). Prioritas 5-Fold OOF."""
    if os.path.exists("models/stage1_oof_logits.npy"):
        fit_logits = np.load("models/stage1_oof_logits.npy")
        fit_true = np.load("models/stage1_oof_true.npy")
        test_logits = np.load("models/stage1_test_logits.npy")
        return fit_logits, fit_true, test_logits, "5fold_oof"
    fit_logits = np.load("models/stage1_val_logits.npy")
    fit_true = np.load("models/stage1_val_true.npy")
    test_logits = np.load("models/stage1_test_logits.npy")
    return fit_logits, fit_true, test_logits, "holdout"


def main():
    log("Loading Stage-1 outputs...")
    fit_logits, fit_true, test_logits, mode = load_stage1()
    log(f"Mode: {mode} | fit_logits: {fit_logits.shape}, test: {test_logits.shape}")

    # Entity IDs
    tr = pd.read_parquet("data/processed/feat_train_entity.parquet")
    te = pd.read_parquet("data/processed/feat_test_entity.parquet")

    if mode == "5fold_oof":
        # OOF mencakup seluruh baris train -> langsung pakai, urutan sama dgn parquet
        if len(tr) != len(fit_true):
            raise ValueError(f"Row mismatch: parquet {len(tr)} vs OOF {len(fit_true)}")
        fit_hub = tr["hub_id"].values
        fit_courier = tr["courier_id"].values
        fit_customer = tr["customer_id"].values
    else:
        # Hold-out: reproduksi split 90/10 yang sama dgn run_stage1_simple.py
        from sklearn.model_selection import train_test_split
        _, val_idx = train_test_split(
            np.arange(len(tr)), test_size=0.1, random_state=42,
            stratify=tr["status_penanganan"],
        )
        val_idx = np.sort(val_idx)
        fit_hub = tr.iloc[val_idx]["hub_id"].values
        fit_courier = tr.iloc[val_idx]["courier_id"].values
        fit_customer = tr.iloc[val_idx]["customer_id"].values

    test_hub = te["hub_id"].values
    test_courier = te["courier_id"].values
    test_customer = te["customer_id"].values

    n_classes = fit_logits.shape[1]
    log(f"Classes: {n_classes}")

    log("Fitting GLMM: Hub effects...")
    hub_corr = fit_glmm_per_class(fit_logits, fit_true, fit_hub, n_classes, "hub", reg=10.0)

    log("Fitting GLMM: Courier effects...")
    courier_corr = fit_glmm_per_class(fit_logits, fit_true, fit_courier, n_classes, "courier", reg=5.0)

    log("Fitting GLMM: Customer effects...")
    customer_corr = fit_glmm_per_class(fit_logits, fit_true, fit_customer, n_classes, "customer", reg=20.0)

    log("Applying GLMM to fit set...")
    fit_corrected = apply_glmm(fit_logits, fit_hub, hub_corr, n_classes)
    fit_corrected = apply_glmm(fit_corrected, fit_courier, courier_corr, n_classes)
    fit_corrected = apply_glmm(fit_corrected, fit_customer, customer_corr, n_classes)

    fit_pred = np.argmax(fit_corrected, axis=1)
    base_f1 = macro_f1(fit_true, np.argmax(fit_logits, axis=1))
    fit_f1 = macro_f1(fit_true, fit_pred)
    log(f"Macro F1 sebelum GLMM: {base_f1:.6f}")
    log(f"Macro F1 setelah GLMM: {fit_f1:.6f}")

    # Per-class F1 dihitung benar: F1 tiap kelas atas seluruh set
    per_class = f1_score(fit_true, fit_pred, average=None, labels=list(range(n_classes)))
    for c in range(n_classes):
        log(f"  Class {c}: F1={per_class[c]:.4f} (n={int((fit_true == c).sum())})")

    log("Applying GLMM to test...")
    test_corrected = apply_glmm(test_logits, test_hub, hub_corr, n_classes)
    test_corrected = apply_glmm(test_corrected, test_courier, courier_corr, n_classes)
    test_corrected = apply_glmm(test_corrected, test_customer, customer_corr, n_classes)
    test_pred = np.argmax(test_corrected, axis=1)

    os.makedirs("models", exist_ok=True)
    np.save("models/stage2_fit_logits.npy", fit_corrected)
    np.save("models/stage2_test_logits.npy", test_corrected)
    np.save("models/stage2_test_pred.npy", test_pred)
    np.save("models/stage2_hub_corr.npy", hub_corr, allow_pickle=True)
    np.save("models/stage2_courier_corr.npy", courier_corr, allow_pickle=True)
    np.save("models/stage2_customer_corr.npy", customer_corr, allow_pickle=True)

    json.dump(
        {"mode": mode, "base_macro_f1": float(base_f1), "glmm_macro_f1": float(fit_f1)},
        open("models/meta_stage2.json", "w"), indent=2,
    )
    log(f"SAVED: stage2_fit_logits, stage2_test_logits, stage2_test_pred (mode={mode})")


if __name__ == "__main__":
    main()
