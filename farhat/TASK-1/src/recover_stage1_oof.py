"""
Recovery Stage-1: rekonstruksi OOF & test logits dari 5 model fold yang SUDAH terlatih.
Tidak melakukan training ulang — hanya prediksi.

Dipakai kalau run_stage1.py crash setelah fold selesai (mis. folder cache/stage1 hilang)
tetapi models/lgb_f0..f4.txt masih ada.

Jalankan dari root TASK-1-exp:
    python src\\recover_stage1_oof.py
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def fast_macro_f1(y_true, y_pred, K):
    f1s = []
    for k in range(K):
        tp = np.sum((y_true == k) & (y_pred == k))
        fp = np.sum((y_true != k) & (y_pred == k))
        fn = np.sum((y_true == k) & (y_pred != k))
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return np.mean(f1s)


def optimize_macro_f1_weights(oof_p, y, n_restarts=2, seed=42):
    K = oof_p.shape[1]
    best_w = np.ones(K, dtype=np.float64)
    best_f1 = fast_macro_f1(y, (oof_p * best_w).argmax(1), K)
    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
        w = np.exp(rng.normal(0, 0.5, K))
        w /= w.mean()
        for _ in range(10):
            improved = False
            for k in range(K):
                for mult in [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2]:
                    w[k] *= mult
                    f1 = fast_macro_f1(y, (oof_p * w).argmax(1), K)
                    if f1 > best_f1 + 1e-6:
                        best_f1 = f1
                        best_w = w.copy()
                        improved = True
                    else:
                        w[k] /= mult
            if not improved:
                break
    return best_w, best_f1


def build_features():
    log("Loading features...")
    tr = pd.read_pickle("data/processed/feat_train.pkl")
    te = pd.read_pickle("data/processed/feat_test.pkl")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)
    log(f"Classes ({K}): {classes}")

    exclude = {"shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id"}
    cat_cols, feat_cols = [], []
    for c in tr.columns:
        if c in exclude:
            continue
        dt = tr[c].dtype
        if dt.name == "category":
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(dt):
            feat_cols.append(c)
        else:
            tr[c] = pd.Categorical(tr[c].astype(str))
            te[c] = pd.Categorical(te[c].astype(str), categories=tr[c].cat.categories)
            cat_cols.append(c)

    log(f"Features: {len(feat_cols)} numeric + {len(cat_cols)} categorical")
    X = tr[feat_cols + cat_cols].copy()
    X_test = te[feat_cols + cat_cols].copy()
    for c in feat_cols:
        X[c] = X[c].fillna(0.0)
        X_test[c] = X_test[c].fillna(0.0)
    return tr, te, X, X_test, y, classes, K


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("cache/stage1", exist_ok=True)

    tr, te, X, X_test, y, classes, K = build_features()

    # Fold split: pakai cache/folds.npy bila valid, kalau tidak reproduksi seed 42
    folds = None
    if os.path.exists("cache/folds.npy"):
        cand = np.load("cache/folds.npy")
        if len(cand) == len(y) and cand.max() == 4 and cand.min() == 0:
            folds = cand
            log("Using existing cache/folds.npy")
    if folds is None:
        log("Recomputing folds (StratifiedKFold 5, seed=42)...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = np.zeros(len(y), dtype=np.int8)
        for fi, (_, va) in enumerate(skf.split(X, y)):
            folds[va] = fi
        np.save("cache/folds.npy", folds)
    log(f"Fold distribution: {np.bincount(folds)}")

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    test_logits = np.zeros((len(te), K), dtype=np.float32)

    for fold in range(5):
        path = f"models/lgb_f{fold}.txt"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model fold hilang: {path}")
        va_idx = np.where(folds == fold)[0]
        log(f"=== FOLD {fold}: predicting {len(va_idx)} val rows ===")
        model = lgb.Booster(model_file=path)

        oof_logits[va_idx] = model.predict(X.iloc[va_idx], raw_score=True).astype(np.float32)
        test_logits += model.predict(X_test, raw_score=True).astype(np.float32) / 5.0

        # Simpan incremental agar tidak hilang lagi kalau crash
        np.save(f"cache/stage1/fold{fold}_val_logits.npy", oof_logits[va_idx])
        np.save(f"cache/stage1/fold{fold}_test_logits.npy", test_logits)
        log(f"Fold {fold} done, best_iter pada model ini (predict memakai semua tree tersimpan)")

    oof_p = softmax(oof_logits, axis=1)
    from sklearn.metrics import log_loss
    full_ll = log_loss(y, oof_p, labels=list(range(K)))
    arg_f1 = fast_macro_f1(y, oof_p.argmax(1), K)
    log(f"OOF LogLoss: {full_ll:.5f} | OOF argmax Macro F1: {arg_f1:.5f}")

    log("Optimizing class multipliers on full OOF...")
    w_opt, cv_f1 = optimize_macro_f1_weights(oof_p, y, n_restarts=5, seed=42)
    log(f"Optimal multipliers: {np.round(w_opt, 4).tolist()}")
    log(f"CV tuned Macro F1: {cv_f1:.5f}")

    # Save artifacts (flattened in models/), mkdir dipastikan tepat sebelum save
    os.makedirs("cache/stage1", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("cache/stage1/oof_logits_all.npy", oof_logits)
    np.save("cache/stage1/test_logits_all.npy", test_logits)
    np.save("cache/stage1/oof_proba.npy", oof_p.astype(np.float32))
    np.save("models/stage1_oof_logits.npy", oof_logits)
    np.save("models/stage1_oof_true.npy", y.astype(np.int64))
    np.save("models/stage1_test_logits.npy", test_logits)
    np.save("models/stage1_test_pred.npy", test_logits.argmax(1).astype(np.int64))
    np.save("models/class_multipliers.npy", w_opt.astype(np.float32))

    meta = {
        "classes": classes,
        "oof_logloss": float(full_ll),
        "oof_argmax_f1": float(arg_f1),
        "cv_tuned_f1": float(cv_f1),
        "class_multipliers": w_opt.tolist(),
        "n_folds": 5,
        "recovered_from": "models/lgb_f0..f4.txt",
    }
    json.dump(meta, open("models/meta_stage1.json", "w"), indent=2)
    log("RECOVERY DONE. Artifacts saved to cache/stage1/ and models/")


if __name__ == "__main__":
    main()
