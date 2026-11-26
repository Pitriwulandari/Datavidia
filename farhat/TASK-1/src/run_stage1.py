"""
Stage-1: 5-Fold LightGBM Training on v5 features (leak-free).
Produces OOF logits and test logits for Stage-2.
Adapted from Ghani's run_stage1_glmm.py pattern.
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import softmax as _softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T0 = time.time()


def softmax(x):
    return _softmax(x, axis=1)


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def fast_macro_f1(y_true, y_pred, K):
    """Fast macro F1 without sklearn overhead."""
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
    """Coordinate ascent per-class multiplier on OOF probabilities."""
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


def main():
    os.makedirs("cache/stage1", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    log("Loading features...")
    tr = pd.read_pickle("data/processed/feat_train.pkl")
    te = pd.read_pickle("data/processed/feat_test.pkl")

    # Target
    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)
    log(f"Classes ({K}): {classes}")

    # Feature columns (exclude IDs, target, strings)
    exclude = {"shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id"}
    cat_cols = []
    feat_cols = []
    for c in tr.columns:
        if c in exclude:
            continue
        dt = tr[c].dtype
        if dt.name == "category":
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(dt):
            feat_cols.append(c)
        else:
            # String/object dtype -> convert to category
            tr[c] = pd.Categorical(tr[c].astype(str))
            te[c] = pd.Categorical(te[c].astype(str), categories=tr[c].cat.categories)
            cat_cols.append(c)

    log(f"Features: {len(feat_cols)} numeric + {len(cat_cols)} categorical = {len(feat_cols) + len(cat_cols)} total")
    log(f"Train shape: {tr.shape}, Test shape: {te.shape}")

    X = tr[feat_cols + cat_cols].copy()
    X_test = te[feat_cols + cat_cols].copy()

    # Fill NaN
    for c in feat_cols:
        X[c] = X[c].fillna(0.0)
        X_test[c] = X_test[c].fillna(0.0)

    # Stratified 5-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.zeros(len(y), dtype=np.int8)
    for fold_idx, (_, va_idx) in enumerate(skf.split(X, y)):
        folds[va_idx] = fold_idx
    np.save("cache/folds.npy", folds)
    log(f"Fold distribution: {np.bincount(folds)}")

    # LightGBM params (tuned for 7-class, 1.2M rows)
    params = {
        "objective": "multiclass",
        "num_class": K,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 63,
        "verbosity": -1,
        "num_threads": 2,
        "seed": 42,
        "deterministic": True,
    }

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    test_logits = np.zeros((len(te), K), dtype=np.float32)

    for fold in range(5):
        log(f"\n=== FOLD {fold} ===")
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(50),
            ],
        )

        best_iter = model.best_iteration
        log(f"Fold {fold}: best_iter={best_iter}")

        # OOF predictions (logits)
        oof_logits[va_idx] = model.predict(X_va, num_iteration=best_iter, raw_score=True).astype(np.float32)

        # Test predictions (accumulate)
        test_logits += model.predict(X_test, num_iteration=best_iter, raw_score=True).astype(np.float32) / 5.0

        # Save model
        model.save_model(f"models/lgb_f{fold}.txt")

        # Save incremental (anti-crash: hasil fold tersimpan segera)
        os.makedirs("cache/stage1", exist_ok=True)
        np.save(f"cache/stage1/fold{fold}_val_logits.npy", oof_logits[va_idx])
        np.save(f"cache/stage1/fold{fold}_test_logits.npy", test_logits)

        # Fold metrics
        oof_p = softmax(oof_logits[va_idx])
        ll = log_loss(y_va, oof_p, labels=list(range(K)))
        arg_f1 = fast_macro_f1(y_va, oof_p.argmax(1), K)
        log(f"Fold {fold}: OOF_ll={ll:.5f} | argmax_F1={arg_f1:.5f}")

        del dtrain, dvalid, model
        gc.collect()

    # Overall OOF metrics
    oof_p = softmax(oof_logits)
    full_ll = log_loss(y, oof_p, labels=list(range(K)))
    arg_f1 = fast_macro_f1(y, oof_p.argmax(1), K)

    log(f"\n=== 5-FOLD STAGE-1 SUMMARY ===")
    log(f"OOF LogLoss: {full_ll:.5f}")
    log(f"argmax Macro F1: {arg_f1:.5f}")

    # Optimize class multipliers on full OOF
    log("Optimizing class multipliers on full OOF...")
    w_opt, cv_f1 = optimize_macro_f1_weights(oof_p, y, n_restarts=5, seed=42)
    log(f"Optimal multipliers: {np.round(w_opt, 4).tolist()}")
    log(f"CV tuned Macro F1: {cv_f1:.5f}")

    # Final OOF with optimized weights
    oof_pred = (oof_p * w_opt).argmax(1)
    from sklearn.metrics import classification_report
    print(classification_report(y, oof_pred, target_names=classes, digits=4), flush=True)

    # Save artifacts (flattened in models/)
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
        "feature_cols": feat_cols,
        "cat_cols": cat_cols,
        "params": params,
        "oof_logloss": float(full_ll),
        "oof_argmax_f1": float(arg_f1),
        "cv_tuned_f1": float(cv_f1),
        "class_multipliers": w_opt.tolist(),
        "folds": folds.tolist(),
        "n_folds": 5,
    }
    json.dump(meta, open("models/meta_stage1.json", "w"), indent=2)
    log("Stage-1 DONE! Artifacts saved to cache/stage1/ and models/")


if __name__ == "__main__":
    main()