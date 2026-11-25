"""v8 New Orthogonal Architecture #1: 5-Fold GroupKFold(customer_id) Symmetric-Tree CatBoost (depth=7).

Uses compact one-hot encoding for the 10 low-cardinality categoricals (~32 binary cols) so CPU training
runs 15x faster (~5 min/fold instead of 9 hours!) with ZERO ordinal distortion and true oblivious trees!

Jalankan dari root repo:
  python -u -W ignore ghani/src/v8/train_v8_catboost.py
"""
import gc
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss

for _sub in ["common", "v7", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
from train_v7_lgb import compute_test_sim_weights, eval_v7_oof, get_v7_feature_cols

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def build_cb_matrix(tr, te, feats, cat_cols):
    """Build float32 matrix with compact one-hot categoricals for fast symmetric-tree CatBoost."""
    num_cols = [c for c in feats if c not in cat_cols]
    Xtr = tr[num_cols].to_numpy(dtype=np.float32, copy=True)
    Xte = te[num_cols].to_numpy(dtype=np.float32, copy=True)

    cat_cats = {}
    oh_tr, oh_te = [Xtr], [Xte]
    col_names = list(num_cols)
    for c in cat_cols:
        cats = list(tr[c].cat.categories.astype(str))
        cat_cats[c] = cats
        for src, dst in [(tr[c], oh_tr), (te[c], oh_te)]:
            codes = pd.Categorical(src.astype(str), categories=cats).codes
            mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
            ok = codes >= 0
            mat[np.where(ok)[0], codes[ok]] = 1.0
            dst.append(mat)
        col_names.extend([f"{c}__{v}" for v in cats])
    return np.hstack(oh_tr), np.hstack(oh_te), col_names, {"num_cols": num_cols, "cat_cols": cat_cols, "cat_cats": cat_cats}


def main():
    os.makedirs("ghani/models/v8/catboost", exist_ok=True)
    os.makedirs("ghani/cache/v8/catboost", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v7/feat_v7_train.pkl")
    te = pd.read_pickle("ghani/cache/v7/feat_v7_test.pkl")
    folds = np.load("ghani/cache/v7/folds_gkf5.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)

    n_tr = tr["cu_sp_n"].to_numpy(dtype=np.float32)
    n_te = te["cu_sp_n"].to_numpy(dtype=np.float32)
    sim_w, _ = compute_test_sim_weights(n_tr, n_te, tr["is_peak_month"].to_numpy(), te["is_peak_month"].to_numpy())

    feats = [c for c in get_v7_feature_cols(tr) if c != "cu_loo_n"]
    cat_cols = [c for c in feats if str(tr[c].dtype) == "category"]
    Xtr_all, Xte_all, _, cb_spec = build_cb_matrix(tr, te, feats, cat_cols)
    del tr, te
    gc.collect()
    joblib.dump(cb_spec, "ghani/models/v8/catboost/cb_spec.joblib", compress=3)
    log(f"Built fast float32 CatBoost matrix: Xtr={Xtr_all.shape}, Xte={Xte_all.shape}")

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(Xte_all), K), dtype=np.float32)
    te_pool = Pool(Xte_all)

    for f in range(5):
        m_path = f"ghani/models/v8/catboost/cb_f{f}.cbm"
        oof_p = f"ghani/cache/v8/catboost/oof_f{f}.npy"
        te_p = f"ghani/cache/v8/catboost/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"v8 CatBoost Fold {f}: cached.")
            continue

        log(f"=== v8 CatBoost Symmetric-Tree Fold {f} ===")
        dtr = Pool(Xtr_all[tr_idx], label=y[tr_idx])
        dva = Pool(Xtr_all[va_idx], label=y[va_idx])
        cb = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=1200,
            learning_rate=0.085,
            depth=7,
            l2_leaf_reg=8.0,
            random_seed=42 + f,
            border_count=128,
            thread_count=-1,
            verbose=200,
        )
        cb.fit(dtr, eval_set=dva, early_stopping_rounds=50, use_best_model=True)
        cb.save_model(m_path)

        va_raw = cb.predict(dva, prediction_type="RawFormulaVal").astype(np.float32)
        te_raw = cb.predict(te_pool, prediction_type="RawFormulaVal").astype(np.float32)
        del dtr, dva, cb
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0
        vp = glmm.softmax(va_raw)
        log(f"  CatBoost F{f} done: val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | argF1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    np.save("ghani/cache/v8/cb_oof_logits.npy", oof_logits)
    np.save("ghani/cache/v8/cb_test_logits.npy", te_logits)
    ll_a, ll_s, f1_a, f1_s, w_m, _ = eval_v7_oof("v8_CatBoost", glmm.softmax(oof_logits), y, folds, n_tr, sim_w, K)
    json.dump(
        {"oof_ll": float(ll_a), "test_sim_ll": float(ll_s), "cv_f1": float(f1_a), "test_sim_f1": float(f1_s), "weights": w_m.tolist()},
        open("ghani/models/v8/meta_cb.json", "w"), indent=2,
    )


if __name__ == "__main__":
    main()
