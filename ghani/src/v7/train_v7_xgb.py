"""v7 Model Family #3: 5-Fold GroupKFold(customer_id) Depth-Wise XGBoost (hist, max_depth=7)
on pure within-split feat_v7 (ZERO customer leakage!).

Jalankan dari root repo:
  python -u -W ignore ghani/src/train_v7_xgb.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm
from train_v7_lgb import compute_test_sim_weights, eval_v7_oof, get_v7_feature_cols

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    os.makedirs("ghani/models/v7/xgb", exist_ok=True)
    os.makedirs("ghani/cache/v7/xgb", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v7/feat_v7_train.pkl")
    te = pd.read_pickle("ghani/cache/v7/feat_v7_test.pkl")
    folds = np.load("ghani/cache/v7/folds_gkf5.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)

    n_tr = tr["cu_sp_n"].to_numpy(dtype=np.float32)
    n_te = te["cu_sp_n"].to_numpy(dtype=np.float32)
    sim_w, _ = compute_test_sim_weights(n_tr, n_te, tr["is_peak_month"].to_numpy(), te["is_peak_month"].to_numpy())
    feats = get_v7_feature_cols(tr)
    log(f"Loaded feat_v7 for XGBoost: tr={tr.shape}, te={te.shape}, feats={len(feats)}")
    tr.drop(columns=[c for c in tr.columns if c not in feats], inplace=True)
    te.drop(columns=[c for c in te.columns if c not in feats], inplace=True)
    gc.collect()

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te), K), dtype=np.float32)
    best_iters = []

    params = dict(
        objective="multi:softprob",
        num_class=K,
        eval_metric="mlogloss",
        tree_method="hist",
        max_bin=128,
        max_depth=7,
        min_child_weight=60.0,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.5,
        colsample_bylevel=0.7,
        reg_lambda=15.0,
        reg_alpha=2.0,
        nthread=os.cpu_count(),
    )

    dte = xgb.DMatrix(te[feats], enable_categorical=True)
    del te
    gc.collect()

    for f in range(5):
        m_path = f"ghani/models/v7/xgb/xgb_f{f}.ubj"
        oof_p = f"ghani/cache/v7/xgb/oof_f{f}.npy"
        te_p = f"ghani/cache/v7/xgb/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            bst = xgb.Booster(model_file=m_path)
            bi = int(bst.best_iteration) + 1 if getattr(bst, "best_iteration", None) is not None else bst.num_boosted_rounds()
            best_iters.append(bi)
            log(f"v7 XGB Fold {f}: cached (bi={bi}).")
            continue

        dtr = xgb.DMatrix(tr.iloc[tr_idx][feats], label=y[tr_idx], enable_categorical=True)
        dva = xgb.DMatrix(tr.iloc[va_idx][feats], label=y[va_idx], enable_categorical=True)

        log(f"=== v7 XGBoost Fold {f} ===")
        model = xgb.train(
            dict(params, seed=42 + f), dtr, num_boost_round=800,
            evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=100,
        )
        model.save_model(m_path)
        bi = model.best_iteration + 1
        best_iters.append(bi)

        va_raw = model.predict(dva, output_margin=True, iteration_range=(0, bi)).astype(np.float32)
        te_raw = model.predict(dte, output_margin=True, iteration_range=(0, bi)).astype(np.float32)
        del dtr, dva, model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0
        vp = glmm.softmax(va_raw)
        log(f"  XGB F{f} done (it={bi}): val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | argF1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    np.save("ghani/cache/v7/xgb_oof_logits.npy", oof_logits)
    np.save("ghani/cache/v7/xgb_test_logits.npy", te_logits)
    ll_a, ll_s, f1_a, f1_s, w_m, _ = eval_v7_oof("v7_XGBoost", glmm.softmax(oof_logits), y, folds, n_tr, sim_w, K)
    json.dump(
        {"oof_ll": float(ll_a), "test_sim_ll": float(ll_s), "cv_f1": float(f1_a), "test_sim_f1": float(f1_s),
         "weights": w_m.tolist(), "xgb_best_iters": best_iters},
        open("ghani/models/v7/meta_xgb.json", "w"), indent=2,
    )


if __name__ == "__main__":
    main()
