"""Stage-1 5-Fold LightGBM for v8 (Zero ID Leakage, Zero Test Pollution, Exact Fold-Matched cu_tr_*).

Each fold f:
  - Builds `cu_tr_*` on `tr_idx` via exact LOO within `tr_idx` (E[n_norm]=1.735, seen=82.3%)
  - Builds `cu_tr_*` on `va_idx` via lookup from `tr_idx`       (E[n_norm]=1.735, seen=82.3%)
  - Predicts `test` using full-train `cu_tr_*`                  (E[n_norm]=1.735, seen=88.6%, 0% test noise!)

Jalankan dari root repo:
  python -u -W ignore ghani/task1/src/v8/run_stage1_v8.py
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

for _sub in ["common", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
import v8_common as v8c

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    os.makedirs("ghani/task1/models/v8/stage1", exist_ok=True)
    os.makedirs("ghani/task1/cache/v8/stage1", exist_ok=True)

    tr = pd.read_pickle("ghani/task1/cache/v8/feat_v8_base_train.pkl")
    te = pd.read_pickle("ghani/task1/cache/v8/feat_v8_base_test.pkl")
    cu_te_full = pd.read_pickle("ghani/task1/cache/v8/cu_tr_test_full.pkl")
    folds = np.load("ghani/task1/cache/v8/folds_v8.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)
    n_cust_all = int(max(tr["customer_id_code"].max(), te["customer_id_code"].max())) + 1

    base_cols = v8c.get_base_feature_cols(tr)
    cat_cols = [c for c in base_cols if str(tr[c].dtype) == "category"]
    s1_feats = base_cols + v8c.CU_TR_FEAT_NAMES
    log(f"Stage-1 v8 features: {len(s1_feats)} (base={len(base_cols)}, cu_tr={len(v8c.CU_TR_FEAT_NAMES)}, cats={len(cat_cols)})")

    X_te_s1 = pd.concat([te[base_cols], cu_te_full], axis=1)

    params = dict(
        objective="multiclass",
        num_class=K,
        learning_rate=0.075,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=12.0,
        reg_alpha=1.5,
        max_bin=127,
        cat_smooth=20.0,
        verbose=-1,
        n_jobs=os.cpu_count(),
    )

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te), K), dtype=np.float32)

    for f in range(5):
        m_path = f"ghani/task1/models/v8/stage1/lgb_f{f}.txt"
        oof_p = f"ghani/task1/cache/v8/stage1/oof_f{f}.npy"
        te_p = f"ghani/task1/cache/v8/stage1/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"Stage-1 v8 Fold {f}: cached.")
            continue

        df_ref = tr.iloc[tr_idx]
        df_va = tr.iloc[va_idx]
        cu_tr = v8c.compute_cu_tr_df(df_ref, df_ref, n_cust_all, is_same=True)
        cu_va = v8c.compute_cu_tr_df(df_ref, df_va, n_cust_all, is_same=False)

        X_tr = pd.concat([df_ref[base_cols], cu_tr], axis=1)
        X_va = pd.concat([df_va[base_cols], cu_va], axis=1)
        del cu_tr, cu_va
        gc.collect()

        dtr = lgb.Dataset(X_tr, y[tr_idx], categorical_feature=cat_cols, free_raw_data=True)
        dva = lgb.Dataset(X_va, y[va_idx], reference=dtr, categorical_feature=cat_cols, free_raw_data=True)
        del X_tr
        gc.collect()

        log(f"=== Stage-1 v8 Fold {f} ===")
        model = lgb.train(
            dict(params, random_state=42 + f),
            dtr,
            num_boost_round=1100,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)],
        )
        del dtr, dva
        gc.collect()

        model.save_model(m_path, num_iteration=model.best_iteration)
        if f == 0:
            imp = pd.Series(model.feature_importance("gain"), index=s1_feats)
            imp = (imp / max(imp.sum(), 1e-9) * 100.0).sort_values(ascending=False)
            imp.to_csv("ghani/task1/cache/v8/stage1/importance_f0.csv")
            log("Top 12 Stage-1 v8 features (% gain): " + str(imp.head(12).round(2).to_dict()))

        va_raw = model.predict(X_va, num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        te_raw = model.predict(X_te_s1, num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        del X_va, model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0

        vp = glmm.softmax(va_raw)
        log(f"Stage-1 v8 Fold {f} done: val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    np.save("ghani/task1/cache/v8/stage1/oof_logits_all.npy", oof_logits)
    np.save("ghani/task1/cache/v8/stage1/test_logits_all.npy", te_logits)

    s1_p = glmm.softmax(oof_logits)
    s1_ll = log_loss(y, s1_p, labels=list(range(K)))
    s1_f1, _, _ = glmm.cv_macro_f1_with_weights(s1_p, y, folds)
    log(f"=== FULL 5-FOLD STAGE-1 v8: OOF_ll={s1_ll:.5f} | arg_F1={glmm.fast_macro_f1(y, s1_p.argmax(1), K):.5f} | CV_tuned_F1={s1_f1:.5f} ===")
    json.dump({"s1_feats": s1_feats, "cat_cols": cat_cols, "oof_ll": float(s1_ll), "cv_f1": float(s1_f1)}, open("ghani/task1/models/v8/meta_stage1.json", "w"), indent=2)


if __name__ == "__main__":
    main()
