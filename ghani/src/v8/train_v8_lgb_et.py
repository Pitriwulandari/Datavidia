"""v8 Orthogonal Model #1: 5-Fold GroupKFold(customer_id) Randomized-Threshold Extra-Trees GBDT (extra_trees=True).

Replaces slow 20-hour CPU CatBoost with a fast (~30 min total!) randomized-threshold GBDT
(extra_trees=True, num_leaves=159, colsample_bytree=0.38, drift-free features without cu_loo_n)
that creates genuinely distinct randomized decision boundaries from greedy LGB_unw and XGBoost!

Jalankan dari root repo:
  python -u -W ignore ghani/src/v8/train_v8_lgb_et.py
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

for _sub in ["common", "v7", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
from train_v7_lgb import compute_test_sim_weights, eval_v7_oof, get_v7_feature_cols

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def main():
    os.makedirs("ghani/models/v8/lgb_et", exist_ok=True)
    os.makedirs("ghani/cache/v8/lgb_et", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v7/feat_v7_train.pkl")
    te = pd.read_pickle("ghani/cache/v7/feat_v7_test.pkl")
    folds = np.load("ghani/cache/v7/folds_gkf5.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)

    n_tr = tr["cu_sp_n"].to_numpy(dtype=np.float32)
    n_te = te["cu_sp_n"].to_numpy(dtype=np.float32)
    sim_w, _ = compute_test_sim_weights(n_tr, n_te, tr["is_peak_month"].to_numpy(), te["is_peak_month"].to_numpy())

    # Drop cu_loo_n split-ratio artifact discovered in adversarial audit!
    feats = [c for c in get_v7_feature_cols(tr) if c != "cu_loo_n"]
    cat_cols = [c for c in feats if str(tr[c].dtype) == "category"]
    tr.drop(columns=[c for c in tr.columns if c not in feats], inplace=True)
    te.drop(columns=[c for c in te.columns if c not in feats], inplace=True)
    gc.collect()
    log(f"Loaded clean drift-free feat_v8 for LGB_ExtraTrees: tr={tr.shape}, te={te.shape}")

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te), K), dtype=np.float32)

    params = dict(
        objective="multiclass",
        num_class=K,
        extra_trees=True,
        learning_rate=0.055,
        num_leaves=159,
        max_depth=11,
        min_child_samples=180,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.38,
        reg_lambda=25.0,
        reg_alpha=3.0,
        max_bin=127,
        cat_smooth=25.0,
        verbose=-1,
        n_jobs=os.cpu_count(),
    )

    for f in range(5):
        m_path = f"ghani/models/v8/lgb_et/lgb_f{f}.txt"
        oof_p = f"ghani/cache/v8/lgb_et/oof_f{f}.npy"
        te_p = f"ghani/cache/v8/lgb_et/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"v8 LGB_ET Fold {f}: cached.")
            continue

        dtr = lgb.Dataset(tr.iloc[tr_idx][feats], y[tr_idx], categorical_feature=cat_cols, free_raw_data=True)
        dva = lgb.Dataset(tr.iloc[va_idx][feats], y[va_idx], reference=dtr, categorical_feature=cat_cols, free_raw_data=True)

        log(f"=== v8 LGB_ExtraTrees Fold {f} ===")
        model = lgb.train(
            dict(params, random_state=500 + f), dtr, num_boost_round=950,
            valid_sets=[dva], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(150)],
        )
        del dtr, dva
        gc.collect()

        model.save_model(m_path, num_iteration=model.best_iteration)
        va_raw = model.predict(tr.iloc[va_idx][feats], num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        te_raw = model.predict(te[feats], num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        del model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0
        vp = glmm.softmax(va_raw)
        log(f"  LGB_ET F{f} done: val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | argF1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    np.save("ghani/cache/v8/lgb_et_oof_logits.npy", oof_logits)
    np.save("ghani/cache/v8/lgb_et_test_logits.npy", te_logits)
    ll_a, ll_s, f1_a, f1_s, w_m, _ = eval_v7_oof("v8_LGB_ExtraTrees", glmm.softmax(oof_logits), y, folds, n_tr, sim_w, K)
    json.dump(
        {"oof_ll": float(ll_a), "test_sim_ll": float(ll_s), "cv_f1": float(f1_a), "test_sim_f1": float(f1_s), "weights": w_m.tolist()},
        open("ghani/models/v8/meta_lgb_et.json", "w"), indent=2,
    )


if __name__ == "__main__":
    main()
