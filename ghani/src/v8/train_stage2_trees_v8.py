"""Stage-2 5-Fold Diverse Tree Models for v8:
  1. `lgb_res`  : Residual-Boosted LightGBM on `glmm_logit`
  2. `lgb_full` : Full-Tree LightGBM from scratch on all 270 zero-shift features
  3. `lgb_et`   : Randomized Extra-Trees Residual LightGBM (`extra_trees=True`)
  4. `xgb`      : Depth-Wise `hist` Residual XGBoost on `glmm_logit`

Jalankan setelah `build_glmm_v8.py`:
  python -u -W ignore ghani/src/v8/train_stage2_trees_v8.py
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
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


def run_lgb_variant(mode, tr_base, te_base, y, folds, K, n_cust_all, cat_cols):
    os.makedirs(f"ghani/models/v8/{mode}", exist_ok=True)
    os.makedirs(f"ghani/cache/v8/{mode}", exist_ok=True)

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_base), K), dtype=np.float32)

    if mode == "lgb_res":
        params = dict(
            objective="multiclass", num_class=K, learning_rate=0.045, num_leaves=31,
            min_child_samples=150, subsample=0.8, subsample_freq=1, colsample_bytree=0.50,
            reg_lambda=20.0, reg_alpha=2.0, max_bin=127, cat_smooth=25.0,
            verbose=-1, n_jobs=os.cpu_count(),
        )
        max_rounds = 550
    elif mode == "lgb_full":
        params = dict(
            objective="multiclass", num_class=K, learning_rate=0.065, num_leaves=63,
            min_child_samples=120, subsample=0.8, subsample_freq=1, colsample_bytree=0.50,
            reg_lambda=15.0, reg_alpha=1.5, max_bin=127, cat_smooth=20.0,
            verbose=-1, n_jobs=os.cpu_count(),
        )
        max_rounds = 900
    elif mode == "lgb_et":
        params = dict(
            objective="multiclass", num_class=K, learning_rate=0.05, num_leaves=95,
            extra_trees=True, min_child_samples=150, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.45, reg_lambda=25.0, reg_alpha=2.5, max_bin=127, cat_smooth=25.0,
            verbose=-1, n_jobs=os.cpu_count(),
        )
        max_rounds = 600
    else:
        raise ValueError(mode)

    seed_base = {"lgb_res": 100, "lgb_full": 200, "lgb_et": 300}[mode]

    for f in range(5):
        m_path = f"ghani/models/v8/{mode}/lgb_f{f}.txt"
        oof_p = f"ghani/cache/v8/{mode}/oof_f{f}.npy"
        te_p = f"ghani/cache/v8/{mode}/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"{mode} Fold {f}: cached.")
            continue

        X_tr, X_va, X_te, blk_tr, blk_va, blk_te = v8c.assemble_stage2_fold(f, tr_base, te_base, folds, n_cust_all)

        if mode == "lgb_res":
            b_tr = blk_tr[:, 7:14].astype(np.float64)
            b_va = blk_va[:, 7:14].astype(np.float64)
            b_te = blk_te[:, 7:14].astype(np.float32)
            dtr = lgb.Dataset(X_tr, y[tr_idx], categorical_feature=cat_cols, init_score=b_tr.ravel(order="F"), free_raw_data=True)
            dva = lgb.Dataset(X_va, y[va_idx], reference=dtr, categorical_feature=cat_cols, init_score=b_va.ravel(order="F"), free_raw_data=True)
        elif mode == "lgb_et":
            b_tr = (0.65 * blk_tr[:, 7:14] + 0.35 * blk_tr[:, 0:7]).astype(np.float64)
            b_va = (0.65 * blk_va[:, 7:14] + 0.35 * blk_va[:, 0:7]).astype(np.float64)
            b_te = (0.65 * blk_te[:, 7:14] + 0.35 * blk_te[:, 0:7]).astype(np.float32)
            dtr = lgb.Dataset(X_tr, y[tr_idx], categorical_feature=cat_cols, init_score=b_tr.ravel(order="F"), free_raw_data=True)
            dva = lgb.Dataset(X_va, y[va_idx], reference=dtr, categorical_feature=cat_cols, init_score=b_va.ravel(order="F"), free_raw_data=True)
        else:
            b_va, b_te = None, None
            dtr = lgb.Dataset(X_tr, y[tr_idx], categorical_feature=cat_cols, free_raw_data=True)
            dva = lgb.Dataset(X_va, y[va_idx], reference=dtr, categorical_feature=cat_cols, free_raw_data=True)
        del X_tr
        gc.collect()

        log(f"=== Stage-2 v8 {mode} Fold {f} ===")
        model = lgb.train(
            dict(params, random_state=seed_base + f),
            dtr,
            num_boost_round=max_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)],
        )
        del dtr, dva
        gc.collect()

        model.save_model(m_path, num_iteration=model.best_iteration)
        va_raw = model.predict(X_va, num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        te_raw = model.predict(X_te, num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        if b_va is not None:
            va_raw = (va_raw + b_va.astype(np.float32)).astype(np.float32)
            te_raw = (te_raw + b_te).astype(np.float32)
        del X_va, X_te, model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0

        vp = glmm.softmax(va_raw)
        log(f"{mode} Fold {f} done: val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    oof_p_all = glmm.softmax(oof_logits)
    ll = log_loss(y, oof_p_all, labels=list(range(K)))
    f1_cv, w_opt, _ = glmm.cv_macro_f1_with_weights(oof_p_all, y, folds)
    log(f"=== STAGE-2 v8 {mode.upper()} 5-FOLD: OOF_ll={ll:.5f} | arg_F1={glmm.fast_macro_f1(y, oof_p_all.argmax(1), K):.5f} | CV_F1={f1_cv:.5f} ===")
    np.save(f"ghani/cache/v8/{mode}_oof_logits.npy", oof_logits)
    np.save(f"ghani/cache/v8/{mode}_test_logits.npy", te_logits)
    json.dump({"oof_ll": float(ll), "cv_f1": float(f1_cv), "weights": w_opt.tolist()}, open(f"ghani/models/v8/meta_{mode}.json", "w"), indent=2)


def run_xgb_res(tr_base, te_base, y, folds, K, n_cust_all):
    os.makedirs("ghani/models/v8/xgb", exist_ok=True)
    os.makedirs("ghani/cache/v8/xgb", exist_ok=True)

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_base), K), dtype=np.float32)
    best_iters = []

    params = dict(
        objective="multi:softprob",
        num_class=K,
        eval_metric="mlogloss",
        tree_method="hist",
        max_bin=128,
        max_depth=6,
        min_child_weight=50.0,
        learning_rate=0.055,
        subsample=0.8,
        colsample_bytree=0.50,
        colsample_bylevel=0.70,
        reg_lambda=15.0,
        reg_alpha=2.0,
        nthread=os.cpu_count(),
    )

    for f in range(5):
        m_path = f"ghani/models/v8/xgb/xgb_f{f}.ubj"
        oof_p = f"ghani/cache/v8/xgb/oof_f{f}.npy"
        te_p = f"ghani/cache/v8/xgb/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            bst = xgb.Booster(model_file=m_path)
            bi = int(bst.best_iteration) + 1 if hasattr(bst, "best_iteration") and bst.best_iteration is not None else bst.num_boosted_rounds()
            best_iters.append(bi)
            log(f"XGB v8 Fold {f}: cached (bi={bi}).")
            continue

        X_tr, X_va, X_te, blk_tr, blk_va, blk_te = v8c.assemble_stage2_fold(f, tr_base, te_base, folds, n_cust_all)
        b_tr = blk_tr[:, 7:14].astype(np.float32)
        b_va = blk_va[:, 7:14].astype(np.float32)
        b_te = blk_te[:, 7:14].astype(np.float32)

        dtr = xgb.DMatrix(X_tr, label=y[tr_idx], base_margin=b_tr, enable_categorical=True)
        dva = xgb.DMatrix(X_va, label=y[va_idx], base_margin=b_va, enable_categorical=True)
        del X_tr, X_va
        gc.collect()

        log(f"=== Stage-2 v8 XGBoost Fold {f} ===")
        model = xgb.train(
            dict(params, seed=42 + f),
            dtr,
            num_boost_round=700,
            evals=[(dva, "val")],
            early_stopping_rounds=40,
            verbose_eval=100,
        )
        model.save_model(m_path)
        bi = int(model.best_iteration) + 1
        best_iters.append(bi)
        va_raw = model.predict(dva, output_margin=True, iteration_range=(0, bi)).astype(np.float32)
        del dtr, dva
        gc.collect()

        dte = xgb.DMatrix(X_te, base_margin=b_te, enable_categorical=True)
        te_raw = model.predict(dte, output_margin=True, iteration_range=(0, bi)).astype(np.float32)
        del X_te, dte, model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0

        vp = glmm.softmax(va_raw)
        log(f"XGB v8 Fold {f} done (iter={bi}): val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    oof_p_all = glmm.softmax(oof_logits)
    ll = log_loss(y, oof_p_all, labels=list(range(K)))
    f1_cv, w_opt, _ = glmm.cv_macro_f1_with_weights(oof_p_all, y, folds)
    log(f"=== STAGE-2 v8 XGBOOST 5-FOLD: OOF_ll={ll:.5f} | arg_F1={glmm.fast_macro_f1(y, oof_p_all.argmax(1), K):.5f} | CV_F1={f1_cv:.5f} ===")
    np.save("ghani/cache/v8/xgb_oof_logits.npy", oof_logits)
    np.save("ghani/cache/v8/xgb_test_logits.npy", te_logits)
    json.dump({"oof_ll": float(ll), "cv_f1": float(f1_cv), "weights": w_opt.tolist(), "best_iters": best_iters}, open("ghani/models/v8/meta_xgb.json", "w"), indent=2)


def main():
    tr_base = pd.read_pickle("ghani/cache/v8/feat_v8_base_train.pkl")
    te_base = pd.read_pickle("ghani/cache/v8/feat_v8_base_test.pkl")
    folds = np.load("ghani/cache/v8/folds_v8.npy")
    classes = sorted(tr_base["status_penanganan"].unique())
    y = tr_base["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)
    n_cust_all = int(max(tr_base["customer_id_code"].max(), te_base["customer_id_code"].max())) + 1
    cat_cols = [c for c in v8c.get_base_feature_cols(tr_base) if str(tr_base[c].dtype) == "category"]

    modes = [m.strip() for m in os.environ.get("MODES", "lgb_res,lgb_full,lgb_et,xgb").split(",") if m.strip()]
    for m in modes:
        if m == "xgb":
            run_xgb_res(tr_base, te_base, y, folds, K, n_cust_all)
        else:
            run_lgb_variant(m, tr_base, te_base, y, folds, K, n_cust_all, cat_cols)


if __name__ == "__main__":
    main()
