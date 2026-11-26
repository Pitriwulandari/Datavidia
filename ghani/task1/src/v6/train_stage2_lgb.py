"""Stage-2 5-Fold LightGBM:
  1. LGB_res : Residual-Boosted LightGBM on top of leak-free Joint GLMM logits (init_score in Fortran order!).
  2. LGB_full: Full-Tree LightGBM from scratch seeing all features + leak-free Joint GLMM random effects (diverse tree topology!).

Jalankan (dari root repo, setelah build_stage2_glmm.py):
  python -u -W ignore ghani/task1/src/train_stage2_lgb.py             # default: jalankan LGB_res & LGB_full
  $env:MODES='res'; python -u -W ignore ghani/task1/src/train_stage2_lgb.py
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm
import stage2_common as s2

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def run_mode(mode, tr_base, te_base, y, folds, classes, te_sid, cat_cols):
    K = len(classes)
    os.makedirs(f"ghani/task1/models/v6/lgb_{mode}", exist_ok=True)
    os.makedirs(f"ghani/task1/cache/v6/lgb_{mode}", exist_ok=True)

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_sid), K), dtype=np.float32)

    if mode == "res":
        params = dict(
            objective="multiclass",
            num_class=K,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=20.0,
            reg_alpha=2.0,
            max_bin=127,
            cat_smooth=25.0,
            verbose=-1,
            n_jobs=os.cpu_count(),
        )
        max_rounds = 600
    else:
        params = dict(
            objective="multiclass",
            num_class=K,
            learning_rate=0.06,
            num_leaves=63,
            min_child_samples=120,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=15.0,
            reg_alpha=1.5,
            max_bin=127,
            cat_smooth=20.0,
            verbose=-1,
            n_jobs=os.cpu_count(),
        )
        max_rounds = 1000

    for f in range(5):
        m_path = f"ghani/task1/models/v6/lgb_{mode}/lgb_f{f}.txt"
        oof_p = f"ghani/task1/cache/v6/lgb_{mode}/oof_f{f}.npy"
        te_p = f"ghani/task1/cache/v6/lgb_{mode}/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"LGB_{mode} Fold {f}: cached.")
            continue

        blk_tr, blk_va, blk_te = s2.get_fold_glmm_arrays(f, folds)
        base_tr = blk_tr[:, 7:14].astype(np.float64)
        base_va = blk_va[:, 7:14].astype(np.float64)
        base_te = blk_te[:, 7:14].astype(np.float64)

        X_tr = s2.attach_glmm_df(tr_base.iloc[tr_idx], blk_tr)
        X_va = s2.attach_glmm_df(tr_base.iloc[va_idx], blk_va)

        if mode == "res":
            dtr = lgb.Dataset(X_tr, y[tr_idx], categorical_feature=cat_cols,
                              init_score=base_tr.ravel(order="F"), free_raw_data=True)
            dva = lgb.Dataset(X_va, y[va_idx], reference=dtr, categorical_feature=cat_cols,
                              init_score=base_va.ravel(order="F"), free_raw_data=True)
        else:
            dtr = lgb.Dataset(X_tr, y[tr_idx], categorical_feature=cat_cols, free_raw_data=True)
            dva = lgb.Dataset(X_va, y[va_idx], reference=dtr, categorical_feature=cat_cols, free_raw_data=True)
        del X_tr
        gc.collect()

        log(f"=== Stage-2 LGB_{mode} Fold {f} ===")
        pf = dict(params, random_state=(100 if mode == "res" else 200) + f)
        model = lgb.train(
            pf, dtr, num_boost_round=max_rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
        )
        del dtr, dva
        gc.collect()

        model.save_model(m_path, num_iteration=model.best_iteration)
        va_raw = model.predict(X_va, num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        if mode == "res":
            va_raw = (va_raw + base_va.astype(np.float32)).astype(np.float32)
        del X_va
        gc.collect()

        X_te = s2.attach_glmm_df(te_base, blk_te)
        te_raw = model.predict(X_te, num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        if mode == "res":
            te_raw = (te_raw + base_te.astype(np.float32)).astype(np.float32)
        del X_te, model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0

        vp = glmm.softmax(va_raw)
        log(f"LGB_{mode} Fold {f} done: val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    oof_p_all = glmm.softmax(oof_logits)
    ll = log_loss(y, oof_p_all, labels=list(range(K)))
    f1_cv, w_opt, _ = glmm.cv_macro_f1_with_weights(oof_p_all, y, folds)
    log(f"=== STAGE-2 LGB_{mode.upper()} 5-FOLD: OOF_ll={ll:.5f} | arg_F1={glmm.fast_macro_f1(y, oof_p_all.argmax(1), K):.5f} | CV_F1={f1_cv:.5f} ===")
    np.save(f"ghani/task1/cache/v6/lgb_{mode}_oof_logits.npy", oof_logits)
    np.save(f"ghani/task1/cache/v6/lgb_{mode}_test_logits.npy", te_logits)
    json.dump({"oof_ll": float(ll), "cv_f1": float(f1_cv), "weights": w_opt.tolist()},
              open(f"ghani/task1/models/v6/meta_lgb_{mode}.json", "w"), indent=2)


def main():
    tr_base, te_base, y, folds, classes, te_sid, cat_cols = s2.load_stage2_base()
    log(f"loaded Stage-2 base: tr={tr_base.shape}, te={te_base.shape}, cats={len(cat_cols)}")
    modes = [m.strip() for m in os.environ.get("MODES", "res,full").split(",") if m.strip()]
    for m in modes:
        run_mode(m, tr_base, te_base, y, folds, classes, te_sid, cat_cols)


if __name__ == "__main__":
    main()
