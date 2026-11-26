"""Stage-2 Residual-Boosted 5-Fold XGBoost (hist, depth-wise trees on top of leak-free Joint GLMM base_margin).

Jalankan (dari root repo, setelah build_stage2_glmm.py):
  python -u -W ignore ghani/task1/src/train_stage2_xgb.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
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


def main():
    os.makedirs("ghani/task1/models/v6/xgb", exist_ok=True)
    os.makedirs("ghani/task1/cache/v6/xgb", exist_ok=True)

    tr_base, te_base, y, folds, classes, te_sid, cat_cols = s2.load_stage2_base()
    K = len(classes)
    log(f"loaded Stage-2 base: tr={tr_base.shape}, te={te_base.shape}, cats={len(cat_cols)}")

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_sid), K), dtype=np.float32)

    params = dict(
        objective="multi:softprob",
        num_class=K,
        eval_metric="mlogloss",
        tree_method="hist",
        max_bin=128,
        max_depth=6,
        min_child_weight=50.0,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        colsample_bylevel=0.7,
        reg_lambda=15.0,
        reg_alpha=2.0,
        nthread=os.cpu_count(),
    )

    for f in range(5):
        m_path = f"ghani/task1/models/v6/xgb/xgb_f{f}.ubj"
        oof_p = f"ghani/task1/cache/v6/xgb/oof_f{f}.npy"
        te_p = f"ghani/task1/cache/v6/xgb/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"XGB Fold {f}: cached.")
            continue

        blk_tr, blk_va, blk_te = s2.get_fold_glmm_arrays(f, folds)
        base_tr = blk_tr[:, 7:14].astype(np.float32)
        base_va = blk_va[:, 7:14].astype(np.float32)
        base_te = blk_te[:, 7:14].astype(np.float32)

        X_tr = s2.attach_glmm_df(tr_base.iloc[tr_idx], blk_tr)
        X_va = s2.attach_glmm_df(tr_base.iloc[va_idx], blk_va)
        dtr = xgb.DMatrix(X_tr, label=y[tr_idx], base_margin=base_tr, enable_categorical=True)
        dva = xgb.DMatrix(X_va, label=y[va_idx], base_margin=base_va, enable_categorical=True)
        del X_tr, X_va
        gc.collect()

        log(f"=== Stage-2 XGBoost Fold {f} ===")
        pf = dict(params, seed=42 + f)
        model = xgb.train(
            pf, dtr, num_boost_round=800,
            evals=[(dva, "val")],
            early_stopping_rounds=40,
            verbose_eval=50,
        )
        model.save_model(m_path)
        bi = model.best_iteration + 1
        va_raw = model.predict(dva, output_margin=True, iteration_range=(0, bi)).astype(np.float32)
        del dtr, dva
        gc.collect()

        X_te = s2.attach_glmm_df(te_base, blk_te)
        dte = xgb.DMatrix(X_te, base_margin=base_te, enable_categorical=True)
        te_raw = model.predict(dte, output_margin=True, iteration_range=(0, bi)).astype(np.float32)
        del X_te, dte, model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0

        vp = glmm.softmax(va_raw)
        log(f"XGB Fold {f} done (iter={bi}): val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    oof_p_all = glmm.softmax(oof_logits)
    ll = log_loss(y, oof_p_all, labels=list(range(K)))
    f1_cv, w_opt, _ = glmm.cv_macro_f1_with_weights(oof_p_all, y, folds)
    log(f"=== STAGE-2 XGBOOST 5-FOLD: OOF_ll={ll:.5f} | arg_F1={glmm.fast_macro_f1(y, oof_p_all.argmax(1), K):.5f} | CV_F1={f1_cv:.5f} ===")
    np.save("ghani/task1/cache/v6/xgb_oof_logits.npy", oof_logits)
    np.save("ghani/task1/cache/v6/xgb_test_logits.npy", te_logits)
    json.dump({"oof_ll": float(ll), "cv_f1": float(f1_cv), "weights": w_opt.tolist()}, open("ghani/task1/models/v6/meta_xgb.json", "w"), indent=2)


if __name__ == "__main__":
    main()
