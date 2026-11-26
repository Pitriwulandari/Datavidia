"""Stage-2 Fast Residual-Boosted 5-Fold CatBoost (Symmetric/Oblivious Trees on top of Joint GLMM baseline).
Uses compact integer-encoded low-cardinality categoricals + border_count=64 + rsm=0.35 -> ~7x faster on CPU!
"""
import gc
import json
import os
import sys
import time

import catboost as cb
import numpy as np
import pandas as pd
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
    os.makedirs("ghani/task1/models/v6/catboost", exist_ok=True)
    os.makedirs("ghani/task1/cache/v6/catboost", exist_ok=True)

    tr_base, te_base, y, folds, classes, te_sid, cat_cols = s2.load_stage2_base()
    K = len(classes)
    # Convert 5 low-cardinality categoricals to clean float32 codes (avoids 10x CPU CTR overhead!)
    for c in cat_cols:
        cats = tr_base[c].cat.categories
        tr_base[c] = tr_base[c].cat.codes.astype(np.float32)
        te_base[c] = pd.Categorical(te_base[c], categories=cats).codes.astype(np.float32)
    log(f"loaded Stage-2 base for fast CatBoost: tr={tr_base.shape}, te={te_base.shape}")

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_sid), K), dtype=np.float32)

    for f in range(5):
        m_path = f"ghani/task1/models/v6/catboost/cb_f{f}.cbm"
        oof_p = f"ghani/task1/cache/v6/catboost/oof_f{f}.npy"
        te_p = f"ghani/task1/cache/v6/catboost/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"CatBoost Fold {f}: cached.")
            continue

        blk_tr, blk_va, blk_te = s2.get_fold_glmm_arrays(f, folds)
        base_tr = blk_tr[:, 7:14].astype(np.float64)
        base_va = blk_va[:, 7:14].astype(np.float64)
        base_te = blk_te[:, 7:14].astype(np.float64)

        X_tr = s2.attach_glmm_df(tr_base.iloc[tr_idx], blk_tr).to_numpy(dtype=np.float32)
        X_va = s2.attach_glmm_df(tr_base.iloc[va_idx], blk_va).to_numpy(dtype=np.float32)

        p_tr = cb.Pool(X_tr, label=y[tr_idx], baseline=base_tr)
        p_va = cb.Pool(X_va, label=y[va_idx], baseline=base_va)
        del X_tr, X_va
        gc.collect()

        log(f"=== Stage-2 Fast CatBoost Fold {f} ===")
        model = cb.CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=550,
            learning_rate=0.10,
            depth=5,
            l2_leaf_reg=15.0,
            random_strength=1.0,
            rsm=0.35,
            border_count=64,
            random_seed=42 + f,
            thread_count=os.cpu_count(),
            early_stopping_rounds=40,
            verbose=50,
        )
        model.fit(p_tr, eval_set=p_va, use_best_model=True)
        model.save_model(m_path)

        va_raw = model.predict(p_va, prediction_type="RawFormulaVal").astype(np.float32)
        del p_tr, p_va
        gc.collect()

        X_te = s2.attach_glmm_df(te_base, blk_te).to_numpy(dtype=np.float32)
        p_te = cb.Pool(X_te, baseline=base_te)
        te_raw = model.predict(p_te, prediction_type="RawFormulaVal").astype(np.float32)
        del X_te, p_te, model
        gc.collect()

        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0

        vp = glmm.softmax(va_raw)
        log(f"CatBoost Fold {f} done: val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")

    oof_p_all = glmm.softmax(oof_logits)
    ll = log_loss(y, oof_p_all, labels=list(range(K)))
    f1_cv, w_opt, _ = glmm.cv_macro_f1_with_weights(oof_p_all, y, folds)
    log(f"=== STAGE-2 CATBOOST 5-FOLD: OOF_ll={ll:.5f} | arg_F1={glmm.fast_macro_f1(y, oof_p_all.argmax(1), K):.5f} | CV_F1={f1_cv:.5f} ===")
    np.save("ghani/task1/cache/v6/cb_oof_logits.npy", oof_logits)
    np.save("ghani/task1/cache/v6/cb_test_logits.npy", te_logits)
    json.dump({"oof_ll": float(ll), "cv_f1": float(f1_cv), "weights": w_opt.tolist()}, open("ghani/task1/models/v6/meta_cb.json", "w"), indent=2)


if __name__ == "__main__":
    main()
