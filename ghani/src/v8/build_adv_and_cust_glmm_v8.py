"""v8 Step 1: Adversarial Covariate-Shift Density-Ratio Estimator (w_adv)
+ De-Aliased Empirical Bayes Customer GLMM (u_cust_v8) on clean v7 GroupKFold logits.

Outputs in ghani/cache/v8/ and ghani/models/v8/:
  - w_adv_train.npy         : (1243617,) smooth P(test|x)/P(train|x) importance weights
  - u_cust_v8_oof.npy       : (1243617, 7) 5-fold Stratified OOF de-aliased customer random effects
  - u_cust_v8_test.npy      : (362245, 7)  full-train de-aliased customer random effects for test
  - glmm_cust_v8_full.npy   : saved full-train customer GLMM state for zero-retraining inference
  - adv_meta.json           : adversarial AUC, top drift features, and tuned customer precisions

Jalankan dari root repo:
  python -u -W ignore ghani/src/v8/build_adv_and_cust_glmm_v8.py
"""
import gc
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

for _sub in ["common", "v7", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
from train_v7_lgb import get_v7_feature_cols

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def build_adversarial_weights(tr, te, feats, cat_cols, folds_gkf):
    log("1/2 Fitting 5-Fold Adversarial Density-Ratio Model P(test | x) ...")
    N_tr, N_te = len(tr), len(te)

    # Assign test rows to 5 folds grouped by customer_id_code as well
    gkf_te = GroupKFold(n_splits=5)
    folds_te = np.zeros(N_te, dtype=np.int32)
    for f, (_, va_i) in enumerate(gkf_te.split(np.zeros(N_te), groups=te["customer_id_code"].to_numpy())):
        folds_te[va_i] = f

    p_adv_tr = np.zeros(N_tr, dtype=np.float32)
    p_adv_te = np.zeros(N_te, dtype=np.float32)
    gains = np.zeros(len(feats), dtype=np.float64)

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.06,
        num_leaves=31,
        min_child_samples=600,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=50.0,
        reg_alpha=5.0,
        max_bin=127,
        verbose=-1,
        n_jobs=os.cpu_count(),
    )

    # Subsample 362,245 train rows per fold (1:1 balanced with test!) for fast, calibrated density ratio
    rng = np.random.RandomState(42)
    for f in range(5):
        tr_pos = np.where(folds_gkf != f)[0]
        va_tr_pos = np.where(folds_gkf == f)[0]
        tr_te_pos = np.where(folds_te != f)[0]
        va_te_pos = np.where(folds_te == f)[0]

        sub_tr_pos = rng.choice(tr_pos, size=len(tr_te_pos), replace=False)
        X_f_tr = pd.concat([tr.iloc[sub_tr_pos][feats], te.iloc[tr_te_pos][feats]], ignore_index=True)
        y_f_tr = np.concatenate([np.zeros(len(sub_tr_pos), dtype=np.int32), np.ones(len(tr_te_pos), dtype=np.int32)])

        sub_va_tr = rng.choice(va_tr_pos, size=len(va_te_pos), replace=False)
        X_f_va = pd.concat([tr.iloc[sub_va_tr][feats], te.iloc[va_te_pos][feats]], ignore_index=True)
        y_f_va = np.concatenate([np.zeros(len(sub_va_tr), dtype=np.int32), np.ones(len(va_te_pos), dtype=np.int32)])

        dtr = lgb.Dataset(X_f_tr, y_f_tr, categorical_feature=cat_cols, free_raw_data=True)
        dva = lgb.Dataset(X_f_va, y_f_va, reference=dtr, categorical_feature=cat_cols, free_raw_data=True)
        bst = lgb.train(
            dict(params, random_state=42 + f), dtr, num_boost_round=250,
            valid_sets=[dva], callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        p_adv_tr[va_tr_pos] = bst.predict(tr.iloc[va_tr_pos][feats], num_iteration=bst.best_iteration).astype(np.float32)
        p_adv_te[va_te_pos] = bst.predict(te.iloc[va_te_pos][feats], num_iteration=bst.best_iteration).astype(np.float32)
        gains += bst.feature_importance("gain")
        del X_f_tr, X_f_va, dtr, dva, bst
        gc.collect()

    y_all = np.concatenate([np.zeros(N_tr), np.ones(N_te)])
    p_all = np.concatenate([p_adv_tr, p_adv_te])
    auc = roc_auc_score(y_all, p_all)

    # Since each fold was trained 1:1 balanced, odds p / (1 - p) IS directly P(x | test) / P(x | train)!
    p_clip = np.clip(p_adv_tr, 0.05, 0.95)
    w_adv = (p_clip / (1.0 - p_clip)).astype(np.float32)
    w_adv = np.clip(w_adv, 0.20, 4.50)
    w_adv /= w_adv.mean()

    imp = pd.Series(gains / gains.sum() * 100.0, index=feats).sort_values(ascending=False)
    top_shift = {k: round(float(v), 2) for k, v in imp.head(12).items()}
    log(f"  Adversarial OOF AUC = {auc:.5f} | w_adv quantiles = {np.round(np.quantile(w_adv, [0.05,0.25,0.5,0.75,0.95]), 3).tolist()}")
    log(f"  Top shift features (% gain): {top_shift}")
    return w_adv, float(auc), top_shift


def build_dealiased_customer_glmm(tr_meta, te_meta, y, folds_gkf, w_adv):
    log("2/2 Fitting De-Aliased Empirical Bayes Customer GLMM on clean v7 GroupKFold logits ...")
    K = 7
    cands = [
        ("ghani/cache/v7/lgb_unw_oof_logits.npy", "ghani/cache/v7/lgb_unw_test_logits.npy", True),
        ("ghani/cache/v7/lgb_tew_oof_logits.npy", "ghani/cache/v7/lgb_tew_test_logits.npy", True),
        ("ghani/cache/v7/glmm_hc_oof_proba.npy",  "ghani/cache/v7/glmm_hc_test_proba.npy",  False),
        ("ghani/cache/v7/xgb_oof_logits.npy",     "ghani/cache/v7/xgb_test_logits.npy",     True),
        ("ghani/cache/v7/nn_oof_logits.npy",      "ghani/cache/v7/nn_test_logits.npy",      True),
    ]
    lo_l, lt_l = [], []
    for po, pt, il in cands:
        ao, at = np.load(po).astype(np.float32), np.load(pt).astype(np.float32)
        if il:
            lo_l.append(ao - ao.mean(1, keepdims=True))
            lt_l.append(at - at.mean(1, keepdims=True))
        else:
            lo = np.log(np.clip(ao, 1e-7, 1.0)); lo -= lo.mean(1, keepdims=True)
            lt = np.log(np.clip(at, 1e-7, 1.0)); lt -= lt.mean(1, keepdims=True)
            lo_l.append(lo)
            lt_l.append(lt)

    Xm_oof = np.hstack(lo_l).astype(np.float32)
    Xm_te = np.hstack(lt_l).astype(np.float32)
    sb = joblib.load("ghani/models/v7/meta_stacker_v7.joblib")
    p_oof = np.zeros((len(y), K), dtype=np.float32)
    p_te = np.zeros((len(te_meta), K), dtype=np.float32)
    for f in range(5):
        p_oof[folds_gkf == f] = sb["models"][f].predict_proba(Xm_oof[folds_gkf == f]).astype(np.float32)
        p_te += sb["models"][f].predict_proba(Xm_te).astype(np.float32) / 5.0

    z_oof = np.log(np.clip(p_oof, 1e-7, 1.0)); z_oof -= z_oof.mean(1, keepdims=True)
    z_te = np.log(np.clip(p_te, 1e-7, 1.0)); z_te -= z_te.mean(1, keepdims=True)

    c_tr = tr_meta["customer_id_code"].to_numpy(np.int32)
    c_te = te_meta["customer_id_code"].to_numpy(np.int32)
    n_cust = int(max(c_tr.max(), c_te.max())) + 1

    # Tune class-specific prior precision lam_cust on Stratified Fold 0 (which mirrors test's ~86% seen customer rate!)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    folds_skf = np.zeros(len(y), dtype=np.int32)
    for f, (_, vi) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds_skf[vi] = f

    tr0, va0 = np.where(folds_skf != 0)[0], np.where(folds_skf == 0)[0]
    seen_rate = np.isin(c_tr[va0], c_tr[tr0]).mean()
    base_ll0 = log_loss(y[va0], p_oof[va0], sample_weight=w_adv[va0], labels=list(range(K)))
    log(f"  Stratified Fold 0 seen-customer rate = {seen_rate*100:.2f}% (test actual = 88.58%) | base w_adv_ll = {base_ll0:.5f}")

    # Conservative base precision vector weighted by residual ICC:
    # High ICC: konfirmasi (2), retur (4), normal (3); Low ICC (heavy shrinkage!): klaim (1), rusak (5), tertunda (6), gagal (0)
    base_lam = np.array([8.0, 25.0, 1.1, 2.2, 0.85, 14.0, 10.0], dtype=np.float64)
    best_ll = base_ll0
    best_lam = base_lam.copy()
    for mult in [0.7, 1.0, 1.4, 2.0, 3.0]:
        cand = {"customer": base_lam * mult}
        st = glmm.fit_joint_glmm(z_oof[tr0], y[tr0], {"customer": c_tr[tr0]}, {"customer": n_cust}, n_classes=K, fixed_lam=cand, n_newton=4, verbose=False)
        _, pv = glmm.predict_glmm(z_oof[va0], {"customer": c_tr[va0]}, st, use_moderation=False)
        ll = log_loss(y[va0], pv, sample_weight=w_adv[va0], labels=list(range(K)))
        log(f"    mult={mult:.2f} -> Fold0 w_adv_ll={ll:.5f}")
        if ll < best_ll:
            best_ll = ll
            best_lam = base_lam * mult

    # 5-fold Stratified OOF u_cust_v8
    u_oof = np.zeros((len(y), K), dtype=np.float32)
    p_with_u = np.zeros((len(y), K), dtype=np.float32)
    for f in range(5):
        ti, vi = np.where(folds_skf != f)[0], np.where(folds_skf == f)[0]
        st = glmm.fit_joint_glmm(z_oof[ti], y[ti], {"customer": c_tr[ti]}, {"customer": n_cust}, n_classes=K, fixed_lam={"customer": best_lam}, n_newton=4, verbose=False)
        u_oof[vi] = st["U"]["customer"][c_tr[vi]] + st["bias"][None, :]
        _, pv = glmm.predict_glmm(z_oof[vi], {"customer": c_tr[vi]}, st, use_moderation=False)
        p_with_u[vi] = pv

    ll_full_u = log_loss(y, p_with_u, sample_weight=w_adv, labels=list(range(K)))
    log(f"  Full 5-fold de-aliased u_cust_v8 w_adv_ll: {log_loss(y, p_oof, sample_weight=w_adv):.5f} -> {ll_full_u:.5f} (delta={ll_full_u - log_loss(y, p_oof, sample_weight=w_adv):+.5f})")

    # Fit full-train customer GLMM for test!
    st_full = glmm.fit_joint_glmm(z_oof, y, {"customer": c_tr}, {"customer": n_cust}, n_classes=K, fixed_lam={"customer": best_lam}, n_newton=5, verbose=False)
    u_te = (st_full["U"]["customer"][c_te] + st_full["bias"][None, :]).astype(np.float32)

    return u_oof, u_te, st_full, best_lam.tolist(), folds_skf


def main():
    os.makedirs("ghani/cache/v8", exist_ok=True)
    os.makedirs("ghani/models/v8", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v7/feat_v7_train.pkl")
    te = pd.read_pickle("ghani/cache/v7/feat_v7_test.pkl")
    folds_gkf = np.load("ghani/cache/v7/folds_gkf5.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    feats = get_v7_feature_cols(tr)
    cat_cols = [c for c in feats if str(tr[c].dtype) == "category"]

    tr_meta = tr[["customer_id_code"]].copy()
    te_meta = te[["customer_id_code"]].copy()

    w_adv, adv_auc, top_shift = build_adversarial_weights(tr, te, feats, cat_cols, folds_gkf)
    del tr, te
    gc.collect()

    u_oof, u_te, st_full, best_lam, folds_skf = build_dealiased_customer_glmm(tr_meta, te_meta, y, folds_gkf, w_adv)

    np.save("ghani/cache/v8/w_adv_train.npy", w_adv)
    np.save("ghani/cache/v8/u_cust_v8_oof.npy", u_oof)
    np.save("ghani/cache/v8/u_cust_v8_test.npy", u_te)
    np.save("ghani/cache/v8/folds_skf5.npy", folds_skf)
    np.save("ghani/models/v8/glmm_cust_v8_full.npy", st_full, allow_pickle=True)
    json.dump(
        {"adv_oof_auc": adv_auc, "top_shift_features": top_shift, "best_cust_lam": best_lam},
        open("ghani/models/v8/adv_meta.json", "w", encoding="utf-8"), indent=2,
    )
    log("DONE build_adv_and_cust_glmm_v8.py!")


if __name__ == "__main__":
    main()
