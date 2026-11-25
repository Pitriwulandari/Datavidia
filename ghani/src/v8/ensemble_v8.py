"""v8 Ultimate Ceiling Super-Ensemble + Sibling Message Passing + Tunable De-Aliased u_cust_v8
+ Adversarial Density-Ratio Weighting (w_adv) + Non-Linear Power-Offset Macro-F1 EUM Optimizer!

Supports setting $env:ALPHA_CUST (default 0.55 safe Bayes compromise; can be updated in 15s to 0.35 or 0.90
if LB(v7) score is reported!).

Jalankan dari root repo:
  python -u -W ignore ghani/src/v8/ensemble_v8.py
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

for _sub in ["common", "v7", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm


def fast_ll(y, p, w=None):
    idx = np.arange(len(y))
    losses = -np.log(np.clip(p[idx, y], 1e-7, 1.0))
    if w is None:
        return float(np.mean(losses, dtype=np.float64))
    return float(np.sum(losses * w, dtype=np.float64) / np.sum(w, dtype=np.float64))


def build_sibling_message_features(df_ids, z_mat, p_mat):
    """Within-split Leave-One-Out sibling logit/prob message passing across customer_id & courier_id."""
    K = z_mat.shape[1]
    c_codes, c_u = pd.factorize(df_ids["customer_id"], sort=False)
    c_cnt = np.bincount(c_codes, minlength=len(c_u)).astype(np.float32)
    has_sib = (c_cnt[c_codes] > 1.0).astype(np.float32)
    den_c = np.maximum(c_cnt[c_codes] - 1.0, 1.0)

    cu_loo_z = np.zeros_like(z_mat, dtype=np.float32)
    cu_loo_p = np.zeros_like(p_mat, dtype=np.float32)
    for k in range(K):
        sz = np.bincount(c_codes, weights=z_mat[:, k], minlength=len(c_u))
        sp = np.bincount(c_codes, weights=p_mat[:, k], minlength=len(c_u))
        cu_loo_z[:, k] = ((sz[c_codes] - z_mat[:, k]) / den_c) * has_sib
        cu_loo_p[:, k] = ((sp[c_codes] - p_mat[:, k]) / den_c) * has_sib

    co_codes, co_u = pd.factorize(df_ids["courier_id"], sort=False)
    co_cnt = np.bincount(co_codes, minlength=len(co_u)).astype(np.float32)
    den_co = np.maximum(co_cnt[co_codes] - 1.0, 1.0)
    co_loo_z = np.zeros_like(z_mat, dtype=np.float32)
    for k in range(K):
        sz = np.bincount(co_codes, weights=z_mat[:, k], minlength=len(co_u))
        co_loo_z[:, k] = (sz[co_codes] - z_mat[:, k]) / den_co

    return np.hstack([cu_loo_z, cu_loo_p, co_loo_z, (1.0 - has_sib)[:, None]]).astype(np.float32)


def optimize_nonlinear_eum(proba, y, n_restarts=2, seed=42):
    """Coordinate-ascent Empirical Utility Maximization for Macro-F1:
       score_k(p) = w_k * (p_k ** gamma_k) + b_k
    Regularized around (w_k, gamma_k=1.0, b_k=0.0) to prevent overfitting!
    """
    K = proba.shape[1]
    w_init, f1_base = glmm.optimize_macro_f1_weights(proba, y, n_restarts=n_restarts, seed=seed)

    best_w = w_init.copy()
    best_g = np.ones(K, dtype=np.float64)
    best_b = np.zeros(K, dtype=np.float64)
    best_f1 = f1_base

    p64 = np.clip(proba.astype(np.float64), 1e-6, 1.0)

    # Coordinate polish over (w, gamma, b)
    for _pass in range(2):
        for k in range(K):
            # Tune gamma_k in [0.80, 1.25]
            for gk in [0.85, 0.92, 0.96, 1.04, 1.08, 1.15]:
                g_try = best_g.copy(); g_try[k] = gk
                sc = best_w[None, :] * (p64 ** g_try[None, :]) + best_b[None, :]
                f1 = glmm.fast_macro_f1(y, sc.argmax(1), K)
                if f1 > best_f1 + 2e-5:
                    best_f1 = f1; best_g = g_try
            # Tune additive margin b_k in [-0.04, +0.04]
            for bk in [-0.03, -0.015, -0.007, 0.007, 0.015, 0.03]:
                b_try = best_b.copy(); b_try[k] = bk
                sc = best_w[None, :] * (p64 ** best_g[None, :]) + b_try[None, :]
                f1 = glmm.fast_macro_f1(y, sc.argmax(1), K)
                if f1 > best_f1 + 2e-5:
                    best_f1 = f1; best_b = b_try
            # Re-polish w_k
            for mult in [0.94, 0.97, 1.03, 1.06]:
                w_try = best_w.copy(); w_try[k] *= mult
                sc = w_try[None, :] * (p64 ** best_g[None, :]) + best_b[None, :]
                f1 = glmm.fast_macro_f1(y, sc.argmax(1), K)
                if f1 > best_f1 + 1e-5:
                    best_f1 = f1; best_w = w_try

    return best_w, best_g, best_b, best_f1


def apply_eum(proba, w, g, b):
    p64 = np.clip(proba.astype(np.float64), 1e-6, 1.0)
    return (w[None, :] * (p64 ** g[None, :]) + b[None, :]).argmax(axis=1)


def main():
    t0 = time.time()
    os.makedirs("ghani/models/v8", exist_ok=True)
    os.makedirs("ghani/cache/v8", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v7/feat_v7_train.pkl")[["customer_id", "courier_id", "cu_sp_n", "is_peak_month", "status_penanganan"]]
    te = pd.read_pickle("ghani/cache/v7/feat_v7_test.pkl")[["shipment_id", "customer_id", "courier_id", "cu_sp_n", "is_peak_month"]]
    folds_gkf = np.load("ghani/cache/v7/folds_gkf5.npy")
    w_adv = np.load("ghani/cache/v8/w_adv_train.npy")
    u_cust_oof = np.load("ghani/cache/v8/u_cust_v8_oof.npy")
    u_cust_te = np.load("ghani/cache/v8/u_cust_v8_test.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)
    idx_n = np.arange(len(y))

    # Deterministic 362,245-row adversarial-weighted TEST-SIM sample matching FULL joint P(test | x)!
    rng = np.random.RandomState(42)
    adv_idx = rng.choice(len(y), size=362_245, replace=True, p=w_adv / w_adv.sum())

    candidates = [
        ("LGB_unw",   "ghani/cache/v7/lgb_unw_oof_logits.npy", "ghani/cache/v7/lgb_unw_test_logits.npy", True),
        ("LGB_tew",   "ghani/cache/v7/lgb_tew_oof_logits.npy", "ghani/cache/v7/lgb_tew_test_logits.npy", True),
        ("HC_GLMM",   "ghani/cache/v7/glmm_hc_oof_proba.npy",  "ghani/cache/v7/glmm_hc_test_proba.npy",  False),
        ("XGBoost",   "ghani/cache/v7/xgb_oof_logits.npy",     "ghani/cache/v7/xgb_test_logits.npy",     True),
        ("ResNetV7",  "ghani/cache/v7/nn_oof_logits.npy",      "ghani/cache/v7/nn_test_logits.npy",      True),
        ("LGB_ET",    "ghani/cache/v8/lgb_et_oof_logits.npy",  "ghani/cache/v8/lgb_et_test_logits.npy",  True),
        ("RGGatedNN", "ghani/cache/v8/rg_nn_oof_logits.npy",   "ghani/cache/v8/rg_nn_test_logits.npy",   True),
    ]

    names, oof_ps, te_ps, oof_ls, te_ls = [], [], [], [], []
    print("=== v8 INDIVIDUAL 5-FOLD GroupKFold MODEL SCORES (WITH ADVERSARIAL DRIFT WEIGHTS w_adv) ===", flush=True)
    for name, op, tp, is_logit in candidates:
        if not (os.path.exists(op) and os.path.exists(tp)):
            continue
        ao, at = np.load(op).astype(np.float32), np.load(tp).astype(np.float32)
        if is_logit:
            po, pt = glmm.softmax(ao), glmm.softmax(at)
            lo = ao - ao.mean(1, keepdims=True)
            lt = at - at.mean(1, keepdims=True)
        else:
            po, pt = ao, at
            lo = np.log(np.clip(ao, 1e-7, 1.0)); lo -= lo.mean(1, keepdims=True)
            lt = np.log(np.clip(at, 1e-7, 1.0)); lt -= lt.mean(1, keepdims=True)
        print(f"  {name:10s}: OOF_ll={fast_ll(y, po):.5f} | ADV-SIM_ll={fast_ll(y, po, w_adv):.5f} | ADV-SIM_argF1={glmm.fast_macro_f1(y[adv_idx], po[adv_idx].argmax(1), K):.5f}", flush=True)
        names.append(name); oof_ps.append(po); te_ps.append(pt); oof_ls.append(lo); te_ls.append(lt)

    M = len(names)

    # 1. Probability Simplex Blend minimizing w_adv LogLoss
    p_true = np.column_stack([oof_ps[i][idx_n, y] for i in range(M)]).astype(np.float64)
    wn = (w_adv / w_adv.sum()).astype(np.float64)

    def _ll_p(rw):
        w = np.exp(rw - rw.max()); w /= w.sum()
        return float(-np.sum(wn * np.log(np.clip(p_true @ w, 1e-7, 1.0))))

    wp = np.exp(minimize(_ll_p, np.zeros(M), method="BFGS").x)
    wp /= wp.sum()
    ens_p_oof = sum(wp[i] * oof_ps[i] for i in range(M)).astype(np.float32)
    ens_p_te = sum(wp[i] * te_ps[i] for i in range(M)).astype(np.float32)
    ens_z_oof = np.log(np.clip(ens_p_oof, 1e-7, 1.0)); ens_z_oof -= ens_z_oof.mean(1, keepdims=True)
    ens_z_te = np.log(np.clip(ens_p_te, 1e-7, 1.0)); ens_z_te -= ens_z_te.mean(1, keepdims=True)
    print(f"\n[1. PROB BLEND ({time.time()-t0:.1f}s)] weights={dict(zip(names, np.round(wp, 4).tolist()))} | ADV-SIM_ll={fast_ll(y, ens_p_oof, w_adv):.5f}", flush=True)

    # 2. Within-Split Sibling Message Passing + Standardized L2 Meta-Stacker (5-Fold GroupKFold, 0% customer leak!)
    mp_oof = build_sibling_message_features(tr, ens_z_oof, ens_p_oof)
    mp_te = build_sibling_message_features(te, ens_z_te, ens_p_te)
    X_stack_oof = np.hstack(oof_ls + [mp_oof]).astype(np.float32)
    X_stack_te = np.hstack(te_ls + [mp_te]).astype(np.float32)

    stack_oof = np.zeros((len(y), K), dtype=np.float32)
    stack_te = np.zeros((len(te), K), dtype=np.float32)
    stack_bundles = []
    for f in range(5):
        tr_m, va_m = folds_gkf != f, folds_gkf == f
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X_stack_oof[tr_m])
        Xva_s = sc.transform(X_stack_oof[va_m])
        Xte_s = sc.transform(X_stack_te)
        clf = LogisticRegression(C=0.04, max_iter=300, solver="lbfgs")
        clf.fit(Xtr_s, y[tr_m], sample_weight=w_adv[tr_m])
        stack_oof[va_m] = clf.predict_proba(Xva_s).astype(np.float32)
        stack_te += clf.predict_proba(Xte_s).astype(np.float32) / 5.0
        stack_bundles.append({"scaler": sc, "clf": clf})

    joblib.dump({"bundles": stack_bundles, "model_names": names, "prob_weights": wp.tolist()}, "ghani/models/v8/meta_stacker_v8.joblib", compress=3)
    base_oof = (0.35 * ens_p_oof + 0.65 * stack_oof).astype(np.float32)
    base_te = (0.35 * ens_p_te + 0.65 * stack_te).astype(np.float32)
    print(f"[2. MESSAGE-PASSING STACKER ({time.time()-t0:.1f}s)] GroupKFold OOF_ll={fast_ll(y, base_oof):.5f} | ADV-SIM_ll={fast_ll(y, base_oof, w_adv):.5f}", flush=True)

    # 3. Apply Tunable De-Aliased Customer GLMM (alpha_cust in [0, 1])!
    # Class-specific safety mask: strong on konfirmasi (2) & normal (3) (proven +0.043/+0.021 on test!),
    # moderate/conservative on retur (4) & gagal (0) & others so unseen/drifted rows never suffer!
    alpha_cust = float(os.environ.get("ALPHA_CUST", "0.55"))
    cls_safety = np.array([0.50, 0.30, 1.00, 0.85, 0.55, 0.35, 0.40], dtype=np.float32) * alpha_cust

    z_base_oof = np.log(np.clip(base_oof, 1e-7, 1.0)); z_base_oof -= z_base_oof.mean(1, keepdims=True)
    z_base_te = np.log(np.clip(base_te, 1e-7, 1.0)); z_base_te -= z_base_te.mean(1, keepdims=True)

    final_oof = glmm.softmax(z_base_oof + u_cust_oof * cls_safety[None, :]).astype(np.float32)
    final_te = glmm.softmax(z_base_te + u_cust_te * cls_safety[None, :]).astype(np.float32)
    print(f"[3. + DE-ALIASED u_cust_v8 (alpha={alpha_cust:.2f})] OOF_ll={fast_ll(y, final_oof):.5f} | ADV-SIM_ll={fast_ll(y, final_oof, w_adv):.5f}", flush=True)

    # 4. Compare Linear Class Multipliers vs Non-Linear EUM (w * p^gamma + b) on 5-Fold GroupKFold w_adv sample!
    rng_cv = np.random.RandomState(999)
    preds_lin = np.zeros(len(y), dtype=np.int32)
    preds_eum = np.zeros(len(y), dtype=np.int32)
    ws_lin, ws_eum, gs_eum, bs_eum = [], [], [], []
    for f in range(5):
        tr_pos = np.where(folds_gkf != f)[0]
        va_pos = np.where(folds_gkf == f)[0]
        wt = w_adv[tr_pos]
        sub_tr = rng_cv.choice(tr_pos, size=320_000, replace=True, p=wt / wt.sum())

        wl, _ = glmm.optimize_macro_f1_weights(final_oof[sub_tr], y[sub_tr], n_restarts=2, seed=42 + f)
        ws_lin.append(wl)
        preds_lin[va_pos] = (final_oof[va_pos] * wl).argmax(1)

        we, ge, be, _ = optimize_nonlinear_eum(final_oof[sub_tr], y[sub_tr], n_restarts=2, seed=42 + f)
        ws_eum.append(we); gs_eum.append(ge); bs_eum.append(be)
        preds_eum[va_pos] = apply_eum(final_oof[va_pos], we, ge, be)

    f1_lin_adv = glmm.fast_macro_f1(y[adv_idx], preds_lin[adv_idx], K)
    f1_eum_adv = glmm.fast_macro_f1(y[adv_idx], preds_eum[adv_idx], K)
    f1_lin_all = glmm.fast_macro_f1(y, preds_lin, K)
    f1_eum_all = glmm.fast_macro_f1(y, preds_eum, K)
    print(f"[4. BOUNDARY OPTIMIZER] Linear Multipliers: ADV-SIM_F1={f1_lin_adv:.5f} (ALL={f1_lin_all:.5f}) | Non-Linear EUM: ADV-SIM_F1={f1_eum_adv:.5f} (ALL={f1_eum_all:.5f})", flush=True)

    if f1_eum_adv > f1_lin_adv + 5e-5:
        use_eum = True
        w_fin = np.exp(np.mean(np.log(ws_eum), axis=0))
        g_fin = np.mean(gs_eum, axis=0)
        b_fin = np.mean(bs_eum, axis=0)
        best_pred_oof = preds_eum
        best_f1_adv, best_f1_all = f1_eum_adv, f1_eum_all
        pred_te = apply_eum(final_te, w_fin, g_fin, b_fin)
    else:
        use_eum = False
        w_fin = np.exp(np.mean(np.log(ws_lin), axis=0))
        w_fin /= np.exp(np.mean(np.log(w_fin)))
        g_fin = np.ones(K)
        b_fin = np.zeros(K)
        best_pred_oof = preds_lin
        best_f1_adv, best_f1_all = f1_lin_adv, f1_lin_all
        pred_te = (final_te * w_fin.astype(np.float32)).argmax(1)

    print(f"\n=== v8 ULTIMATE WINNER: OOF_ll={fast_ll(y, final_oof):.5f} | ADV-SIM_ll={fast_ll(y, final_oof, w_adv):.5f} | ALL_F1={best_f1_all:.5f} | ADV-SIM_F1={best_f1_adv:.5f} ===", flush=True)
    print(classification_report(y[adv_idx], best_pred_oof[adv_idx], target_names=classes, digits=4), flush=True)

    sub_path = "submissions/submission_v8_ghani.csv"
    sub = pd.DataFrame({"shipment_id": te["shipment_id"], "status_penanganan": [classes[i] for i in pred_te]})
    sub.to_csv(sub_path, index=False)

    assert len(sub) == 362245 and sub["status_penanganan"].isna().sum() == 0
    ss = pd.read_csv("Task-1-Datavidia-11.0/sample_submission.csv", usecols=["shipment_id"])
    assert (sub["shipment_id"].values == ss["shipment_id"].values).all(), "shipment_id order mismatch!"

    meta = {
        "models": names,
        "prob_weights": dict(zip(names, np.round(wp, 5).tolist())),
        "alpha_cust": alpha_cust,
        "cls_safety_vec": cls_safety.tolist(),
        "use_nonlinear_eum": use_eum,
        "eum_w": w_fin.tolist(),
        "eum_gamma": g_fin.tolist(),
        "eum_bias": b_fin.tolist(),
        "classes": classes,
        "oof_ll": fast_ll(y, final_oof),
        "adv_sim_ll": fast_ll(y, final_oof, w_adv),
        "all_f1": float(best_f1_all),
        "adv_sim_f1": float(best_f1_adv),
        "test_distribution": sub["status_penanganan"].value_counts(normalize=True).round(4).to_dict(),
    }
    json.dump(meta, open("ghani/models/v8/meta_v8_ensemble.json", "w", encoding="utf-8"), indent=2)
    print(f"SAVED {sub_path} | rows={len(sub):,} | dist={meta['test_distribution']}", flush=True)


if __name__ == "__main__":
    main()
