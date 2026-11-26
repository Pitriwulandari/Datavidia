"""v7 Final Super-Ensemble Blender + GroupKFold L2 Meta-Stacker + Test-Regime Macro-F1 Optimizer.

Evaluates ALL available v7 5-Fold GroupKFold(customer_id) models (ZERO customer leakage!)
using exact TEST-SIM importance weighting matching test's (n_cust_in_split, is_peak_month) distribution.
Also saves meta_stacker_v7.joblib & meta_v7_ensemble.json for 100% zero-retraining inference!

Jalankan dari root repo:
  python -u -W ignore ghani/task1/src/ensemble_v7.py
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

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm
from train_v7_lgb import compute_test_sim_weights


def fast_ll(y, p, w=None):
    idx = np.arange(len(y))
    losses = -np.log(np.clip(p[idx, y], 1e-7, 1.0))
    if w is None:
        return float(np.mean(losses, dtype=np.float64))
    return float(np.sum(losses * w, dtype=np.float64) / np.sum(w, dtype=np.float64))


def main():
    t0 = time.time()
    tr = pd.read_pickle("ghani/task1/cache/v7/feat_v7_train.pkl")[["status_penanganan", "cu_sp_n", "is_peak_month"]]
    te = pd.read_pickle("ghani/task1/cache/v7/feat_v7_test.pkl")[["shipment_id", "cu_sp_n", "is_peak_month"]]
    folds = np.load("ghani/task1/cache/v7/folds_gkf5.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)
    idx_n = np.arange(len(y))

    n_tr = tr["cu_sp_n"].to_numpy(dtype=np.float32)
    n_te = te["cu_sp_n"].to_numpy(dtype=np.float32)
    sim_w, _ = compute_test_sim_weights(n_tr, n_te, tr["is_peak_month"].to_numpy(), te["is_peak_month"].to_numpy())

    # Deterministic 362,245-row TEST-SIM evaluation index matching exact test distribution
    rng_eval = np.random.RandomState(42)
    sim_idx = rng_eval.choice(len(y), size=362_245, replace=True, p=sim_w / sim_w.sum())

    candidates = [
        ("LGB_unw",  "ghani/task1/cache/v7/lgb_unw_oof_logits.npy",  "ghani/task1/cache/v7/lgb_unw_test_logits.npy",  True),
        ("LGB_tew",  "ghani/task1/cache/v7/lgb_tew_oof_logits.npy",  "ghani/task1/cache/v7/lgb_tew_test_logits.npy",  True),
        ("HC_GLMM",  "ghani/task1/cache/v7/glmm_hc_oof_proba.npy",   "ghani/task1/cache/v7/glmm_hc_test_proba.npy",   False),
        ("XGBoost",  "ghani/task1/cache/v7/xgb_oof_logits.npy",      "ghani/task1/cache/v7/xgb_test_logits.npy",      True),
        ("ResNetV7", "ghani/task1/cache/v7/nn_oof_logits.npy",       "ghani/task1/cache/v7/nn_test_logits.npy",       True),
        ("RGResV7",  "ghani/task1/cache/v7/nn2_oof_logits.npy",      "ghani/task1/cache/v7/nn2_test_logits.npy",      True),
    ]

    names, oof_ps, te_ps, oof_ls, te_ls = [], [], [], [], []
    print("=== INDIVIDUAL v7 5-FOLD GroupKFold(customer_id) MODEL SCORES ===", flush=True)
    for name, op, tp, is_logit in candidates:
        if not (os.path.exists(op) and os.path.exists(tp)):
            continue
        ao = np.load(op).astype(np.float32)
        at = np.load(tp).astype(np.float32)
        if is_logit:
            po, pt = glmm.softmax(ao), glmm.softmax(at)
            lo = ao - ao.mean(axis=1, keepdims=True)
            lt = at - at.mean(axis=1, keepdims=True)
        else:
            po, pt = ao, at
            lo = np.log(np.clip(ao, 1e-7, 1.0))
            lo -= lo.mean(axis=1, keepdims=True)
            lt = np.log(np.clip(at, 1e-7, 1.0))
            lt -= lt.mean(axis=1, keepdims=True)
        ll_all = fast_ll(y, po)
        ll_sim = fast_ll(y, po, sim_w)
        f1_sim = glmm.fast_macro_f1(y[sim_idx], po[sim_idx].argmax(1), K)
        print(f"  {name:10s}: OOF_ll={ll_all:.5f} | TEST-SIM_ll={ll_sim:.5f} | TEST-SIM_argmaxF1={f1_sim:.5f}", flush=True)
        names.append(name)
        oof_ps.append(po)
        te_ps.append(pt)
        oof_ls.append(lo)
        te_ls.append(lt)

    M = len(names)
    assert M >= 1, "No v7 models found!"

    # 1. Probability Simplex Blend minimizing exact TEST-SIM LogLoss!
    p_true_mat = np.column_stack([oof_ps[i][idx_n, y] for i in range(M)]).astype(np.float64)
    w_norm = (sim_w / sim_w.sum()).astype(np.float64)

    def _ll_prob_sim(raw_w):
        w = np.exp(raw_w - raw_w.max())
        w /= w.sum()
        pt = p_true_mat @ w
        return float(-np.sum(w_norm * np.log(np.clip(pt, 1e-7, 1.0))))

    res_p = minimize(_ll_prob_sim, np.zeros(M), method="BFGS", options={"maxiter": 200})
    wp = np.exp(res_p.x - res_p.x.max())
    wp /= wp.sum()
    ens_p_oof = sum(wp[i] * oof_ps[i] for i in range(M)).astype(np.float32)
    ens_p_te = sum(wp[i] * te_ps[i] for i in range(M)).astype(np.float32)
    ll_p_sim = fast_ll(y, ens_p_oof, sim_w)

    def _cv_f1_sim(oof_p, split_n1=False):
        rng_o = np.random.RandomState(123)
        preds = np.zeros(len(y), dtype=np.int32)
        ws_a, ws_b = [], []
        for f in range(5):
            tr_pos = np.where(folds != f)[0]
            va_pos = np.where(folds == f)[0]
            if not split_n1:
                w_tr = sim_w[tr_pos]
                sub_tr = rng_o.choice(tr_pos, size=300_000, replace=True, p=w_tr / w_tr.sum())
                wf, _ = glmm.optimize_macro_f1_weights(oof_p[sub_tr], y[sub_tr], n_restarts=2, seed=42 + f)
                ws_a.append(wf)
                preds[va_pos] = (oof_p[va_pos] * wf).argmax(1)
            else:
                # Separate class multipliers for n==1 vs n>=2 (strictly out-of-fold!)
                m1_tr = tr_pos[n_tr[tr_pos] == 1]
                m2_tr = tr_pos[n_tr[tr_pos] >= 2]
                w1, _ = glmm.optimize_macro_f1_weights(oof_p[m1_tr], y[m1_tr], n_restarts=2, seed=42 + f)
                w2_tr = sim_w[m2_tr]
                sub2 = rng_o.choice(m2_tr, size=250_000, replace=True, p=w2_tr / w2_tr.sum())
                w2, _ = glmm.optimize_macro_f1_weights(oof_p[sub2], y[sub2], n_restarts=2, seed=142 + f)
                ws_a.append(w1)
                ws_b.append(w2)
                va1 = va_pos[n_tr[va_pos] == 1]
                va2 = va_pos[n_tr[va_pos] >= 2]
                preds[va1] = (oof_p[va1] * w1).argmax(1)
                preds[va2] = (oof_p[va2] * w2).argmax(1)

        f1_sim = glmm.fast_macro_f1(y[sim_idx], preds[sim_idx], K)
        f1_all = glmm.fast_macro_f1(y, preds, K)
        wm_a = np.exp(np.mean(np.log(ws_a), axis=0))
        wm_a /= np.exp(np.mean(np.log(wm_a)))
        if not split_n1:
            return f1_sim, f1_all, {"global": wm_a.tolist()}, preds
        wm_b = np.exp(np.mean(np.log(ws_b), axis=0))
        wm_b /= np.exp(np.mean(np.log(wm_b)))
        return f1_sim, f1_all, {"n1": wm_a.tolist(), "n2p": wm_b.tolist()}, preds

    results = []
    f1_ps, f1_pa, w_p, pred_p = _cv_f1_sim(ens_p_oof, split_n1=False)
    print(f"\n[PROB BLEND ({time.time()-t0:.1f}s)] weights={dict(zip(names, np.round(wp, 3).tolist()))}", flush=True)
    print(f"  TEST-SIM_ll={ll_p_sim:.5f} | CV_F1={f1_pa:.5f} | TEST-SIM_F1={f1_ps:.5f}", flush=True)
    results.append(("prob_blend", f1_ps, ll_p_sim, ens_p_oof, ens_p_te, w_p, pred_p, dict(zip(names, np.round(wp, 4).tolist()))))

    # 2. 5-Fold GroupKFold L2 Logistic Meta-Stacker (with test-sim sample weights!) + 50/50 Hybrid
    stack_models = []
    if M >= 2:
        X_meta_oof = np.hstack(oof_ls).astype(np.float32)
        X_meta_te = np.hstack(te_ls).astype(np.float32)
        meta_oof = np.zeros((len(y), K), dtype=np.float32)
        meta_te = np.zeros((len(te), K), dtype=np.float32)
        for f in range(5):
            tr_m, va_m = folds != f, folds == f
            clf = LogisticRegression(C=0.05, max_iter=150, solver="lbfgs")
            clf.fit(X_meta_oof[tr_m], y[tr_m], sample_weight=sim_w[tr_m])
            meta_oof[va_m] = clf.predict_proba(X_meta_oof[va_m]).astype(np.float32)
            meta_te += clf.predict_proba(X_meta_te).astype(np.float32) / 5.0
            stack_models.append(clf)

        joblib.dump({"models": stack_models, "model_names": names, "C": 0.05}, "ghani/task1/models/v7/meta_stacker_v7.joblib", compress=3)
        ll_ms = fast_ll(y, meta_oof, sim_w)
        f1_ms, f1_ma, w_m, pred_m = _cv_f1_sim(meta_oof, split_n1=False)
        print(f"\n[META STACKER ({time.time()-t0:.1f}s)] TEST-SIM_ll={ll_ms:.5f} | CV_F1={f1_ma:.5f} | TEST-SIM_F1={f1_ms:.5f}", flush=True)
        results.append(("meta_stacker", f1_ms, ll_ms, meta_oof, meta_te, w_m, pred_m, {"C": 0.05, "models": names}))

        hyb_oof = (0.5 * ens_p_oof + 0.5 * meta_oof).astype(np.float32)
        hyb_te = (0.5 * ens_p_te + 0.5 * meta_te).astype(np.float32)
        ll_hs = fast_ll(y, hyb_oof, sim_w)
        f1_hs, f1_ha, w_h, pred_h = _cv_f1_sim(hyb_oof, split_n1=False)
        print(f"\n[HYBRID 50/50 GLOBAL-W ({time.time()-t0:.1f}s)] TEST-SIM_ll={ll_hs:.5f} | CV_F1={f1_ha:.5f} | TEST-SIM_F1={f1_hs:.5f}", flush=True)
        results.append(("hybrid_global_w", f1_hs, ll_hs, hyb_oof, hyb_te, w_h, pred_h, {"prob_weights": dict(zip(names, np.round(wp, 4).tolist())), "meta_C": 0.05}))

        # Also test regime-specific (n==1 vs n>=2) class multipliers on Hybrid!
        f1_h2s, f1_h2a, w_h2, pred_h2 = _cv_f1_sim(hyb_oof, split_n1=True)
        print(f"[HYBRID 50/50 SPLIT-N1-W ({time.time()-t0:.1f}s)] TEST-SIM_ll={ll_hs:.5f} | CV_F1={f1_h2a:.5f} | TEST-SIM_F1={f1_h2s:.5f}", flush=True)
        results.append(("hybrid_split_n1_w", f1_h2s, ll_hs, hyb_oof, hyb_te, w_h2, pred_h2, {"prob_weights": dict(zip(names, np.round(wp, 4).tolist())), "meta_C": 0.05}))

    results.sort(key=lambda r: (r[1], -r[2]), reverse=True)
    mode, best_f1_sim, best_ll_sim, best_oof, best_te, best_w, best_pred, info = results[0]
    best_ll_all = fast_ll(y, best_oof)
    best_f1_all = glmm.fast_macro_f1(y, best_pred, K)

    print(f"\n=== v7 FINAL WINNER ({mode.upper()}): TEST-SIM_ll={best_ll_sim:.5f} | TEST-SIM_F1={best_f1_sim:.5f} | ALL_OOF_ll={best_ll_all:.5f} | ALL_CV_F1={best_f1_all:.5f} ===", flush=True)
    for nv in [1, 2, 3]:
        print(f"  Breakdown n_cust={nv}: F1={glmm.fast_macro_f1(y[n_tr==nv], best_pred[n_tr==nv], K):.5f}", flush=True)
    print(f"  Breakdown n_cust>=4: F1={glmm.fast_macro_f1(y[n_tr>=4], best_pred[n_tr>=4], K):.5f}", flush=True)
    print(classification_report(y[sim_idx], best_pred[sim_idx], target_names=classes, digits=4), flush=True)

    # Apply winning multipliers to test
    if "global" in best_w:
        w_arr = np.array(best_w["global"], dtype=np.float32)
        pred_te = (best_te * w_arr).argmax(1)
    else:
        w1 = np.array(best_w["n1"], dtype=np.float32)
        w2 = np.array(best_w["n2p"], dtype=np.float32)
        pred_te = np.zeros(len(te), dtype=np.int32)
        m1 = n_te == 1
        pred_te[m1] = (best_te[m1] * w1).argmax(1)
        pred_te[~m1] = (best_te[~m1] * w2).argmax(1)

    tag = os.environ.get("TAG", "v7")
    sub = pd.DataFrame({"shipment_id": te["shipment_id"], "status_penanganan": [classes[i] for i in pred_te]})
    sub_path = f"submissions/task1/submission_{tag}_ghani.csv"
    sub.to_csv(sub_path, index=False)

    assert len(sub) == 362245, f"Wrong row count: {len(sub)}"
    assert sub["status_penanganan"].isna().sum() == 0, "NaNs found!"
    assert set(sub["status_penanganan"].unique()).issubset(set(classes)), "Invalid labels!"
    ss = pd.read_csv("Task-1-Datavidia-11.0/sample_submission.csv", usecols=["shipment_id"])
    assert (sub["shipment_id"].values == ss["shipment_id"].values).all(), "shipment_id order mismatch!"

    meta = {
        "mode": mode, "models": names, "blend_info": info, "classes": classes,
        "class_multipliers": best_w,
        "oof_ll": float(best_ll_all), "cv_macro_f1": float(best_f1_all),
        "test_sim_ll": float(best_ll_sim), "test_sim_f1": float(best_f1_sim),
        "test_distribution": sub["status_penanganan"].value_counts(normalize=True).round(4).to_dict(),
    }
    json.dump(meta, open("ghani/task1/models/v7/meta_v7_ensemble.json", "w"), indent=2)
    print(f"SAVED {sub_path} | rows={len(sub):,} | dist={meta['test_distribution']}", flush=True)


if __name__ == "__main__":
    main()
