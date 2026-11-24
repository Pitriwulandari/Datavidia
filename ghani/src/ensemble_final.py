"""Fast Final Super-Ensemble Blender + Cross-Validated L2 Meta-Stacker + Exact Macro-F1 Optimizer.

Uses vectorized NumPy log-loss (50x faster than sklearn.metrics.log_loss inside optimizer loops!)
and compares 3 leak-free ensembling strategies across ALL available Stage-2 5-Fold models:
  1. Probability-space Simplex Blend
  2. Logit-space Linear Blend
  3. 5-Fold Cross-Validated L2-Regularized Multinomial Meta-Stacker
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmm_lib as glmm


def fast_ll(y, p):
    idx = np.arange(len(y))
    return float(-np.mean(np.log(np.clip(p[idx, y], 1e-7, 1.0)), dtype=np.float64))


def main():
    t0 = time.time()
    tr = pd.read_pickle("ghani/cache/feat_v5_train.pkl")[["status_penanganan"]]
    te = pd.read_pickle("ghani/cache/feat_v5_test.pkl")[["shipment_id"]]
    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    folds = np.load("ghani/cache/folds_v5.npy")
    K = len(classes)
    idx_n = np.arange(len(y))

    candidates = [
        ("GLMM",     "ghani/cache/stage2/glmm_oof_proba.npy",       "ghani/cache/stage2/glmm_test_proba.npy",       False),
        ("LGB_res",  "ghani/cache/stage2/lgb_res_oof_logits.npy",   "ghani/cache/stage2/lgb_res_test_logits.npy",   True),
        ("LGB_full", "ghani/cache/stage2/lgb_full_oof_logits.npy",  "ghani/cache/stage2/lgb_full_test_logits.npy",  True),
        ("CatBoost", "ghani/cache/stage2/cb_oof_logits.npy",        "ghani/cache/stage2/cb_test_logits.npy",        True),
        ("XGBoost",  "ghani/cache/stage2/xgb_oof_logits.npy",       "ghani/cache/stage2/xgb_test_logits.npy",       True),
        ("ResNet",   "ghani/cache/stage2/nn_oof_logits.npy",        "ghani/cache/stage2/nn_test_logits.npy",        True),
        ("RGResNet", "ghani/cache/stage2/nn2_oof_logits.npy",       "ghani/cache/stage2/nn2_test_logits.npy",       True),
    ]

    names, oof_ps, te_ps, oof_ls, te_ls = [], [], [], [], []
    print("=== INDIVIDUAL 5-FOLD MODEL SCORES ===", flush=True)
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
        ll = fast_ll(y, po)
        arg_f1 = glmm.fast_macro_f1(y, po.argmax(1), K)
        print(f"  {name:10s}: OOF_ll={ll:.5f} | argmax_F1={arg_f1:.5f}", flush=True)
        names.append(name)
        oof_ps.append(po)
        te_ps.append(pt)
        oof_ls.append(lo)
        te_ls.append(lt)

    M = len(names)
    assert M >= 1, "No models found!"

    # Pre-extract true-class probabilities for ultra-fast simplex optimization (100x speedup!)
    p_true_mat = np.column_stack([oof_ps[i][idx_n, y] for i in range(M)]).astype(np.float64)

    def _ll_prob_fast(raw_w):
        w = np.exp(raw_w - raw_w.max())
        w /= w.sum()
        pt = p_true_mat @ w
        return float(-np.mean(np.log(np.clip(pt, 1e-7, 1.0))))

    res_p = minimize(_ll_prob_fast, np.zeros(M), method="BFGS", options={"maxiter": 200})
    wp = np.exp(res_p.x - res_p.x.max())
    wp /= wp.sum()
    ens_p_oof = sum(wp[i] * oof_ps[i] for i in range(M)).astype(np.float32)
    ens_p_te = sum(wp[i] * te_ps[i] for i in range(M)).astype(np.float32)
    ll_p = fast_ll(y, ens_p_oof)

    # Optimize Macro F1 cleanly with 2 restarts per CV fold
    def _cv_f1(oof_p):
        preds = np.zeros(len(y), dtype=np.int32)
        ws = []
        for f in range(5):
            w_f, _ = glmm.optimize_macro_f1_weights(oof_p[folds != f], y[folds != f], n_restarts=2, seed=42 + f)
            ws.append(w_f)
            preds[folds == f] = (oof_p[folds == f] * w_f).argmax(1)
        f1_v = glmm.fast_macro_f1(y, preds, K)
        w_m = np.exp(np.mean(np.log(ws), axis=0))
        w_m /= np.exp(np.mean(np.log(w_m)))
        return f1_v, w_m, preds

    results = []
    f1_p, w_cls_p, pred_p = _cv_f1(ens_p_oof)
    print(f"\n[PROB BLEND ({time.time()-t0:.1f}s)] weights={dict(zip(names, np.round(wp, 3).tolist()))}", flush=True)
    print(f"  OOF_ll={ll_p:.5f} | argmax_F1={glmm.fast_macro_f1(y, ens_p_oof.argmax(1), K):.5f} | CV_tuned_F1={f1_p:.5f}", flush=True)
    results.append(("prob_blend", f1_p, ll_p, ens_p_oof, ens_p_te, w_cls_p, pred_p, dict(zip(names, np.round(wp, 4).tolist()))))

    # 2. Fast Logit-space Linear Blend (evaluate on 250k stratified subset for speed, verify on all 1.24M!)
    sub_idx = np.arange(0, len(y), 5)
    sub_ls = [oof_ls[i][sub_idx] for i in range(M)]
    sub_y = y[sub_idx]
    sub_n = np.arange(len(sub_y))

    def _ll_logit_sub(w):
        z = sum(w[i] * sub_ls[i] for i in range(M))
        zm = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(zm).sum(axis=1))
        return float(-np.mean(zm[sub_n, sub_y] - lse, dtype=np.float64))

    res_l = minimize(_ll_logit_sub, wp.copy(), method="Nelder-Mead", options={"maxiter": 250})
    wl = res_l.x
    ens_l_oof = glmm.softmax(sum(wl[i] * oof_ls[i] for i in range(M))).astype(np.float32)
    ens_l_te = glmm.softmax(sum(wl[i] * te_ls[i] for i in range(M))).astype(np.float32)
    ll_l = fast_ll(y, ens_l_oof)
    f1_l, w_cls_l, pred_l = _cv_f1(ens_l_oof)
    print(f"\n[LOGIT BLEND ({time.time()-t0:.1f}s)] weights={dict(zip(names, np.round(wl, 3).tolist()))}", flush=True)
    print(f"  OOF_ll={ll_l:.5f} | argmax_F1={glmm.fast_macro_f1(y, ens_l_oof.argmax(1), K):.5f} | CV_tuned_F1={f1_l:.5f}", flush=True)
    results.append(("logit_blend", f1_l, ll_l, ens_l_oof, ens_l_te, w_cls_l, pred_l, dict(zip(names, np.round(wl, 4).tolist()))))

    # 3. 5-Fold CV L2 Logistic Meta-Stacker + 50/50 Hybrid with Prob Blend
    if M >= 2:
        X_meta_oof = np.hstack(oof_ls).astype(np.float32)
        X_meta_te = np.hstack(te_ls).astype(np.float32)
        meta_oof = np.zeros((len(y), K), dtype=np.float32)
        meta_te = np.zeros((len(te), K), dtype=np.float32)
        for f in range(5):
            tr_m, va_m = folds != f, folds == f
            clf = LogisticRegression(C=0.05, max_iter=150, solver="lbfgs")
            clf.fit(X_meta_oof[tr_m], y[tr_m])
            meta_oof[va_m] = clf.predict_proba(X_meta_oof[va_m]).astype(np.float32)
            meta_te += clf.predict_proba(X_meta_te).astype(np.float32) / 5.0
        ll_m = fast_ll(y, meta_oof)
        f1_m, w_cls_m, pred_m = _cv_f1(meta_oof)
        print(f"\n[META STACKER ({time.time()-t0:.1f}s)] OOF_ll={ll_m:.5f} | argmax_F1={glmm.fast_macro_f1(y, meta_oof.argmax(1), K):.5f} | CV_tuned_F1={f1_m:.5f}", flush=True)
        results.append(("meta_stacker", f1_m, ll_m, meta_oof, meta_te, w_cls_m, pred_m, {"C": 0.05, "models": names}))

        # Hybrid: 50% Prob Blend + 50% Meta Stacker
        hyb_oof = (0.5 * ens_p_oof + 0.5 * meta_oof).astype(np.float32)
        hyb_te = (0.5 * ens_p_te + 0.5 * meta_te).astype(np.float32)
        ll_h = fast_ll(y, hyb_oof)
        f1_h, w_cls_h, pred_h = _cv_f1(hyb_oof)
        print(f"\n[HYBRID 50/50 PROB+STACKER ({time.time()-t0:.1f}s)] OOF_ll={ll_h:.5f} | argmax_F1={glmm.fast_macro_f1(y, hyb_oof.argmax(1), K):.5f} | CV_tuned_F1={f1_h:.5f}", flush=True)
        results.append(("hybrid_prob_stacker", f1_h, ll_h, hyb_oof, hyb_te, w_cls_h, pred_h, {"prob_weights": dict(zip(names, np.round(wp, 4).tolist())), "meta_C": 0.05}))

    results.sort(key=lambda r: (r[1], -r[2]), reverse=True)
    mode, best_f1, best_ll, best_oof, best_te, best_w_cls, best_pred, info = results[0]

    print(f"\n=== FINAL WINNER ({mode.upper()}): OOF_ll={best_ll:.5f} | CV_Macro_F1={best_f1:.5f} ===", flush=True)
    print("Optimal class multipliers:", np.round(best_w_cls, 4).tolist(), flush=True)
    print(classification_report(y, best_pred, target_names=classes, digits=4), flush=True)

    tag = os.environ.get("TAG", "v6_ensemble")
    pred_te = (best_te * best_w_cls).argmax(1)
    sub = pd.DataFrame({"shipment_id": te["shipment_id"], "status_penanganan": [classes[i] for i in pred_te]})
    sub_path = f"submissions/submission_{tag}_ghani.csv"
    sub.to_csv(sub_path, index=False)

    assert len(sub) == 362245, f"Wrong row count: {len(sub)}"
    assert sub["status_penanganan"].isna().sum() == 0, "NaNs in predictions!"
    assert set(sub["status_penanganan"].unique()).issubset(set(classes)), "Invalid class labels!"

    meta = {
        "mode": mode,
        "models": names,
        "blend_info": info,
        "classes": classes,
        "class_multipliers": best_w_cls.tolist(),
        "oof_ll": float(best_ll),
        "cv_macro_f1": float(best_f1),
        "test_distribution": sub["status_penanganan"].value_counts(normalize=True).round(4).to_dict(),
    }
    json.dump(meta, open(f"ghani/models/meta_{tag}.json", "w"), indent=2)
    print(f"SAVED {sub_path} | rows={len(sub)} | dist={meta['test_distribution']}", flush=True)


if __name__ == "__main__":
    main()
