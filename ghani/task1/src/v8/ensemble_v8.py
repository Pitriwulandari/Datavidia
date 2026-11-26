"""Final v8 Super-Ensemble Blender + Cross-Validated L2 Meta-Stacker + Exact Macro-F1 Optimizer.
Also exports `ghani/task1/models/v8/feature_spec_v8.json` and `meta_stacker_v8.joblib` so `infer_v8.py`
runs 100% zero-retraining inference with 0.000000 drift!

Jalankan setelah model-model Stage-2 v8 selesai:
  python -u -W ignore ghani/task1/src/v8/ensemble_v8.py
"""
import gc
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

for _sub in ["common", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
import v8_common as v8c


def fast_ll(y, p):
    idx = np.arange(len(y))
    return float(-np.mean(np.log(np.clip(p[idx, y], 1e-7, 1.0)), dtype=np.float64))


def main():
    t0 = time.time()
    tr = pd.read_pickle("ghani/task1/cache/v8/feat_v8_base_train.pkl")
    te = pd.read_pickle("ghani/task1/cache/v8/feat_v8_base_test.pkl")[["shipment_id", "customer_id"]]
    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    folds = np.load("ghani/task1/cache/v8/folds_v8.npy")
    K = len(classes)
    idx_n = np.arange(len(y))

    candidates = [
        ("GLMM",     "ghani/task1/cache/v8/glmm_oof_proba.npy",      "ghani/task1/cache/v8/glmm_test_proba.npy",      False),
        ("LGB_res",  "ghani/task1/cache/v8/lgb_res_oof_logits.npy",  "ghani/task1/cache/v8/lgb_res_test_logits.npy",  True),
        ("LGB_full", "ghani/task1/cache/v8/lgb_full_oof_logits.npy", "ghani/task1/cache/v8/lgb_full_test_logits.npy", True),
        ("LGB_et",   "ghani/task1/cache/v8/lgb_et_oof_logits.npy",   "ghani/task1/cache/v8/lgb_et_test_logits.npy",   True),
        ("XGBoost",  "ghani/task1/cache/v8/xgb_oof_logits.npy",      "ghani/task1/cache/v8/xgb_test_logits.npy",      True),
        ("ResNet",   "ghani/task1/cache/v8/nn_oof_logits.npy",       "ghani/task1/cache/v8/nn_test_logits.npy",       True),
        ("RGResNet", "ghani/task1/cache/v8/nn2_oof_logits.npy",      "ghani/task1/cache/v8/nn2_test_logits.npy",      True),
    ]

    names, oof_ps, te_ps, oof_ls, te_ls = [], [], [], [], []
    print("=== INDIVIDUAL 5-FOLD v8 MODEL SCORES ===", flush=True)
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
    assert M >= 1, "No v8 models found!"

    p_true_mat = np.column_stack([oof_ps[i][idx_n, y] for i in range(M)]).astype(np.float64)

    def _ll_prob_fast(raw_w):
        w = np.exp(raw_w - raw_w.max())
        w /= w.sum()
        return float(-np.mean(np.log(np.clip(p_true_mat @ w, 1e-7, 1.0))))

    res_p = minimize(_ll_prob_fast, np.zeros(M), method="BFGS", options={"maxiter": 250})
    wp = np.exp(res_p.x - res_p.x.max())
    wp /= wp.sum()
    ens_p_oof = sum(wp[i] * oof_ps[i] for i in range(M)).astype(np.float32)
    ens_p_te = sum(wp[i] * te_ps[i] for i in range(M)).astype(np.float32)
    ll_p = fast_ll(y, ens_p_oof)

    def _cv_f1(oof_p):
        preds = np.zeros(len(y), dtype=np.int32)
        ws = []
        for f in range(5):
            w_f, _ = glmm.optimize_macro_f1_weights(oof_p[folds != f], y[folds != f], n_restarts=3, seed=42 + f)
            ws.append(w_f)
            preds[folds == f] = (oof_p[folds == f] * w_f).argmax(1)
        f1_v = glmm.fast_macro_f1(y, preds, K)
        w_m = np.exp(np.mean(np.log(ws), axis=0))
        w_m /= np.exp(np.mean(np.log(w_m)))
        return f1_v, w_m, preds

    results = []
    f1_p, w_cls_p, pred_p = _cv_f1(ens_p_oof)
    print(f"\n[PROB BLEND ({time.time()-t0:.1f}s)] weights={dict(zip(names, np.round(wp, 4).tolist()))}", flush=True)
    print(f"  OOF_ll={ll_p:.5f} | argmax_F1={glmm.fast_macro_f1(y, ens_p_oof.argmax(1), K):.5f} | CV_tuned_F1={f1_p:.5f}", flush=True)
    results.append(("prob_blend", f1_p, ll_p, ens_p_oof, ens_p_te, w_cls_p, pred_p, {"prob_weights": dict(zip(names, np.round(wp, 6).tolist()))}))

    # 5-Fold L2 Meta-Stacker + Hybrid 50/50
    stack_models = []
    if M >= 2:
        X_meta_oof = np.hstack(oof_ls).astype(np.float32)
        X_meta_te = np.hstack(te_ls).astype(np.float32)
        meta_oof = np.zeros((len(y), K), dtype=np.float32)
        meta_te = np.zeros((len(te), K), dtype=np.float32)
        for f in range(5):
            tr_m, va_m = folds != f, folds == f
            clf = LogisticRegression(C=0.05, max_iter=180, solver="lbfgs")
            clf.fit(X_meta_oof[tr_m], y[tr_m])
            stack_models.append(clf)
            meta_oof[va_m] = clf.predict_proba(X_meta_oof[va_m]).astype(np.float32)
            meta_te += clf.predict_proba(X_meta_te).astype(np.float32) / 5.0
        joblib.dump({"models": stack_models, "model_names": names, "C": 0.05}, "ghani/task1/models/v8/meta_stacker_v8.joblib", compress=3)

        ll_m = fast_ll(y, meta_oof)
        f1_m, w_cls_m, pred_m = _cv_f1(meta_oof)
        print(f"\n[META STACKER ({time.time()-t0:.1f}s)] OOF_ll={ll_m:.5f} | argmax_F1={glmm.fast_macro_f1(y, meta_oof.argmax(1), K):.5f} | CV_tuned_F1={f1_m:.5f}", flush=True)
        results.append(("meta_stacker", f1_m, ll_m, meta_oof, meta_te, w_cls_m, pred_m, {"prob_weights": dict(zip(names, np.round(wp, 6).tolist())), "meta_C": 0.05}))

        hyb_oof = (0.5 * ens_p_oof + 0.5 * meta_oof).astype(np.float32)
        hyb_te = (0.5 * ens_p_te + 0.5 * meta_te).astype(np.float32)
        ll_h = fast_ll(y, hyb_oof)
        f1_h, w_cls_h, pred_h = _cv_f1(hyb_oof)
        print(f"\n[HYBRID 50/50 PROB+STACKER ({time.time()-t0:.1f}s)] OOF_ll={ll_h:.5f} | argmax_F1={glmm.fast_macro_f1(y, hyb_oof.argmax(1), K):.5f} | CV_tuned_F1={f1_h:.5f}", flush=True)
        results.append(("hybrid_prob_stacker", f1_h, ll_h, hyb_oof, hyb_te, w_cls_h, pred_h, {"prob_weights": dict(zip(names, np.round(wp, 6).tolist())), "meta_C": 0.05}))

    results.sort(key=lambda r: (r[1], -r[2]), reverse=True)
    mode, best_f1, best_ll, best_oof, best_te, best_w_cls, best_pred, info = results[0]

    print(f"\n=== FINAL v8 WINNER ({mode.upper()}): OOF_ll={best_ll:.5f} | CV_Macro_F1={best_f1:.5f} ===", flush=True)
    print("Optimal class multipliers:", np.round(best_w_cls, 4).tolist(), flush=True)
    print(classification_report(y, best_pred, target_names=classes, digits=4), flush=True)

    pred_te = (best_te * best_w_cls).argmax(1)
    sub = pd.DataFrame({"shipment_id": te["shipment_id"], "status_penanganan": [classes[i] for i in pred_te]})
    sub_path = "submissions/task1/submission_v8_ghani.csv"
    sub.to_csv(sub_path, index=False)

    assert len(sub) == 362245, f"Wrong row count: {len(sub)}"
    assert sub["status_penanganan"].isna().sum() == 0, "NaNs in predictions!"
    assert set(sub["status_penanganan"].unique()).issubset(set(classes)), "Invalid class labels!"
    ss = pd.read_csv("Task-1-Datavidia-11.0/sample_submission.csv", usecols=["shipment_id"])
    assert (sub["shipment_id"].values == ss["shipment_id"].values).all(), "shipment_id order mismatch!"

    # Export feature_spec_v8.json for standalone zero-retraining inference
    base_cols = v8c.get_base_feature_cols(tr)
    cat_cols = [c for c in base_cols if str(tr[c].dtype) == "category"]
    cat_categories = {c: list(tr[c].cat.categories.astype(str)) for c in cat_cols}
    xgb_iters = [0] * 5
    if os.path.exists("ghani/task1/models/v8/meta_xgb.json"):
        xgb_iters = json.load(open("ghani/task1/models/v8/meta_xgb.json"))["best_iters"]

    moda_ref_speed = None
    rs_path = "Task-1-Datavidia-11.0/train/route_segments.csv"
    if os.path.exists(rs_path):
        d = pd.read_csv(rs_path, usecols=["moda", "jarak_segmen_km", "durasi_jam"])
        valid = d["durasi_jam"].notna() & (d["durasi_jam"] > 0)
        spd = d.loc[valid, "jarak_segmen_km"] / d.loc[valid, "durasi_jam"]
        moda_ref_speed = {str(k): float(v) for k, v in spd.groupby(d.loc[valid, "moda"]).median().to_dict().items()}
        del d
        gc.collect()

    spec = {
        "classes": classes,
        "base_cols": base_cols,
        "cu_tr_cols": v8c.CU_TR_FEAT_NAMES,
        "glmm_cols": v8c.GLMM_V8_FEAT_NAMES,
        "cat_cols": cat_cols,
        "cat_categories": cat_categories,
        "xgb_best_iters": xgb_iters,
        "moda_ref_speed": moda_ref_speed,
        "n_full_train": int(len(tr)),
    }
    json.dump(spec, open("ghani/task1/models/v8/feature_spec_v8.json", "w"), indent=2)

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
    json.dump(meta, open("ghani/task1/models/v8/meta_v8_ensemble.json", "w"), indent=2)
    np.save("ghani/task1/cache/v8/ens_oof_proba.npy", best_oof)
    np.save("ghani/task1/cache/v8/ens_test_proba.npy", best_te)
    print(f"SAVED {sub_path} | rows={len(sub):,} | dist={meta['test_distribution']}", flush=True)

    # Compare directly against v6 (LB=0.60163) across seen vs unseen customers!
    if os.path.exists("submissions/task1/submission_v6_ghani.csv"):
        v6 = pd.read_csv("submissions/task1/submission_v6_ghani.csv")["status_penanganan"]
        acc_all = (sub["status_penanganan"] == v6).mean()
        f1_all = f1_score(v6, sub["status_penanganan"], average="macro")
        print(f"Comparison vs v6 (LB=0.60163): overall_agree={acc_all*100:.2f}% | macro_F1(v6, v8)={f1_all:.4f}", flush=True)


if __name__ == "__main__":
    main()
