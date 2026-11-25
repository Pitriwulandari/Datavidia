"""v7 Model Family #1 & #2: 5-Fold GroupKFold(customer_id) LightGBM (Unweighted + Test-Regime Weighted)
+ Persistent Hub/Courier Joint GLMM (ZERO customer target leakage!).

Produces:
  - ghani/models/v7/lgb_unw/lgb_f0..4.txt
  - ghani/models/v7/lgb_tew/lgb_f0..4.txt
  - ghani/models/v7/glmm_hc_full.npy
  - ghani/cache/v7/{lgb_unw,lgb_tew}_{oof,test}_logits.npy
  - submissions/submission_v7_lgb_ghani.csv (immediate fast #1 contender!)

Jalankan dari root repo:
  python -u -W ignore ghani/src/train_v7_lgb.py
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def get_v7_feature_cols(tr):
    drop = {
        "shipment_id", "status_penanganan",
        "customer_id", "courier_id", "hub_id", "package_id",
        "customer_id_code", "courier_id_code", "hub_id_code", "package_id_code",
    }
    return [c for c in tr.columns if c not in drop]


def compute_test_sim_weights(n_tr, n_te, peak_tr, peak_te):
    """Exact importance weights matching test's (n_cust_in_split bucket, is_peak_month) joint distribution."""
    b_tr = np.minimum(n_tr.astype(np.int32), 4)
    b_te = np.minimum(n_te.astype(np.int32), 4)
    k_tr = (b_tr - 1) * 2 + peak_tr.astype(np.int32)
    k_te = (b_te - 1) * 2 + peak_te.astype(np.int32)
    p_tr = np.bincount(k_tr, minlength=8) / len(k_tr)
    p_te = np.bincount(k_te, minlength=8) / len(k_te)
    ratio = p_te / np.maximum(p_tr, 1e-6)
    return ratio[k_tr].astype(np.float32), p_te


def eval_v7_oof(name, proba, y, folds, n_tr, sim_w, K=7):
    ll_all = log_loss(y, proba, labels=list(range(K)))
    ll_sim = log_loss(y, proba, sample_weight=sim_w, labels=list(range(K)))
    arg_all = glmm.fast_macro_f1(y, proba.argmax(1), K)

    # Leak-free 5-fold CV class multipliers: compare global vs separate (n==1 vs n>=2) on test-sim distribution!
    # Draw fixed deterministic test-matched subsample per fold for exact unbiased TEST-SIM F1
    rng = np.random.RandomState(42)
    # Build deterministic test-matched index mask (weighted sampling without replacement per bucket)
    p_prob = sim_w / sim_w.sum()
    sim_idx = rng.choice(len(y), size=362_245, replace=True, p=p_prob)

    preds_glob = np.zeros(len(y), dtype=np.int32)
    ws_glob = []
    for f in range(5):
        tr_m = folds != f
        # Optimize on test-matched subsample of train folds so multipliers match test's n_cust_te distribution!
        tr_pos = np.where(tr_m)[0]
        w_tr = sim_w[tr_pos]
        sub_tr = rng.choice(tr_pos, size=300_000, replace=True, p=w_tr / w_tr.sum())
        wf, _ = glmm.optimize_macro_f1_weights(proba[sub_tr], y[sub_tr], n_restarts=2, seed=42 + f)
        ws_glob.append(wf)
        preds_glob[folds == f] = (proba[folds == f] * wf).argmax(1)

    w_mean = np.exp(np.mean(np.log(ws_glob), axis=0))
    w_mean /= np.exp(np.mean(np.log(w_mean)))

    f1_all = glmm.fast_macro_f1(y, preds_glob, K)
    f1_sim = glmm.fast_macro_f1(y[sim_idx], preds_glob[sim_idx], K)
    parts = [f"n={nv}:{glmm.fast_macro_f1(y[n_tr==nv], preds_glob[n_tr==nv], K):.4f}" for nv in [1, 2, 3]]
    parts.append(f"n>=4:{glmm.fast_macro_f1(y[n_tr>=4], preds_glob[n_tr>=4], K):.4f}")
    log(f"==> {name:24s} | OOF_ll={ll_all:.5f} (sim={ll_sim:.5f}) | argF1={arg_all:.5f} | CV_F1={f1_all:.5f} | TEST-SIM_F1={f1_sim:.5f} | {' '.join(parts)}")
    return ll_all, ll_sim, f1_all, f1_sim, w_mean, preds_glob


def build_hc_entities(tr_meta, te_meta):
    """Build Hub + Hub_Peak + Courier entity codes ONLY (100% persistent across train & test!)."""
    ents_tr, ents_te, n_levels = {}, {}, {}
    for name, col in [("hub", "hub_id_code"), ("courier", "courier_id_code")]:
        c_tr = tr_meta[col].to_numpy(dtype=np.int32)
        c_te = te_meta[col].to_numpy(dtype=np.int32)
        ents_tr[name] = c_tr
        ents_te[name] = c_te
        n_levels[name] = int(max(c_tr.max(), c_te.max())) + 1
    hp_tr = (tr_meta["hub_id_code"].to_numpy(np.int32) * 2 + tr_meta["is_peak_month"].to_numpy(np.int32)).astype(np.int32)
    hp_te = (te_meta["hub_id_code"].to_numpy(np.int32) * 2 + te_meta["is_peak_month"].to_numpy(np.int32)).astype(np.int32)
    ents_tr["hub_peak"] = hp_tr
    ents_te["hub_peak"] = hp_te
    n_levels["hub_peak"] = n_levels["hub"] * 2
    return ents_tr, ents_te, n_levels


def train_lgb_mode(mode, tr, te, feats, cat_cols, y, folds, sample_weight, K):
    os.makedirs(f"ghani/models/v7/lgb_{mode}", exist_ok=True)
    os.makedirs(f"ghani/cache/v7/lgb_{mode}", exist_ok=True)

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te), K), dtype=np.float32)

    if mode == "unw":
        params = dict(
            objective="multiclass", num_class=K, learning_rate=0.06, num_leaves=63,
            min_child_samples=120, subsample=0.8, subsample_freq=1, colsample_bytree=0.55,
            reg_lambda=15.0, reg_alpha=2.0, max_bin=127, cat_smooth=20.0,
            verbose=-1, n_jobs=os.cpu_count(),
        )
        seed_base = 100
    else:
        # Test-regime weighted + deeper leaves & stronger colsample diversity!
        params = dict(
            objective="multiclass", num_class=K, learning_rate=0.055, num_leaves=95,
            min_child_samples=160, subsample=0.8, subsample_freq=1, colsample_bytree=0.45,
            reg_lambda=20.0, reg_alpha=3.0, max_bin=127, cat_smooth=25.0,
            verbose=-1, n_jobs=os.cpu_count(),
        )
        seed_base = 300

    for f in range(5):
        m_path = f"ghani/models/v7/lgb_{mode}/lgb_f{f}.txt"
        oof_p = f"ghani/cache/v7/lgb_{mode}/oof_f{f}.npy"
        te_p = f"ghani/cache/v7/lgb_{mode}/te_f{f}.npy"
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        if os.path.exists(m_path) and os.path.exists(oof_p) and os.path.exists(te_p) and os.environ.get("FORCE", "0") != "1":
            oof_logits[va_idx] = np.load(oof_p)
            te_logits += np.load(te_p) / 5.0
            log(f"v7 LGB_{mode} Fold {f}: cached.")
            continue

        w_tr = sample_weight[tr_idx] if sample_weight is not None else None
        w_va = sample_weight[va_idx] if sample_weight is not None else None
        dtr = lgb.Dataset(tr.iloc[tr_idx][feats], y[tr_idx], weight=w_tr, categorical_feature=cat_cols, free_raw_data=True)
        dva = lgb.Dataset(tr.iloc[va_idx][feats], y[va_idx], weight=w_va, reference=dtr, categorical_feature=cat_cols, free_raw_data=True)

        log(f"=== v7 LGB_{mode} Fold {f} (train={len(tr_idx):,}, val={len(va_idx):,}) ===")
        model = lgb.train(
            dict(params, random_state=seed_base + f), dtr, num_boost_round=900,
            valid_sets=[dva], callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)],
        )
        del dtr, dva
        gc.collect()

        model.save_model(m_path, num_iteration=model.best_iteration)
        va_raw = model.predict(tr.iloc[va_idx][feats], num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        te_raw = model.predict(te[feats], num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        np.save(oof_p, va_raw)
        np.save(te_p, te_raw)
        oof_logits[va_idx] = va_raw
        te_logits += te_raw / 5.0

        vp = glmm.softmax(va_raw)
        log(f"  LGB_{mode} F{f} done (it={model.best_iteration}): val_ll={log_loss(y[va_idx], vp, labels=list(range(K))):.5f} | argF1={glmm.fast_macro_f1(y[va_idx], vp.argmax(1), K):.5f}")
        del model
        gc.collect()

    np.save(f"ghani/cache/v7/lgb_{mode}_oof_logits.npy", oof_logits)
    np.save(f"ghani/cache/v7/lgb_{mode}_test_logits.npy", te_logits)
    return oof_logits, te_logits


def main():
    os.makedirs("ghani/models/v7", exist_ok=True)
    os.makedirs("ghani/cache/v7", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v7/feat_v7_train.pkl")
    te = pd.read_pickle("ghani/cache/v7/feat_v7_test.pkl")
    folds = np.load("ghani/cache/v7/folds_gkf5.npy")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    K = len(classes)

    te_sid = te["shipment_id"].copy()
    n_tr = tr["cu_sp_n"].to_numpy(dtype=np.float32)
    n_te = te["cu_sp_n"].to_numpy(dtype=np.float32)
    sim_w, _ = compute_test_sim_weights(n_tr, n_te, tr["is_peak_month"].to_numpy(), te["is_peak_month"].to_numpy())
    # Soft sqrt weight for LGB_tew training (focuses splits on n=1,2 without high variance!)
    soft_w = np.sqrt(sim_w).astype(np.float32)
    soft_w /= soft_w.mean()

    tr_meta = tr[["hub_id_code", "courier_id_code", "is_peak_month"]].copy()
    te_meta = te[["hub_id_code", "courier_id_code", "is_peak_month"]].copy()

    feats = get_v7_feature_cols(tr)
    cat_cols = [c for c in feats if str(tr[c].dtype) == "category"]
    log(f"Loaded feat_v7: tr={tr.shape}, te={te.shape}, feats={len(feats)}, cats={len(cat_cols)}")

    # Save v7 feature spec immediately for reproducible zero-retraining inference
    cat_categories = {c: list(tr[c].cat.categories.astype(str)) for c in cat_cols}
    json.dump(
        {"classes": classes, "features": feats, "cat_cols": cat_cols, "cat_categories": cat_categories},
        open("ghani/models/v7/feature_spec_v7.json", "w", encoding="utf-8"), indent=2,
    )

    # Free non-feature ID columns immediately to keep RAM lean!
    tr_drop = [c for c in tr.columns if c not in feats]
    te_drop = [c for c in te.columns if c not in feats]
    tr.drop(columns=tr_drop, inplace=True)
    te.drop(columns=te_drop, inplace=True)
    gc.collect()

    oof_unw, te_unw = train_lgb_mode("unw", tr, te, feats, cat_cols, y, folds, None, K)
    eval_v7_oof("v7_LGB_unw", glmm.softmax(oof_unw), y, folds, n_tr, sim_w, K)

    oof_tew, te_tew = train_lgb_mode("tew", tr, te, feats, cat_cols, y, folds, soft_w, K)
    eval_v7_oof("v7_LGB_tew (sqrt-w)", glmm.softmax(oof_tew), y, folds, n_tr, sim_w, K)

    del tr, te
    gc.collect()

    # Combine LGB_unw + LGB_tew + Persistent Hub/Courier Joint GLMM (ZERO u_cust!)
    blend_oof_l = (0.5 * oof_unw + 0.5 * oof_tew).astype(np.float32)
    blend_te_l = (0.5 * te_unw + 0.5 * te_tew).astype(np.float32)
    ll_b, ll_bs, f1_b, f1_bs, w_b, _ = eval_v7_oof("v7_LGB_unw+tew", glmm.softmax(blend_oof_l), y, folds, n_tr, sim_w, K)

    # Check if persistent Hub+Courier Joint GLMM on top improves GroupKFold OOF further
    ents_tr, ents_te, n_levels = build_hc_entities(tr_meta, te_meta)
    hc_lam = {
        "hub":      np.array([30.0,  7.0, 60.0, 22.0, 60.0, 45.0, 14.0], dtype=np.float64),
        "hub_peak": np.array([90.0, 90.0, 90.0, 55.0, 90.0, 90.0, 28.0], dtype=np.float64),
        "courier":  np.array([15.0, 54.0, 135.0, 21.0, 135.0, 21.0, 15.0], dtype=np.float64),
    }
    gl_oof = np.zeros((len(y), K), dtype=np.float32)
    for f in range(5):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        st = glmm.fit_joint_glmm(blend_oof_l[tr_idx], y[tr_idx], {k: v[tr_idx] for k, v in ents_tr.items()}, n_levels, n_classes=K, fixed_lam=hc_lam, n_newton=4, verbose=False)
        _, p_va = glmm.predict_glmm(blend_oof_l[va_idx], {k: v[va_idx] for k, v in ents_tr.items()}, st, use_moderation=False)
        gl_oof[va_idx] = p_va
    st_full = glmm.fit_joint_glmm(blend_oof_l, y, ents_tr, n_levels, n_classes=K, fixed_lam=hc_lam, n_newton=5, verbose=False)
    gl_te_eta, gl_te_p = glmm.predict_glmm(blend_te_l, ents_te, st_full, use_moderation=False)
    np.save("ghani/models/v7/glmm_hc_full.npy", st_full, allow_pickle=True)
    np.save("ghani/cache/v7/glmm_hc_oof_proba.npy", gl_oof)
    np.save("ghani/cache/v7/glmm_hc_test_proba.npy", gl_te_p)

    ll_g, ll_gs, f1_g, f1_gs, w_g, pred_g = eval_v7_oof("v7_LGB+HC_GLMM", gl_oof, y, folds, n_tr, sim_w, K)

    # Pick whichever has better TEST-SIM Macro F1 / LogLoss for immediate v7_lgb submission
    if f1_gs >= f1_bs:
        use_te, use_w, tag_w, best_sim, best_ll = gl_te_p, w_g, "LGB_unw+tew+HC_GLMM", f1_gs, ll_gs
        best_pred = pred_g
    else:
        use_te, use_w, tag_w, best_sim, best_ll = glmm.softmax(blend_te_l), w_b, "LGB_unw+tew", f1_bs, ll_bs
        best_pred = (glmm.softmax(blend_oof_l) * w_b).argmax(1)

    print(classification_report(y, best_pred, target_names=classes, digits=4), flush=True)
    pred_te = (use_te * use_w).argmax(1)
    sub = pd.DataFrame({"shipment_id": te_sid, "status_penanganan": [classes[i] for i in pred_te]})
    sub_path = "submissions/submission_v7_lgb_ghani.csv"
    sub.to_csv(sub_path, index=False)
    assert len(sub) == 362245 and sub["status_penanganan"].isna().sum() == 0
    meta = {
        "winner": tag_w, "classes": classes, "class_multipliers": use_w.tolist(),
        "test_sim_ll": float(best_ll), "test_sim_f1": float(best_sim),
        "test_distribution": sub["status_penanganan"].value_counts(normalize=True).round(4).to_dict(),
    }
    json.dump(meta, open("ghani/models/v7/meta_v7_lgb.json", "w"), indent=2)
    log(f"SAVED {sub_path} | TEST-SIM_F1={best_sim:.5f} | dist={meta['test_distribution']}")


if __name__ == "__main__":
    main()
