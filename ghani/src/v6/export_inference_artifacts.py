"""Export lightweight preprocessing & meta-stacker artifacts so infer_v6.py can run
100% inference WITHOUT retraining any Stage-1, GLMM, or Stage-2 base model.

Exports into ghani/models/v6/:
  1. feature_spec.json          : exact Stage-1/Stage-2 columns, categorical levels, XGBoost best_iterations
  2. meta_stacker.joblib        : 5-fold L2 LogisticRegression (C=0.05) meta-stacker models
  3. nn/scaler_f0..4.joblib     : exact median/signed-log1p/mean/std/one-hot params for ResNet
  4. nn2/scaler_f0..4.joblib    : exact median/QuantileTransformer(RankGauss)/one-hot params for RGResNet

Jalankan dari root repo:
  python -u -W ignore ghani/src/export_inference_artifacts.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm
import stage2_common as s2
from build_stage2_glmm import GLMM_FEAT_NAMES
from run_stage1_glmm import get_feature_cols

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def export_feature_spec():
    log("1/4 Exporting feature_spec.json ...")
    tr = pd.read_pickle("ghani/cache/v5/feat_v5_train.pkl")
    s1_feats = get_feature_cols(tr)
    classes = sorted(tr["status_penanganan"].unique())

    imp = pd.read_csv("ghani/cache/v5/stage1/importance_f0.csv", index_col=0).iloc[:, 0]
    keep_base = [c for c in imp.index if imp[c] >= 0.02]
    for must in ["has_klaim_inv", "is_cod", "inv_any_penalty", "base_inv_ps", "scan_has_selesai", "scan_q_min"]:
        if must in tr.columns and must not in keep_base:
            keep_base.append(must)

    s1_cat_cols = [c for c in s1_feats if str(tr[c].dtype) == "category"]
    cat_categories = {c: list(tr[c].cat.categories.astype(str)) for c in s1_cat_cols}

    # Check XGBoost saved best_iteration attributes
    xgb_best_iters = []
    for f in range(5):
        bst = xgb.Booster(model_file=f"ghani/models/v6/xgb/xgb_f{f}.ubj")
        bi = int(bst.best_iteration) + 1 if hasattr(bst, "best_iteration") and bst.best_iteration is not None else bst.num_boosted_rounds()
        xgb_best_iters.append(bi)
        log(f"  XGB fold {f}: best_iteration+1={bi}, total_rounds={bst.num_boosted_rounds()}")

    # Also compute exact moda_ref_speed from train/route_segments.csv so test-only build is deterministic
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
        "stage1_features": s1_feats,
        "stage1_cat_cols": s1_cat_cols,
        "stage2_keep_base": keep_base,
        "glmm_feat_names": GLMM_FEAT_NAMES,
        "cat_categories": cat_categories,
        "xgb_best_iters": xgb_best_iters,
        "moda_ref_speed": moda_ref_speed,
    }
    out_path = "ghani/models/v6/feature_spec.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    log(f"  Saved {out_path} (s1_feats={len(s1_feats)}, keep_base={len(keep_base)}, cats={len(s1_cat_cols)})")
    del tr
    gc.collect()


def export_meta_stacker():
    log("2/4 Exporting 5-fold L2 Logistic Meta-Stacker ...")
    tr = pd.read_pickle("ghani/cache/v5/feat_v5_train.pkl")[["status_penanganan"]]
    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    folds = np.load("ghani/cache/v5/folds_v5.npy")

    candidates = [
        ("GLMM",     "ghani/cache/v6/glmm_oof_proba.npy",       False),
        ("LGB_res",  "ghani/cache/v6/lgb_res_oof_logits.npy",   True),
        ("LGB_full", "ghani/cache/v6/lgb_full_oof_logits.npy",  True),
        ("XGBoost",  "ghani/cache/v6/xgb_oof_logits.npy",       True),
        ("ResNet",   "ghani/cache/v6/nn_oof_logits.npy",        True),
        ("RGResNet", "ghani/cache/v6/nn2_oof_logits.npy",       True),
    ]
    oof_ls = []
    names = []
    for name, op, is_logit in candidates:
        ao = np.load(op).astype(np.float32)
        if is_logit:
            lo = ao - ao.mean(axis=1, keepdims=True)
        else:
            lo = np.log(np.clip(ao, 1e-7, 1.0))
            lo -= lo.mean(axis=1, keepdims=True)
        oof_ls.append(lo)
        names.append(name)

    X_meta_oof = np.hstack(oof_ls).astype(np.float32)
    models = []
    for f in range(5):
        tr_m = folds != f
        clf = LogisticRegression(C=0.05, max_iter=150, solver="lbfgs")
        clf.fit(X_meta_oof[tr_m], y[tr_m])
        models.append(clf)
        log(f"  Fit meta-stacker fold {f} (n_iter={clf.n_iter_})")

    out_path = "ghani/models/v6/meta_stacker.joblib"
    joblib.dump({"models": models, "model_names": names, "C": 0.05}, out_path, compress=3)
    log(f"  Saved {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
    del X_meta_oof, oof_ls, tr
    gc.collect()


def export_nn_scalers():
    log("3/4 & 4/4 Exporting ResNet & RGResNet fold scalers ...")
    tr_base, _, _, folds, _, _, cat_cols = s2.load_stage2_base()

    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        blk_tr, _, _ = s2.get_fold_glmm_arrays(f, folds)
        df_tr = s2.attach_glmm_df(tr_base.iloc[tr_idx], blk_tr)
        del blk_tr

        num_cols = [c for c in df_tr.columns if c not in cat_cols]
        cat_cats = {c: list(pd.Categorical(df_tr[c].astype(str)).categories) for c in cat_cols}

        X_raw = df_tr[num_cols].to_numpy(dtype=np.float32, copy=True)
        del df_tr
        gc.collect()

        med = np.nanmedian(X_raw, axis=0)
        med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
        na = np.isnan(X_raw)
        if na.any():
            X_raw[na] = np.take(med, np.where(na)[1])

        # --- Scaler 1: ResNet (signed log1p + z-score) ---
        X1 = X_raw.copy()
        heavy = np.std(X1, axis=0) > 100.0
        if heavy.any():
            X1[:, heavy] = np.sign(X1[:, heavy]) * np.log1p(np.abs(X1[:, heavy]))
        mu = X1.mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = np.maximum(X1.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
        del X1

        p1 = f"ghani/models/v6/nn/scaler_f{f}.joblib"
        joblib.dump(
            {"num_cols": num_cols, "cat_cols": cat_cols, "cat_cats": cat_cats,
             "med": med, "heavy": heavy, "mu": mu, "sd": sd},
            p1, compress=3,
        )

        # --- Scaler 2: RGResNet (QuantileTransformer RankGauss, exact seed=100+f) ---
        qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", subsample=150_000, random_state=100 + f)
        qt.fit(X_raw)
        del X_raw
        gc.collect()

        p2 = f"ghani/models/v6/nn2/scaler_f{f}.joblib"
        joblib.dump(
            {"num_cols": num_cols, "cat_cols": cat_cols, "cat_cats": cat_cats,
             "med": med, "qt": qt},
            p2, compress=3,
        )
        log(f"  Fold {f} saved: {p1} ({os.path.getsize(p1)/1024:.1f} KB) | {p2} ({os.path.getsize(p2)/1024:.1f} KB)")

    del tr_base
    gc.collect()
    log("ALL INFERENCE ARTIFACTS EXPORTED SUCCESSFULLY!")


def main():
    export_feature_spec()
    export_meta_stacker()
    export_nn_scalers()


if __name__ == "__main__":
    main()
