"""Standalone Zero-Retraining Inference Script for Task-1 Datavidia 11.0 (v7 Final Winner).

Loads ALL saved v7 models from ghani/models/v7/ and generates predictions for all 362,245 test rows
WITHOUT retraining any model (100.000% bit-for-bit exact reproduction):
  1. 5-Fold GroupKFold LightGBM Unweighted   (ghani/models/v7/lgb_unw/lgb_f0..4.txt)
  2. 5-Fold GroupKFold LightGBM Test-Weight  (ghani/models/v7/lgb_tew/lgb_f0..4.txt)
  3. Persistent Hub/Courier Joint GLMM       (ghani/models/v7/glmm_hc_full.npy)
  4. 5-Fold GroupKFold Depth-Wise XGBoost    (ghani/models/v7/xgb/xgb_f0..4.ubj)
  5. 5-Fold GroupKFold PyTorch ResNetV7+SWA  (ghani/models/v7/nn/resnet_f0..4.pt + scaler_f0..4.joblib)
  6. 5-Fold GroupKFold L2 Meta-Stacker       (ghani/models/v7/meta_stacker_v7.joblib)
  7. Optimal Test-Regime Class Multipliers   (ghani/models/v7/meta_v7_ensemble.json)

Penggunaan (dari root repo):
  python -u -W ignore ghani/src/v7/infer_v7.py
  python -u -W ignore ghani/src/v7/infer_v7.py --rebuild-features --output submissions/submission_v7_ghani.csv
"""
import argparse
import gc
import json
import os
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

for _sub in ["common", "v1", "v2", "v5", "v6", "v7"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_features_v5 as bf5
import build_features_v7 as bf7
import glmm_lib as glmm
from train_v7_nn import TabularResNetV7

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_or_build_v7_test(spec, rebuild=False):
    cache_te = "ghani/cache/v7/feat_v7_test.pkl"
    if not rebuild and os.path.exists(cache_te):
        log(f"Loading cached v7 test features from {cache_te} ...")
        te = pd.read_pickle(cache_te)
    else:
        log("Rebuilding v7 features directly from raw Task-1-Datavidia-11.0/ CSVs (zero target labels)...")
        tr_b, moda_ref = bf5.build_split("train", None)
        te_b, _ = bf5.build_split("test", moda_ref)
        tr_b, te_b = bf5.add_unsupervised_entity_profiles(tr_b, te_b)
        id_cols = {"shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id"}
        for c in tr_b.columns:
            if c not in id_cols and tr_b[c].dtype == object:
                cats = pd.Categorical(pd.concat([tr_b[c], te_b[c]]).astype(str)).categories
                tr_b[c] = pd.Categorical(tr_b[c].astype(str), categories=cats)
                te_b[c] = pd.Categorical(te_b[c].astype(str), categories=cats)
        for eid in ["customer_id", "courier_id", "hub_id", "package_id"]:
            all_v = pd.concat([tr_b[eid], te_b[eid]]).astype(str)
            codes, _ = pd.factorize(all_v, sort=False)
            tr_b[eid + "_code"] = codes[: len(tr_b)].astype(np.int32)
            te_b[eid + "_code"] = codes[len(tr_b):].astype(np.int32)
        _, te = bf7.build_v7_from_base(tr_b, te_b)
        del tr_b, te_b
        gc.collect()

    for c, cats in spec["cat_categories"].items():
        if c in te.columns:
            te[c] = pd.Categorical(te[c].astype(str), categories=cats)
    log(f"v7 test feature matrix ready: {te.shape}")
    return te


def transform_nn_test_v7(df_te, scaler):
    Xte = df_te[scaler["num_cols"]].to_numpy(dtype=np.float32, copy=True)
    na_te = np.isnan(Xte[:, scaler["na_cols"]]).astype(np.float32)

    med = scaler["med"]
    na = np.isnan(Xte)
    if na.any():
        Xte[na] = np.take(med, np.where(na)[1])

    heavy = scaler["heavy"]
    if heavy.any():
        Xte[:, heavy] = np.sign(Xte[:, heavy]) * np.log1p(np.abs(Xte[:, heavy]))

    Xte -= scaler["mu"]
    Xte /= scaler["sd"]
    np.clip(Xte, -8.0, 8.0, out=Xte)

    parts = [Xte, na_te]
    for c in scaler["cat_cols"]:
        cats = scaler["cat_cats"][c]
        codes = pd.Categorical(df_te[c].astype(str), categories=cats).codes
        mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
        ok = codes >= 0
        mat[np.where(ok)[0], codes[ok]] = 1.0
        parts.append(mat)
    return np.hstack(parts)


def run_inference_v7(output_path="submissions/submission_v7_reproduced_ghani.csv", rebuild_features=False, verify_path=None):
    torch.set_num_threads(os.cpu_count() or 8)
    spec = json.load(open("ghani/models/v7/feature_spec_v7.json", encoding="utf-8"))
    meta_xgb = json.load(open("ghani/models/v7/meta_xgb.json", encoding="utf-8"))
    meta = json.load(open("ghani/models/v7/meta_v7_ensemble.json", encoding="utf-8"))

    classes = meta["classes"]
    K = len(classes)
    feats = spec["features"]

    te = load_or_build_v7_test(spec, rebuild=rebuild_features)
    te_sid = te["shipment_id"].copy()
    n_te = te["cu_sp_n"].to_numpy(dtype=np.float32)
    N_te = len(te)

    ents_te = {
        "hub": te["hub_id_code"].to_numpy(dtype=np.int32),
        "courier": te["courier_id_code"].to_numpy(dtype=np.int32),
        "hub_peak": (te["hub_id_code"].to_numpy(dtype=np.int32) * 2 + te["is_peak_month"].to_numpy(dtype=np.int32)).astype(np.int32),
    }

    X_te = te[feats]
    del te
    gc.collect()

    # 1. LGB_unw (5 folds)
    log("Stage 1/5: Running 5-Fold v7 LGB_unw inference ...")
    te_unw = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/models/v7/lgb_unw/lgb_f{f}.txt")
        te_unw += bst.predict(X_te, raw_score=True).astype(np.float32) / 5.0
        del bst

    # 2. LGB_tew (5 folds)
    log("Stage 2/5: Running 5-Fold v7 LGB_tew inference ...")
    te_tew = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/models/v7/lgb_tew/lgb_f{f}.txt")
        te_tew += bst.predict(X_te, raw_score=True).astype(np.float32) / 5.0
        del bst

    # 3. Persistent Hub/Courier Joint GLMM
    log("Stage 3/5: Applying saved Persistent Hub/Courier Joint GLMM ...")
    st_hc = np.load("ghani/models/v7/glmm_hc_full.npy", allow_pickle=True).item()
    _, gl_te_p = glmm.predict_glmm((0.5 * te_unw + 0.5 * te_tew).astype(np.float32), ents_te, st_hc, use_moderation=False)
    gl_te_p = gl_te_p.astype(np.float32)
    del st_hc, ents_te

    # 4. XGBoost (5 folds)
    log("Stage 4/5: Running 5-Fold v7 XGBoost inference ...")
    dte_xgb = xgb.DMatrix(X_te, enable_categorical=True)
    xgb_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = xgb.Booster(model_file=f"ghani/models/v7/xgb/xgb_f{f}.ubj")
        bi = meta_xgb["xgb_best_iters"][f]
        xgb_te += bst.predict(dte_xgb, output_margin=True, iteration_range=(0, bi)).astype(np.float32) / 5.0
        del bst
    del dte_xgb
    gc.collect()

    # 5. PyTorch TabularResNetV7 + Weight-Space SWA (5 folds)
    log("Stage 5/5: Running 5-Fold v7 PyTorch TabularResNetV7 (SWA) inference ...")
    nn_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        scaler = joblib.load(f"ghani/models/v7/nn/scaler_f{f}.joblib")
        Xte_nn = transform_nn_test_v7(X_te, scaler)
        model = TabularResNetV7(Xte_nn.shape[1], n_classes=K, hidden=320, n_blocks=3, dropout=0.18)
        state = torch.load(f"ghani/models/v7/nn/resnet_f{f}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        chunks = []
        with torch.no_grad():
            for s in range(0, N_te, 16384):
                chunks.append(model(torch.from_numpy(Xte_nn[s : s + 16384])).numpy())
        nn_te += np.vstack(chunks).astype(np.float32) / 5.0
        del Xte_nn, model, state
        gc.collect()

    del X_te
    gc.collect()

    # 6. Combine via saved 5-Fold GroupKFold L2 Meta-Stacker + Optimal Class Multipliers
    log(f"Combining 5 models via saved '{meta['mode']}' + class multipliers ...")
    model_outputs = [
        ("LGB_unw",  te_unw,  True),
        ("LGB_tew",  te_tew,  True),
        ("HC_GLMM",  gl_te_p, False),
        ("XGBoost",  xgb_te,  True),
        ("ResNetV7", nn_te,   True),
    ]
    te_ps, te_ls = {}, []
    for name, arr, is_logit in model_outputs:
        if is_logit:
            te_ps[name] = glmm.softmax(arr).astype(np.float32)
            l = (arr - arr.mean(axis=1, keepdims=True)).astype(np.float32)
        else:
            te_ps[name] = arr.astype(np.float32)
            l = np.log(np.clip(arr, 1e-7, 1.0)).astype(np.float32)
            l -= l.mean(axis=1, keepdims=True)
        te_ls.append(l)

    stack_bundle = joblib.load("ghani/models/v7/meta_stacker_v7.joblib")
    X_meta_te = np.hstack(te_ls).astype(np.float32)
    meta_te = np.zeros((N_te, K), dtype=np.float32)
    for clf in stack_bundle["models"]:
        meta_te += clf.predict_proba(X_meta_te).astype(np.float32) / len(stack_bundle["models"])

    mode = meta["mode"]
    if mode == "meta_stacker":
        final_p = meta_te
    elif mode == "prob_blend":
        pw = meta["blend_info"]
        final_p = sum(float(pw[k]) * te_ps[k] for k in pw).astype(np.float32)
    else:
        pw = meta["blend_info"]["prob_weights"]
        ens_p = sum(float(pw[k]) * te_ps[k] for k in pw).astype(np.float32)
        final_p = (0.5 * ens_p + 0.5 * meta_te).astype(np.float32)

    cm = meta["class_multipliers"]
    if "global" in cm:
        w_cls = np.array(cm["global"], dtype=np.float32)
        pred_idx = (final_p * w_cls).argmax(axis=1)
    else:
        w1 = np.array(cm["n1"], dtype=np.float32)
        w2 = np.array(cm["n2p"], dtype=np.float32)
        pred_idx = np.zeros(N_te, dtype=np.int32)
        m1 = n_te == 1
        pred_idx[m1] = (final_p[m1] * w1).argmax(axis=1)
        pred_idx[~m1] = (final_p[~m1] * w2).argmax(axis=1)

    sub = pd.DataFrame({"shipment_id": te_sid.values, "status_penanganan": [classes[i] for i in pred_idx]})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)

    assert len(sub) == 362245 and sub["status_penanganan"].isna().sum() == 0
    ss = pd.read_csv("Task-1-Datavidia-11.0/sample_submission.csv", usecols=["shipment_id"])
    assert (sub["shipment_id"].values == ss["shipment_id"].values).all(), "shipment_id mismatch!"

    dist = sub["status_penanganan"].value_counts(normalize=True).round(4).to_dict()
    log(f"SAVED {output_path} | rows={len(sub):,} | dist={dist}")

    if verify_path and os.path.exists(verify_path):
        ref = pd.read_csv(verify_path)
        agree = float((sub["status_penanganan"].values == ref["status_penanganan"].values).mean())
        log(f"VERIFICATION vs {verify_path}: exact label agreement = {agree * 100:.4f}% ({int((sub['status_penanganan'].values == ref['status_penanganan'].values).sum()):,}/{len(sub):,})")
        for cname, arr, cpath in [
            ("LGB_unw",  te_unw,  "ghani/cache/v7/lgb_unw_test_logits.npy"),
            ("LGB_tew",  te_tew,  "ghani/cache/v7/lgb_tew_test_logits.npy"),
            ("HC_GLMM",  gl_te_p, "ghani/cache/v7/glmm_hc_test_proba.npy"),
            ("XGBoost",  xgb_te,  "ghani/cache/v7/xgb_test_logits.npy"),
            ("ResNetV7", nn_te,   "ghani/cache/v7/nn_test_logits.npy"),
        ]:
            if os.path.exists(cpath):
                diff = np.abs(arr - np.load(cpath))
                log(f"  [{cname:10s}] mean_abs_diff={diff.mean():.6f} | max_abs_diff={diff.max():.6f}")

    return sub


def main():
    ap = argparse.ArgumentParser(description="Zero-retraining inference for Datavidia Task-1 v7 Super-Ensemble")
    ap.add_argument("--output", default="submissions/submission_v7_reproduced_ghani.csv")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--verify-against", default="submissions/submission_v7_ghani.csv")
    args = ap.parse_args()
    run_inference_v7(output_path=args.output, rebuild_features=args.rebuild_features, verify_path=args.verify_against)


if __name__ == "__main__":
    main()
