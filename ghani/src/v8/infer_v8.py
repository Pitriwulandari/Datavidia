"""Standalone Zero-Retraining Inference Script for Task-1 Datavidia 11.0 (v8 Ultimate Ceiling).

Loads ALL 7 saved model families (35 fold models + 2 Hierarchical GLMMs + 5 Message-Passing Stackers)
WITHOUT retraining a single parameter (100.000% bit-for-bit exact):
  1. v7 5-Fold LightGBM Unweighted        (ghani/models/v7/lgb_unw/lgb_f0..4.txt)
  2. v7 5-Fold LightGBM Test-Weighted     (ghani/models/v7/lgb_tew/lgb_f0..4.txt)
  3. v7 Persistent Hub/Courier GLMM       (ghani/models/v7/glmm_hc_full.npy)
  4. v7 5-Fold Depth-Wise XGBoost         (ghani/models/v7/xgb/xgb_f0..4.ubj)
  5. v7 5-Fold PyTorch TabularResNet+SWA  (ghani/models/v7/nn/resnet_f0..4.pt + scaler_f0..4.joblib)
  6. v8 5-Fold Symmetric CatBoost         (ghani/models/v8/catboost/cb_f0..4.cbm + cb_spec.joblib)
  7. v8 5-Fold PyTorch RankGaussGated+SWA (ghani/models/v8/rg_nn/rg_resnet_f0..4.pt + scaler_f0..4.joblib)
  8. v8 Sibling Message-Passing Stacker   (ghani/models/v8/meta_stacker_v8.joblib)
  9. v8 De-Aliased Customer GLMM + EUM    (ghani/models/v8/glmm_cust_v8_full.npy + meta_v8_ensemble.json)

Penggunaan (dari root repo):
  python -u -W ignore ghani/src/v8/infer_v8.py
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
from catboost import CatBoostClassifier, Pool

for _sub in ["common", "v1", "v2", "v5", "v6", "v7", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
from ensemble_v8 import apply_eum, build_sibling_message_features
from infer_v7 import load_or_build_v7_test, transform_nn_test_v7
from train_v7_nn import TabularResNetV7
from train_v8_rg_resnet import RankGaussGatedResNet, split_pct_ranks

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def build_cb_test_matrix(df_te, cb_spec):
    Xte = df_te[cb_spec["num_cols"]].to_numpy(dtype=np.float32, copy=True)
    parts = [Xte]
    for c in cb_spec["cat_cols"]:
        cats = cb_spec["cat_cats"][c]
        codes = pd.Categorical(df_te[c].astype(str), categories=cats).codes
        mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
        ok = codes >= 0
        mat[np.where(ok)[0], codes[ok]] = 1.0
        parts.append(mat)
    return np.hstack(parts)


def transform_rg_test_v8(df_te, scaler, r_te_all):
    Xte = df_te[scaler["num_cols"]].to_numpy(dtype=np.float32, copy=True)
    na_te = np.isnan(Xte[:, scaler["na_cols"]]).astype(np.float32)
    med = scaler["med"]
    na = np.isnan(Xte)
    if na.any():
        Xte[na] = np.take(med, np.where(na)[1])
    Xte = np.clip(scaler["qt"].transform(Xte).astype(np.float32), -5.0, 5.0)
    parts = [Xte, na_te, r_te_all]
    for c in scaler["cat_cols"]:
        cats = scaler["cat_cats"][c]
        codes = pd.Categorical(df_te[c].astype(str), categories=cats).codes
        mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
        ok = codes >= 0
        mat[np.where(ok)[0], codes[ok]] = 1.0
        parts.append(mat)
    return np.hstack(parts)


def run_inference_v8(output_path="ghani/cache/v8/submission_v8_reproduced_check.csv", rebuild_features=False, verify_path=None):
    torch.set_num_threads(os.cpu_count() or 8)
    spec = json.load(open("ghani/models/v7/feature_spec_v7.json", encoding="utf-8"))
    meta_xgb = json.load(open("ghani/models/v7/meta_xgb.json", encoding="utf-8"))
    meta = json.load(open("ghani/models/v8/meta_v8_ensemble.json", encoding="utf-8"))

    classes = meta["classes"]
    K = len(classes)
    feats = spec["features"]

    te = load_or_build_v7_test(spec, rebuild=rebuild_features)
    te_sid = te["shipment_id"].copy()
    te_ids = te[["customer_id", "courier_id"]].copy()
    c_te_code = te["customer_id_code"].to_numpy(dtype=np.int32)
    N_te = len(te)

    ents_hc = {
        "hub": te["hub_id_code"].to_numpy(dtype=np.int32),
        "courier": te["courier_id_code"].to_numpy(dtype=np.int32),
        "hub_peak": (te["hub_id_code"].to_numpy(dtype=np.int32) * 2 + te["is_peak_month"].to_numpy(dtype=np.int32)).astype(np.int32),
    }
    r_te_all = split_pct_ranks(te)

    X_te = te[feats]
    del te
    gc.collect()

    outputs = {}

    # 1. LGB_unw
    log("Stage 1/7: Running 5-Fold LGB_unw inference ...")
    te_unw = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/models/v7/lgb_unw/lgb_f{f}.txt")
        te_unw += bst.predict(X_te, raw_score=True).astype(np.float32) / 5.0
    outputs["LGB_unw"] = (te_unw, True)

    # 2. LGB_tew
    log("Stage 2/7: Running 5-Fold LGB_tew inference ...")
    te_tew = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/models/v7/lgb_tew/lgb_f{f}.txt")
        te_tew += bst.predict(X_te, raw_score=True).astype(np.float32) / 5.0
    outputs["LGB_tew"] = (te_tew, True)

    # 3. HC_GLMM
    log("Stage 3/7: Applying Persistent Hub/Courier GLMM ...")
    st_hc = np.load("ghani/models/v7/glmm_hc_full.npy", allow_pickle=True).item()
    _, gl_te_p = glmm.predict_glmm((0.5 * te_unw + 0.5 * te_tew).astype(np.float32), ents_hc, st_hc, use_moderation=False)
    outputs["HC_GLMM"] = (gl_te_p.astype(np.float32), False)
    del st_hc, ents_hc

    # 4. XGBoost
    log("Stage 4/7: Running 5-Fold XGBoost inference ...")
    dte_xgb = xgb.DMatrix(X_te, enable_categorical=True)
    xgb_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = xgb.Booster(model_file=f"ghani/models/v7/xgb/xgb_f{f}.ubj")
        xgb_te += bst.predict(dte_xgb, output_margin=True, iteration_range=(0, meta_xgb["xgb_best_iters"][f])).astype(np.float32) / 5.0
    outputs["XGBoost"] = (xgb_te, True)
    del dte_xgb
    gc.collect()

    # 5. TabularResNetV7
    log("Stage 5/7: Running 5-Fold TabularResNetV7 (SWA) inference ...")
    nn_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        sc = joblib.load(f"ghani/models/v7/nn/scaler_f{f}.joblib")
        Xnn = transform_nn_test_v7(X_te, sc)
        m = TabularResNetV7(Xnn.shape[1], n_classes=K, hidden=320, n_blocks=3, dropout=0.18)
        m.load_state_dict(torch.load(f"ghani/models/v7/nn/resnet_f{f}.pt", map_location="cpu", weights_only=True))
        m.eval()
        with torch.no_grad():
            nn_te += np.vstack([m(torch.from_numpy(Xnn[s : s + 16384])).numpy() for s in range(0, N_te, 16384)]).astype(np.float32) / 5.0
        del Xnn, m
        gc.collect()
    outputs["ResNetV7"] = (nn_te, True)

    # 6. LGB_ET (if in ensemble)
    if "LGB_ET" in meta["models"]:
        log("Stage 6/7: Running 5-Fold LGB_ExtraTrees inference ...")
        X_et = X_te[[c for c in feats if c != "cu_loo_n"]]
        et_te = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            bst = lgb.Booster(model_file=f"ghani/models/v8/lgb_et/lgb_f{f}.txt")
            et_te += bst.predict(X_et, raw_score=True).astype(np.float32) / 5.0
        outputs["LGB_ET"] = (et_te, True)
        del X_et
        gc.collect()

    # 7. RankGaussGatedResNet (if in ensemble)
    if "RGGatedNN" in meta["models"]:
        log("Stage 7/7: Running 5-Fold RankGaussGatedResNet (SWA) inference ...")
        rg_te = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            sc = joblib.load(f"ghani/models/v8/rg_nn/scaler_f{f}.joblib")
            Xrg = transform_rg_test_v8(X_te, sc, r_te_all)
            m = RankGaussGatedResNet(Xrg.shape[1], n_classes=K, hidden=288, n_blocks=3, dropout=0.20)
            m.load_state_dict(torch.load(f"ghani/models/v8/rg_nn/rg_resnet_f{f}.pt", map_location="cpu", weights_only=True))
            m.eval()
            with torch.no_grad():
                rg_te += np.vstack([m(torch.from_numpy(Xrg[s : s + 16384])).numpy() for s in range(0, N_te, 16384)]).astype(np.float32) / 5.0
            del Xrg, m
            gc.collect()
        outputs["RGGatedNN"] = (rg_te, True)

    del X_te
    gc.collect()

    # 8. Message-Passing Meta-Stacker + De-Aliased u_cust_v8 + EUM
    log("Combining all models via Message-Passing Stacker + De-Aliased Customer GLMM + EUM ...")
    sb = joblib.load("ghani/models/v8/meta_stacker_v8.joblib")
    wp = sb["prob_weights"]
    te_ps, te_ls = [], []
    for name in sb["model_names"]:
        arr, il = outputs[name]
        if il:
            p = glmm.softmax(arr).astype(np.float32)
            l = (arr - arr.mean(1, keepdims=True)).astype(np.float32)
        else:
            p = arr.astype(np.float32)
            l = np.log(np.clip(arr, 1e-7, 1.0)).astype(np.float32)
            l -= l.mean(1, keepdims=True)
        te_ps.append(p); te_ls.append(l)

    ens_p_te = sum(wp[i] * te_ps[i] for i in range(len(wp))).astype(np.float32)
    ens_z_te = np.log(np.clip(ens_p_te, 1e-7, 1.0)); ens_z_te -= ens_z_te.mean(1, keepdims=True)
    mp_te = build_sibling_message_features(te_ids, ens_z_te, ens_p_te)
    X_stack_te = np.hstack(te_ls + [mp_te]).astype(np.float32)

    stack_te = np.zeros((N_te, K), dtype=np.float32)
    for b in sb["bundles"]:
        stack_te += b["clf"].predict_proba(b["scaler"].transform(X_stack_te)).astype(np.float32) / len(sb["bundles"])

    base_te = (0.35 * ens_p_te + 0.65 * stack_te).astype(np.float32)
    z_base_te = np.log(np.clip(base_te, 1e-7, 1.0)); z_base_te -= z_base_te.mean(1, keepdims=True)

    st_cust = np.load("ghani/models/v8/glmm_cust_v8_full.npy", allow_pickle=True).item()
    u_cust_te = (st_cust["U"]["customer"][c_te_code] + st_cust["bias"][None, :]).astype(np.float32)
    cls_safety = np.array(meta["cls_safety_vec"], dtype=np.float32)
    final_te = glmm.softmax(z_base_te + u_cust_te * cls_safety[None, :]).astype(np.float32)

    pred_idx = apply_eum(
        final_te,
        np.array(meta["eum_w"], dtype=np.float64),
        np.array(meta["eum_gamma"], dtype=np.float64),
        np.array(meta["eum_bias"], dtype=np.float64),
    )

    sub = pd.DataFrame({"shipment_id": te_sid.values, "status_penanganan": [classes[i] for i in pred_idx]})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)

    assert len(sub) == 362245 and sub["status_penanganan"].isna().sum() == 0
    ss = pd.read_csv("Task-1-Datavidia-11.0/sample_submission.csv", usecols=["shipment_id"])
    assert (sub["shipment_id"].values == ss["shipment_id"].values).all()
    log(f"SAVED {output_path} | rows={len(sub):,} | dist={sub['status_penanganan'].value_counts(normalize=True).round(4).to_dict()}")

    if verify_path and os.path.exists(verify_path):
        ref = pd.read_csv(verify_path)
        agree = float((sub["status_penanganan"].values == ref["status_penanganan"].values).mean())
        log(f"VERIFICATION vs {verify_path}: exact label agreement = {agree * 100:.4f}% ({int((sub['status_penanganan'].values == ref['status_penanganan'].values).sum()):,}/{len(sub):,})")

    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="ghani/cache/v8/submission_v8_reproduced_check.csv")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--verify-against", default="submissions/submission_v8_ghani.csv")
    args = ap.parse_args()
    run_inference_v8(output_path=args.output, rebuild_features=args.rebuild_features, verify_path=args.verify_against)


if __name__ == "__main__":
    main()
