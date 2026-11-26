"""Standalone Zero-Retraining Inference Script for Task-1 Datavidia 11.0 (v6 Super-Ensemble).

Loads ALL saved models from ghani/task1/models/ and generates predictions for all 362,245 test rows
WITHOUT retraining any Stage-1, GLMM, Stage-2, or Meta-Stacker model:
  1. Stage-1 5-Fold LightGBM             (ghani/task1/models/v5/stage1/lgb_f0..4.txt)
  2. Full-Train Joint Hierarchical GLMM  (ghani/task1/models/v6/glmm_state_full.npy)
  3. Stage-2 5-Fold Residual LightGBM    (ghani/task1/models/v6/lgb_res/lgb_f0..4.txt)
  4. Stage-2 5-Fold Full LightGBM        (ghani/task1/models/v6/lgb_full/lgb_f0..4.txt)
  5. Stage-2 5-Fold Residual XGBoost     (ghani/task1/models/v6/xgb/xgb_f0..4.ubj)
  6. Stage-2 5-Fold Tabular ResNet       (ghani/task1/models/v6/nn/resnet_f0..4.pt + scaler_f0..4.joblib)
  7. Stage-2 5-Fold RankGauss ResNet     (ghani/task1/models/v6/nn2/rg_resnet_f0..4.pt + scaler_f0..4.joblib)
  8. 5-Fold L2 Logistic Meta-Stacker     (ghani/task1/models/v6/meta_stacker.joblib)
  9. Hybrid Prob+Stacker & Multipliers   (ghani/task1/models/v6/meta_v6_ensemble.json)

Penggunaan (dari root repo):
  python -u -W ignore ghani/task1/src/infer_v6.py
  python -u -W ignore ghani/task1/src/infer_v6.py --rebuild-features --output submissions/task1/submission_v6_reproduced_ghani.csv
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

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import build_features_v5 as bf
import glmm_lib as glmm
from build_stage2_glmm import extract_glmm_block
from stage2_common import attach_glmm_df
from train_stage2_nn import TabularResNet
from train_stage2_nn2 import RankGaussResNet

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def load_or_build_test_features(spec, rebuild=False):
    cache_te = "ghani/task1/cache/v5/feat_v5_test.pkl"
    if not rebuild and os.path.exists(cache_te):
        log(f"Loading cached test features from {cache_te} ...")
        te = pd.read_pickle(cache_te)
    else:
        log("Rebuilding features directly from raw Task-1-Datavidia-11.0/ CSVs (unsupervised profiles, zero target labels)...")
        tr_raw, moda_ref = bf.build_split("train", spec.get("moda_ref_speed"))
        te_raw, _ = bf.build_split("test", moda_ref)
        tr_raw, te_raw = bf.add_unsupervised_entity_profiles(tr_raw, te_raw)
        # Replicate exact factorized entity ID codes & categoricals
        for eid in ["customer_id", "courier_id", "hub_id", "package_id"]:
            all_v = pd.concat([tr_raw[eid], te_raw[eid]]).astype(str)
            codes, _ = pd.factorize(all_v, sort=False)
            te_raw[eid + "_code"] = codes[len(tr_raw):].astype(np.int32)
        del tr_raw
        gc.collect()
        te = te_raw

    # Enforce exact categorical levels saved in feature_spec.json
    for c, cats in spec["cat_categories"].items():
        if c in te.columns:
            te[c] = pd.Categorical(te[c].astype(str), categories=cats)
    log(f"Test feature table ready: {te.shape}")
    return te


def build_stage2_base_test(te, spec):
    df = te[spec["stage2_keep_base"]].copy()
    df["co_p_q_x_dist"] = ((5.0 - df["co_p_scan_q_mean"]) * df["log_distance_km"]).astype(np.float32)
    df["co_p_gap_x_peak"] = (df["co_p_scan_gap_mean"] * (1.0 + df["is_peak_month"])).astype(np.float32)
    df["hu_p_gap_x_peak"] = (df["hu_p_scan_gap_mean"] * (1.0 + df["is_peak_month"])).astype(np.float32)
    df["comp_x_has_klaim"] = (df["cu_complaint_index"] * (1.0 + 2.0 * df["has_klaim_inv"])).astype(np.float32)
    df["comp_x_cod"] = (df["cu_complaint_index"] * (1.0 + df["is_cod"])).astype(np.float32)
    df["comp_over_credit"] = (df["cu_complaint_index"] / (df["cu_credit_score"].fillna(560.0) / 560.0)).astype(np.float32)
    df["bad_addr_x_upaya"] = ((6.0 - df["cu_address_quality"]) * (1.0 + df["scan_cnt_upaya"])).astype(np.float32)
    df["bad_addr_x_catatan"] = ((6.0 - df["cu_address_quality"]) * (1.0 + df["has_catatan"])).astype(np.float32)
    df["co_p_q_x_fragile"] = ((5.0 - df["co_p_scan_q_mean"]) * df["sp_fragile_score"]).astype(np.float32)
    df["co_p_q_x_night"] = ((5.0 - df["co_p_scan_q_mean"]) * (1.0 + df["is_night_ship"])).astype(np.float32)
    n_other = np.maximum(df["cu_loo_n"].to_numpy(np.float32) - 1.0, 0.0)
    df["cu_loo_sum_upaya"] = (df["cu_loo_scan_cnt_upaya"].fillna(0.0).to_numpy(np.float32) * n_other).astype(np.float32)
    df["cu_loo_sum_klaim"] = (df["cu_loo_has_klaim_inv"].fillna(0.0).to_numpy(np.float32) * n_other).astype(np.float32)
    df["cu_loo_sum_catatan"] = (df["cu_loo_has_catatan"].fillna(0.0).to_numpy(np.float32) * n_other).astype(np.float32)
    df["cu_loo_sum_no_sel"] = (df["cu_loo_no_selesai"].fillna(0.0).to_numpy(np.float32) * n_other).astype(np.float32)
    return df


def transform_for_nn(df_te, scaler, kind="resnet"):
    num_cols = scaler["num_cols"]
    Xte = df_te[num_cols].to_numpy(dtype=np.float32, copy=True)
    med = scaler["med"]
    na = np.isnan(Xte)
    if na.any():
        Xte[na] = np.take(med, np.where(na)[1])

    if kind == "resnet":
        heavy = scaler["heavy"]
        if heavy.any():
            Xte[:, heavy] = np.sign(Xte[:, heavy]) * np.log1p(np.abs(Xte[:, heavy]))
        Xte -= scaler["mu"]
        Xte /= scaler["sd"]
        np.clip(Xte, -8.0, 8.0, out=Xte)
    else:
        Xte = scaler["qt"].transform(Xte).astype(np.float32)
        np.clip(Xte, -5.0, 5.0, out=Xte)

    cat_cols = scaler["cat_cols"]
    if cat_cols:
        oh = []
        for c in cat_cols:
            cats = scaler["cat_cats"][c]
            codes = pd.Categorical(df_te[c].astype(str), categories=cats).codes
            mat = np.zeros((len(codes), len(cats)), dtype=np.float32)
            ok = codes >= 0
            mat[np.where(ok)[0], codes[ok]] = 1.0
            oh.append(mat)
        Xte = np.hstack([Xte] + oh)
    return Xte


def run_inference(output_path="submissions/task1/submission_v6_inferred_ghani.csv", rebuild_features=False, verify_path=None):
    torch.set_num_threads(os.cpu_count() or 8)
    with open("ghani/task1/models/v6/feature_spec.json", encoding="utf-8") as fh:
        spec = json.load(fh)
    with open("ghani/task1/models/v6/meta_v6_ensemble.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    classes = meta["classes"]
    K = len(classes)

    # 1. Load or build test feature matrix
    te = load_or_build_test_features(spec, rebuild=rebuild_features)
    te_sid = te["shipment_id"].copy()
    N_te = len(te)

    # Extract entity codes for GLMM
    ents_te = {
        "hub": te["hub_id_code"].to_numpy(dtype=np.int32),
        "courier": te["courier_id_code"].to_numpy(dtype=np.int32),
        "customer": te["customer_id_code"].to_numpy(dtype=np.int32),
        "hub_peak": (te["hub_id_code"].to_numpy(dtype=np.int32) * 2 + te["is_peak_month"].to_numpy(dtype=np.int32)).astype(np.int32),
    }

    # 2. Stage-1 5-Fold LightGBM inference
    log("Stage 1/7: Running 5-Fold Stage-1 LightGBM inference ...")
    X_s1 = te[spec["stage1_features"]]
    s1_te_logits = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/task1/models/v5/stage1/lgb_f{f}.txt")
        s1_te_logits += bst.predict(X_s1, raw_score=True).astype(np.float32) / 5.0
        del bst
    del X_s1
    gc.collect()

    # 3. Joint Hierarchical GLMM inference (from saved full-train state)
    log("Stage 2/7: Extracting 49-col Joint Hierarchical GLMM block from saved state ...")
    st_full = np.load("ghani/task1/models/v6/glmm_state_full.npy", allow_pickle=True).item()
    te_block, glmm_te_eta, glmm_te_p = extract_glmm_block(s1_te_logits, ents_te, st_full, n_classes=K)
    del st_full, ents_te

    # Build Stage-2 DataFrame
    te_base = build_stage2_base_test(te, spec)
    del te
    gc.collect()
    X_te_s2 = attach_glmm_df(te_base, te_block)
    del te_base
    gc.collect()
    s1_te_blk = te_block[:, 0:7].astype(np.float32)
    base_te_glmm = te_block[:, 7:14].astype(np.float32)
    log(f"Stage-2 test matrix ready: {X_te_s2.shape}")

    # 4. Stage-2 LGB_res (5 folds)
    log("Stage 3/7: Running 5-Fold Stage-2 LGB_res inference ...")
    lgb_res_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/task1/models/v6/lgb_res/lgb_f{f}.txt")
        pred = bst.predict(X_te_s2, raw_score=True).astype(np.float32) + base_te_glmm
        lgb_res_te += pred / 5.0
        del bst

    # 5. Stage-2 LGB_full (5 folds)
    log("Stage 4/7: Running 5-Fold Stage-2 LGB_full inference ...")
    lgb_full_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/task1/models/v6/lgb_full/lgb_f{f}.txt")
        lgb_full_te += bst.predict(X_te_s2, raw_score=True).astype(np.float32) / 5.0
        del bst

    # 6. Stage-2 XGBoost (5 folds)
    log("Stage 5/7: Running 5-Fold Stage-2 XGBoost inference ...")
    dte_xgb = xgb.DMatrix(X_te_s2, base_margin=base_te_glmm, enable_categorical=True)
    xgb_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = xgb.Booster(model_file=f"ghani/task1/models/v6/xgb/xgb_f{f}.ubj")
        bi = spec["xgb_best_iters"][f]
        xgb_te += bst.predict(dte_xgb, output_margin=True, iteration_range=(0, bi)).astype(np.float32) / 5.0
        del bst
    del dte_xgb
    gc.collect()

    # 7. Stage-2 TabularResNet (5 folds)
    log("Stage 6/7: Running 5-Fold Stage-2 TabularResNet (PyTorch) inference ...")
    nn_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        scaler = joblib.load(f"ghani/task1/models/v6/nn/scaler_f{f}.joblib")
        Xte_nn = transform_for_nn(X_te_s2, scaler, kind="resnet")
        model = TabularResNet(Xte_nn.shape[1], n_classes=K, hidden=256, n_blocks=2, dropout=0.20)
        state = torch.load(f"ghani/task1/models/v6/nn/resnet_f{f}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        chunks = []
        with torch.no_grad():
            for s in range(0, N_te, 16384):
                xb = torch.from_numpy(Xte_nn[s : s + 16384])
                gb = torch.from_numpy(base_te_glmm[s : s + 16384])
                chunks.append(model(xb, gb).numpy())
        nn_te += np.vstack(chunks).astype(np.float32) / 5.0
        del Xte_nn, model, state
        gc.collect()

    # 8. Stage-2 RankGaussResNet (5 folds)
    log("Stage 7/7: Running 5-Fold Stage-2 RankGaussResNet (PyTorch) inference ...")
    nn2_te = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        scaler = joblib.load(f"ghani/task1/models/v6/nn2/scaler_f{f}.joblib")
        Xte_rg = transform_for_nn(X_te_s2, scaler, kind="rgresnet")
        model = RankGaussResNet(Xte_rg.shape[1], n_classes=K, hidden=256, n_blocks=2, dropout=0.20)
        state = torch.load(f"ghani/task1/models/v6/nn2/rg_resnet_f{f}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        chunks = []
        with torch.no_grad():
            for s in range(0, N_te, 16384):
                xb = torch.from_numpy(Xte_rg[s : s + 16384])
                gb = torch.from_numpy(base_te_glmm[s : s + 16384])
                sb = torch.from_numpy(s1_te_blk[s : s + 16384])
                chunks.append(model(xb, gb, sb).numpy())
        nn2_te += np.vstack(chunks).astype(np.float32) / 5.0
        del Xte_rg, model, state
        gc.collect()

    del X_te_s2
    gc.collect()

    # 9. Hybrid Prob Blend (50%) + 5-Fold L2 Meta-Stacker (50%) + Optimal Class Multipliers
    log("Combining 6 models via Hybrid Prob Blend + Saved 5-Fold L2 Meta-Stacker ...")
    model_outputs = [
        ("GLMM",     glmm_te_p,   False),
        ("LGB_res",  lgb_res_te,  True),
        ("LGB_full", lgb_full_te, True),
        ("XGBoost",  xgb_te,      True),
        ("ResNet",   nn_te,       True),
        ("RGResNet", nn2_te,      True),
    ]
    prob_weights = meta["blend_info"]["prob_weights"]
    ens_p_te = np.zeros((N_te, K), dtype=np.float32)
    te_ls = []
    for name, arr, is_logit in model_outputs:
        if is_logit:
            p = glmm.softmax(arr).astype(np.float32)
            l = (arr - arr.mean(axis=1, keepdims=True)).astype(np.float32)
        else:
            p = arr.astype(np.float32)
            l = np.log(np.clip(arr, 1e-7, 1.0)).astype(np.float32)
            l -= l.mean(axis=1, keepdims=True)
        ens_p_te += float(prob_weights[name]) * p
        te_ls.append(l)

    stack_bundle = joblib.load("ghani/task1/models/v6/meta_stacker.joblib")
    X_meta_te = np.hstack(te_ls).astype(np.float32)
    meta_te = np.zeros((N_te, K), dtype=np.float32)
    for clf in stack_bundle["models"]:
        meta_te += clf.predict_proba(X_meta_te).astype(np.float32) / len(stack_bundle["models"])

    hyb_te = (0.5 * ens_p_te + 0.5 * meta_te).astype(np.float32)
    w_cls = np.array(meta["class_multipliers"], dtype=np.float32)
    pred_idx = (hyb_te * w_cls).argmax(axis=1)

    sub = pd.DataFrame({
        "shipment_id": te_sid.values,
        "status_penanganan": [classes[i] for i in pred_idx],
    })
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)

    # Mandatory integrity checks
    assert len(sub) == 362245, f"Expected 362245 rows, got {len(sub)}"
    assert sub["status_penanganan"].isna().sum() == 0, "NaNs found!"
    assert set(sub["status_penanganan"].unique()).issubset(set(classes)), "Unknown class labels!"
    sample_path = "Task-1-Datavidia-11.0/sample_submission.csv"
    if os.path.exists(sample_path):
        ss = pd.read_csv(sample_path, usecols=["shipment_id"])
        assert (sub["shipment_id"].values == ss["shipment_id"].values).all(), "shipment_id order mismatch!"

    dist = sub["status_penanganan"].value_counts(normalize=True).round(4).to_dict()
    log(f"SAVED {output_path} | rows={len(sub):,} | dist={dist}")

    # Optional verification against reference submission
    if verify_path and os.path.exists(verify_path):
        ref = pd.read_csv(verify_path)
        agree = float((sub["status_penanganan"].values == ref["status_penanganan"].values).mean())
        log(f"VERIFICATION vs {verify_path}: exact label agreement = {agree * 100:.3f}% ({int((sub['status_penanganan'].values == ref['status_penanganan'].values).sum()):,}/{len(sub):,})")
        # Per-component drift diagnostics if cached logits exist
        checks = [
            ("Stage1_LGB", s1_te_logits, "ghani/task1/cache/v5/stage1/test_logits_all.npy"),
            ("GLMM_proba", glmm_te_p,    "ghani/task1/cache/v6/glmm_test_proba.npy"),
            ("LGB_res",    lgb_res_te,   "ghani/task1/cache/v6/lgb_res_test_logits.npy"),
            ("LGB_full",   lgb_full_te,  "ghani/task1/cache/v6/lgb_full_test_logits.npy"),
            ("XGBoost",    xgb_te,       "ghani/task1/cache/v6/xgb_test_logits.npy"),
            ("ResNet",     nn_te,        "ghani/task1/cache/v6/nn_test_logits.npy"),
            ("RGResNet",   nn2_te,       "ghani/task1/cache/v6/nn2_test_logits.npy"),
        ]
        for cname, arr, cpath in checks:
            if os.path.exists(cpath):
                ref_a = np.load(cpath)
                diff = np.abs(arr - ref_a)
                log(f"  [{cname:10s}] mean_abs_diff={diff.mean():.6f} | max_abs_diff={diff.max():.6f}")

    return sub


def main():
    ap = argparse.ArgumentParser(description="Zero-retraining inference for Datavidia Task-1 v6 Super-Ensemble")
    ap.add_argument("--output", default="submissions/task1/submission_v6_inferred_ghani.csv", help="Output CSV path")
    ap.add_argument("--rebuild-features", action="store_true", help="Rebuild features from raw Task-1-Datavidia-11.0 CSVs")
    ap.add_argument("--verify-against", default="submissions/task1/submission_v6_ghani.csv", help="Reference CSV for agreement check")
    args = ap.parse_args()
    run_inference(output_path=args.output, rebuild_features=args.rebuild_features, verify_path=args.verify_against)


if __name__ == "__main__":
    main()
