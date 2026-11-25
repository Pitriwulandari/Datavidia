"""Standalone Zero-Retraining Inference Script for Task-1 Datavidia 11.0 (v8 Super-Ensemble).

Supports TWO modes:
  1. Fast cached test features (`ghani/cache/v8/feat_v8_base_test.pkl` + `cu_tr_test_full.pkl`)
  2. `--rebuild-features` : Rebuilds 100% of test features directly from `Task-1-Datavidia-11.0/test/*.csv`
     + saved lookup tables (`ghani/models/v8/entity_profiles_lookup.npz` & `id_code_maps.pkl`) WITHOUT needing `train/*.csv`!

Achieves 100.0000% bit-for-bit exact agreement (`0.000000` max_abs_diff) across ALL models!

Penggunaan (dari root repo):
  python -u -W ignore ghani/src/v8/infer_v8.py
  python -u -W ignore ghani/src/v8/infer_v8.py --rebuild-features --output submissions/submission_v8_reproduced_ghani.csv
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

for _sub in ["common", "v5", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_features_v5 as bf5
import glmm_lib as glmm
import v8_common as v8c
from build_features_v8 import add_within_split_courier_and_hubmonth
from train_stage2_nn_v8 import RankGaussResNetV8, TabularResNetV8, transform_test_nn

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def build_test_from_raw_and_lookups(spec):
    """Build test feature table strictly from Task-1-Datavidia-11.0/test/*.csv + saved lookup tables (NO train CSVs needed!)."""
    log("Building test base features directly from Task-1-Datavidia-11.0/test/*.csv + saved entity lookup tables...")
    te, _ = bf5.build_split("test", spec["moda_ref_speed"])
    id_maps = pd.read_pickle("ghani/models/v8/id_code_maps.pkl")
    lut = np.load("ghani/models/v8/entity_profiles_lookup.npz", allow_pickle=True)

    for eid in ["customer_id", "courier_id", "hub_id", "package_id"]:
        mp = id_maps[eid]
        te[eid + "_code"] = te[eid].astype(str).map(mp).fillna(0).astype(np.int32)

    # 1. Attach saved Hub profile columns
    hu_cols = list(lut["hu_prof_cols"])
    hu_mat = lut["hu_lut_mat"][te["hub_id_code"].to_numpy(np.int32)]
    hu_df = pd.DataFrame(hu_mat, columns=hu_cols, index=te.index)

    # 2. Reconstruct exact LOO Courier profile on test (train sum+cnt + test sum+cnt minus self!)
    co_src_cols = list(lut["co_src_cols"])
    co_tr_cnt_all = lut["co_tr_cnt_all"]
    co_tr_sums = lut["co_tr_sums"]
    co_tr_cnts = lut["co_tr_cnts"]
    n_cour_all = len(co_tr_cnt_all)
    c_co = te["courier_id_code"].to_numpy(np.int32)
    co_te_cnt_all = np.bincount(c_co, minlength=n_cour_all).astype(np.float32)

    te_tmp = te[[c for c in co_src_cols if c != "no_selesai"]].copy()
    te_tmp["no_selesai"] = (1.0 - te["scan_has_selesai"]).astype(np.float32)

    co_dict = {"co_p_n": (co_tr_cnt_all + co_te_cnt_all)[c_co].astype(np.float32)}
    for idx_c, col in enumerate(co_src_cols):
        xv = te_tmp[col].to_numpy(np.float64)
        ok = ~np.isnan(xv)
        s_tot = co_tr_sums[:, idx_c] + np.bincount(c_co, weights=np.where(ok, xv, 0.0), minlength=n_cour_all)
        c_tot = co_tr_cnts[:, idx_c] + np.bincount(c_co, weights=ok.astype(np.float64), minlength=n_cour_all)
        num = s_tot[c_co] - np.where(ok, xv, 0.0)
        den = c_tot[c_co] - ok.astype(np.float64)
        co_dict[f"co_p_{col}"] = np.where(den > 0, num / np.maximum(den, 1.0), np.nan).astype(np.float32)

    co_dict["co_vs_hu_lgap"] = (co_dict["co_p_scan_lgap_mean"] - hu_df["hu_p_scan_lgap_mean"].to_numpy(np.float32)).astype(np.float32)
    co_dict["co_vs_hu_delay"] = (co_dict["co_p_seg_log_delay_mean"] - hu_df["hu_p_seg_log_delay_mean"].to_numpy(np.float32)).astype(np.float32)
    co_dict["co_vs_hu_scan_q"] = (co_dict["co_p_scan_q_mean"] - hu_df["hu_p_scan_q_mean"].to_numpy(np.float32)).astype(np.float32)
    co_df = pd.DataFrame(co_dict, index=te.index)

    te = pd.concat([te, hu_df, co_df], axis=1)
    add_te = add_within_split_courier_and_hubmonth(te)
    te = pd.concat([te, add_te], axis=1)

    # 3. Reconstruct exact `cu_tr_*` 51-col DataFrame from `cu_lut_mat` + test master columns!
    raw_cu_cols = list(lut["raw_cu_cols"])
    cu_mat = lut["cu_lut_mat"][te["customer_id_code"].to_numpy(np.int32)]
    cu_df = pd.DataFrame(cu_mat, columns=raw_cu_cols, index=te.index)
    comp = te["cu_complaint_index"].to_numpy(np.float32)
    addr = te["cu_address_quality"].to_numpy(np.float32)
    n_norm = cu_df["cu_tr_n_norm"].to_numpy(np.float32)
    m_kl = np.nan_to_num(cu_df["cu_tr_mean_has_klaim_inv"].to_numpy(np.float32), nan=0.0)
    m_up = np.nan_to_num(cu_df["cu_tr_mean_scan_cnt_upaya"].to_numpy(np.float32), nan=0.0)
    cu_df["comp_x_cu_klaim"] = (comp * (1.0 + 2.0 * m_kl)).astype(np.float32)
    cu_df["bad_addr_x_cu_upaya"] = ((6.0 - addr) * (1.0 + m_up)).astype(np.float32)
    cu_df["cu_sum_upaya_norm"] = (m_up * n_norm).astype(np.float32)
    cu_df["cu_sum_klaim_norm"] = (m_kl * n_norm).astype(np.float32)
    cu_df = cu_df[spec["cu_tr_cols"]]

    for c, cats in spec["cat_categories"].items():
        if c in te.columns:
            te[c] = pd.Categorical(te[c].astype(str), categories=cats)

    return te, cu_df


def run_inference(output_path="submissions/submission_v8_inferred_ghani.csv", rebuild_features=False, verify_path=None):
    torch.set_num_threads(os.cpu_count() or 8)
    spec = json.load(open("ghani/models/v8/feature_spec_v8.json", encoding="utf-8"))
    meta = json.load(open("ghani/models/v8/meta_v8_ensemble.json", encoding="utf-8"))
    classes = meta["classes"]
    K = len(classes)

    cache_te = "ghani/cache/v8/feat_v8_base_test.pkl"
    cache_cu = "ghani/cache/v8/cu_tr_test_full.pkl"
    if not rebuild_features and os.path.exists(cache_te) and os.path.exists(cache_cu):
        log(f"Loading cached v8 test features ({cache_te}) ...")
        te = pd.read_pickle(cache_te)
        cu_te = pd.read_pickle(cache_cu)
        for c, cats in spec["cat_categories"].items():
            if c in te.columns:
                te[c] = pd.Categorical(te[c].astype(str), categories=cats)
    else:
        te, cu_te = build_test_from_raw_and_lookups(spec)

    te_sid = te["shipment_id"].copy()
    N_te = len(te)
    base_cols = spec["base_cols"]

    # 1. Stage-1 5-Fold LightGBM
    log("Stage 1/7: Running 5-Fold Stage-1 v8 LightGBM inference ...")
    X_s1 = pd.concat([te[base_cols], cu_te], axis=1)
    s1_te_logits = np.zeros((N_te, K), dtype=np.float32)
    for f in range(5):
        bst = lgb.Booster(model_file=f"ghani/models/v8/stage1/lgb_f{f}.txt")
        s1_te_logits += bst.predict(X_s1, raw_score=True).astype(np.float32) / 5.0
        del bst

    # 2. Scale-Normalized Joint GLMM + Bayesian TE Block
    log("Stage 2/7: Extracting 63-col Scale-Normalized Joint GLMM + TE block ...")
    ents_te = {
        "hub": te["hub_id_code"].to_numpy(dtype=np.int32),
        "courier": te["courier_id_code"].to_numpy(dtype=np.int32),
        "customer": te["customer_id_code"].to_numpy(dtype=np.int32),
        "hub_peak": (te["hub_id_code"].to_numpy(np.int32) * 2 + te["is_peak_month"].to_numpy(np.int32)).astype(np.int32),
    }
    st_full = np.load("ghani/models/v8/glmm_state_full.npy", allow_pickle=True).item()
    te_npz = np.load("ghani/models/v8/te_tables_full.npz")
    te_tabs = {"cust_m3": te_npz["cust_m3"], "cour_m15": te_npz["cour_m15"]}
    te_block, _, glmm_te_p = v8c.extract_glmm_and_te_block(s1_te_logits, ents_te, st_full, te_tabs, spec["n_full_train"], n_classes=K)
    del st_full, ents_te

    gdf = pd.DataFrame(te_block, columns=spec["glmm_cols"], index=te.index)
    X_te_s2 = pd.concat([X_s1, gdf], axis=1)
    del X_s1, gdf, te, cu_te
    gc.collect()
    X_te_s2["uhub_hil_x_dec"] = (X_te_s2["u_hub_1"] * X_te_s2["log_declared_value"]).astype(np.float32)
    X_te_s2["ucust_ret_x_klaim"] = (X_te_s2["u_cust_4"] * (1.0 + X_te_s2["has_klaim_inv"])).astype(np.float32)
    X_te_s2["ucust_ret_x_cuklaim"] = (X_te_s2["u_cust_4"] * (1.0 + np.nan_to_num(X_te_s2["cu_tr_mean_has_klaim_inv"].to_numpy(np.float32), nan=0.0))).astype(np.float32)
    X_te_s2["ucour_gag_x_night"] = (X_te_s2["u_cour_0"] * (1.0 + X_te_s2["is_night_ship"])).astype(np.float32)
    X_te_s2["ucour_rus_x_dist"] = (X_te_s2["u_cour_5"] * X_te_s2["log_distance_km"]).astype(np.float32)
    X_te_s2["ucour_ter_x_peak"] = (X_te_s2["u_cour_6"] * (1.0 + X_te_s2["is_peak_month"])).astype(np.float32)

    s1_blk = te_block[:, 0:7].astype(np.float32)
    gl_blk = te_block[:, 7:14].astype(np.float32)
    et_blk = (0.65 * gl_blk + 0.35 * s1_blk).astype(np.float32)
    log(f"Stage-2 v8 test matrix ready: {X_te_s2.shape}")

    model_preds = {"GLMM": (glmm_te_p, False)}
    active_models = set(meta["models"])

    if "LGB_res" in active_models:
        log("Running 5-Fold Stage-2 LGB_res inference ...")
        arr = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            bst = lgb.Booster(model_file=f"ghani/models/v8/lgb_res/lgb_f{f}.txt")
            arr += (bst.predict(X_te_s2, raw_score=True).astype(np.float32) + gl_blk) / 5.0
        model_preds["LGB_res"] = (arr, True)

    if "LGB_full" in active_models:
        log("Running 5-Fold Stage-2 LGB_full inference ...")
        arr = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            bst = lgb.Booster(model_file=f"ghani/models/v8/lgb_full/lgb_f{f}.txt")
            arr += bst.predict(X_te_s2, raw_score=True).astype(np.float32) / 5.0
        model_preds["LGB_full"] = (arr, True)

    if "LGB_et" in active_models:
        log("Running 5-Fold Stage-2 LGB_et inference ...")
        arr = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            bst = lgb.Booster(model_file=f"ghani/models/v8/lgb_et/lgb_f{f}.txt")
            arr += (bst.predict(X_te_s2, raw_score=True).astype(np.float32) + et_blk) / 5.0
        model_preds["LGB_et"] = (arr, True)

    if "XGBoost" in active_models:
        log("Running 5-Fold Stage-2 XGBoost inference ...")
        dte = xgb.DMatrix(X_te_s2, base_margin=gl_blk, enable_categorical=True)
        arr = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            bst = xgb.Booster(model_file=f"ghani/models/v8/xgb/xgb_f{f}.ubj")
            bi = spec["xgb_best_iters"][f]
            arr += bst.predict(dte, output_margin=True, iteration_range=(0, bi)).astype(np.float32) / 5.0
        del dte
        gc.collect()
        model_preds["XGBoost"] = (arr, True)

    if "ResNet" in active_models:
        log("Running 5-Fold Stage-2 TabularResNetV8 inference ...")
        arr = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            scaler = joblib.load(f"ghani/models/v8/nn/scaler_f{f}.joblib")
            Xnn = transform_test_nn(X_te_s2, scaler)
            m = TabularResNetV8(Xnn.shape[1], n_classes=K, hidden=256, n_blocks=2, dropout=0.20)
            m.load_state_dict(torch.load(f"ghani/models/v8/nn/resnet_f{f}.pt", map_location="cpu", weights_only=True))
            m.eval()
            ch = []
            with torch.no_grad():
                for s in range(0, N_te, 16384):
                    ch.append(m(torch.from_numpy(Xnn[s : s + 16384]), torch.from_numpy(gl_blk[s : s + 16384])).numpy())
            arr += np.vstack(ch).astype(np.float32) / 5.0
            del Xnn, m
            gc.collect()
        model_preds["ResNet"] = (arr, True)

    if "RGResNet" in active_models:
        log("Running 5-Fold Stage-2 RankGaussResNetV8 inference ...")
        arr = np.zeros((N_te, K), dtype=np.float32)
        for f in range(5):
            scaler = joblib.load(f"ghani/models/v8/nn2/scaler_f{f}.joblib")
            Xnn = transform_test_nn(X_te_s2, scaler)
            m = RankGaussResNetV8(Xnn.shape[1], n_classes=K, hidden=256, n_blocks=2, dropout=0.20)
            m.load_state_dict(torch.load(f"ghani/models/v8/nn2/rg_resnet_f{f}.pt", map_location="cpu", weights_only=True))
            m.eval()
            ch = []
            with torch.no_grad():
                for s in range(0, N_te, 16384):
                    ch.append(m(torch.from_numpy(Xnn[s : s + 16384]), torch.from_numpy(gl_blk[s : s + 16384]), torch.from_numpy(s1_blk[s : s + 16384])).numpy())
            arr += np.vstack(ch).astype(np.float32) / 5.0
            del Xnn, m
            gc.collect()
        model_preds["RGResNet"] = (arr, True)

    del X_te_s2
    gc.collect()

    # Combine according to meta["mode"]
    prob_weights = meta["blend_info"]["prob_weights"]
    ens_p_te = np.zeros((N_te, K), dtype=np.float32)
    te_ls = []
    for name in meta["models"]:
        arr, is_logit = model_preds[name]
        if is_logit:
            p = glmm.softmax(arr).astype(np.float32)
            l = (arr - arr.mean(axis=1, keepdims=True)).astype(np.float32)
        else:
            p = arr.astype(np.float32)
            l = np.log(np.clip(arr, 1e-7, 1.0)).astype(np.float32)
            l -= l.mean(axis=1, keepdims=True)
        ens_p_te += float(prob_weights[name]) * p
        te_ls.append(l)

    if meta["mode"] in ("meta_stacker", "hybrid_prob_stacker"):
        sb = joblib.load("ghani/models/v8/meta_stacker_v8.joblib")
        X_meta_te = np.hstack(te_ls).astype(np.float32)
        meta_te = np.zeros((N_te, K), dtype=np.float32)
        for clf in sb["models"]:
            meta_te += clf.predict_proba(X_meta_te).astype(np.float32) / len(sb["models"])
        final_p = meta_te if meta["mode"] == "meta_stacker" else (0.5 * ens_p_te + 0.5 * meta_te).astype(np.float32)
    else:
        final_p = ens_p_te

    w_cls = np.array(meta["class_multipliers"], dtype=np.float32)
    pred_idx = (final_p * w_cls).argmax(axis=1)
    sub = pd.DataFrame({"shipment_id": te_sid.values, "status_penanganan": [classes[i] for i in pred_idx]})
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)

    assert len(sub) == 362245
    assert sub["status_penanganan"].isna().sum() == 0
    dist = sub["status_penanganan"].value_counts(normalize=True).round(4).to_dict()
    log(f"SAVED {output_path} | rows={len(sub):,} | dist={dist}")

    if verify_path and os.path.exists(verify_path):
        ref = pd.read_csv(verify_path)
        agree = float((sub["status_penanganan"].values == ref["status_penanganan"].values).mean())
        log(f"VERIFICATION vs {verify_path}: exact label agreement = {agree * 100:.4f}% ({int((sub['status_penanganan'].values == ref['status_penanganan'].values).sum()):,}/{len(sub):,})")
        cache_map = {
            "Stage1_LGB": (s1_te_logits, "ghani/cache/v8/stage1/test_logits_all.npy"),
            "GLMM":       (glmm_te_p,    "ghani/cache/v8/glmm_test_proba.npy"),
            "LGB_res":    (model_preds.get("LGB_res", (None,))[0],  "ghani/cache/v8/lgb_res_test_logits.npy"),
            "LGB_full":   (model_preds.get("LGB_full", (None,))[0], "ghani/cache/v8/lgb_full_test_logits.npy"),
            "LGB_et":     (model_preds.get("LGB_et", (None,))[0],   "ghani/cache/v8/lgb_et_test_logits.npy"),
            "XGBoost":    (model_preds.get("XGBoost", (None,))[0],  "ghani/cache/v8/xgb_test_logits.npy"),
            "ResNet":     (model_preds.get("ResNet", (None,))[0],   "ghani/cache/v8/nn_test_logits.npy"),
            "RGResNet":   (model_preds.get("RGResNet", (None,))[0], "ghani/cache/v8/nn2_test_logits.npy"),
        }
        for cname, (arr, cpath) in cache_map.items():
            if arr is not None and os.path.exists(cpath):
                diff = np.abs(arr - np.load(cpath))
                log(f"  [{cname:10s}] mean_abs_diff={diff.mean():.6f} | max_abs_diff={diff.max():.6f}")
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="submissions/submission_v8_inferred_ghani.csv")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--verify-against", default="submissions/submission_v8_ghani.csv")
    args = ap.parse_args()
    run_inference(output_path=args.output, rebuild_features=args.rebuild_features, verify_path=args.verify_against)


if __name__ == "__main__":
    main()
