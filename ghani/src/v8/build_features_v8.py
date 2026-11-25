"""Feature Engineering v8 (Task-1 Datavidia 11.0):
- Removes 100% of mixed train+test customer LOO features (`cu_loo_*`) and broken test-only `cu_sp_*`.
- Replaces drifted mixed-split `shp_vs_co_*` and `hm_p_*` with drift-free within-split versions.
- Exports compact entity lookup tables (`ghani/models/v8/entity_profiles_lookup.npz`) so `infer_v8.py`
  and `final_datavidia_task1_v8_ghani.ipynb` can run 100% zero-retraining inference from `test/*.csv` alone!

Jalankan dari root repo:
  python -u -W ignore ghani/src/v8/build_features_v8.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

for _sub in ["common", "v5", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_features_v5 as bf5
import v8_common as v8c

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def add_within_split_courier_and_hubmonth(df):
    """Compute drift-free within-split courier deviations and hub-month congestion profiles."""
    # 1. Within-split courier LOO deviations (~52 rows/courier in train, ~15 rows/courier in test)
    co_c, co_u = pd.factorize(df["courier_id"], sort=False)
    n_co = len(co_u)
    new_cols = {}
    for col, out_name in [
        ("scan_q_mean", "shp_vs_co_scan_q_sp"),
        ("scan_lgap_mean", "shp_vs_co_lgap_sp"),
        ("seg_log_delay_mean", "shp_vs_co_delay_sp"),
    ]:
        x = df[col].to_numpy(dtype=np.float64)
        ok = ~np.isnan(x)
        xv = np.where(ok, x, 0.0)
        s = np.bincount(co_c, weights=xv, minlength=n_co)
        c = np.bincount(co_c, weights=ok.astype(np.float64), minlength=n_co)
        den = c[co_c] - ok.astype(np.float64)
        co_loo = np.where(den > 0, (s[co_c] - xv) / np.maximum(den, 1.0), np.nan)
        new_cols[out_name] = (x - co_loo).astype(np.float32)

    # 2. Within-split Hub x Month normalized deviations (invariant to train vs test size & peak shift)
    hm_str = df["hub_id"].astype(str) + "_" + df["ship_month"].astype(str)
    hm_c, hm_u = pd.factorize(hm_str, sort=False)
    hu_c, hu_u = pd.factorize(df["hub_id"], sort=False)
    n_hm, n_hu = len(hm_u), len(hu_u)
    hm_cnt = np.bincount(hm_c, minlength=n_hm).astype(np.float32)
    hu_cnt = np.bincount(hu_c, minlength=n_hu).astype(np.float32)
    new_cols["hm_sp_n_ratio"] = (hm_cnt[hm_c] / (hu_cnt[hu_c] / 12.0 + 1e-6)).astype(np.float32)

    for col, short in [("scan_lgap_mean", "lgap"), ("seg_log_delay_mean", "delay"), ("inv_days_max", "invd")]:
        x = df[col].to_numpy(dtype=np.float64)
        ok = ~np.isnan(x)
        xv = np.where(ok, x, 0.0)
        s_hm = np.bincount(hm_c, weights=xv, minlength=n_hm)
        c_hm = np.bincount(hm_c, weights=ok.astype(np.float64), minlength=n_hm)
        den_hm = c_hm[hm_c] - ok.astype(np.float64)
        hm_loo = np.where(den_hm > 0, (s_hm[hm_c] - xv) / np.maximum(den_hm, 1.0), np.nan)

        s_hu = np.bincount(hu_c, weights=xv, minlength=n_hu)
        c_hu = np.bincount(hu_c, weights=ok.astype(np.float64), minlength=n_hu)
        hu_mean = np.where(c_hu[hu_c] > 0, s_hu[hu_c] / np.maximum(c_hu[hu_c], 1.0), np.nan)
        new_cols[f"hm_sp_{short}"] = hm_loo.astype(np.float32)
        if short in ("lgap", "delay"):
            new_cols[f"hm_sp_{short}_vs_hu"] = (hm_loo - hu_mean).astype(np.float32)

    # 3. Clean shipment/courier/hub domain interactions (zero customer leakage)
    new_cols["co_p_q_x_dist"] = ((5.0 - df["co_p_scan_q_mean"].to_numpy(np.float32)) * df["log_distance_km"].to_numpy(np.float32)).astype(np.float32)
    new_cols["co_p_gap_x_peak"] = (df["co_p_scan_gap_mean"].to_numpy(np.float32) * (1.0 + df["is_peak_month"].to_numpy(np.float32))).astype(np.float32)
    new_cols["hu_p_gap_x_peak"] = (df["hu_p_scan_gap_mean"].to_numpy(np.float32) * (1.0 + df["is_peak_month"].to_numpy(np.float32))).astype(np.float32)
    new_cols["comp_x_has_klaim"] = (df["cu_complaint_index"].to_numpy(np.float32) * (1.0 + 2.0 * df["has_klaim_inv"].to_numpy(np.float32))).astype(np.float32)
    new_cols["comp_x_cod"] = (df["cu_complaint_index"].to_numpy(np.float32) * (1.0 + df["is_cod"].to_numpy(np.float32))).astype(np.float32)
    new_cols["comp_over_credit"] = (df["cu_complaint_index"].to_numpy(np.float32) / (df["cu_credit_score"].fillna(560.0).to_numpy(np.float32) / 560.0)).astype(np.float32)
    new_cols["bad_addr_x_upaya"] = ((6.0 - df["cu_address_quality"].to_numpy(np.float32)) * (1.0 + df["scan_cnt_upaya"].to_numpy(np.float32))).astype(np.float32)
    new_cols["bad_addr_x_catatan"] = ((6.0 - df["cu_address_quality"].to_numpy(np.float32)) * (1.0 + df["has_catatan"].to_numpy(np.float32))).astype(np.float32)
    new_cols["co_p_q_x_fragile"] = ((5.0 - df["co_p_scan_q_mean"].to_numpy(np.float32)) * df["sp_fragile_score"].to_numpy(np.float32)).astype(np.float32)
    new_cols["co_p_q_x_night"] = ((5.0 - df["co_p_scan_q_mean"].to_numpy(np.float32)) * (1.0 + df["is_night_ship"].to_numpy(np.float32))).astype(np.float32)

    return pd.DataFrame(new_cols, index=df.index)


def main():
    os.makedirs("ghani/cache/v8", exist_ok=True)
    os.makedirs("ghani/models/v8", exist_ok=True)

    if os.path.exists("ghani/cache/v5/feat_v5_train.pkl") and os.path.exists("ghani/cache/v5/feat_v5_test.pkl") and os.environ.get("FROM_RAW", "0") != "1":
        log("Loading cached feat_v5 tables for fast clean v8 base construction...")
        tr = pd.read_pickle("ghani/cache/v5/feat_v5_train.pkl")
        te = pd.read_pickle("ghani/cache/v5/feat_v5_test.pkl")
    else:
        log("Building base tables directly from raw Task-1-Datavidia-11.0/ CSVs...")
        tr, moda_ref = bf5.build_split("train", None)
        te, _ = bf5.build_split("test", moda_ref)
        tr, te = bf5.add_unsupervised_entity_profiles(tr, te)
        id_cols = {"shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id"}
        for c in tr.columns:
            if c not in id_cols and tr[c].dtype == object:
                cats = pd.Categorical(pd.concat([tr[c], te[c]]).astype(str)).categories
                tr[c] = pd.Categorical(tr[c].astype(str), categories=cats)
                te[c] = pd.Categorical(te[c].astype(str), categories=cats)
        for eid in ["customer_id", "courier_id", "hub_id", "package_id"]:
            all_v = pd.concat([tr[eid], te[eid]]).astype(str)
            codes, _ = pd.factorize(all_v, sort=False)
            tr[eid + "_code"] = codes[: len(tr)].astype(np.int32)
            te[eid + "_code"] = codes[len(tr):].astype(np.int32)

    # Drop ALL mixed cu_loo_* AND drifted mixed shp_vs_co_* / hm_p_* columns!
    drop_cols = [c for c in tr.columns if c.startswith("cu_loo_") or c.startswith("hm_") or c in ("shp_vs_co_scan_q", "shp_vs_co_lgap", "shp_vs_co_delay")]
    log(f"Dropping {len(drop_cols)} mixed/drifted columns: {drop_cols}")

    add_tr = add_within_split_courier_and_hubmonth(tr)
    add_te = add_within_split_courier_and_hubmonth(te)
    tr = pd.concat([tr.drop(columns=drop_cols), add_tr], axis=1)
    te = pd.concat([te.drop(columns=drop_cols), add_te], axis=1)
    del add_tr, add_te
    gc.collect()

    # Stratified 5-Fold CV (exact match to folds_v5 for apple-to-apple comparison)
    if os.path.exists("ghani/cache/v5/folds_v5.npy"):
        folds = np.load("ghani/cache/v5/folds_v5.npy")
    else:
        classes = sorted(tr["status_penanganan"].unique())
        y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = np.zeros(len(tr), dtype=np.int32)
        for f, (_, va_i) in enumerate(skf.split(np.zeros(len(tr)), y)):
            folds[va_i] = f
    np.save("ghani/cache/v8/folds_v8.npy", folds)

    # Pre-compute full-train -> test customer behavioral history (`cu_tr_test_full.pkl`)
    n_cust_all = int(max(tr["customer_id_code"].max(), te["customer_id_code"].max())) + 1
    cu_te_full = v8c.compute_cu_tr_df(tr, te, n_cust_all, is_same=False)
    cu_te_full.to_pickle("ghani/cache/v8/cu_tr_test_full.pkl")
    log(f"Pre-computed pure train->test customer history: {cu_te_full.shape} | seen_rate={cu_te_full['cu_tr_seen'].mean()*100:.2f}% | mean_n_norm={cu_te_full['cu_tr_n_norm'].mean():.4f}")

    # Also export standalone lookup artifacts so `infer_v8.py --rebuild-from-test-only` needs ZERO train CSVs!
    log("Exporting standalone entity lookup tables to ghani/models/v8/entity_profiles_lookup.npz ...")
    # Build full-customer lookup table for all unique customers (0..n_cust_all-1)
    # Use dummy 1-row-per-customer frame for the 47 raw cu_tr columns (excluding the 4 master interactions which use test master cols)
    dummy_cu = pd.DataFrame({
        "customer_id_code": np.arange(n_cust_all, dtype=np.int32),
        "cu_complaint_index": np.zeros(n_cust_all, dtype=np.float32),
        "cu_address_quality": np.zeros(n_cust_all, dtype=np.float32),
    })
    cu_lut_df = v8c.compute_cu_tr_df(tr, dummy_cu, n_cust_all, is_same=False)
    raw_cu_cols = [c for c in v8c.CU_TR_FEAT_NAMES if c not in ("comp_x_cu_klaim", "bad_addr_x_cu_upaya", "cu_sum_upaya_norm", "cu_sum_klaim_norm")]
    cu_lut_mat = cu_lut_df[raw_cu_cols].to_numpy(dtype=np.float32)

    hu_prof_cols = [c for c in tr.columns if c.startswith("hu_p_")]
    co_prof_cols = [c for c in tr.columns if c.startswith("co_p_") or c.startswith("co_vs_hu_")]
    all_hu = pd.concat([tr[["hub_id_code"] + hu_prof_cols], te[["hub_id_code"] + hu_prof_cols]]).drop_duplicates("hub_id_code").set_index("hub_id_code")
    n_hub_all = int(max(tr["hub_id_code"].max(), te["hub_id_code"].max())) + 1
    hu_lut_mat = all_hu.reindex(np.arange(n_hub_all))[hu_prof_cols].to_numpy(dtype=np.float32)

    # For courier LOO on test, store exact test-side courier profile matrix directly indexed by shipment_id order OR courier mean!
    # Wait: in v5, co_p_* was LOO across 1.6M rows (~67.5 rows/courier, so LOO vs mean differs by < 1/67 = 0.002!).
    # Even better: store exact courier total sum & valid count across train so combining with test gives 100.000% BIT-FOR-BIT exact LOO on test!
    co_src_cols = [
        "scan_lgap_mean", "scan_gap_mean", "scan_q_mean", "scan_q_pct_le2", "scan_manual_mean",
        "seg_log_delay_mean", "seg_total_delay_ratio", "seg_dur_sum",
        "inv_days_max", "has_klaim_inv", "scan_cnt_upaya", "no_selesai", "inv_any_penalty", "log_declared_value",
    ]
    tr_tmp = tr[["courier_id_code"] + [c for c in co_src_cols if c != "no_selesai"]].copy()
    tr_tmp["no_selesai"] = (1.0 - tr["scan_has_selesai"]).astype(np.float32)
    n_cour_all = int(max(tr["courier_id_code"].max(), te["courier_id_code"].max())) + 1
    c_co = tr_tmp["courier_id_code"].to_numpy(np.int32)
    co_tr_cnt_all = np.bincount(c_co, minlength=n_cour_all).astype(np.float32)
    co_tr_sums = np.zeros((n_cour_all, len(co_src_cols)), dtype=np.float64)
    co_tr_cnts = np.zeros((n_cour_all, len(co_src_cols)), dtype=np.float64)
    for idx_c, col in enumerate(co_src_cols):
        xv = tr_tmp[col].to_numpy(np.float64)
        ok = ~np.isnan(xv)
        co_tr_sums[:, idx_c] = np.bincount(c_co, weights=np.where(ok, xv, 0.0), minlength=n_cour_all)
        co_tr_cnts[:, idx_c] = np.bincount(c_co, weights=ok.astype(np.float64), minlength=n_cour_all)

    # Save ID string -> int32 code maps
    id_maps = {}
    for eid in ["customer_id", "courier_id", "hub_id", "package_id"]:
        comb = pd.concat([tr[[eid, eid + "_code"]], te[[eid, eid + "_code"]]]).drop_duplicates(eid)
        id_maps[eid] = dict(zip(comb[eid].astype(str), comb[eid + "_code"].astype(int)))
    joblib_path = "ghani/models/v8/id_code_maps.pkl"
    pd.to_pickle(id_maps, joblib_path)

    np.savez_compressed(
        "ghani/models/v8/entity_profiles_lookup.npz",
        cu_lut_mat=cu_lut_mat,
        raw_cu_cols=np.array(raw_cu_cols, dtype=object),
        hu_lut_mat=hu_lut_mat,
        hu_prof_cols=np.array(hu_prof_cols, dtype=object),
        co_tr_cnt_all=co_tr_cnt_all,
        co_tr_sums=co_tr_sums,
        co_tr_cnts=co_tr_cnts,
        co_src_cols=np.array(co_src_cols, dtype=object),
    )
    log(f"Saved standalone lookup bundle ({os.path.getsize('ghani/models/v8/entity_profiles_lookup.npz')/1e6:.1f} MB)")

    tr.to_pickle("ghani/cache/v8/feat_v8_base_train.pkl")
    te.to_pickle("ghani/cache/v8/feat_v8_base_test.pkl")
    log(f"SAVED feat_v8_base_train.pkl {tr.shape} & feat_v8_base_test.pkl {te.shape}")


if __name__ == "__main__":
    main()
