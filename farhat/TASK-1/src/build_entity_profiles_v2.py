"""
Entity Profiles LOO (Leak-Free) for TASK-1-exp v2.
Uses existing TASK-1 aggregated features + adds Ghani's LOO entity profiles.
"""
import gc
import os
import time

import numpy as np
import pandas as pd

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _group_mean_and_cnt(df, grp_col, cols, prefix, loo=False):
    codes, uniques = pd.factorize(df[grp_col], sort=False)
    codes = codes.astype(np.int64)
    n_u = len(uniques)
    cnt_all = np.bincount(codes, minlength=n_u).astype(np.float32)
    res = {f"{prefix}_n": cnt_all[codes]}
    for col in cols:
        x = df[col].to_numpy()
        if x.dtype.kind not in ('f', 'i', 'u'):
            x = pd.to_numeric(x, errors='coerce').to_numpy(dtype=np.float64)
        else:
            x = x.astype(np.float64)
        ok = ~np.isnan(x)
        weights = np.where(ok, x, 0.0).astype(np.float64)
        s_ok = np.bincount(codes, weights=weights, minlength=n_u)
        c_ok = np.bincount(codes, weights=ok.astype(np.float64), minlength=n_u)
        if loo:
            num = s_ok[codes] - np.where(ok, x, 0.0)
            den = c_ok[codes] - ok.astype(np.float64)
            val = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
        else:
            val = np.where(c_ok[codes] > 0, s_ok[codes] / np.maximum(c_ok[codes], 1.0), np.nan)
        res[f"{prefix}_{col}"] = val.astype(np.float32)
    return res


def main():
    log("Loading TASK-1 features...")
    tr = pd.read_parquet("data/processed/feat_train.parquet")
    te = pd.read_parquet("data/processed/feat_test.parquet")
    log(f"Train: {tr.shape}, Test: {te.shape}")

    # Source columns for profiling (numeric, predictive, exist in both)
    prof_src = [
        # Scan
        "sc_n", "gap_mean", "gap_std", "q_mean", "q_std", "q_min", "q_max",
        "man_mean", "ev_upaya_antar_n", "ev_selesai_n", "no_selesai",
        # Route
        "rs_n", "rs_km_sum", "rs_dur_sum", "rs_dur_mean", "rs_dur_std",
        "moda_darat_n", "moda_udara_n", "moda_laut_n", "moda_kereta_n",
        "dur_per_km", "cost_per_km",
        # Invoice
        "inv_n", "inv_amt_sum", "inv_klaim_n", "inv_langganan_n", "inv_pengiriman_n",
        "pay_problem_n", "inv_disc_rate",
        # Shipment attrs
        "log_value", "distance_km", "ship_month", "ship_hour", "is_cod",
        "order_to_ship_hours", "prioritas_pengiriman", "jumlah_catatan", "flag_alamat_baru",
        "weight_kg", "volume_l", "qty_items", "coupon_count",
        # Customer
        "cus_addr_q", "cus_complaint", "cus_credit", "cus_tenure",
        # Courier
        "co_score", "co_exp", "co_capacity",
        # Hub
        "hub_auto", "hub_cap", "hub_density", "hub_hours",
    ]
    prof_src = [c for c in prof_src if c in tr.columns and c in te.columns]
    # Remove duplicates
    prof_src = list(dict.fromkeys(prof_src))
    log(f"Profiling on {len(prof_src)} source columns")

    key_cols = ["customer_id", "courier_id", "hub_id"]  # ship_month already in prof_src
    all_df = pd.concat([tr[key_cols + prof_src], te[key_cols + prof_src]], ignore_index=True)
    log(f"Combined for profiling: {all_df.shape}")

    # 1. Hub profiles (NO LOO - many rows per hub)
    hu_cols = [
        "gap_mean", "q_mean", "rs_dur_sum", "dur_per_km",
        "inv_amt_sum", "inv_klaim_n", "pay_problem_n",
        "log_value", "distance_km", "ship_month",
    ]
    hu_cols = [c for c in hu_cols if c in all_df.columns]
    hu_p = _group_mean_and_cnt(all_df, "hub_id", hu_cols, "hu_p", loo=False)
    log(f"Hub profiles: {len([k for k in hu_p if k.startswith('hu_p_')])} features")

    # 2. Courier profiles (LOO!)
    co_cols = [
        "gap_mean", "gap_std", "q_mean", "q_std", "man_mean",
        "ev_upaya_antar_n", "ev_selesai_n", "no_selesai",
        "rs_dur_sum", "dur_per_km", "inv_amt_sum", "inv_klaim_n",
        "log_value", "distance_km", "ship_month",
    ]
    co_cols = [c for c in co_cols if c in all_df.columns]
    co_p = _group_mean_and_cnt(all_df, "courier_id", co_cols, "co_p", loo=True)
    log(f"Courier profiles (LOO): {len([k for k in co_p if k.startswith('co_p_')])} features")

    # 3. Hub x Month profiles (LOO!)
    all_df = all_df.copy()
    all_df["hub_month"] = all_df["hub_id"].astype(str) + "_" + all_df["ship_month"].astype(str)
    hm_cols = ["gap_mean", "rs_dur_sum", "inv_amt_sum"]
    hm_cols = [c for c in hm_cols if c in all_df.columns]
    hm_p = _group_mean_and_cnt(all_df, "hub_month", hm_cols, "hm_p", loo=True)
    hm_p["hm_p_n_ratio"] = (hm_p["hm_p_n"] / (hu_p["hu_p_n"] / 12.0 + 1e-6)).astype(np.float32)
    hm_p["hm_gap_vs_hu"] = (hm_p["hm_p_gap_mean"] - hu_p["hu_p_gap_mean"]).astype(np.float32)
    hm_p["hm_dur_vs_hu"] = (hm_p["hm_p_rs_dur_sum"] - hu_p["hu_p_rs_dur_sum"]).astype(np.float32)
    log(f"Hub×Month profiles: {len([k for k in hm_p if k.startswith('hm_p_')])} features")

    # 4. Courier vs Hub & Shipment vs Courier deviations
    dev = {
        "co_vs_hu_gap": (co_p["co_p_gap_mean"] - hu_p["hu_p_gap_mean"]).astype(np.float32),
        "co_vs_hu_dur": (co_p["co_p_rs_dur_sum"] - hu_p["hu_p_rs_dur_sum"]).astype(np.float32),
        "co_vs_hu_q": (co_p["co_p_q_mean"] - hu_p["hu_p_q_mean"]).astype(np.float32),
        "shp_vs_co_gap": (all_df["gap_mean"].to_numpy(np.float32) - co_p["co_p_gap_mean"]).astype(np.float32),
        "shp_vs_co_q": (all_df["q_mean"].to_numpy(np.float32) - co_p["co_p_q_mean"]).astype(np.float32),
        "shp_vs_co_dur": (all_df["rs_dur_sum"].to_numpy(np.float32) - co_p["co_p_rs_dur_sum"]).astype(np.float32),
    }
    log(f"Deviation features: {len(dev)} features")

    # 5. Customer LOO profiles (raw + courier-residualized)
    all_df["q_res_co"] = dev["shp_vs_co_q"]
    all_df["gap_res_co"] = dev["shp_vs_co_gap"]
    all_df["dur_res_co"] = dev["shp_vs_co_dur"]
    cu_cols = [
        "jumlah_catatan", "flag_alamat_baru", "ev_upaya_antar_n", "ev_selesai_n",
        "no_selesai", "inv_klaim_n", "inv_amt_sum", "pay_problem_n",
        "q_mean", "q_res_co", "gap_mean", "gap_res_co",
        "man_mean", "dur_res_co", "distance_km", "is_cod",
        "log_value", "ship_hour",
    ]
    cu_cols = [c for c in cu_cols if c in all_df.columns]
    cu_p = _group_mean_and_cnt(all_df, "customer_id", cu_cols, "cu_loo", loo=True)
    log(f"Customer profiles (LOO): {len([k for k in cu_p if k.startswith('cu_loo_')])} features")

    # Combine all new profile columns
    new_cols = {}
    for d in (hu_p, co_p, hm_p, dev, cu_p):
        new_cols.update(d)
    prof_df = pd.DataFrame(new_cols)
    log(f"Total new profile features: {prof_df.shape[1]}")

    n_tr = len(tr)
    tr = pd.concat([tr.reset_index(drop=True), prof_df.iloc[:n_tr].reset_index(drop=True)], axis=1)
    te = pd.concat([te.reset_index(drop=True), prof_df.iloc[n_tr:].reset_index(drop=True)], axis=1)
    log(f"Extended Train: {tr.shape}, Test: {te.shape}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    tr.to_parquet("data/processed/feat_train_entity.parquet")
    te.to_parquet("data/processed/feat_test_entity.parquet")
    log("SAVED feat_train_entity.parquet & feat_test_entity.parquet")


if __name__ == "__main__":
    main()