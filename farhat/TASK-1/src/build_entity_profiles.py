"""
Entity Profiles LOO (Leak-Free) for TASK-1-exp.
Adapted from Ghani's build_features_v5.py add_unsupervised_entity_profiles().
Input: TASK-1 feat_train.parquet + feat_test.parquet
Output: Extended features with entity profiles, saved to data/processed/
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
    n_u = len(uniques)
    cnt_all = np.bincount(codes, minlength=n_u).astype(np.float32)
    res = {f"{prefix}_n": cnt_all[codes]}
    for col in cols:
        x = df[col].to_numpy(dtype=np.float64)
        ok = ~np.isnan(x)
        s_ok = np.bincount(codes, weights=np.where(ok, x, 0.0), minlength=n_u)
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

    # Identify key numeric columns for entity profiling
    # From TASK-1 features, pick the most predictive ones
    # Ghani used: scan stats, route delay, invoice stats, shipment attrs
    prof_src = [
        "log_declared_value", "log_distance_km", "ship_month", "ship_hour", "is_cod",
        "order_to_ship_hours", "prioritas_pengiriman", "jumlah_catatan", "flag_alamat_baru",
        "weight_kg", "volume_l", "qty_items", "coupon_count",
        # Scan features (if exist)
        "scan_n", "scan_gap_mean", "scan_gap_std", "scan_q_mean", "scan_q_std",
        "scan_q_pct_le2", "scan_manual_mean", "scan_cnt_upaya", "scan_has_selesai",
        # Route features (if exist)
        "seg_n", "seg_total_delay_ratio", "seg_log_delay_mean", "seg_dur_sum",
        # Invoice features (if exist)
        "base_inv_amount", "has_klaim_inv", "inv_days_max", "inv_any_penalty",
    ]
    # Filter to existing columns
    prof_src = [c for c in prof_src if c in tr.columns]
    log(f"Profiling on {len(prof_src)} source columns: {prof_src}")

    key_cols = ["customer_id", "courier_id", "hub_id", "ship_month"]
    all_df = pd.concat([tr[key_cols + prof_src], te[key_cols + prof_src]], ignore_index=True)
    log(f"Combined for profiling: {all_df.shape}")

    # 1. Hub profiles (~2455 rows/hub) - NO LOO (hub has many rows)
    hu_cols = [
        "scan_gap_mean", "seg_log_delay_mean", "seg_total_delay_ratio", "seg_dur_sum",
        "scan_q_mean", "scan_manual_mean", "inv_days_max", "has_klaim_inv",
        "log_declared_value", "base_inv_amount", "inv_any_penalty",
    ]
    hu_cols = [c for c in hu_cols if c in all_df.columns]
    hu_p = _group_mean_and_cnt(all_df, "hub_id", hu_cols, "hu_p", loo=False)
    log(f"Hub profiles: {len([k for k in hu_p if k.startswith('hu_p_')])} features")

    # 2. Courier profiles (~67.5 rows/courier) - LOO!
    co_cols = [
        "scan_gap_mean", "scan_q_mean", "scan_q_pct_le2", "scan_manual_mean",
        "seg_log_delay_mean", "seg_total_delay_ratio", "seg_dur_sum",
        "inv_days_max", "has_klaim_inv", "scan_cnt_upaya",
        "log_declared_value",
    ]
    co_cols = [c for c in co_cols if c in all_df.columns]
    co_p = _group_mean_and_cnt(all_df, "courier_id", co_cols, "co_p", loo=True)
    log(f"Courier profiles (LOO): {len([k for k in co_p if k.startswith('co_p_')])} features")

    # 3. Hub x Month profiles (~205 rows/group) - LOO!
    all_df["hub_month"] = all_df["hub_id"].astype(str) + "_" + all_df["ship_month"].astype(str)
    hm_cols = ["scan_gap_mean", "seg_log_delay_mean", "inv_days_max"]
    hm_cols = [c for c in hm_cols if c in all_df.columns]
    hm_p = _group_mean_and_cnt(all_df, "hub_month", hm_cols, "hm_p", loo=True)
    hm_p["hm_p_n_ratio"] = (hm_p["hm_p_n"] / (hu_p["hu_p_n"] / 12.0 + 1e-6)).astype(np.float32)
    hm_p["hm_lgap_vs_hu"] = (hm_p["hm_p_scan_gap_mean"] - hu_p["hu_p_scan_gap_mean"]).astype(np.float32)
    hm_p["hm_delay_vs_hu"] = (hm_p["hm_p_seg_log_delay_mean"] - hu_p["hu_p_seg_log_delay_mean"]).astype(np.float32)
    log(f"Hub×Month profiles: {len([k for k in hm_p if k.startswith('hm_p_')])} features")

    # 4. Courier vs Hub & Shipment vs Courier deviations
    dev = {
        "co_vs_hu_gap": (co_p["co_p_scan_gap_mean"] - hu_p["hu_p_scan_gap_mean"]).astype(np.float32),
        "co_vs_hu_delay": (co_p["co_p_seg_log_delay_mean"] - hu_p["hu_p_seg_log_delay_mean"]).astype(np.float32),
        "co_vs_hu_scan_q": (co_p["co_p_scan_q_mean"] - hu_p["hu_p_scan_q_mean"]).astype(np.float32),
        "shp_vs_co_gap": (all_df["scan_gap_mean"].to_numpy(np.float32) - co_p["co_p_scan_gap_mean"]).astype(np.float32),
        "shp_vs_co_scan_q": (all_df["scan_q_mean"].to_numpy(np.float32) - co_p["co_p_scan_q_mean"]).astype(np.float32),
        "shp_vs_co_delay": (all_df["seg_log_delay_mean"].to_numpy(np.float32) - co_p["co_p_seg_log_delay_mean"]).astype(np.float32),
    }
    log(f"Deviation features: {len(dev)} features")

    # 5. Customer LOO profiles (raw + courier-residualized)
    all_df["scan_q_res_co"] = dev["shp_vs_co_scan_q"]
    all_df["gap_res_co"] = dev["shp_vs_co_gap"]
    all_df["delay_res_co"] = dev["shp_vs_co_delay"]
    cu_cols = [
        "jumlah_catatan", "scan_cnt_upaya", "scan_has_selesai", "has_klaim_inv",
        "inv_days_max", "base_inv_amount", "scan_q_mean", "scan_q_res_co",
        "scan_manual_mean", "gap_res_co", "delay_res_co",
        "distance_km", "flag_alamat_baru", "is_cod", "is_night_ship", "log_declared_value",
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