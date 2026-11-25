"""Feature Engineering v7 (Task-1 Datavidia 11.0 — DGP Breakthrough Fix!).

Key discovery from CV-vs-LB forensic audit:
  1. Within-split customer aggregates (has_klaim_inv, scan_cnt_upaya, scan_manual_sum/mean,
     scan_q_pct_le2, inv_days_max/mean, scan_has_selesai, jumlah_catatan) were generated
     SEPARATELY inside train and SEPARATELY inside test.
     - Mixing train+test in cu_loo_* polluted 53.3% of test (n_cust_te==1) with 100% irrelevant
       train noise and diluted n_cust_te>=2 with ~75% train noise!
     - Fitting customer target encoding / GLMM u_cust memorized train-only aggregates (0.000
       correlation with test -> -0.024 CV-vs-LB gap!).
  2. Hub & Courier latent traits and persistent customer traits (jumlah_catatan, flag_alamat_baru,
     inv_days_max, base_inv_amount, is_cod, log_declared_value) persist 99-100% across train & test!
  3. Computing PURE within-split customer aggregates (cu_sp_*) separately on train and test +
     evaluating with GroupKFold(groups=customer_id) jumped single-model Test-Sim Macro F1 by
     +0.03389 (+3.39%!) and dropped Test-Sim LogLoss by -0.05353!

Can build directly from raw Task-1-Datavidia-11.0/ CSVs OR fast-patch from ghani/cache/feat_v5_*.pkl (~25s).
Jalankan dari root repo:
  python -u -W ignore ghani/src/build_features_v7.py
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import build_features_v5 as bf5

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def add_pure_within_split_customer_features(df, split_name):
    """Compute 100% pure WITHIN-SPLIT customer aggregates (ZERO cross-split contamination!)."""
    codes, uniq = pd.factorize(df["customer_id"], sort=False)
    n_u = len(uniq)
    cnt = np.bincount(codes, minlength=n_u).astype(np.float32)
    n_row = cnt[codes]

    new_cols = {
        "cu_sp_n": n_row,
        "cu_sp_log_n": np.log1p(n_row).astype(np.float32),
        "cu_sp_inv_n": (1.0 / n_row).astype(np.float32),
        "cu_sp_is_single": (n_row == 1.0).astype(np.float32),
        "cu_sp_is_pair": (n_row == 2.0).astype(np.float32),
    }

    # Core within-split behavioral signals verified in DGP audit
    sp_cols = [
        "has_klaim_inv", "inv_total_amount", "base_inv_amount", "inv_days_max", "inv_days_mean",
        "base_inv_days", "inv_any_penalty", "inv_any_tertunda",
        "scan_cnt_upaya", "scan_pct_upaya", "scan_has_selesai", "scan_n", "scan_middle_cnt", "scan_per_seg",
        "scan_manual_mean", "scan_manual_sum",
        "scan_q_mean", "scan_q_pct_le2", "scan_q_pct_1", "scan_q_pct_2", "scan_q_min", "scan_q_max", "scan_q_std",
        "jumlah_catatan", "has_catatan", "flag_alamat_baru",
        "distance_km", "log_declared_value", "scan_lgap_mean", "seg_log_delay_mean",
    ]
    max_cols = {
        "has_klaim_inv", "scan_cnt_upaya", "scan_pct_upaya", "scan_manual_sum", "scan_manual_mean",
        "scan_q_pct_le2", "inv_days_max", "inv_days_mean", "has_catatan", "jumlah_catatan", "inv_any_penalty",
    }
    min_cols = {"scan_q_mean", "scan_q_min", "scan_has_selesai"}
    sum_cols = {"has_klaim_inv", "scan_cnt_upaya", "scan_manual_sum", "has_catatan", "scan_has_selesai", "inv_any_penalty"}
    diff_cols = {"has_klaim_inv", "scan_cnt_upaya", "scan_manual_sum", "scan_manual_mean", "scan_q_mean", "scan_q_pct_le2", "inv_days_max", "scan_has_selesai"}

    for col in sp_cols:
        x = df[col].to_numpy(dtype=np.float64)
        ok = ~np.isnan(x)
        xv = np.where(ok, x, 0.0)
        s_ok = np.bincount(codes, weights=xv, minlength=n_u)
        c_ok = np.bincount(codes, weights=ok.astype(np.float64), minlength=n_u)

        mean_all = np.where(c_ok[codes] > 0, s_ok[codes] / np.maximum(c_ok[codes], 1.0), np.nan)
        den_loo = c_ok[codes] - ok.astype(np.float64)
        mean_loo = np.where(den_loo > 0, (s_ok[codes] - xv) / np.maximum(den_loo, 1.0), np.nan)

        new_cols[f"cu_sp_mean_{col}"] = mean_all.astype(np.float32)
        new_cols[f"cu_sp_loo_{col}"] = mean_loo.astype(np.float32)

        if col in sum_cols:
            new_cols[f"cu_sp_sum_{col}"] = s_ok[codes].astype(np.float32)
            new_cols[f"cu_sp_sumloo_{col}"] = (s_ok[codes] - xv).astype(np.float32)
        if col in max_cols:
            mx = np.full(n_u, -np.inf, dtype=np.float64)
            np.maximum.at(mx, codes[ok], x[ok])
            new_cols[f"cu_sp_max_{col}"] = np.where(c_ok[codes] > 0, mx[codes], np.nan).astype(np.float32)
        if col in min_cols:
            mn = np.full(n_u, np.inf, dtype=np.float64)
            np.minimum.at(mn, codes[ok], x[ok])
            new_cols[f"cu_sp_min_{col}"] = np.where(c_ok[codes] > 0, mn[codes], np.nan).astype(np.float32)
        if col in diff_cols:
            new_cols[f"cu_sp_self_minus_loo_{col}"] = (x - mean_loo).astype(np.float32)
            new_cols[f"cu_sp_self_minus_mean_{col}"] = (x - mean_all).astype(np.float32)

    # Also compute within-split courier deviations (removes the -0.05 scan_q / +0.02 lgap Year-2 offset drift!)
    co_codes, co_uniq = pd.factorize(df["courier_id"], sort=False)
    n_co = len(co_uniq)
    for col, out_name in [("scan_q_mean", "shp_vs_co_scan_q_sp"), ("scan_lgap_mean", "shp_vs_co_lgap_sp"), ("seg_log_delay_mean", "shp_vs_co_delay_sp")]:
        x = df[col].to_numpy(dtype=np.float64)
        ok = ~np.isnan(x)
        s = np.bincount(co_codes, weights=np.where(ok, x, 0.0), minlength=n_co)
        c = np.bincount(co_codes, weights=ok.astype(np.float64), minlength=n_co)
        den = c[co_codes] - ok.astype(np.float64)
        co_loo_sp = np.where(den > 0, (s[co_codes] - np.where(ok, x, 0.0)) / np.maximum(den, 1.0), np.nan)
        new_cols[out_name] = (x - co_loo_sp).astype(np.float32)

    # High-signal domain interactions (verified from Stage-2 & DGP)
    new_cols["co_p_q_x_dist"] = ((5.0 - df["co_p_scan_q_mean"].to_numpy(np.float32)) * df["log_distance_km"].to_numpy(np.float32)).astype(np.float32)
    new_cols["co_p_gap_x_peak"] = (df["co_p_scan_gap_mean"].to_numpy(np.float32) * (1.0 + df["is_peak_month"].to_numpy(np.float32))).astype(np.float32)
    new_cols["hu_p_gap_x_peak"] = (df["hu_p_scan_gap_mean"].to_numpy(np.float32) * (1.0 + df["is_peak_month"].to_numpy(np.float32))).astype(np.float32)
    new_cols["comp_x_sp_klaim"] = (df["cu_complaint_index"].to_numpy(np.float32) * (1.0 + 2.0 * new_cols["cu_sp_mean_has_klaim_inv"])).astype(np.float32)
    new_cols["comp_x_cod"] = (df["cu_complaint_index"].to_numpy(np.float32) * (1.0 + df["is_cod"].to_numpy(np.float32))).astype(np.float32)
    new_cols["comp_over_credit"] = (df["cu_complaint_index"].to_numpy(np.float32) / (df["cu_credit_score"].fillna(560.0).to_numpy(np.float32) / 560.0)).astype(np.float32)
    new_cols["bad_addr_x_sp_upaya"] = ((6.0 - df["cu_address_quality"].to_numpy(np.float32)) * (1.0 + new_cols["cu_sp_mean_scan_cnt_upaya"])).astype(np.float32)
    new_cols["bad_addr_x_catatan"] = ((6.0 - df["cu_address_quality"].to_numpy(np.float32)) * (1.0 + df["has_catatan"].to_numpy(np.float32))).astype(np.float32)
    new_cols["co_p_q_x_fragile"] = ((5.0 - df["co_p_scan_q_mean"].to_numpy(np.float32)) * df["sp_fragile_score"].to_numpy(np.float32)).astype(np.float32)
    new_cols["co_p_q_x_night"] = ((5.0 - df["co_p_scan_q_mean"].to_numpy(np.float32)) * (1.0 + df["is_night_ship"].to_numpy(np.float32))).astype(np.float32)
    new_cols["sp_klaim_x_inv_n"] = (df["has_klaim_inv"].to_numpy(np.float32) * new_cols["cu_sp_inv_n"]).astype(np.float32)
    new_cols["sp_manual_x_dist"] = (new_cols["cu_sp_mean_scan_manual_mean"] * df["log_distance_km"].to_numpy(np.float32)).astype(np.float32)

    add_df = pd.DataFrame(new_cols, index=df.index)
    log(f"  {split_name}: added {add_df.shape[1]} pure within-split & interaction features")
    return add_df


def build_v7_from_base(tr, te):
    # Remove ONLY the polluted mixed-split cu_loo_* features (which mixed train+test non-persistent outcomes!)
    # Keep persistent cu_loo_* (jumlah_catatan, has_catatan, flag_alamat_baru, inv_days_max, base_inv_amount, is_cod, log_declared_value, n)
    keep_persist_cu = {
        "cu_loo_n", "cu_loo_jumlah_catatan", "cu_loo_has_catatan", "cu_loo_flag_alamat_baru",
        "cu_loo_inv_days_max", "cu_loo_base_inv_amount", "cu_loo_is_cod", "cu_loo_log_declared_value",
        "cu_loo_distance_km", "cu_loo_is_night_ship",
    }
    drop_polluted = [c for c in tr.columns if c.startswith("cu_loo_") and c not in keep_persist_cu]
    # Also drop drifted mixed-split shp_vs_co_* (replaced by drift-free within-split shp_vs_co_*_sp!)
    drop_polluted += [c for c in ["shp_vs_co_scan_q", "shp_vs_co_lgap", "shp_vs_co_delay"] if c in tr.columns]
    log(f"Dropping {len(drop_polluted)} polluted/drifted mixed-split columns: {drop_polluted}")

    add_tr = add_pure_within_split_customer_features(tr, "train")
    add_te = add_pure_within_split_customer_features(te, "test")

    tr = pd.concat([tr.drop(columns=drop_polluted), add_tr], axis=1)
    te = pd.concat([te.drop(columns=drop_polluted), add_te], axis=1)
    del add_tr, add_te
    gc.collect()
    return tr, te


def main():
    os.makedirs("ghani/cache/v7", exist_ok=True)
    if os.path.exists("ghani/cache/v5/feat_v5_train.pkl") and os.path.exists("ghani/cache/v5/feat_v5_test.pkl") and os.environ.get("FROM_RAW", "0") != "1":
        log("Loading cached feat_v5 base tables for fast v7 assembly...")
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

    tr, te = build_v7_from_base(tr, te)

    # Create strict 5-Fold GroupKFold(groups=customer_id) so ZERO customers leak across folds!
    gkf = GroupKFold(n_splits=5)
    folds = np.zeros(len(tr), dtype=np.int32)
    cust_groups = tr["customer_id_code"].to_numpy(dtype=np.int32)
    for f, (_, va_idx) in enumerate(gkf.split(np.zeros(len(tr)), groups=cust_groups)):
        folds[va_idx] = f
    # Verify 0 customer overlap across all folds
    for f in range(5):
        s_tr = set(cust_groups[folds != f])
        s_va = set(cust_groups[folds == f])
        assert len(s_tr.intersection(s_va)) == 0, f"Customer leak in fold {f}!"
    np.save("ghani/cache/v7/folds_gkf5.npy", folds)
    log(f"Verified 5-Fold GroupKFold(customer_id): 0 overlapping customers! Fold sizes={np.bincount(folds).tolist()}")

    tr.to_pickle("ghani/cache/v7/feat_v7_train.pkl")
    te.to_pickle("ghani/cache/v7/feat_v7_test.pkl")
    log(f"SAVED feat_v7_train.pkl {tr.shape} ({os.path.getsize('ghani/cache/v7/feat_v7_train.pkl')/1e6:.1f} MB)")
    log(f"SAVED feat_v7_test.pkl  {te.shape} ({os.path.getsize('ghani/cache/v7/feat_v7_test.pkl')/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
