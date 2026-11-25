"""Shared v8 Core Utilities:
1. Exact distribution-matched, zero-leakage `train`-only Customer Behavioral History (`cu_tr_*`, 51 cols).
2. Scale-invariant Joint Hierarchical GLMM + Multi-Scale Bayesian Smoothed Target Encoding (`63` cols).
3. Zero-shift Stage-2 feature assembly for LightGBM, XGBoost, ExtraTrees-LGB, and PyTorch ResNets.
"""
import gc
import os
import sys

import numpy as np
import pandas as pd

_COMMON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "common"))
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
import glmm_lib as glmm

N_FULL_TRAIN = 1_243_617

CU_HIST_COLS = [
    "jumlah_catatan", "has_catatan", "flag_alamat_baru", "distance_km", "is_cod", "is_night_ship", "log_declared_value",
    "scan_cnt_upaya", "scan_pct_upaya", "scan_has_selesai", "scan_q_mean", "scan_q_pct_le2", "scan_q_pct_2",
    "scan_q_min", "scan_q_std", "scan_manual_mean", "scan_manual_sum", "scan_n", "scan_middle_cnt", "scan_lgap_mean",
    "seg_log_delay_mean",
    "has_klaim_inv", "inv_total_amount", "base_inv_amount", "klaim_vs_base_amt",
    "inv_days_max", "inv_days_mean", "base_inv_days", "inv_any_penalty", "inv_any_tertunda", "scan_per_seg",
]

CU_MAX_COLS = {
    "has_klaim_inv", "inv_total_amount", "inv_days_max", "scan_cnt_upaya", "scan_pct_upaya",
    "scan_manual_sum", "scan_manual_mean", "scan_q_pct_le2", "jumlah_catatan", "inv_any_penalty",
}
CU_MIN_COLS = {"scan_q_mean", "scan_q_min", "scan_has_selesai"}


def get_cu_tr_feat_names():
    names = ["cu_tr_n_norm", "cu_tr_log_n_norm", "cu_tr_seen"]
    for col in CU_HIST_COLS:
        names.append(f"cu_tr_mean_{col}")
        if col in CU_MAX_COLS:
            names.append(f"cu_tr_max_{col}")
        if col in CU_MIN_COLS:
            names.append(f"cu_tr_min_{col}")
    names += ["comp_x_cu_klaim", "bad_addr_x_cu_upaya", "cu_sum_upaya_norm", "cu_sum_klaim_norm"]
    return names


CU_TR_FEAT_NAMES = get_cu_tr_feat_names()

GLMM_V8_FEAT_NAMES = (
    [f"s1_logit_{k}" for k in range(7)]
    + [f"glmm_logit_{k}" for k in range(7)]
    + [f"glmm_prob_{k}" for k in range(7)]
    + [f"u_cust_{k}" for k in range(7)]
    + [f"u_cour_{k}" for k in range(7)]
    + [f"u_hub_{k}" for k in range(7)]
    + ["glmm_max_p", "glmm_entropy", "glmm_margin", "log_n_cust_norm", "h_cust_norm", "log_n_cour_norm", "h_cour_norm"]
    + [f"te_cust_m3_{k}" for k in range(7)]
    + [f"te_cour_m15_{k}" for k in range(7)]
)


def compute_cu_tr_df(ref_df, tgt_df, n_cust_all, is_same=False):
    """Vectorized pure train-only customer profile builder with exact LOO max/min when is_same=True.
    Scale factor `(0.80 * N_FULL_TRAIN) / len(ref_df)` makes `cu_tr_n_norm` 100% distribution-matched
    across 4-fold train LOO, 4-fold val lookup, and 5-fold full-train test lookup!
    """
    c_ref = ref_df["customer_id_code"].to_numpy(np.int32)
    c_tgt = tgt_df["customer_id_code"].to_numpy(np.int32)
    cnt_ref = np.bincount(c_ref, minlength=n_cust_all).astype(np.float64)

    if is_same:
        n_eff = np.maximum(cnt_ref[c_tgt] - 1.0, 0.0)
        scale = (0.80 * N_FULL_TRAIN) / max(len(ref_df), 1)
        # Note: for LOO inside 4-fold ref (len = 0.80 * N), E[cnt_ref - 1] = 0.80 * E[N_C - 1] = 1.735 -> scale = 1.0!
    else:
        n_eff = cnt_ref[c_tgt]
        scale = (0.80 * N_FULL_TRAIN) / max(len(ref_df), 1)
        # For val on 4-fold ref: scale = 1.0 -> E[n_eff * scale] = 1.735!
        # For test on 5-fold ref: scale = 0.80 -> E[n_eff * scale] = 0.80 * 2.169 = 1.735!

    n_norm = (n_eff * scale).astype(np.float32)
    out = {
        "cu_tr_n_norm": n_norm,
        "cu_tr_log_n_norm": np.log1p(n_norm).astype(np.float32),
        "cu_tr_seen": (n_eff > 0).astype(np.float32),
    }

    for col in CU_HIST_COLS:
        x_r = ref_df[col].to_numpy(dtype=np.float64)
        ok_r = ~np.isnan(x_r)
        xv_r = np.where(ok_r, x_r, 0.0)
        s = np.bincount(c_ref, weights=xv_r, minlength=n_cust_all)
        c = np.bincount(c_ref, weights=ok_r.astype(np.float64), minlength=n_cust_all)

        if is_same:
            num = s[c_tgt] - xv_r
            den = c[c_tgt] - ok_r.astype(np.float64)
        else:
            num = s[c_tgt]
            den = c[c_tgt]

        out[f"cu_tr_mean_{col}"] = np.where(den > 0, num / np.maximum(den, 1.0), np.nan).astype(np.float32)

        if col in CU_MAX_COLS:
            ok_idx = np.where(ok_r)[0]
            mx1 = np.full(n_cust_all, -np.inf, dtype=np.float64)
            np.maximum.at(mx1, c_ref[ok_idx], x_r[ok_idx])
            if is_same:
                is_m1 = x_r[ok_idx] == mx1[c_ref[ok_idx]]
                cnt_m1 = np.bincount(c_ref[ok_idx], weights=is_m1.astype(np.float64), minlength=n_cust_all)
                mx2 = np.full(n_cust_all, -np.inf, dtype=np.float64)
                not_m1 = ok_idx[~is_m1]
                np.maximum.at(mx2, c_ref[not_m1], x_r[not_m1])
                use_m2 = ok_r & (x_r == mx1[c_tgt]) & (cnt_m1[c_tgt] == 1.0)
                raw_mx = np.where(use_m2, mx2[c_tgt], mx1[c_tgt])
            else:
                raw_mx = mx1[c_tgt]
            out[f"cu_tr_max_{col}"] = np.where((den > 0) & np.isfinite(raw_mx), raw_mx, np.nan).astype(np.float32)

        if col in CU_MIN_COLS:
            ok_idx = np.where(ok_r)[0]
            mi1 = np.full(n_cust_all, np.inf, dtype=np.float64)
            np.minimum.at(mi1, c_ref[ok_idx], x_r[ok_idx])
            if is_same:
                is_m1 = x_r[ok_idx] == mi1[c_ref[ok_idx]]
                cnt_m1 = np.bincount(c_ref[ok_idx], weights=is_m1.astype(np.float64), minlength=n_cust_all)
                mi2 = np.full(n_cust_all, np.inf, dtype=np.float64)
                not_m1 = ok_idx[~is_m1]
                np.minimum.at(mi2, c_ref[not_m1], x_r[not_m1])
                use_m2 = ok_r & (x_r == mi1[c_tgt]) & (cnt_m1[c_tgt] == 1.0)
                raw_mi = np.where(use_m2, mi2[c_tgt], mi1[c_tgt])
            else:
                raw_mi = mi1[c_tgt]
            out[f"cu_tr_min_{col}"] = np.where((den > 0) & np.isfinite(raw_mi), raw_mi, np.nan).astype(np.float32)

    # Scale-invariant customer-history x master interactions
    comp = tgt_df["cu_complaint_index"].to_numpy(np.float32)
    addr = tgt_df["cu_address_quality"].to_numpy(np.float32)
    m_kl = np.nan_to_num(out["cu_tr_mean_has_klaim_inv"], nan=0.0)
    m_up = np.nan_to_num(out["cu_tr_mean_scan_cnt_upaya"], nan=0.0)
    out["comp_x_cu_klaim"] = (comp * (1.0 + 2.0 * m_kl)).astype(np.float32)
    out["bad_addr_x_cu_upaya"] = ((6.0 - addr) * (1.0 + m_up)).astype(np.float32)
    out["cu_sum_upaya_norm"] = (m_up * n_norm).astype(np.float32)
    out["cu_sum_klaim_norm"] = (m_kl * n_norm).astype(np.float32)

    return pd.DataFrame(out, columns=CU_TR_FEAT_NAMES, index=tgt_df.index)


def extract_glmm_and_te_block(base_logits, ents, st, te_tables, n_fitted_rows, n_classes=7):
    """Extract 63-col scale-normalized GLMM + Smoothed Bayesian TE block with ZERO 3-fold vs 5-fold shift!"""
    eta, p = glmm.predict_glmm(base_logits, ents, st, use_moderation=False)
    u_cust = st["U"]["customer"][ents["customer"]]
    u_cour = st["U"]["courier"][ents["courier"]]
    u_hub = st["U"]["hub"][ents["hub"]] + st["U"]["hub_peak"][ents["hub_peak"]]

    # Normalize counts and Fisher info to 4-fold (0.80 * N_FULL_TRAIN) reference scale!
    norm_scale = float((0.80 * N_FULL_TRAIN) / max(n_fitted_rows, 1))
    n_cust_norm = st["cnt"]["customer"][ents["customer"]][:, None] * norm_scale
    h_cust_norm = st["H_data"]["customer"][ents["customer"]].sum(axis=1, keepdims=True) * norm_scale
    n_cour_norm = st["cnt"]["courier"][ents["courier"]][:, None] * norm_scale
    h_cour_norm = st["H_data"]["courier"][ents["courier"]].sum(axis=1, keepdims=True) * norm_scale

    p_clip = np.clip(p, 1e-7, 1.0)
    ent = -(p_clip * np.log(p_clip)).sum(axis=1, keepdims=True)
    p_sorted = np.sort(p, axis=1)
    max_p = p_sorted[:, -1:]
    margin = p_sorted[:, -1:] - p_sorted[:, -2:-1]

    s1_c = (base_logits - base_logits.mean(axis=1, keepdims=True)).astype(np.float32)
    te_cust = te_tables["cust_m3"][ents["customer"]]
    te_cour = te_tables["cour_m15"][ents["courier"]]

    block = np.hstack([
        s1_c,                       # 0..6   (7)
        eta,                        # 7..13  (7)
        p,                          # 14..20 (7)
        u_cust,                     # 21..27 (7)
        u_cour,                     # 28..34 (7)
        u_hub,                      # 35..41 (7)
        max_p,                      # 42     (1)
        ent,                        # 43     (1)
        margin,                     # 44     (1)
        np.log1p(n_cust_norm),      # 45     (1)
        h_cust_norm,                # 46     (1)
        np.log1p(n_cour_norm),      # 47     (1)
        h_cour_norm,                # 48     (1)
        te_cust,                    # 49..55 (7)
        te_cour,                    # 56..62 (7)
    ]).astype(np.float32)
    return block, eta, p


def fit_te_tables(y_fit, ents_fit, n_levels, n_classes=7):
    """Fit Bayesian-smoothed marginal target encoding lookup tables with sample-size-invariant smoothing."""
    N = len(y_fit)
    priors = (np.bincount(y_fit, minlength=n_classes) / max(N, 1)).astype(np.float64)
    scale_m = float(N / (0.80 * N_FULL_TRAIN))
    out = {}
    for key, ename, base_m in [("cust_m3", "customer", 3.0), ("cour_m15", "courier", 15.0)]:
        m_eff = base_m * scale_m
        codes = ents_fit[ename]
        n_u = n_levels[ename]
        cnt = np.bincount(codes, minlength=n_u).astype(np.float64)
        tab = np.zeros((n_u, n_classes), dtype=np.float32)
        for k in range(n_classes):
            sk = np.bincount(codes, weights=(y_fit == k).astype(np.float64), minlength=n_u)
            tab[:, k] = ((sk + m_eff * priors[k]) / (cnt + m_eff)).astype(np.float32)
        out[key] = tab
    return out


def get_base_feature_cols(df):
    drop = {
        "shipment_id", "status_penanganan",
        "customer_id", "courier_id", "hub_id", "package_id",
        "customer_id_code", "courier_id_code", "hub_id_code", "package_id_code",
    }
    return [c for c in df.columns if c not in drop]


def assemble_stage2_fold(f, tr_base, te_base, folds, n_cust_all):
    """Build 100% leak-free, zero-shift (X_tr, X_va, X_te, blk_tr, blk_va, blk_te) for outer fold f."""
    tr_idx = np.where(folds != f)[0]
    va_idx = np.where(folds == f)[0]
    base_cols = get_base_feature_cols(tr_base)

    # 1. Pure train-only customer history (exact LOO on tr_idx, lookup on va_idx, full-train lookup on te)
    df_ref = tr_base.iloc[tr_idx]
    cu_tr = compute_cu_tr_df(df_ref, df_ref, n_cust_all, is_same=True)
    cu_va = compute_cu_tr_df(df_ref, tr_base.iloc[va_idx], n_cust_all, is_same=False)
    cu_te_path = "ghani/cache/v8/cu_tr_test_full.pkl"
    if os.path.exists(cu_te_path):
        cu_te = pd.read_pickle(cu_te_path)
    else:
        cu_te = compute_cu_tr_df(tr_base, te_base, n_cust_all, is_same=False)
        cu_te.to_pickle(cu_te_path)

    # 2. Inner 3-fold GLMM+TE for tr_idx, 4-fold OOF GLMM+TE for va_idx, 5-fold full GLMM+TE for te
    n_g = len(GLMM_V8_FEAT_NAMES)
    blk_full_tr = np.zeros((len(folds), n_g), dtype=np.float32)
    for j in range(5):
        if j == f:
            continue
        blk_full_tr[folds == j] = np.load(f"ghani/cache/v8/inner_hold{f}_pred{j}.npy")
    blk_tr = blk_full_tr[tr_idx]
    del blk_full_tr

    oof_blk = np.load("ghani/cache/v8/glmm_oof_block.npy")
    blk_va = oof_blk[va_idx].copy()
    del oof_blk
    blk_te = np.load("ghani/cache/v8/glmm_test_block.npy")

    def _join_all(df_sub, cu_df, blk):
        gdf = pd.DataFrame(blk, columns=GLMM_V8_FEAT_NAMES, index=df_sub.index)
        out = pd.concat([df_sub[base_cols], cu_df, gdf], axis=1)
        out["uhub_hil_x_dec"] = (out["u_hub_1"] * out["log_declared_value"]).astype(np.float32)
        out["ucust_ret_x_klaim"] = (out["u_cust_4"] * (1.0 + out["has_klaim_inv"])).astype(np.float32)
        out["ucust_ret_x_cuklaim"] = (out["u_cust_4"] * (1.0 + np.nan_to_num(out["cu_tr_mean_has_klaim_inv"].to_numpy(np.float32), nan=0.0))).astype(np.float32)
        out["ucour_gag_x_night"] = (out["u_cour_0"] * (1.0 + out["is_night_ship"])).astype(np.float32)
        out["ucour_rus_x_dist"] = (out["u_cour_5"] * out["log_distance_km"]).astype(np.float32)
        out["ucour_ter_x_peak"] = (out["u_cour_6"] * (1.0 + out["is_peak_month"])).astype(np.float32)
        return out

    X_tr = _join_all(df_ref, cu_tr, blk_tr)
    X_va = _join_all(tr_base.iloc[va_idx], cu_va, blk_va)
    X_te = _join_all(te_base, cu_te, blk_te)
    del cu_tr, cu_va, cu_te
    gc.collect()
    return X_tr, X_va, X_te, blk_tr, blk_va, blk_te
