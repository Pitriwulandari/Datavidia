"""Shared Stage-2 feature assembly & leak-free fold builder for LightGBM, CatBoost, XGBoost, and PyTorch ResNet."""
import gc
import os

import numpy as np
import pandas as pd

from build_stage2_glmm import GLMM_FEAT_NAMES


def load_stage2_base():
    """Load pruned feat_v5 + add 14 high-signal domain interaction features. Returns (X_tr, X_te, y, folds, classes, te_sid, cat_cols)."""
    tr = pd.read_pickle("ghani/task1/cache/v5/feat_v5_train.pkl")
    te = pd.read_pickle("ghani/task1/cache/v5/feat_v5_test.pkl")

    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    folds = np.load("ghani/task1/cache/v5/folds_v5.npy")
    te_sid = te["shipment_id"].copy()

    # Prune ~35 zero/near-zero gain noise columns (keep all columns with gain >= 0.02% in Stage-1 + key binary flags)
    imp = pd.read_csv("ghani/task1/cache/v5/stage1/importance_f0.csv", index_col=0).iloc[:, 0]
    keep_base = [c for c in imp.index if imp[c] >= 0.02]
    # Always keep key domain flags even if split via NaN in Stage-1
    for must in ["has_klaim_inv", "is_cod", "inv_any_penalty", "base_inv_ps", "scan_has_selesai", "scan_q_min"]:
        if must in tr.columns and must not in keep_base:
            keep_base.append(must)

    tr = tr[keep_base].copy()
    te = te[keep_base].copy()
    gc.collect()

    # Add 14 new verified high-signal domain interactions!
    for df in (tr, te):
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

    cat_cols = [c for c in tr.columns if str(tr[c].dtype) == "category"]
    return tr, te, y, folds, classes, te_sid, cat_cols


def get_fold_glmm_arrays(f, folds):
    """Return strictly leak-free (blk_tr, blk_va, blk_te) float32 arrays for outer fold f."""
    tr_idx = np.where(folds != f)[0]
    va_idx = np.where(folds == f)[0]

    # Assemble inner-fold leak-free GLMM block for tr_idx
    blk_full_tr = np.zeros((len(folds), len(GLMM_FEAT_NAMES)), dtype=np.float32)
    for j in range(5):
        if j == f:
            continue
        idx_j = np.where(folds == j)[0]
        blk_full_tr[idx_j] = np.load(f"ghani/task1/cache/v6/inner_hold{f}_pred{j}.npy")
    blk_tr = blk_full_tr[tr_idx]
    del blk_full_tr

    oof_blk = np.load("ghani/task1/cache/v6/glmm_oof_block.npy")
    blk_va = oof_blk[va_idx].copy()
    del oof_blk

    blk_te = np.load("ghani/task1/cache/v6/glmm_test_block.npy")
    return blk_tr, blk_va, blk_te


def attach_glmm_df(df_base, glmm_blk):
    """Attach 49-col GLMM block + 5 key GLMM x domain interaction features to DataFrame."""
    gdf = pd.DataFrame(glmm_blk, columns=GLMM_FEAT_NAMES, index=df_base.index)
    out = pd.concat([df_base, gdf], axis=1)
    # High-value non-linear interactions between Joint GLMM random effects & shipment features!
    out["uhub_hil_x_dec"] = (out["u_hub_1"] * out["log_declared_value"]).astype(np.float32)
    out["ucust_ret_x_klaim"] = (out["u_cust_4"] * (1.0 + out["has_klaim_inv"])).astype(np.float32)
    out["ucour_gag_x_night"] = (out["u_cour_0"] * (1.0 + out["is_night_ship"])).astype(np.float32)
    out["ucour_rus_x_dist"] = (out["u_cour_5"] * out["log_distance_km"]).astype(np.float32)
    out["ucour_ter_x_peak"] = (out["u_cour_6"] * (1.0 + out["is_peak_month"])).astype(np.float32)
    return out
