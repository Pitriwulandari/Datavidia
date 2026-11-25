"""Stage-2 Leak-Free Scale-Normalized Joint GLMM + Multi-Scale Bayesian TE Builder for v8.

Fixes BOTH bugs of v6's GLMM stage:
  1. Stage-1 `cu_tr_*` never touched `test` rows -> ZERO negative self-cancellation (`corr = 0.000`, positive alignment on test!).
  2. Counts & Fisher information (`log_n_cust_norm`, `h_cust_norm`, `log_n_cour_norm`, `h_cour_norm`) AND Bayesian TE smoothing
     are normalized to exact 4-fold scale (`0.80 * N_train`) across 3-fold inner, 4-fold val, and 5-fold full test -> 0.000 shift!

Jalankan setelah `run_stage1_v8.py`:
  python -u -W ignore ghani/src/v8/build_glmm_v8.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss

for _sub in ["common", "v8"]:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glmm_lib as glmm
import v8_common as v8c

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def build_entity_codes(tr, te):
    ents_tr, ents_te, n_levels = {}, {}, {}
    for name, col in [("hub", "hub_id_code"), ("courier", "courier_id_code"), ("customer", "customer_id_code")]:
        c_tr = tr[col].to_numpy(dtype=np.int32)
        c_te = te[col].to_numpy(dtype=np.int32)
        ents_tr[name] = c_tr
        ents_te[name] = c_te
        n_levels[name] = int(max(c_tr.max(), c_te.max())) + 1
    ents_tr["hub_peak"] = (tr["hub_id_code"].to_numpy(np.int32) * 2 + tr["is_peak_month"].to_numpy(np.int32)).astype(np.int32)
    ents_te["hub_peak"] = (te["hub_id_code"].to_numpy(np.int32) * 2 + te["is_peak_month"].to_numpy(np.int32)).astype(np.int32)
    n_levels["hub_peak"] = n_levels["hub"] * 2
    return ents_tr, ents_te, n_levels


def main():
    os.makedirs("ghani/cache/v8", exist_ok=True)
    os.makedirs("ghani/models/v8", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v8/feat_v8_base_train.pkl")[["status_penanganan", "hub_id_code", "courier_id_code", "customer_id_code", "is_peak_month"]]
    te = pd.read_pickle("ghani/cache/v8/feat_v8_base_test.pkl")[["shipment_id", "hub_id_code", "courier_id_code", "customer_id_code", "is_peak_month", "jumlah_catatan", "scan_cnt_upaya", "has_klaim_inv"]]
    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    folds = np.load("ghani/cache/v8/folds_v8.npy")
    K = len(classes)

    s1_oof = np.load("ghani/cache/v8/stage1/oof_logits_all.npy")
    s1_te = np.load("ghani/cache/v8/stage1/test_logits_all.npy")
    ents_tr, ents_te, n_levels = build_entity_codes(tr, te)
    log(f"Loaded Stage-1 v8 logits: oof={s1_oof.shape}, test={s1_te.shape}")

    base_lam = {
        "hub":      np.array([30.0,  7.0, 60.0, 22.0, 60.0, 45.0, 14.0], dtype=np.float64),
        "hub_peak": np.array([90.0, 90.0, 90.0, 55.0, 90.0, 90.0, 28.0], dtype=np.float64),
        "courier":  np.array([ 5.0, 18.0, 45.0,  7.0, 45.0,  7.0,  5.0], dtype=np.float64),
        "customer": np.array([ 1.8,  6.5,  0.55, 1.05, 0.40, 2.0,  3.5], dtype=np.float64),
    }

    tr0, va0 = np.where(folds != 0)[0], np.where(folds == 0)[0]
    e_tr0 = {k: v[tr0] for k, v in ents_tr.items()}
    e_va0 = {k: v[va0] for k, v in ents_tr.items()}
    best_ll0 = 999.0
    chosen_lam = base_lam
    for cust_mult in [0.8, 1.2, 1.8, 2.6, 3.6]:
        for cour_mult in [1.0, 1.8, 3.0]:
            cand = {
                "hub": base_lam["hub"],
                "hub_peak": base_lam["hub_peak"],
                "courier": base_lam["courier"] * cour_mult,
                "customer": base_lam["customer"] * cust_mult,
            }
            st = glmm.fit_joint_glmm(s1_oof[tr0], y[tr0], e_tr0, n_levels, n_classes=K, fixed_lam=cand, n_newton=3, verbose=False)
            _, p0 = glmm.predict_glmm(s1_oof[va0], e_va0, st, use_moderation=False)
            ll0 = log_loss(y[va0], p0, labels=list(range(K)))
            if ll0 < best_ll0:
                best_ll0 = ll0
                chosen_lam = cand
    for k in range(K):
        bv = chosen_lam["customer"][k]
        for m in [0.65, 1.55]:
            cand = {ek: ev.copy() for ek, ev in chosen_lam.items()}
            cand["customer"][k] = bv * m
            st = glmm.fit_joint_glmm(s1_oof[tr0], y[tr0], e_tr0, n_levels, n_classes=K, fixed_lam=cand, n_newton=3, verbose=False)
            _, p0 = glmm.predict_glmm(s1_oof[va0], e_va0, st, use_moderation=False)
            ll0 = log_loss(y[va0], p0, labels=list(range(K)))
            if ll0 < best_ll0 - 1e-5:
                best_ll0 = ll0
                chosen_lam = cand
    log(f"Selected chosen_lam (Fold0 val_ll={best_ll0:.5f}): " + str({k: np.round(v, 2).tolist() for k, v in chosen_lam.items()}))

    # 1. Fit 5 outer-fold GLMM + TE blocks (4 folds -> 1 val fold)
    n_g = len(v8c.GLMM_V8_FEAT_NAMES)
    oof_block = np.zeros((len(y), n_g), dtype=np.float32)
    glmm_oof_p = np.zeros((len(y), K), dtype=np.float32)
    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        e_tr = {k: v[tr_idx] for k, v in ents_tr.items()}
        e_va = {k: v[va_idx] for k, v in ents_tr.items()}
        st = glmm.fit_joint_glmm(s1_oof[tr_idx], y[tr_idx], e_tr, n_levels, n_classes=K, fixed_lam=chosen_lam, n_newton=4, verbose=False)
        te_tabs = v8c.fit_te_tables(y[tr_idx], e_tr, n_levels, n_classes=K)
        blk, _, p_v = v8c.extract_glmm_and_te_block(s1_oof[va_idx], e_va, st, te_tabs, len(tr_idx), n_classes=K)
        oof_block[va_idx] = blk
        glmm_oof_p[va_idx] = p_v
        log(f"Outer GLMM v8 Fold {f}: val_ll={log_loss(y[va_idx], p_v, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], p_v.argmax(1), K):.5f}")

    full_ll = log_loss(y, glmm_oof_p, labels=list(range(K)))
    honest_f1, w_opt, oof_pred = glmm.cv_macro_f1_with_weights(glmm_oof_p, y, folds)
    log(f"=== 5-FOLD GLMM v8: OOF_ll={full_ll:.5f} | arg_F1={glmm.fast_macro_f1(y, glmm_oof_p.argmax(1), K):.5f} | CV_tuned_F1={honest_f1:.5f} ===")
    print(classification_report(y, oof_pred, target_names=classes, digits=4), flush=True)

    # 2. Fit Full-Train GLMM + TE (100% of train -> test), with exact scale normalization!
    st_full = glmm.fit_joint_glmm(s1_oof, y, ents_tr, n_levels, n_classes=K, fixed_lam=chosen_lam, n_newton=5, verbose=False)
    te_tabs_full = v8c.fit_te_tables(y, ents_tr, n_levels, n_classes=K)
    te_block, _, glmm_te_p = v8c.extract_glmm_and_te_block(s1_te, ents_te, st_full, te_tabs_full, len(y), n_classes=K)

    # Mandatory sanity check: ZERO negative self-cancellation on test AND 0.000 shift on normalized counts!
    u_te = st_full["U"]["customer"][ents_te["customer"]]
    log(f"SANITY CHECK u_cust_v8 on TEST (must be >= 0, NOT -0.105 like v6!):")
    log(f"  corr(u_te_konf, te.scan_cnt_upaya) = {np.corrcoef(u_te[:, 2], te['scan_cnt_upaya'])[0, 1]:+.4f}")
    log(f"  corr(u_te_konf, te.jumlah_catatan) = {np.corrcoef(u_te[:, 2], te['jumlah_catatan'])[0, 1]:+.4f}")
    log(f"  corr(u_te_ret,  te.has_klaim_inv ) = {np.corrcoef(u_te[:, 4], te['has_klaim_inv'])[0, 1]:+.4f}")
    log(f"  Scale check: oof log_n_cour_norm={oof_block[:,47].mean():.4f} vs te={te_block[:,47].mean():.4f} | oof h_cour_norm={oof_block[:,48].mean():.4f} vs te={te_block[:,48].mean():.4f}")

    np.save("ghani/cache/v8/glmm_oof_block.npy", oof_block)
    np.save("ghani/cache/v8/glmm_test_block.npy", te_block)
    np.save("ghani/cache/v8/glmm_oof_proba.npy", glmm_oof_p)
    np.save("ghani/cache/v8/glmm_test_proba.npy", glmm_te_p)
    np.save("ghani/models/v8/glmm_state_full.npy", st_full, allow_pickle=True)
    np.savez_compressed("ghani/models/v8/te_tables_full.npz", **te_tabs_full)

    # 3. C(5,2) = 10 leak-free inner 3-fold GLMM+TE blocks (exact same normalized scale!)
    log("Building C(5,2)=10 leak-free scale-normalized 3-fold inner blocks...")
    for i in range(5):
        for j in range(i + 1, 5):
            tr3 = np.where((folds != i) & (folds != j))[0]
            e_tr3 = {k: v[tr3] for k, v in ents_tr.items()}
            st3 = glmm.fit_joint_glmm(s1_oof[tr3], y[tr3], e_tr3, n_levels, n_classes=K, fixed_lam=chosen_lam, n_newton=4, verbose=False)
            te3 = v8c.fit_te_tables(y[tr3], e_tr3, n_levels, n_classes=K)
            for target_f, holdout_f in [(i, j), (j, i)]:
                idx_t = np.where(folds == target_f)[0]
                e_t = {k: v[idx_t] for k, v in ents_tr.items()}
                blk_t, _, _ = v8c.extract_glmm_and_te_block(s1_oof[idx_t], e_t, st3, te3, len(tr3), n_classes=K)
                np.save(f"ghani/cache/v8/inner_hold{holdout_f}_pred{target_f}.npy", blk_t)
            log(f"  done pair ({i},{j}) | inner3 h_cour_norm={blk_t[:,48].mean():.4f}")

    json.dump(
        {
            "classes": classes,
            "weights": w_opt.tolist(),
            "glmm_oof_ll": float(full_ll),
            "glmm_cv_f1": float(honest_f1),
            "chosen_lam": {k: v.tolist() for k, v in chosen_lam.items()},
            "glmm_feat_names": v8c.GLMM_V8_FEAT_NAMES,
            "n_fitted_rows_full": int(len(y)),
        },
        open("ghani/models/v8/meta_glmm.json", "w"),
        indent=2,
    )
    log("DONE build_glmm_v8.py!")


if __name__ == "__main__":
    main()
