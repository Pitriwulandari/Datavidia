"""Stage-2 Leak-Free GLMM Feature Generator (uses C(5,2)=10 symmetry for 100% leak-free inner-4-fold GLMM).

Also performs coordinate optimization of the class-specific Joint GLMM precision matrix on full 5-fold OOF
to squeeze every last 0.001 out of the pure Bayesian GLMM model before feeding into Stage-2 ensembles!

Jalankan setelah Stage-1 5-Fold selesai:
  python -u -W ignore ghani/src/build_stage2_glmm.py
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, log_loss

for _sub in ['common', 'v1', 'v2', 'v5', 'v6', 'v7']:
    _p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', _sub))
    if _p not in sys.path:
        sys.path.insert(0, _p)
import glmm_lib as glmm
from run_stage1_glmm import build_entity_codes

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def extract_glmm_block(base_logits, ents, st, n_classes=7):
    """Extract 49-column float32 GLMM feature block for any subset of rows."""
    eta, p = glmm.predict_glmm(base_logits, ents, st, use_moderation=False)
    K = n_classes
    u_cust = st["U"]["customer"][ents["customer"]]
    u_cour = st["U"]["courier"][ents["courier"]]
    u_hub = st["U"]["hub"][ents["hub"]] + st["U"]["hub_peak"][ents["hub_peak"]]

    # Exact Fisher information & seen counts (tells Stage-2 how reliable U is!)
    n_cust = st["cnt"]["customer"][ents["customer"]][:, None]
    h_cust = st["H_data"]["customer"][ents["customer"]].sum(axis=1, keepdims=True)
    n_cour = st["cnt"]["courier"][ents["courier"]][:, None]
    h_cour = st["H_data"]["courier"][ents["courier"]].sum(axis=1, keepdims=True)

    p_clip = np.clip(p, 1e-7, 1.0)
    ent = -(p_clip * np.log(p_clip)).sum(axis=1, keepdims=True)
    p_sorted = np.sort(p, axis=1)
    max_p = p_sorted[:, -1:]
    margin = p_sorted[:, -1:] - p_sorted[:, -2:-1]

    s1_c = (base_logits - base_logits.mean(axis=1, keepdims=True)).astype(np.float32)
    block = np.hstack([
        s1_c,          # 0..6   (7)
        eta,           # 7..13  (7)
        p,             # 14..20 (7)
        u_cust,        # 21..27 (7)
        u_cour,        # 28..34 (7)
        u_hub,         # 35..41 (7)
        max_p,         # 42     (1)
        ent,           # 43     (1)
        margin,        # 44     (1)
        np.log1p(n_cust),  # 45 (1)
        h_cust,        # 46     (1)
        np.log1p(n_cour),  # 47 (1)
        h_cour,        # 48     (1)
    ]).astype(np.float32)
    return block, eta, p


GLMM_FEAT_NAMES = (
    [f"s1_logit_{k}" for k in range(7)]
    + [f"glmm_logit_{k}" for k in range(7)]
    + [f"glmm_prob_{k}" for k in range(7)]
    + [f"u_cust_{k}" for k in range(7)]
    + [f"u_cour_{k}" for k in range(7)]
    + [f"u_hub_{k}" for k in range(7)]
    + ["glmm_max_p", "glmm_entropy", "glmm_margin", "log_n_cust_tr", "h_cust_tot", "log_n_cour_tr", "h_cour_tot"]
)


def main():
    os.makedirs("ghani/cache/v6", exist_ok=True)
    os.makedirs("ghani/models/v6", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/v5/feat_v5_train.pkl")[["status_penanganan", "hub_id_code", "courier_id_code", "customer_id_code", "is_peak_month"]]
    te = pd.read_pickle("ghani/cache/v5/feat_v5_test.pkl")[["shipment_id", "hub_id_code", "courier_id_code", "customer_id_code", "is_peak_month"]]
    classes = sorted(tr["status_penanganan"].unique())
    y = tr["status_penanganan"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int32)
    folds = np.load("ghani/cache/v5/folds_v5.npy")
    K = len(classes)

    s1_oof = np.load("ghani/cache/v5/stage1/oof_logits_all.npy")
    s1_te = np.load("ghani/cache/v5/stage1/test_logits_all.npy")
    ents_tr, ents_te, n_levels = build_entity_codes(tr, te)
    log(f"loaded Stage-1 logits: oof={s1_oof.shape}, test={s1_te.shape}")

    # Step 1: Fine-tune per-entity, per-class prior precisions on Fold 0 OOF LogLoss (quick & surgical!)
    # Classes: [0:gagal, 1:klaim, 2:konfirmasi, 3:normal, 4:retur, 5:rusak, 6:tertunda]
    best_lam = {
        "hub":      np.array([30.0,  7.0, 60.0, 22.0, 60.0, 45.0, 14.0], dtype=np.float64),
        "hub_peak": np.array([90.0, 90.0, 90.0, 55.0, 90.0, 90.0, 28.0], dtype=np.float64),
        "courier":  np.array([ 5.0, 18.0, 45.0,  7.0, 45.0,  7.0,  5.0], dtype=np.float64),
        "customer": np.array([ 1.8,  6.5,  0.55, 1.05, 0.40, 2.0,  3.5], dtype=np.float64),
    }

    # Check wider customer/courier multipliers on Fold 0 using true OOF Stage-1 logits!
    # (OOF logits have larger residuals than in-sample logits, so optimal lambda is higher!)
    tr0, va0 = np.where(folds != 0)[0], np.where(folds == 0)[0]
    e_tr0 = {k: v[tr0] for k, v in ents_tr.items()}
    e_va0 = {k: v[va0] for k, v in ents_tr.items()}
    best_ll0 = 999.0
    chosen_lam = best_lam
    for cust_mult in [1.0, 1.5, 2.2, 3.2]:
        for cour_mult in [1.0, 1.8, 3.0]:
            cand = {
                "hub": best_lam["hub"],
                "hub_peak": best_lam["hub_peak"],
                "courier": best_lam["courier"] * cour_mult,
                "customer": best_lam["customer"] * cust_mult,
            }
            st = glmm.fit_joint_glmm(s1_oof[tr0], y[tr0], e_tr0, n_levels, n_classes=K, fixed_lam=cand, n_newton=3, verbose=False)
            _, p0 = glmm.predict_glmm(s1_oof[va0], e_va0, st, use_moderation=False)
            ll0 = log_loss(y[va0], p0, labels=list(range(K)))
            f1_a = glmm.fast_macro_f1(y[va0], p0.argmax(1), K)
            log(f"  grid cust_x={cust_mult:.1f} cour_x={cour_mult:.1f} -> F0 val_ll={ll0:.5f} arg_F1={f1_a:.5f}")
            if ll0 < best_ll0:
                best_ll0 = ll0
                chosen_lam = cand
    # Also test per-class customer_lam coordinate refinement (1 pass) on Fold 0!
    log(f"Coarse best (F0 val_ll={best_ll0:.5f}); doing 1-pass per-class customer_lam polish...")
    for k in range(K):
        base_v = chosen_lam["customer"][k]
        for m in [0.6, 1.65]:
            cand = {ek: ev.copy() for ek, ev in chosen_lam.items()}
            cand["customer"][k] = base_v * m
            st = glmm.fit_joint_glmm(s1_oof[tr0], y[tr0], e_tr0, n_levels, n_classes=K, fixed_lam=cand, n_newton=3, verbose=False)
            _, p0 = glmm.predict_glmm(s1_oof[va0], e_va0, st, use_moderation=False)
            ll0 = log_loss(y[va0], p0, labels=list(range(K)))
            if ll0 < best_ll0 - 2e-5:
                best_ll0 = ll0
                chosen_lam = cand
                log(f"  improved cust_lam[{k}] -> {chosen_lam['customer'][k]:.2f} | F0 val_ll={best_ll0:.5f}")
    log(f"Selected best_lam (F0 val_ll={best_ll0:.5f}): " + str({k: np.round(v, 2).tolist() for k, v in chosen_lam.items()}))

    # Step 2: Fit all 5 outer-fold GLMMs (4 folds -> 1 val fold) AND 1 full GLMM (all 5 folds -> test)
    oof_block = np.zeros((len(y), len(GLMM_FEAT_NAMES)), dtype=np.float32)
    glmm_oof_p = np.zeros((len(y), K), dtype=np.float32)
    glmm_oof_eta = np.zeros((len(y), K), dtype=np.float32)
    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        e_tr = {k: v[tr_idx] for k, v in ents_tr.items()}
        e_va = {k: v[va_idx] for k, v in ents_tr.items()}
        st = glmm.fit_joint_glmm(s1_oof[tr_idx], y[tr_idx], e_tr, n_levels, n_classes=K, fixed_lam=chosen_lam, n_newton=4, verbose=False)
        blk, eta_v, p_v = extract_glmm_block(s1_oof[va_idx], e_va, st, K)
        oof_block[va_idx] = blk
        glmm_oof_eta[va_idx] = eta_v
        glmm_oof_p[va_idx] = p_v
        log(f"Outer GLMM Fold {f}: val_ll={log_loss(y[va_idx], p_v, labels=list(range(K))):.5f} arg_F1={glmm.fast_macro_f1(y[va_idx], p_v.argmax(1), K):.5f}")

    full_ll = log_loss(y, glmm_oof_p, labels=list(range(K)))
    honest_f1, w_opt, oof_pred = glmm.cv_macro_f1_with_weights(glmm_oof_p, y, folds)
    log(f"=== TUNED 5-FOLD GLMM: OOF_ll={full_ll:.5f} | arg_F1={glmm.fast_macro_f1(y, glmm_oof_p.argmax(1), K):.5f} | CV_tuned_F1={honest_f1:.5f} ===")
    print(classification_report(y, oof_pred, target_names=classes, digits=4), flush=True)

    st_full = glmm.fit_joint_glmm(s1_oof, y, ents_tr, n_levels, n_classes=K, fixed_lam=chosen_lam, n_newton=5, verbose=False)
    te_block, glmm_te_eta, glmm_te_p = extract_glmm_block(s1_te, ents_te, st_full, K)

    np.save("ghani/cache/v6/glmm_oof_block.npy", oof_block)
    np.save("ghani/cache/v6/glmm_test_block.npy", te_block)
    np.save("ghani/cache/v6/glmm_oof_proba.npy", glmm_oof_p)
    np.save("ghani/cache/v6/glmm_test_proba.npy", glmm_te_p)
    np.save("ghani/models/v6/glmm_state_full.npy", st_full, allow_pickle=True)

    # Step 3: C(5,2) = 10 leak-free inner-fold GLMM blocks!
    # For each unordered pair (i, j) with i < j, fit GLMM on the 3 folds {0..4} \ {i, j}.
    # Save block for fold i (used when outer val is j) and block for fold j (used when outer val is i)!
    log("Building C(5,2)=10 leak-free 3-fold GLMM inner blocks for Stage-2 training...")
    for i in range(5):
        for j in range(i + 1, 5):
            tr3 = np.where((folds != i) & (folds != j))[0]
            e_tr3 = {k: v[tr3] for k, v in ents_tr.items()}
            st3 = glmm.fit_joint_glmm(s1_oof[tr3], y[tr3], e_tr3, n_levels, n_classes=K, fixed_lam=chosen_lam, n_newton=4, verbose=False)
            for target_f, holdout_f in [(i, j), (j, i)]:
                idx_t = np.where(folds == target_f)[0]
                e_t = {k: v[idx_t] for k, v in ents_tr.items()}
                blk_t, _, _ = extract_glmm_block(s1_oof[idx_t], e_t, st3, K)
                np.save(f"ghani/cache/v6/inner_hold{holdout_f}_pred{target_f}.npy", blk_t)
            log(f"  done pair ({i},{j})")

    # Save tuned 5-Fold Stage-1 + Full-Data Joint GLMM submission (v5)!
    pred_te = (glmm_te_p * w_opt).argmax(1)
    sub = pd.DataFrame({"shipment_id": te["shipment_id"], "status_penanganan": [classes[i] for i in pred_te]})
    sub.to_csv("submissions/submission_v5_ghani.csv", index=False)
    json.dump(
        {"classes": classes, "weights": w_opt.tolist(), "glmm_oof_ll": float(full_ll),
         "glmm_cv_f1": float(honest_f1), "chosen_lam": {k: v.tolist() for k, v in chosen_lam.items()},
         "glmm_feat_names": GLMM_FEAT_NAMES},
        open("ghani/models/v6/meta_glmm.json", "w"), indent=2,
    )
    log("DONE build_stage2_glmm.py!")


if __name__ == "__main__":
    main()
