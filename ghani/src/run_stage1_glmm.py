"""Stage-1 5-Fold LightGBM (Non-ID Prior) + Class-Specific Joint Hierarchical Multinomial GLMM.

Jalankan (dari root repo):
  python -u -W ignore ghani/src/run_stage1_glmm.py
  $env:FOLDS='1,2,3,4'; python -u -W ignore ghani/src/run_stage1_glmm.py
"""
import gc
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glmm_lib as glmm

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def get_feature_cols(tr):
    drop = {
        "shipment_id", "status_penanganan",
        "customer_id", "courier_id", "hub_id", "package_id",
        "customer_id_code", "courier_id_code", "hub_id_code", "package_id_code",
    }
    return [c for c in tr.columns if c not in drop]


def build_entity_codes(tr, te):
    """Build compact int32 entity code dicts for Joint GLMM."""
    ents_tr, ents_te, n_levels = {}, {}, {}
    for name, col in [("hub", "hub_id_code"), ("courier", "courier_id_code"), ("customer", "customer_id_code")]:
        c_tr = tr[col].to_numpy(dtype=np.int32)
        c_te = te[col].to_numpy(dtype=np.int32)
        ents_tr[name] = c_tr
        ents_te[name] = c_te
        n_levels[name] = int(max(c_tr.max(), c_te.max())) + 1

    hp_tr = (tr["hub_id_code"].to_numpy(np.int32) * 2 + tr["is_peak_month"].to_numpy(np.int32)).astype(np.int32)
    hp_te = (te["hub_id_code"].to_numpy(np.int32) * 2 + te["is_peak_month"].to_numpy(np.int32)).astype(np.int32)
    ents_tr["hub_peak"] = hp_tr
    ents_te["hub_peak"] = hp_te
    n_levels["hub_peak"] = n_levels["hub"] * 2
    return ents_tr, ents_te, n_levels


def main():
    os.makedirs("ghani/models/stage1", exist_ok=True)
    os.makedirs("ghani/cache/stage1", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    tr = pd.read_pickle("ghani/cache/feat_v5_train.pkl")
    te = pd.read_pickle("ghani/cache/feat_v5_test.pkl")
    log(f"loaded train {tr.shape}, test {te.shape}")

    classes = sorted(tr["status_penanganan"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    y = tr["status_penanganan"].map(cmap).to_numpy(dtype=np.int32)
    K = len(classes)

    te_sid = te["shipment_id"].copy()
    ents_tr, ents_te, n_levels = build_entity_codes(tr, te)
    feats = get_feature_cols(tr)
    cat_cols = [c for c in feats if str(tr[c].dtype) == "category"]

    # Free heavy string ID columns immediately to minimize RAM!
    drop_ids = [c for c in ["shipment_id", "status_penanganan", "customer_id", "courier_id", "hub_id", "package_id",
                            "customer_id_code", "courier_id_code", "hub_id_code", "package_id_code"] if c in tr.columns]
    tr.drop(columns=drop_ids, inplace=True)
    te.drop(columns=[c for c in drop_ids if c in te.columns], inplace=True)
    gc.collect()

    folds_path = "ghani/cache/folds_v5.npy"
    if os.path.exists(folds_path):
        folds = np.load(folds_path)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = np.zeros(len(tr), dtype=np.int32)
        for f, (_, va_idx) in enumerate(skf.split(np.zeros(len(tr)), y)):
            folds[va_idx] = f
        np.save(folds_path, folds)

    log(f"Stage-1 features: {len(feats)} (cat={len(cat_cols)})")

    req_folds = os.environ.get("FOLDS", "0,1,2,3,4")
    run_folds = [int(x.strip()) for x in req_folds.split(",") if x.strip() != ""]

    params = dict(
        objective="multiclass",
        num_class=K,
        learning_rate=0.075,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=10.0,
        reg_alpha=1.0,
        max_bin=255,
        cat_smooth=20.0,
        verbose=-1,
        n_jobs=os.cpu_count(),
    )

    for f in run_folds:
        m_path = f"ghani/models/stage1/lgb_f{f}.txt"
        oof_f_path = f"ghani/cache/stage1/oof_logits_f{f}.npy"
        te_f_path = f"ghani/cache/stage1/test_logits_f{f}.npy"
        if os.path.exists(m_path) and os.path.exists(oof_f_path) and os.path.exists(te_f_path) and os.environ.get("FORCE", "0") != "1":
            log(f"Fold {f}: already cached, skipping.")
            continue

        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        log(f"=== Stage-1 Fold {f}: train={len(tr_idx)}, val={len(va_idx)} ===")

        dtr = lgb.Dataset(tr.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols, free_raw_data=True)
        dva = lgb.Dataset(tr.iloc[va_idx], y[va_idx], reference=dtr, categorical_feature=cat_cols, free_raw_data=True)
        p_f = dict(params, random_state=42 + f)
        model = lgb.train(
            p_f, dtr, num_boost_round=1200, valid_sets=[dva],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
        )
        del dtr, dva
        gc.collect()

        model.save_model(m_path, num_iteration=model.best_iteration)
        va_logits = model.predict(tr.iloc[va_idx], num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        te_logits = model.predict(te, num_iteration=model.best_iteration, raw_score=True).astype(np.float32)
        np.save(oof_f_path, va_logits)
        np.save(te_f_path, te_logits)

        va_p = glmm.softmax(va_logits)
        ll = log_loss(y[va_idx], va_p, labels=list(range(K)))
        f1_arg = glmm.fast_macro_f1(y[va_idx], va_p.argmax(1), K)
        _, f1_tun = glmm.optimize_macro_f1_weights(va_p, y[va_idx], n_restarts=2)
        log(f"Stage-1 Fold {f} done (iter={model.best_iteration}): val_ll={ll:.5f} | argmax_F1={f1_arg:.5f} | tuned_F1={f1_tun:.5f}")
        del model
        gc.collect()

    all_exist = all(os.path.exists(f"ghani/cache/stage1/oof_logits_f{f}.npy") for f in range(5))
    if not all_exist:
        log("Not all 5 folds done yet; stopping here.")
        return

    del tr, te
    gc.collect()

    oof_logits = np.zeros((len(y), K), dtype=np.float32)
    te_logits = np.zeros((len(te_sid), K), dtype=np.float32)
    for f in range(5):
        va_idx = np.where(folds == f)[0]
        oof_logits[va_idx] = np.load(f"ghani/cache/stage1/oof_logits_f{f}.npy")
        te_logits += np.load(f"ghani/cache/stage1/test_logits_f{f}.npy") / 5.0

    np.save("ghani/cache/stage1/oof_logits_all.npy", oof_logits)
    np.save("ghani/cache/stage1/test_logits_all.npy", te_logits)

    s1_p = glmm.softmax(oof_logits)
    s1_ll = log_loss(y, s1_p, labels=list(range(K)))
    s1_arg = glmm.fast_macro_f1(y, s1_p.argmax(1), K)
    s1_honest, _, _ = glmm.cv_macro_f1_with_weights(s1_p, y, folds)
    log(f"=== FULL 5-FOLD STAGE-1: OOF_ll={s1_ll:.5f} | argmax_F1={s1_arg:.5f} | CV_tuned_F1={s1_honest:.5f} ===")

    # Class-specific Joint GLMM (validated optimal precisions)
    # Classes: [0:gagal, 1:klaim, 2:konfirmasi, 3:normal, 4:retur, 5:rusak, 6:tertunda]
    fixed_lam = {
        "hub":      np.array([25.0,  6.0, 50.0, 20.0, 50.0, 40.0, 12.0]),
        "hub_peak": np.array([80.0, 80.0, 80.0, 50.0, 80.0, 80.0, 25.0]),
        "courier":  np.array([ 4.0, 15.0, 40.0,  6.0, 40.0,  6.0,  4.0]),
        "customer": np.array([ 2.0,  8.0,  0.5,  1.1,  0.38, 2.2,  4.0]),
    }

    glmm_oof_logits = np.zeros((len(y), K), dtype=np.float32)
    glmm_oof_proba = np.zeros((len(y), K), dtype=np.float32)
    for f in range(5):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        e_tr = {k: v[tr_idx] for k, v in ents_tr.items()}
        e_va = {k: v[va_idx] for k, v in ents_tr.items()}
        st = glmm.fit_joint_glmm(oof_logits[tr_idx], y[tr_idx], e_tr, n_levels, n_classes=K, fixed_lam=fixed_lam, n_newton=4, verbose=False)
        va_eta, va_p = glmm.predict_glmm(oof_logits[va_idx], e_va, st, use_moderation=False)
        glmm_oof_logits[va_idx] = va_eta
        glmm_oof_proba[va_idx] = va_p
        log(f"  Joint GLMM Fold {f}: val_ll={log_loss(y[va_idx], va_p, labels=list(range(K))):.5f} | arg_F1={glmm.fast_macro_f1(y[va_idx], va_p.argmax(1), K):.5f}")

    g_ll = log_loss(y, glmm_oof_proba, labels=list(range(K)))
    g_arg = glmm.fast_macro_f1(y, glmm_oof_proba.argmax(1), K)
    g_honest, w_opt, oof_pred = glmm.cv_macro_f1_with_weights(glmm_oof_proba, y, folds)
    log(f"=== FULL 5-FOLD STAGE1+GLMM: OOF_ll={g_ll:.5f} | argmax_F1={g_arg:.5f} | CV_tuned_F1={g_honest:.5f} ===")
    print(classification_report(y, oof_pred, target_names=classes, digits=4), flush=True)

    # Fit FULL-TRAIN Joint GLMM on 100% of train (1,243,617 rows -> +25% more customer/courier data for test!)
    st_full = glmm.fit_joint_glmm(oof_logits, y, ents_tr, n_levels, n_classes=K, fixed_lam=fixed_lam, n_newton=5, verbose=True)
    glmm_te_logits_full, glmm_te_proba_full = glmm.predict_glmm(te_logits, ents_te, st_full, use_moderation=False)

    np.save("ghani/cache/stage1/glmm_oof_logits.npy", glmm_oof_logits)
    np.save("ghani/cache/stage1/glmm_oof_proba.npy", glmm_oof_proba)
    np.save("ghani/cache/stage1/glmm_test_logits_full.npy", glmm_te_logits_full)
    np.save("ghani/cache/stage1/glmm_test_proba_full.npy", glmm_te_proba_full)
    np.save("ghani/models/stage1/glmm_state_full.npy", st_full, allow_pickle=True)

    pred_te = (glmm_te_proba_full * w_opt).argmax(1)
    sub = pd.DataFrame({"shipment_id": te_sid, "status_penanganan": [classes[i] for i in pred_te]})
    sub.to_csv("submissions/submission_v5_ghani.csv", index=False)
    json.dump(
        {"classes": classes, "weights": w_opt.tolist(), "features": feats,
         "stage1_oof_ll": float(s1_ll), "stage1_cv_f1": float(s1_honest),
         "glmm_oof_ll": float(g_ll), "glmm_cv_f1": float(g_honest),
         "fixed_lam": {k: v.tolist() for k, v in fixed_lam.items()}},
        open("ghani/models/meta_v5.json", "w"), indent=2,
    )
    log("SAVED submissions/submission_v5_ghani.csv " + str(sub["status_penanganan"].value_counts(normalize=True).round(3).to_dict()))


if __name__ == "__main__":
    main()
