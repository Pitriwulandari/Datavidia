"""
Stage-1 LightGBM (Single Train/Valid + Early Stopping) for TASK-1-exp.
Uses entity-augmented features. Produces OOF-style preds on valid + test preds.
"""
import gc
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def main():
    log("Loading entity-augmented features...")
    tr = pd.read_parquet("data/processed/feat_train_entity.parquet")
    te = pd.read_parquet("data/processed/feat_test_entity.parquet")
    log(f"Train: {tr.shape}, Test: {te.shape}")

    target = "status_penanganan"
    y = tr[target].astype("category")
    y_codes = y.cat.codes
    n_classes = len(y.cat.categories)
    log(f"Classes ({n_classes}): {list(y.cat.categories)}")

    # Drop non-feature columns
    drop_cols = [target, "shipment_id"]
    # Also drop any remaining object/string columns
    X = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
    X_te = te.drop(columns=[c for c in ["shipment_id"] if c in te.columns])

    # Handle categorical: LightGBM native
    cat_cols = X.select_dtypes(include=["category", "object", "string"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        if c in X_te.columns:
            X_te[c] = X_te[c].astype("category")
    log(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # Numeric: fill NaN with median
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)
        if c in X_te.columns:
            X_te[c] = X_te[c].fillna(med)

    # Align columns
    common_cols = [c for c in X.columns if c in X_te.columns]
    X = X[common_cols]
    X_te = X_te[common_cols]
    log(f"Final features: {len(common_cols)}")

    # Train/Valid split (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_codes, test_size=0.1, random_state=42, stratify=y_codes
    )
    log(f"Train: {X_tr.shape}, Valid: {X_val.shape}")

    # LightGBM params (from Ghani v7 but simpler)
    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "n_jobs": 2,
        "seed": 42,
        "force_col_wise": True,
    }

    # Class weights via sample_weight
    class_weight_map = {
        0: 1.2,  # gagal_antar_awal
        1: 1.5,  # klaim_hilang
        2: 0.7,  # konfirmasi_alamat
        3: 0.7,  # normal
        4: 1.5,  # retur_pengirim
        5: 1.5,  # rusak_perjalanan
        6: 0.7,  # tertunda_operasional
    }
    tr_weight = np.array([class_weight_map[int(v)] for v in y_tr])
    val_weight = np.array([class_weight_map[int(v)] for v in y_val])

    log("Training LightGBM...")
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=tr_weight, categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, weight=val_weight, categorical_feature=cat_cols, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.log_evaluation(50),
            lgb.early_stopping(100, verbose=True),
        ],
    )
    best_iter = model.best_iteration
    log(f"Best iteration: {best_iter}")

    # Predict on valid (for OOF-style)
    val_logits = model.predict(X_val, num_iteration=best_iter)
    val_pred = np.argmax(val_logits, axis=1)
    val_f1 = macro_f1(y_val, val_pred)
    log(f"Valid Macro F1: {val_f1:.6f}")

    # Per-class F1
    for i, cls in enumerate(y.cat.categories):
        mask = y_val == i
        if mask.sum() > 0:
            f1c = f1_score(y_val[mask], val_pred[mask], average="macro")
            log(f"  {cls}: F1={f1c:.4f} (n={mask.sum()})")

    # Predict on test
    test_logits = model.predict(X_te, num_iteration=best_iter)
    test_pred = np.argmax(test_logits, axis=1)

    # Save
    os.makedirs("models", exist_ok=True)
    model.save_model("models/stage1_lgbm.txt")
    np.save("models/stage1_val_logits.npy", val_logits)
    np.save("models/stage1_val_true.npy", y_val.to_numpy())
    np.save("models/stage1_test_logits.npy", test_logits)
    np.save("models/stage1_test_pred.npy", test_pred)
    log("SAVED: stage1_lgbm.txt, val_logits, val_true, test_logits, test_pred")

    # Feature importance
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv("models/stage1_feature_importance.csv", index=False)
    log("Top 20 features:")
    print(imp.head(20).to_string(index=False))


if __name__ == "__main__":
    main()