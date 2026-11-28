import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

SUB_PATH = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
BASE_FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "pitri/task2/cache/features_v2_spatial.parquet"

MODEL_DIR = "pitri/task2/models/v6"
OUT_CSV = "pitri/submissions/task2/submission_v6_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"

CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]


def zscore_arr(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-6 else 1.0)


def build_engineered_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c != "Id"]
    X_base = np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Fitur rasio & interaksi tambahan untuk mempertajam batas antar-fenomena
    tail_to_med_diff = (df["diff_mag_p95"].values + 1e-4) / (df["diff_mag_mean"].values + 1e-4)
    local_spike_x_tail = df["grid_max_to_med"].values * df["peak_to_mean_ratio"].values
    burn_x_veg_loss = df["dark_post_frac"].values * np.maximum(0.0, -df["dexg_mean"].values)
    ash_x_desat = df["ash_post_frac"].values * np.maximum(0.0, -df["dsat_mean"].values)
    water_x_edge_drop = df["dblue_mean"].values * np.maximum(0.0, -df["dedge_mean"].values)
    quake_edge_x_var = df["dedge_mean"].values * df["diff_mag_std"].values

    extra = np.column_stack([
        tail_to_med_diff,
        local_spike_x_tail,
        burn_x_veg_loss,
        ash_x_desat,
        water_x_edge_drop,
        quake_edge_x_var,
    ]).astype(np.float32)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([X_base, extra]).astype(np.float32)


def build_anchors(df: pd.DataFrame) -> np.ndarray:
    s_explosion = (
        1.55 * zscore_arr(df["peak_to_mean_ratio"].values)
        + 1.35 * zscore_arr(df["grid_max_to_med"].values)
        + 0.70 * zscore_arr(df["grid_std_to_mean"].values)
        - 0.50 * zscore_arr(df["diff_mag_mean"].values)
    )
    s_fire = (
        1.45 * zscore_arr(df["dark_post_frac"].values)
        - 1.15 * zscore_arr(df["dexg_mean"].values)
        - 0.55 * zscore_arr(df["dexg_p10"].values)
        - 0.25 * zscore_arr(df["grid_max_to_med"].values)
    )
    s_tsunami = (
        1.45 * zscore_arr(df["dblue_mean"].values)
        + 0.90 * zscore_arr(df["dblue_p90"].values)
        - 0.35 * zscore_arr(df["dedge_mean"].values)
    )
    s_volcano = (
        1.45 * zscore_arr(df["ash_post_frac"].values)
        + 0.65 * zscore_arr(df["dval_mean"].values)
        - 0.65 * zscore_arr(df["dsat_mean"].values)
        - 0.25 * zscore_arr(df["grid_max_to_med"].values)
    )
    s_hurricane = (
        0.95 * zscore_arr(df["diff_mag_mean"].values)
        - 0.75 * zscore_arr(df["dexg_mean"].values)
        - 0.45 * zscore_arr(df["grid_max_to_med"].values)
        + 0.30 * zscore_arr(df["dsat_mean"].values)
    )
    s_earthquake = (
        0.85 * zscore_arr(df["dedge_mean"].values)
        + 0.50 * zscore_arr(df["edge_post_mean"].values)
        + 0.45 * zscore_arr(df["diff_mag_std"].values)
        - 0.75 * zscore_arr(df["ash_post_frac"].values)
        - 0.70 * zscore_arr(df["dark_post_frac"].values)
    )
    scores = np.column_stack([s_hurricane, s_earthquake, s_volcano, s_fire, s_tsunami, s_explosion])

    anchor_quota = {
        "Hurricane": 275,
        "Earthquake": 220,
        "Volcano": 205,
        "Fire": 195,
        "Tsunami": 145,
        "Explosion": 32,
    }
    seed_y = np.full(len(df), -1, dtype=int)
    used = np.zeros(len(df), dtype=bool)
    for c_idx in [5, 4, 3, 2, 0, 1]:
        k = anchor_quota[CLASSES[c_idx]]
        avail = np.where(~used)[0]
        chosen = avail[np.argsort(-scores[avail, c_idx])[:k]]
        seed_y[chosen] = c_idx
        used[chosen] = True
    return seed_y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    sub = pd.read_csv(SUB_PATH)[["Id"]].copy()
    base_df = pd.read_parquet(BASE_FEAT_PATH)
    sp_df = pd.read_parquet(SPATIAL_FEAT_PATH)
    df = sub.merge(base_df, on="Id", how="left").merge(sp_df, on="Id", how="left")

    X_eng = build_engineered_matrix(df)
    rob_scaler = RobustScaler()
    std_scaler = StandardScaler()
    X_rob = np.clip(rob_scaler.fit_transform(X_eng), -10.0, 10.0)
    X_std = np.nan_to_num(std_scaler.fit_transform(X_eng), nan=0.0, posinf=0.0, neginf=0.0)

    gmm = GaussianMixture(
        n_components=8,
        covariance_type="diag",
        reg_covar=1e-2,
        n_init=3,
        max_iter=300,
        random_state=SEED,
    )
    gmm.fit(X_std)
    gmm_probs = gmm.predict_proba(X_std).astype(np.float32)

    X_all = np.hstack([X_std, X_rob, gmm_probs]).astype(np.float32)
    y_seed = build_anchors(df)
    train_mask = y_seed >= 0
    X_tr, y_tr = X_all[train_mask], y_seed[train_mask]

    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(192, 96), alpha=1.0, max_iter=350, early_stopping=True, random_state=SEED)
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=15,
        colsample_bytree=0.75,
        subsample=0.8,
        random_state=SEED,
        verbose=-1,
    )

    clf_lr.fit(X_tr, y_tr)
    clf_mlp.fit(X_tr, y_tr)
    clf_lgb.fit(X_tr, y_tr)

    probs = (
        0.30 * clf_lr.predict_proba(X_all)
        + 0.35 * clf_mlp.predict_proba(X_all)
        + 0.35 * clf_lgb.predict_proba(X_all)
    )

    target_priors = np.array([0.275, 0.245, 0.185, 0.172, 0.110, 0.013], dtype=np.float32)
    emp_priors = np.clip(probs.mean(axis=0), 1e-4, 1.0)
    cal_probs = probs * (target_priors / emp_priors) ** 0.85
    pred_idx = np.argmax(cal_probs, axis=1)

    if (pred_idx == 5).sum() < 32:
        top_exp = np.argsort(-cal_probs[:, 5])[:38]
        pred_idx[top_exp] = 5

    pred_labels = [CLASSES[i] for i in pred_idx]
    out_df = pd.DataFrame({"Id": df["Id"], "Phenomenon": pred_labels})
    assert len(out_df) == 3180 and (out_df["Id"] == sub["Id"]).all()
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    out_df.to_csv(OUT_CSV, index=False)
    joblib.dump(
        {
            "rob_scaler": rob_scaler,
            "std_scaler": std_scaler,
            "gmm": gmm,
            "clf_lr": clf_lr,
            "clf_mlp": clf_mlp,
            "clf_lgb": clf_lgb,
            "target_priors": target_priors,
            "emp_priors": emp_priors,
            "classes": CLASSES,
        },
        os.path.join(MODEL_DIR, "ensemble_v6.joblib"),
    )
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "v6_change_focus_cpu",
                "seed": SEED,
                "features": "Base + spatial/edge features + 6 domain interaction ratios + GMM(8) + 3-family ensemble",
            },
            f,
            indent=2,
        )

    counts = out_df["Phenomenon"].value_counts()
    print("Saved:", OUT_CSV)
    print("\nDistribusi prediksi v6_change_focus:")
    for k, v in counts.items():
        print(f"- {k:10s}: {v:4d} ({v/len(out_df)*100:5.1f}%)")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v6 Change-Focus Interaction + GMM(8) + 3-Family Ensemble (2026-11-28)\n")
        f.write(f"- Seed: {SEED}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={v}" for k, v in counts.items()]) + "\n")


if __name__ == "__main__":
    main()
