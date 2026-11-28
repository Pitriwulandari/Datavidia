import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

SUB_PATH = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
BASE_FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "pitri/task2/cache/features_v2_spatial.parquet"

MODEL_DIR = "pitri/task2/models/v3"
OUT_CSV = "pitri/submissions/task2/submission_v3_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"

CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]


def zscore_arr(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-6 else 1.0)


def compute_physical_scores(df: pd.DataFrame) -> np.ndarray:
    s_explosion = (
        1.50 * zscore_arr(df["peak_to_mean_ratio"].values)
        + 1.25 * zscore_arr(df["grid_max_to_med"].values)
        + 0.65 * zscore_arr(df["grid_std_to_mean"].values)
        - 0.45 * zscore_arr(df["diff_mag_mean"].values)
    )
    s_fire = (
        1.35 * zscore_arr(df["dark_post_frac"].values)
        - 1.05 * zscore_arr(df["dexg_mean"].values)
        - 0.55 * zscore_arr(df["dexg_p10"].values)
        - 0.25 * zscore_arr(df["grid_max_to_med"].values)
    )
    s_tsunami = (
        1.40 * zscore_arr(df["dblue_mean"].values)
        + 0.85 * zscore_arr(df["dblue_p90"].values)
        - 0.35 * zscore_arr(df["dedge_mean"].values)
    )
    s_volcano = (
        1.40 * zscore_arr(df["ash_post_frac"].values)
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
        0.90 * zscore_arr(df["dedge_mean"].values)
        + 0.55 * zscore_arr(df["edge_post_mean"].values)
        + 0.45 * zscore_arr(df["diff_mag_std"].values)
        - 0.65 * zscore_arr(df["ash_post_frac"].values)
        - 0.60 * zscore_arr(df["dark_post_frac"].values)
    )
    return np.column_stack([s_hurricane, s_earthquake, s_volcano, s_fire, s_tsunami, s_explosion])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    sub = pd.read_csv(SUB_PATH)[["Id"]].copy()
    base_df = pd.read_parquet(BASE_FEAT_PATH)
    sp_df = pd.read_parquet(SPATIAL_FEAT_PATH)
    df = sub.merge(base_df, on="Id", how="left").merge(sp_df, on="Id", how="left")

    feat_cols = [c for c in df.columns if c != "Id"]
    X_raw = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = np.nan_to_num(scaler.fit_transform(X_raw), nan=0.0, posinf=0.0, neginf=0.0)

    # reg_covar=1e-2 mencegah singular/collapsed covariance pada fitur skewed/sparse
    gmm = GaussianMixture(
        n_components=8,
        covariance_type="diag",
        reg_covar=1e-2,
        n_init=3,
        max_iter=300,
        random_state=SEED,
    )
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled).astype(np.float32)

    X_all = np.hstack([X_scaled, gmm_probs]).astype(np.float32)

    phys_scores = compute_physical_scores(df)
    anchor_quota = {
        "Hurricane": 280,
        "Earthquake": 240,
        "Volcano": 195,
        "Fire": 185,
        "Tsunami": 135,
        "Explosion": 28,
    }
    seed_y = np.full(len(df), -1, dtype=int)
    used = np.zeros(len(df), dtype=bool)
    for c_idx in [5, 4, 3, 2, 0, 1]:
        k = anchor_quota[CLASSES[c_idx]]
        avail = np.where(~used)[0]
        chosen = avail[np.argsort(-phys_scores[avail, c_idx])[:k]]
        seed_y[chosen] = c_idx
        used[chosen] = True

    train_mask = seed_y >= 0
    X_tr, y_tr = X_all[train_mask], seed_y[train_mask]

    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1.0, max_iter=300, early_stopping=True, random_state=SEED)
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
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
        0.35 * clf_lr.predict_proba(X_all)
        + 0.35 * clf_mlp.predict_proba(X_all)
        + 0.30 * clf_lgb.predict_proba(X_all)
    )

    target_priors = np.array([0.270, 0.240, 0.188, 0.175, 0.115, 0.012], dtype=np.float32)
    emp_priors = np.clip(probs.mean(axis=0), 1e-4, 1.0)
    cal_probs = probs * (target_priors / emp_priors) ** 0.55
    pred_idx = np.argmax(cal_probs, axis=1)

    if (pred_idx == 5).sum() < 30:
        top_exp = np.argsort(-cal_probs[:, 5])[:34]
        pred_idx[top_exp] = 5

    pred_labels = [CLASSES[i] for i in pred_idx]
    out_df = pd.DataFrame({"Id": df["Id"], "Phenomenon": pred_labels})
    assert len(out_df) == 3180
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    out_df.to_csv(OUT_CSV, index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "gmm": gmm,
            "clf_lr": clf_lr,
            "clf_mlp": clf_mlp,
            "clf_lgb": clf_lgb,
            "target_priors": target_priors,
            "emp_priors": emp_priors,
            "classes": CLASSES,
            "feat_cols": feat_cols,
        },
        os.path.join(MODEL_DIR, "ensemble_v3_fast.joblib"),
    )
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "v3_fast_cpu",
                "seed": SEED,
                "features": "RGB/HSV/ExG/dblue/dark_post/ash_post/Grid4x4/Sobel + GMM(8, reg_covar=1e-2)",
            },
            f,
            indent=2,
        )

    counts = out_df["Phenomenon"].value_counts()
    print("Saved:", OUT_CSV)
    print("\nDistribusi prediksi v3_fast_cpu:")
    for k, v in counts.items():
        print(f"- {k}: {v} ({v/len(out_df)*100:.1f}%)")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v3 Fast CPU Physical Color-Spatial + GMM + 3-Family Ensemble (2026-11-28)\n")
        f.write(f"- Seed: {SEED}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={v}" for k, v in counts.items()]) + "\n")


if __name__ == "__main__":
    main()
