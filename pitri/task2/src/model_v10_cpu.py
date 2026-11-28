import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

SUB_SAMPLE = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
BASE_FEAT_PATH = "/content/Datavidia/pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "/content/Datavidia/pitri/task2/cache/features_v2_spatial.parquet"

CANDIDATE_SUBS = [
    ("/content/Datavidia/pitri/submissions/task2/submission_v9_pitri.csv", 1.35),
    ("/content/Datavidia/pitri/submissions/task2/submission_merged_available_pitri.csv", 1.20),
    ("/content/Datavidia/pitri/submissions/task2/submission_v4_pitri.csv", 1.10),
    ("/content/Datavidia/pitri/submissions/task2/submission_v6_pitri.csv", 0.85),
    ("/content/Datavidia/pitri/submissions/task2/submission_v3_pitri.csv", 0.70),
]

OUT_CSV = "/content/Datavidia/pitri/submissions/task2/submission_v10_pitri.csv"
LOG_PATH = "/content/Datavidia/pitri/task2/results/LOG.md"
CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]
CLS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_aligned(path: str, sample_ids: pd.Series):
    df = pd.read_csv(path)
    m = pd.DataFrame({"Id": sample_ids}).merge(df[["Id", "Phenomenon"]], on="Id", how="left")
    assert m["Phenomenon"].isna().sum() == 0, f"Ada NaN di {path}"
    assert m["Phenomenon"].isin(CLASSES).all(), f"Ada label tidak valid di {path}"
    return m["Phenomenon"].map(CLS_TO_IDX).to_numpy()


def main():
    sample = pd.read_csv(SUB_SAMPLE)[["Id"]]
    sample_ids = sample["Id"]
    n = len(sample_ids)

    df = (
        pd.DataFrame({"Id": sample_ids})
        .merge(pd.read_parquet(BASE_FEAT_PATH), on="Id", how="left")
        .merge(pd.read_parquet(SPATIAL_FEAT_PATH), on="Id", how="left")
    )
    feat_cols = [c for c in df.columns if c != "Id"]
    X_raw = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = np.clip(RobustScaler().fit_transform(X_raw), -8.0, 8.0)

    vote_scores = np.zeros((n, len(CLASSES)), dtype=np.float32)
    agree_counts = np.zeros((n, len(CLASSES)), dtype=int)
    used = []

    for path, w in CANDIDATE_SUBS:
        if os.path.exists(path):
            arr = load_aligned(path, sample_ids)
            for i, c in enumerate(arr):
                vote_scores[i, c] += w
                agree_counts[i, c] += 1
            used.append((os.path.basename(path), w))

    assert len(used) >= 2, "Minimal harus ada 2 file submission tersedia."

    pseudo_y = np.argmax(vote_scores, axis=1)
    max_agree = agree_counts[np.arange(n), pseudo_y]
    sorted_scores = np.sort(vote_scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]

    train_mask = (max_agree >= 2) | (margin >= 1.10)
    for c in range(len(CLASSES)):
        idx_c = np.where(pseudo_y == c)[0]
        if len(idx_c) > 0:
            order = np.argsort(-vote_scores[idx_c, c])
            train_mask[idx_c[order[: min(15, len(idx_c))]]] = True

    X_tr = X_scaled[train_mask]
    y_tr = pseudo_y[train_mask]
    print(f"Sumber dipakai: {used}")
    print(f"Total sampel latih konsensus (CPU): {len(X_tr)}/{n}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cb_test_probs = np.zeros((n, len(CLASSES)), dtype=np.float32)
    lgb_test_probs = np.zeros((n, len(CLASSES)), dtype=np.float32)
    mlp_test_probs = np.zeros((n, len(CLASSES)), dtype=np.float32)

    for fold, (tr_i, va_i) in enumerate(skf.split(X_tr, y_tr), 1):
        cb = CatBoostClassifier(
            iterations=450,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=SEED + fold,
            task_type="CPU",
            thread_count=-1,
            verbose=False,
        )
        cb.fit(X_tr[tr_i], y_tr[tr_i], eval_set=(X_tr[va_i], y_tr[va_i]), early_stopping_rounds=50, verbose=False)
        cb_test_probs += cb.predict_proba(X_scaled) / skf.n_splits

        lgbm = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.04,
            num_leaves=15,
            class_weight="balanced",
            colsample_bytree=0.8,
            subsample=0.85,
            random_state=SEED + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(pd.DataFrame(X_tr[tr_i], columns=feat_cols), y_tr[tr_i])
        lgb_test_probs += lgbm.predict_proba(pd.DataFrame(X_scaled, columns=feat_cols)) / skf.n_splits

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1.0,
            max_iter=250,
            early_stopping=True,
            random_state=SEED + fold,
        )
        mlp.fit(X_tr[tr_i], y_tr[tr_i])
        mlp_test_probs += mlp.predict_proba(X_scaled) / skf.n_splits

    blend_probs = 0.50 * cb_test_probs + 0.30 * lgb_test_probs + 0.20 * mlp_test_probs

    target_priors = np.array([0.29, 0.24, 0.16, 0.15, 0.14, 0.02], dtype=np.float32)
    emp_priors = np.clip(blend_probs.mean(axis=0), 1e-4, 1.0)
    cal_probs = blend_probs * (target_priors / emp_priors) ** 0.65
    pred_idx = np.argmax(cal_probs, axis=1)

    out_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": [CLASSES[i] for i in pred_idx]})
    assert len(out_df) == 3180 and (out_df["Id"] == sample_ids).all()
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    vc = out_df["Phenomenon"].value_counts()
    print("\nSaved:", OUT_CSV)
    print("Distribusi v10 CPU (CatBoost Balanced + LGBM + MLP):")
    for k in CLASSES:
        cnt = int(vc.get(k, 0))
        print(f"- {k:10s}: {cnt:4d} ({cnt/len(out_df)*100:5.1f}%)")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v10 CPU CatBoost Balanced + LGBM + MLP (2026-11-28)\n")
        f.write(f"- Train samples: {len(X_tr)}/{n}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={int(vc.get(k, 0))}" for k in CLASSES]) + "\n")


if __name__ == "__main__":
    main()
