import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import pairwise_distances
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

SUB_SAMPLE = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
BASE_FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "pitri/task2/cache/features_v2_spatial.parquet"

OUT_CSV = "pitri/submissions/task2/submission_v9_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"
CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]


def find_external_csv(keyword: str):
    hits = [
        p for p in glob.glob("**/*.csv", recursive=True)
        if keyword.lower() in p.lower() and not p.startswith("pitri/")
    ]
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def load_aligned(path: str, sample_ids: pd.Series):
    df = pd.read_csv(path)
    m = pd.DataFrame({"Id": sample_ids}).merge(df[["Id", "Phenomenon"]], on="Id", how="left")
    if m["Phenomenon"].isna().sum() > 0 or not m["Phenomenon"].isin(CLASSES).all():
        return None
    return m["Phenomenon"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def main():
    sample = pd.read_csv(SUB_SAMPLE)[["Id"]]
    sample_ids = sample["Id"]

    df = (
        pd.DataFrame({"Id": sample_ids})
        .merge(pd.read_parquet(BASE_FEAT_PATH), on="Id", how="left")
        .merge(pd.read_parquet(SPATIAL_FEAT_PATH), on="Id", how="left")
    )
    feat_cols = [c for c in df.columns if c != "Id"]
    X_raw = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = np.clip(RobustScaler().fit_transform(X_raw), -8.0, 8.0)

    # Kumpulkan sumber prediksi yang tersedia (eksternal Ghani/Farhat + internal Pitri)
    sources = []
    for kw, w in [("ghani", 1.5), ("farhat", 1.4)]:
        p = find_external_csv(kw)
        if p:
            arr = load_aligned(p, sample_ids)
            if arr is not None:
                sources.append((kw, arr, w))

    pitri_cands = [
        ("pitri/submissions/task2/submission_v8_pitri.csv", 1.2),
        ("pitri/submissions/task2/submission_merged_available_pitri.csv", 1.1),
        ("pitri/submissions/task2/submission_v4_pitri.csv", 1.0),
        ("pitri/submissions/task2/submission_v6_pitri.csv", 0.8),
    ]
    for p, w in pitri_cands:
        if os.path.exists(p):
            arr = load_aligned(p, sample_ids)
            if arr is not None:
                sources.append((os.path.basename(p), arr, w))

    assert len(sources) >= 2, "Butuh minimal 2 sumber prediksi untuk konsensus."

    n = len(sample_ids)
    vote_scores = np.zeros((n, len(CLASSES)), dtype=np.float32)
    agree_count = np.zeros((n, len(CLASSES)), dtype=int)

    for _, arr, w in sources:
        for i, c in enumerate(arr):
            vote_scores[i, c] += w
            agree_count[i, c] += 1

    # Cek konsistensi dengan tetangga terdekat di ruang fitur
    dist = pairwise_distances(X_scaled, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argmin(dist, axis=1)

    best_c = np.argmax(vote_scores, axis=1)
    best_agree = agree_count[np.arange(n), best_c]
    nn_same = (best_c == best_c[nn_idx])

    # Pilih sampel konsensus tinggi sebagai pseudo-label latih
    min_agree = 2 if len(sources) <= 3 else 3
    high_conf_mask = (best_agree >= min_agree) & nn_same
    # Pastikan tiap kelas punya minimal beberapa sampel latih jika tersedia
    for c in range(len(CLASSES)):
        idx_c = np.where(best_c == c)[0]
        if len(idx_c) > 0 and high_conf_mask[idx_c].sum() < 5:
            order = np.argsort(-vote_scores[idx_c, c])
            high_conf_mask[idx_c[order[: min(10, len(idx_c))]]] = True

    X_tr, y_tr = X_scaled[high_conf_mask], best_c[high_conf_mask]

    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(160, 80), alpha=1.0, max_iter=300, early_stopping=True, random_state=SEED)
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.04, num_leaves=15,
        colsample_bytree=0.8, subsample=0.85, random_state=SEED, verbose=-1
    )

    clf_lr.fit(X_tr, y_tr)
    clf_mlp.fit(X_tr, y_tr)
    clf_lgb.fit(X_tr, y_tr)

    probs = (
        0.30 * clf_lr.predict_proba(X_scaled)
        + 0.35 * clf_mlp.predict_proba(X_scaled)
        + 0.35 * clf_lgb.predict_proba(X_scaled)
    )

    # Kalibrasi prior agar kelas minoritas tidak terhapus
    target_priors = np.array([0.28, 0.23, 0.17, 0.16, 0.14, 0.02], dtype=np.float32)
    emp_priors = np.clip(probs.mean(axis=0), 1e-4, 1.0)
    cal_probs = probs * (target_priors / emp_priors) ** 0.75
    pred_idx = np.argmax(cal_probs, axis=1)

    out_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": [CLASSES[i] for i in pred_idx]})
    assert len(out_df) == 3180 and (out_df["Id"] == sample_ids).all()
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    vc = out_df["Phenomenon"].value_counts()
    print(f"Dipakai {len(sources)} sumber: {[s[0] for s in sources]}")
    print(f"Sampel high-confidence pseudo-label: {high_conf_mask.sum()}/{n}")
    print("Saved:", OUT_CSV)
    for k in CLASSES:
        cnt = int(vc.get(k, 0))
        print(f"- {k:10s}: {cnt:4d} ({cnt/len(out_df)*100:5.1f}%)")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v9 Consensus Self-Training Ensemble (2026-11-28)\n")
        f.write(f"- Sources: {[s[0] for s in sources]}; High-conf: {int(high_conf_mask.sum())}/{n}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={int(vc.get(k, 0))}" for k in CLASSES]) + "\n")


if __name__ == "__main__":
    main()
