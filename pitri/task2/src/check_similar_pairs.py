import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

SUB_SAMPLE = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
FEAT_BASE = "pitri/task2/cache/features_task2.parquet"
FEAT_SPATIAL = "pitri/task2/cache/features_v2_spatial.parquet"
V8_CSV = "pitri/submissions/task2/submission_v8_pitri.csv"
OUT_AUDIT = "pitri/task2/results/similar_audit.csv"


def main():
    os.makedirs(os.path.dirname(OUT_AUDIT), exist_ok=True)
    sample = pd.read_csv(SUB_SAMPLE)[["Id"]].copy()
    df = sample.merge(pd.read_parquet(FEAT_BASE), on="Id", how="left").merge(
        pd.read_parquet(FEAT_SPATIAL), on="Id", how="left"
    )

    if os.path.exists(V8_CSV):
        v8 = pd.read_csv(V8_CSV)
        df = df.merge(v8, on="Id", how="left")
        pred_col = "Phenomenon"
    else:
        pred_col = None

    feat_cols = [c for c in df.columns if c not in ("Id", "Phenomenon")]
    X = np.nan_to_num(df[feat_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = StandardScaler().fit_transform(X)

    # Cari tetangga terdekat (nearest neighbor selain diri sendiri)
    dist = pairwise_distances(X_scaled, metric="euclidean")
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argmin(dist, axis=1)
    nn_dist = dist[np.arange(len(df)), nn_idx]

    audit_df = pd.DataFrame({
        "Id": df["Id"],
        "nearest_Id": df["Id"].iloc[nn_idx].values,
        "nn_distance": np.round(nn_dist, 4),
    })

    if pred_col:
        audit_df["pred_v8"] = df[pred_col].values
        audit_df["nearest_pred_v8"] = df[pred_col].iloc[nn_idx].values
        audit_df["same_label_as_nearest"] = audit_df["pred_v8"] == audit_df["nearest_pred_v8"]

        conflict = audit_df[~audit_df["same_label_as_nearest"]].sort_values("nn_distance")
        print(f"Persentase sampel yang label v8-nya sama dengan tetangga paling mirip: {audit_df['same_label_as_nearest'].mean()*100:.1f}%")
        print("\n10 pasangan paling mirip tapi label v8-nya beda (kandidat cek manual):")
        print(conflict.head(10).to_string(index=False))

    audit_df.to_csv(OUT_AUDIT, index=False)
    print("\nTersimpan di:", OUT_AUDIT)


if __name__ == "__main__":
    main()
