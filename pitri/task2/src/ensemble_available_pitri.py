import os
import numpy as np
import pandas as pd

SUB_SAMPLE = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
BASE_FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "pitri/task2/cache/features_v2_spatial.parquet"

CANDIDATES = [
    ("pitri/submissions/task2/submission_v8_pitri.csv", 1.30),
    ("pitri/submissions/task2/submission_v4_pitri.csv", 1.15),
    ("pitri/submissions/task2/submission_v6_pitri.csv", 0.85),
    ("pitri/submissions/task2/submission_v3_pitri.csv", 0.70),
]

OUT_CSV = "pitri/submissions/task2/submission_merged_available_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"

# 0=Hurricane, 1=Earthquake, 2=Volcano, 3=Fire, 4=Tsunami, 5=Explosion
CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]


def zscore_arr(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-6 else 1.0)


def build_physical_tiebreak(sample_ids: pd.Series) -> np.ndarray:
    base_df = pd.read_parquet(BASE_FEAT_PATH)
    sp_df = pd.read_parquet(SPATIAL_FEAT_PATH)
    df = pd.DataFrame({"Id": sample_ids}).merge(base_df, on="Id", how="left").merge(sp_df, on="Id", how="left")

    s_hurricane = zscore_arr(df["diff_mag_mean"].values) - 0.7 * zscore_arr(df["dexg_mean"].values)
    s_earthquake = 0.8 * zscore_arr(df["dedge_mean"].values) - 0.7 * zscore_arr(df["ash_post_frac"].values)
    s_volcano = 1.4 * zscore_arr(df["ash_post_frac"].values) - 0.6 * zscore_arr(df["dsat_mean"].values)
    s_fire = 1.4 * zscore_arr(df["dark_post_frac"].values) - 1.1 * zscore_arr(df["dexg_mean"].values)
    s_tsunami = 1.4 * zscore_arr(df["dblue_mean"].values) + 0.8 * zscore_arr(df["dblue_p90"].values)
    s_explosion = 1.5 * zscore_arr(df["peak_to_mean_ratio"].values) + 1.3 * zscore_arr(df["grid_max_to_med"].values)

    raw = np.column_stack([s_hurricane, s_earthquake, s_volcano, s_fire, s_tsunami, s_explosion])
    exp_s = np.exp(raw - np.max(raw, axis=1, keepdims=True))
    return 0.20 * (exp_s / np.sum(exp_s, axis=1, keepdims=True))


def main():
    sample = pd.read_csv(SUB_SAMPLE)[["Id"]]
    sample_ids = sample["Id"]

    scores = build_physical_tiebreak(sample_ids)
    used_files = []

    for path, weight in CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path)
            merged = pd.DataFrame({"Id": sample_ids}).merge(df[["Id", "Phenomenon"]], on="Id", how="left")
            assert merged["Phenomenon"].isna().sum() == 0, f"Ada NaN pada {path}"
            assert merged["Phenomenon"].isin(CLASSES).all(), f"Ada label tidak valid pada {path}"
            idx = merged["Phenomenon"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
            for row_i, c_i in enumerate(idx):
                scores[row_i, c_i] += weight
            used_files.append((path, weight))
            print(f"Dipakai (bobot {weight:.2f}): {path}")

    assert len(used_files) >= 2, "Minimal harus ada 2 file submission yang tersedia untuk digabung."

    # Penyesuaian halus agar 6 kelas tetap seimbang di Macro F1
    class_factor = np.array([1.00, 0.95, 1.04, 1.04, 1.06, 1.15], dtype=np.float32)
    final_idx = np.argmax(scores * class_factor, axis=1)
    final_labels = [CLASSES[i] for i in final_idx]

    out_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": final_labels})
    assert len(out_df) == 3180 and (out_df["Id"] == sample_ids).all()
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    counts = out_df["Phenomenon"].value_counts()
    print("\nSaved:", OUT_CSV)
    print("\nDistribusi hasil gabungan file yang tersedia:")
    for k, v in counts.items():
        print(f"- {k:10s}: {v:4d} ({v/len(out_df)*100:5.1f}%)")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## Merged Available Pitri Submissions + Physical Tie-Break (2026-11-28)\n")
        f.write(f"- Sources: {[u[0] for u in used_files]}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={v}" for k, v in counts.items()]) + "\n")


if __name__ == "__main__":
    main()
