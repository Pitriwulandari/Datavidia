import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SUB_PATH = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
OUT_CSV = "pitri/submissions/task2/submission_v1_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"

CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]
# Anchor kuota awal mengikuti v3 (LB 0.37895) untuk 3.180 sampel
ANCHOR_COUNTS = {
    "Hurricane": 900,
    "Earthquake": 772,
    "Volcano": 589,
    "Fire": 543,
    "Tsunami": 351,
    "Explosion": 25,
}

def zscore(s: pd.Series) -> np.ndarray:
    arr = s.fillna(0.0).to_numpy(dtype=np.float32)
    std = arr.std()
    return (arr - arr.mean()) / (std if std > 1e-6 else 1.0)

def main():
    df = pd.read_parquet(FEAT_PATH)
    sub = pd.read_csv(SUB_PATH)[["Id"]].copy()
    df = sub.merge(df, on="Id", how="left")

    # Skor afinitas domain-fisik (6 kelas)
    s_fire = 1.2 * zscore(df["dark_post_frac"]) - 1.0 * zscore(df["dexg_mean"]) - 0.5 * zscore(df["dexg_p10"])
    s_tsunami = 1.3 * zscore(df["dblue_mean"]) + 0.8 * zscore(df["dblue_p90"])
    s_volcano = 1.3 * zscore(df["ash_post_frac"]) + 0.6 * zscore(df["dval_mean"]) - 0.6 * zscore(df["dsat_mean"])
    s_explosion = 1.5 * zscore(df["peak_to_mean_ratio"]) + 0.7 * zscore(df["diff_mag_p95"]) - 0.5 * zscore(df["diff_mag_mean"])
    s_hurricane = 0.9 * zscore(df["diff_mag_mean"]) - 0.7 * zscore(df["dexg_mean"]) - 0.5 * zscore(df["peak_to_mean_ratio"])
    s_earthquake = 0.8 * zscore(df["diff_mag_std"]) - 0.6 * zscore(df["ash_post_frac"]) - 0.6 * zscore(df["dark_post_frac"])

    scores = np.column_stack([s_hurricane, s_earthquake, s_volcano, s_fire, s_tsunami, s_explosion])
    n = len(df)
    assigned = np.full(n, "", dtype=object)
    used = np.zeros(n, dtype=bool)

    # Alokasikan kelas paling spesifik/langka dulu berdasarkan skor tertinggi
    order_alloc = ["Explosion", "Tsunami", "Fire", "Volcano", "Hurricane", "Earthquake"]
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    for c in order_alloc:
        idx = cls_to_idx[c]
        k = ANCHOR_COUNTS[c]
        available_idx = np.where(~used)[0]
        if len(available_idx) == 0:
            break
        k_take = min(k, len(available_idx))
        top_local = np.argsort(-scores[available_idx, idx])[:k_take]
        chosen = available_idx[top_local]
        assigned[chosen] = c
        used[chosen] = True

    # Fallback jika masih ada sisa baris
    rem = np.where(~used)[0]
    if len(rem) > 0:
        assigned[rem] = "Earthquake"

    out_df = pd.DataFrame({"Id": df["Id"], "Phenomenon": assigned})
    assert len(out_df) == 3180, f"Expected 3180 rows, got {len(out_df)}"
    assert out_df["Phenomenon"].isin(CLASSES).all(), "Invalid class label found"
    assert out_df["Id"].isna().sum() == 0 and out_df["Phenomenon"].isna().sum() == 0

    out_df.to_csv(OUT_CSV, index=False)
    counts = out_df["Phenomenon"].value_counts()
    print("Saved:", OUT_CSV)
    print("\nDistribusi prediksi v1:")
    for k, v in counts.items():
        print(f"- {k}: {v} ({v/len(out_df)*100:.1f}%)")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v1 Baseline Domain-Physical Heuristic + v3 Anchor (2026-11-28)\n")
        f.write(f"- Seed: {SEED}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={v}" for k, v in counts.items()]) + "\n")

if __name__ == "__main__":
    main()
