import numpy as np
import pandas as pd

SUB_SAMPLE = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2/sample_submission.csv"
V1_CSV = "pitri/submissions/task2/submission_v1_pitri.csv"
V2_CSV = "pitri/submissions/task2/submission_v2_pitri.csv"
V3_CSV = "pitri/submissions/task2/submission_v3_pitri.csv"
FEAT_BASE = "pitri/task2/cache/features_task2.parquet"
FEAT_SPATIAL = "pitri/task2/cache/features_v2_spatial.parquet"

sample = pd.read_csv(SUB_SAMPLE)
v3 = pd.read_csv(V3_CSV)

print("=== 1. FORMAT & INTEGRITAS SUBMISSION V3 ===")
print("Jumlah baris:", len(v3))
print("Kolom:", list(v3.columns))
print("Urutan Id identik dengan sample_submission:", (v3["Id"] == sample["Id"]).all())
print("Missing values:", int(v3.isna().sum().sum()))
print("Duplikat Id:", int(v3["Id"].duplicated().sum()))

print("\n=== 2. DISTRIBUSI V3 VS V1 & V2 ===")
for name, path in [("v1", V1_CSV), ("v2", V2_CSV), ("v3", V3_CSV)]:
    try:
        df_s = pd.read_csv(path)
        vc = df_s["Phenomenon"].value_counts()
        print(f"\n[{name}]")
        for k, v in vc.items():
            print(f"  - {k:10s}: {v:4d} ({v/len(df_s)*100:5.1f}%)")
    except FileNotFoundError:
        print(f"\n[{name}] file belum ada.")

v2 = pd.read_csv(V2_CSV)
agree_v2_v3 = (v2["Phenomenon"] == v3["Phenomenon"]).mean() * 100
print(f"\nKecocokan label v2 vs v3: {agree_v2_v3:.1f}%")

# Cek sampel yang diprediksi Earthquake di v3 tapi punya sinyal fisik kuat di kelas lain
df_feat = sample[["Id"]].merge(pd.read_parquet(FEAT_BASE), on="Id", how="left").merge(
    pd.read_parquet(FEAT_SPATIAL), on="Id", how="left"
)
df_feat["pred_v3"] = v3["Phenomenon"]
eq_mask = df_feat["pred_v3"] == "Earthquake"
print(f"\n=== 3. CEK DI DALAM {eq_mask.sum()} PREDIKSI EARTHQUAKE (V3) ===")
print("Punya ash_post_frac > p85:", int((df_feat.loc[eq_mask, "ash_post_frac"] > df_feat["ash_post_frac"].quantile(0.85)).sum()))
print("Punya dark_post_frac > p85:", int((df_feat.loc[eq_mask, "dark_post_frac"] > df_feat["dark_post_frac"].quantile(0.85)).sum()))
print("Punya dblue_mean > p85:", int((df_feat.loc[eq_mask, "dblue_mean"] > df_feat["dblue_mean"].quantile(0.85)).sum()))
print("Punya peak_to_mean_ratio > p90:", int((df_feat.loc[eq_mask, "peak_to_mean_ratio"] > df_feat["peak_to_mean_ratio"].quantile(0.90)).sum()))
