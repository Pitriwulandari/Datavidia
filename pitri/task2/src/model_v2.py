import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)

DATA_DIR = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2"
FENOMENA_DIR = os.path.join(DATA_DIR, "Fenomena")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
BASE_FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "pitri/task2/cache/features_v2_spatial.parquet"
OUT_CSV = "pitri/submissions/task2/submission_v2_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"

CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]

def extract_spatial_and_edge_features(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> dict:
    pre = cv2.resize(cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
    post = cv2.resize(cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)

    diff_rgb = np.abs(post - pre).mean(axis=-1)  # (512, 512)

    # Grid 4x4 (masing-masing 128x128) untuk mengukur fokus/lokalisasi perubahan
    grid_means = []
    for r in range(0, 512, 128):
        for c in range(0, 512, 128):
            grid_means.append(float(diff_rgb[r:r+128, c:c+128].mean()))
    grid_arr = np.array(grid_means, dtype=np.float32)
    grid_max_to_med = float(grid_arr.max() / (np.median(grid_arr) + 1e-6))
    grid_std_to_mean = float(grid_arr.std() / (grid_arr.mean() + 1e-6))

    # Gradien tepi (Sobel) untuk menangkap perubahan struktur/reruntuhan vs abu/air halus
    pre_gray = cv2.cvtColor(pre.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    post_gray = cv2.cvtColor(post.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    gx_pre = cv2.Sobel(pre_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_pre = cv2.Sobel(pre_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_pre = np.sqrt(gx_pre**2 + gy_pre**2)

    gx_post = cv2.Sobel(post_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_post = cv2.Sobel(post_gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_post = np.sqrt(gx_post**2 + gy_post**2)

    dedge_mean = float((edge_post - edge_pre).mean())
    edge_post_mean = float(edge_post.mean())

    return {
        "grid_max_to_med": grid_max_to_med,
        "grid_std_to_mean": grid_std_to_mean,
        "dedge_mean": dedge_mean,
        "edge_post_mean": edge_post_mean,
    }

def zscore(s: pd.Series) -> np.ndarray:
    arr = s.fillna(0.0).to_numpy(dtype=np.float32)
    std = arr.std()
    return (arr - arr.mean()) / (std if std > 1e-6 else 1.0)

def main():
    sub = pd.read_csv(SUB_PATH)[["Id"]].copy()
    base_df = pd.read_parquet(BASE_FEAT_PATH)

    if os.path.exists(SPATIAL_FEAT_PATH):
        sp_df = pd.read_parquet(SPATIAL_FEAT_PATH)
    else:
        rows = []
        for row_id in tqdm(sub["Id"].astype(str), desc="Extracting v2 spatial/edge features"):
            pre_bgr = cv2.imread(os.path.join(FENOMENA_DIR, row_id, "pre.jpg"), cv2.IMREAD_COLOR)
            post_bgr = cv2.imread(os.path.join(FENOMENA_DIR, row_id, "post.jpg"), cv2.IMREAD_COLOR)
            feat = extract_spatial_and_edge_features(pre_bgr, post_bgr)
            feat["Id"] = row_id
            rows.append(feat)
        sp_df = pd.DataFrame(rows)
        sp_df.to_parquet(SPATIAL_FEAT_PATH, index=False)

    df = sub.merge(base_df, on="Id", how="left").merge(sp_df, on="Id", how="left")

    # Skor kelas v2 (menambah grid localization + perubahan gradien tepi)
    s_explosion = (
        1.4 * zscore(df["peak_to_mean_ratio"])
        + 1.1 * zscore(df["grid_max_to_med"])
        + 0.6 * zscore(df["grid_std_to_mean"])
        - 0.5 * zscore(df["diff_mag_mean"])
    )
    s_fire = (
        1.25 * zscore(df["dark_post_frac"])
        - 1.05 * zscore(df["dexg_mean"])
        - 0.55 * zscore(df["dexg_p10"])
        - 0.35 * zscore(df["grid_max_to_med"])
    )
    s_tsunami = (
        1.35 * zscore(df["dblue_mean"])
        + 0.85 * zscore(df["dblue_p90"])
        - 0.35 * zscore(df["dedge_mean"])
    )
    s_volcano = (
        1.35 * zscore(df["ash_post_frac"])
        + 0.65 * zscore(df["dval_mean"])
        - 0.65 * zscore(df["dsat_mean"])
        - 0.30 * zscore(df["grid_max_to_med"])
    )
    s_hurricane = (
        0.95 * zscore(df["diff_mag_mean"])
        - 0.75 * zscore(df["dexg_mean"])
        - 0.45 * zscore(df["grid_max_to_med"])
        + 0.30 * zscore(df["dsat_mean"])
    )
    s_earthquake = (
        0.85 * zscore(df["dedge_mean"])
        + 0.55 * zscore(df["edge_post_mean"])
        + 0.45 * zscore(df["diff_mag_std"])
        - 0.65 * zscore(df["ash_post_frac"])
        - 0.60 * zscore(df["dark_post_frac"])
    )

    scores = np.column_stack([s_hurricane, s_earthquake, s_volcano, s_fire, s_tsunami, s_explosion])

    # Kalibrasi kuota awal mendekati v3 dengan sedikit ruang penyesuaian pada kelas ber-sinyal kuat
    targets = {
        "Explosion": 28,
        "Tsunami": 360,
        "Fire": 545,
        "Volcano": 585,
        "Hurricane": 890,
        "Earthquake": 772,
    }
    n = len(df)
    assigned = np.full(n, "", dtype=object)
    used = np.zeros(n, dtype=bool)
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    for c in ["Explosion", "Tsunami", "Fire", "Volcano", "Hurricane", "Earthquake"]:
        idx = cls_to_idx[c]
        k = targets[c]
        avail = np.where(~used)[0]
        if len(avail) == 0:
            break
        k_take = min(k, len(avail))
        top_local = np.argsort(-scores[avail, idx])[:k_take]
        chosen = avail[top_local]
        assigned[chosen] = c
        used[chosen] = True

    rem = np.where(~used)[0]
    if len(rem) > 0:
        assigned[rem] = "Earthquake"

    out_df = pd.DataFrame({"Id": df["Id"], "Phenomenon": assigned})
    assert len(out_df) == 3180
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    out_df.to_csv(OUT_CSV, index=False)
    counts = out_df["Phenomenon"].value_counts()
    print("Saved:", OUT_CSV)
    print("\nDistribusi prediksi v2:")
    for k, v in counts.items():
        print(f"- {k}: {v} ({v/len(out_df)*100:.1f}%)")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v2 Spatial Grid 4x4 + Sobel Edge + Color/Spectral Features (2026-11-28)\n")
        f.write(f"- Seed: {SEED}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={v}" for k, v in counts.items()]) + "\n")

if __name__ == "__main__":
    main()
