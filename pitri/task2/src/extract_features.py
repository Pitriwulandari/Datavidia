import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2"
FENOMENA_DIR = os.path.join(DATA_DIR, "Fenomena")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = "pitri/task2/cache/features_task2.parquet"

def compute_pair_features(pre_bgr: np.ndarray, post_bgr: np.ndarray) -> dict:
    # Resize ke 512x512 agar ekstraksi cepat namun tetap menangkap pola spasial & warna
    pre = cv2.resize(cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
    post = cv2.resize(cv2.cvtColor(post_bgr, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)

    r_pre, g_pre, b_pre = pre[..., 0], pre[..., 1], pre[..., 2]
    r_post, g_post, b_post = post[..., 0], post[..., 1], post[..., 2]

    # Indeks vegetasi (Excess Green) & indeks air/kebiruan
    exg_pre = 2.0 * g_pre - r_pre - b_pre
    exg_post = 2.0 * g_post - r_post - b_post
    dexg = exg_post - exg_pre

    blue_ratio_pre = (b_pre + 1.0) / (r_pre + g_pre + b_pre + 3.0)
    blue_ratio_post = (b_post + 1.0) / (r_post + g_post + b_post + 3.0)
    dblue = blue_ratio_post - blue_ratio_pre

    # Saturasi & kecerahan (HSV)
    hsv_pre = cv2.cvtColor(pre.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_post = cv2.cvtColor(post.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    dsat = hsv_post[..., 1] - hsv_pre[..., 1]
    dval = hsv_post[..., 2] - hsv_pre[..., 2]

    # Magnitud perubahan & rasio puncak terhadap rata-rata (konsentrasi spasial / ledakan)
    diff_mag = np.sqrt((r_post - r_pre) ** 2 + (g_post - g_pre) ** 2 + (b_post - b_pre) ** 2)
    mean_mag = float(diff_mag.mean()) + 1e-6
    p95_mag = float(np.percentile(diff_mag, 95))
    peak_to_mean = p95_mag / mean_mag

    # Gelap / burn scar (post gelap & saturasi rendah)
    dark_post = ((post.mean(axis=-1) < 55.0) & (hsv_post[..., 1] < 90.0)).astype(np.float32)
    # Cerah/abu-abu seragam (post terang/abu & saturasi rendah -> abu vulkanik/awan)
    ash_post = ((post.mean(axis=-1) > 130.0) & (hsv_post[..., 1] < 45.0)).astype(np.float32)

    return {
        "diff_mag_mean": float(diff_mag.mean()),
        "diff_mag_std": float(diff_mag.std()),
        "diff_mag_p90": float(np.percentile(diff_mag, 90)),
        "diff_mag_p95": p95_mag,
        "peak_to_mean_ratio": float(peak_to_mean),
        "dr_mean": float((r_post - r_pre).mean()),
        "dg_mean": float((g_post - g_pre).mean()),
        "db_mean": float((b_post - b_pre).mean()),
        "dexg_mean": float(dexg.mean()),
        "dexg_p10": float(np.percentile(dexg, 10)),
        "dblue_mean": float(dblue.mean()),
        "dblue_p90": float(np.percentile(dblue, 90)),
        "dsat_mean": float(dsat.mean()),
        "dval_mean": float(dval.mean()),
        "dark_post_frac": float(dark_post.mean()),
        "ash_post_frac": float(ash_post.mean()),
    }

def main():
    sub = pd.read_csv(SUB_PATH)
    rows = []
    for row_id in tqdm(sub["Id"].astype(str), desc="Extracting Task-2 features"):
        pre_bgr = cv2.imread(os.path.join(FENOMENA_DIR, row_id, "pre.jpg"), cv2.IMREAD_COLOR)
        post_bgr = cv2.imread(os.path.join(FENOMENA_DIR, row_id, "post.jpg"), cv2.IMREAD_COLOR)
        feat = compute_pair_features(pre_bgr, post_bgr)
        feat["Id"] = row_id
        rows.append(feat)

    df_feat = pd.DataFrame(rows)
    cols = ["Id"] + [c for c in df_feat.columns if c != "Id"]
    df_feat = df_feat[cols]
    df_feat.to_parquet(OUT_PATH, index=False)
    print("Saved features:", df_feat.shape, "->", OUT_PATH)

if __name__ == "__main__":
    main()
