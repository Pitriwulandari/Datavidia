import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2"
FENOMENA_DIR = os.path.join(DATA_DIR, "Fenomena")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_CSV = "pitri/task2/cache/eda_summary.csv"

sub = pd.read_csv(SUB_PATH)
print("=== 1. CEK SAMPLE SUBMISSION ===")
print("Shape:", sub.shape)
print("Missing values per kolom:\n", sub.isna().sum())
print("Duplikat Id:", sub["Id"].duplicated().sum())
print("Nilai unik Phenomenon saat ini:", sub["Phenomenon"].value_counts(dropna=False).to_dict())

folder_ids = set(os.listdir(FENOMENA_DIR)) if os.path.exists(FENOMENA_DIR) else set()
sub_ids = set(sub["Id"].astype(str))
print("\n=== 2. CEK KESESUAIAN ID FOLDER VS SUBMISSION ===")
print("Jumlah folder di Fenomena/:", len(folder_ids))
print("Id di sample_submission tapi tidak ada foldernya:", len(sub_ids - folder_ids))
print("Folder di Fenomena/ tapi tidak ada di sample_submission:", len(folder_ids - sub_ids))

records = []
for row_id in tqdm(sub["Id"].astype(str), desc="Auditing image pairs"):
    p_dir = os.path.join(FENOMENA_DIR, row_id)
    pre_path = os.path.join(p_dir, "pre.jpg")
    post_path = os.path.join(p_dir, "post.jpg")

    pre_exists = os.path.isfile(pre_path)
    post_exists = os.path.isfile(post_path)

    pre_img = cv2.imread(pre_path, cv2.IMREAD_COLOR) if pre_exists else None
    post_img = cv2.imread(post_path, cv2.IMREAD_COLOR) if post_exists else None

    pre_ok = pre_img is not None and pre_img.size > 0
    post_ok = post_img is not None and post_img.size > 0

    pre_h, pre_w, pre_c = pre_img.shape if pre_ok else (np.nan, np.nan, np.nan)
    post_h, post_w, post_c = post_img.shape if post_ok else (np.nan, np.nan, np.nan)

    if pre_ok and post_ok:
        pre_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        post_rgb = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB).astype(np.float32)

        pre_std = float(pre_rgb.std())
        post_std = float(post_rgb.std())
        pre_mean = float(pre_rgb.mean())
        post_mean = float(post_rgb.mean())

        if (pre_h, pre_w) != (post_h, post_w):
            post_cmp = cv2.resize(post_rgb, (pre_w, pre_h), interpolation=cv2.INTER_LINEAR)
        else:
            post_cmp = post_rgb

        diff_abs = np.abs(post_cmp - pre_rgb)
        mean_abs_diff = float(diff_abs.mean())
        std_abs_diff = float(diff_abs.std())
    else:
        pre_std = post_std = pre_mean = post_mean = mean_abs_diff = std_abs_diff = np.nan

    records.append({
        "Id": row_id,
        "pre_exists": pre_exists,
        "post_exists": post_exists,
        "pre_ok": pre_ok,
        "post_ok": post_ok,
        "pre_h": pre_h, "pre_w": pre_w, "pre_c": pre_c,
        "post_h": post_h, "post_w": post_w, "post_c": post_c,
        "same_shape_as_1024": (
            pre_h == 1024 and pre_w == 1024 and pre_c == 3 and
            post_h == 1024 and post_w == 1024 and post_c == 3
        ),
        "pre_mean": pre_mean, "post_mean": post_mean,
        "pre_std": pre_std, "post_std": post_std,
        "mean_abs_diff": mean_abs_diff, "std_abs_diff": std_abs_diff,
    })

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

print("\n=== 3. RINGKASAN MISSING / CORRUPT / DIMENSI ===")
print("pre.jpg hilang:", (~df["pre_exists"]).sum())
print("post.jpg hilang:", (~df["post_exists"]).sum())
print("pre.jpg gagal dibaca/korup:", (~df["pre_ok"]).sum())
print("post.jpg gagal dibaca/korup:", (~df["post_ok"]).sum())
print("Pasangan tidak tepat 1024x1024x3:", (~df["same_shape_as_1024"]).sum())
if (~df["same_shape_as_1024"]).sum() > 0:
    print("Contoh ukuran non-1024x1024x3:")
    print(df.loc[~df["same_shape_as_1024"], ["Id", "pre_h", "pre_w", "pre_c", "post_h", "post_w", "post_c"]].head(10))

print("\n=== 4. CEK GAMBAR HAMPIR POLOS / LOW VARIANCE (std < 5) ===")
print("pre_std < 5:", (df["pre_std"] < 5).sum())
print("post_std < 5:", (df["post_std"] < 5).sum())

print("\n=== 5. STATISTIK NUMERIK RINGKAS ===")
print(df[["pre_mean", "post_mean", "pre_std", "post_std", "mean_abs_diff", "std_abs_diff"]].describe().round(2))
print("\nTersimpan ke:", OUT_CSV)
