import os
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import timm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2"
FENOMENA_DIR = os.path.join(DATA_DIR, "Fenomena")
SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

BASE_FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "pitri/task2/cache/features_v2_spatial.parquet"
EMB_CACHE_PATH = "pitri/task2/cache/dinov2_vitb14_emb.npz"

MODEL_DIR = "pitri/task2/models/v3"
OUT_CSV = "pitri/submissions/task2/submission_v3_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"

CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"


def zscore_arr(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-6 else 1.0)


@torch.no_grad()
def extract_dinov2_embeddings(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(EMB_CACHE_PATH):
        data = np.load(EMB_CACHE_PATH)
        return data["pre_emb"], data["post_emb"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    batch_size = 32 if device == "cuda" else 8
    pre_list, post_list = [], []

    for i in tqdm(range(0, len(ids), batch_size), desc=f"Extracting {MODEL_NAME}"):
        batch_ids = ids[i : i + batch_size]
        pre_tensors, post_tensors = [], []
        pre_flip_tensors, post_flip_tensors = [], []

        for rid in batch_ids:
            p_pre = Image.open(os.path.join(FENOMENA_DIR, rid, "pre.jpg")).convert("RGB")
            p_post = Image.open(os.path.join(FENOMENA_DIR, rid, "post.jpg")).convert("RGB")

            pre_tensors.append(transform(p_pre))
            post_tensors.append(transform(p_post))
            pre_flip_tensors.append(transform(p_pre.transpose(Image.FLIP_LEFT_RIGHT)))
            post_flip_tensors.append(transform(p_post.transpose(Image.FLIP_LEFT_RIGHT)))

        x_pre = torch.stack(pre_tensors).to(device)
        x_post = torch.stack(post_tensors).to(device)
        x_pre_f = torch.stack(pre_flip_tensors).to(device)
        x_post_f = torch.stack(post_flip_tensors).to(device)

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            e_pre = F.normalize(0.5 * (model(x_pre) + model(x_pre_f)), dim=-1)
            e_post = F.normalize(0.5 * (model(x_post) + model(x_post_f)), dim=-1)

        pre_list.append(e_pre.cpu().numpy().astype(np.float32))
        post_list.append(e_post.cpu().numpy().astype(np.float32))

    pre_emb = np.vstack(pre_list)
    post_emb = np.vstack(post_list)
    np.savez_compressed(EMB_CACHE_PATH, pre_emb=pre_emb, post_emb=post_emb)
    return pre_emb, post_emb


def compute_physical_scores(df: pd.DataFrame, cos_dist: np.ndarray) -> np.ndarray:
    cd_z = zscore_arr(cos_dist)
    s_explosion = (
        1.45 * zscore_arr(df["peak_to_mean_ratio"].values)
        + 1.15 * zscore_arr(df["grid_max_to_med"].values)
        + 0.65 * zscore_arr(df["grid_std_to_mean"].values)
        - 0.45 * zscore_arr(df["diff_mag_mean"].values)
    )
    s_fire = (
        1.30 * zscore_arr(df["dark_post_frac"].values)
        - 1.05 * zscore_arr(df["dexg_mean"].values)
        - 0.55 * zscore_arr(df["dexg_p10"].values)
        + 0.25 * cd_z
    )
    s_tsunami = (
        1.40 * zscore_arr(df["dblue_mean"].values)
        + 0.85 * zscore_arr(df["dblue_p90"].values)
        - 0.35 * zscore_arr(df["dedge_mean"].values)
        + 0.25 * cd_z
    )
    s_volcano = (
        1.40 * zscore_arr(df["ash_post_frac"].values)
        + 0.65 * zscore_arr(df["dval_mean"].values)
        - 0.65 * zscore_arr(df["dsat_mean"].values)
        + 0.25 * cd_z
    )
    s_hurricane = (
        0.95 * zscore_arr(df["diff_mag_mean"].values)
        - 0.75 * zscore_arr(df["dexg_mean"].values)
        - 0.45 * zscore_arr(df["grid_max_to_med"].values)
        + 0.35 * cd_z
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
    sub = pd.read_csv(SUB_PATH)[["Id"]].copy()
    ids = sub["Id"].astype(str).tolist()

    base_df = pd.read_parquet(BASE_FEAT_PATH)
    sp_df = pd.read_parquet(SPATIAL_FEAT_PATH)
    df = sub.merge(base_df, on="Id", how="left").merge(sp_df, on="Id", how="left")

    pre_emb, post_emb = extract_dinov2_embeddings(ids)
    diff_emb = post_emb - pre_emb
    cos_sim = (pre_emb * post_emb).sum(axis=1)
    cos_dist = 1.0 - cos_sim

    phys_cols = [c for c in df.columns if c != "Id"]
    X_phys = df[phys_cols].fillna(0.0).to_numpy(dtype=np.float32)
    X_phys = np.column_stack([X_phys, cos_dist[:, None]])

    scaler_phys = StandardScaler()
    X_phys_scaled = scaler_phys.fit_transform(X_phys)

    pca_post = PCA(n_components=96, random_state=SEED)
    pca_diff = PCA(n_components=96, random_state=SEED)
    Z_post = pca_post.fit_transform(post_emb)
    Z_diff = pca_diff.fit_transform(diff_emb)

    X_all = np.hstack([Z_post, Z_diff, X_phys_scaled]).astype(np.float32)

    # Bentuk anchor pseudo-label dari skor fisik-visual paling yakin per kelas
    phys_scores = compute_physical_scores(df, cos_dist)
    anchor_quota = {
        "Hurricane": 280,
        "Earthquake": 240,
        "Volcano": 190,
        "Fire": 180,
        "Tsunami": 130,
        "Explosion": 20,
    }
    seed_y = np.full(len(df), -1, dtype=int)
    used = np.zeros(len(df), dtype=bool)
    for c_idx in [5, 4, 3, 2, 0, 1]:
        cname = CLASSES[c_idx]
        k = anchor_quota[cname]
        avail = np.where(~used)[0]
        chosen = avail[np.argsort(-phys_scores[avail, c_idx])[:k]]
        seed_y[chosen] = c_idx
        used[chosen] = True

    train_mask = seed_y >= 0
    X_tr, y_tr = X_all[train_mask], seed_y[train_mask]

    # 3-Family Ensemble (LR + MLP + LightGBM) dengan seed tetap
    clf_lr = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(192, 64), alpha=1.0, max_iter=300, early_stopping=True, random_state=SEED)
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=15,
        subsample_for_bin=2000,
        colsample_bytree=0.7,
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

    # Prior calibration ringan untuk menjaga keseimbangan Macro F1 (6 kelas)
    target_priors = np.array([0.275, 0.240, 0.185, 0.175, 0.115, 0.010], dtype=np.float32)
    emp_priors = np.clip(probs.mean(axis=0), 1e-4, 1.0)
    cal_probs = probs * (target_priors / emp_priors) ** 0.55
    pred_idx = np.argmax(cal_probs, axis=1)

    # Pastikan kelas Explosion minimal 24-30 sampel teratas jika terlalu sedikit
    if (pred_idx == 5).sum() < 22:
        top_exp = np.argsort(-cal_probs[:, 5])[:26]
        pred_idx[top_exp] = 5

    pred_labels = [CLASSES[i] for i in pred_idx]
    out_df = pd.DataFrame({"Id": df["Id"], "Phenomenon": pred_labels})
    assert len(out_df) == 3180
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    out_df.to_csv(OUT_CSV, index=False)

    # Simpan semua artefak model untuk zero-retraining inference
    joblib.dump(
        {
            "scaler_phys": scaler_phys,
            "pca_post": pca_post,
            "pca_diff": pca_diff,
            "clf_lr": clf_lr,
            "clf_mlp": clf_mlp,
            "clf_lgb": clf_lgb,
            "target_priors": target_priors,
            "emp_priors": emp_priors,
            "classes": CLASSES,
        },
        os.path.join(MODEL_DIR, "ensemble_v3.joblib"),
    )
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": MODEL_NAME,
                "source": "timm / Meta AI DINOv2 (Apache-2.0)",
                "seed": SEED,
                "features": "DINOv2 ViT-B/14 (post + diff, HFlip TTA) + RGB/HSV/ExG/Sobel/Grid4x4 physical features",
            },
            f,
            indent=2,
        )

    counts = out_df["Phenomenon"].value_counts()
    print("Saved:", OUT_CSV)
    print("\nDistribusi prediksi v3:")
    for k, v in counts.items():
        print(f"- {k}: {v} ({v/len(out_df)*100:.1f}%)")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v3 DINOv2 ViT-B/14 + Physical Color-Spatial Fusion + 3-Family Ensemble (2026-11-28)\n")
        f.write(f"- Seed: {SEED}; Backbone: `{MODEL_NAME}`; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={v}" for k, v in counts.items()]) + "\n")


if __name__ == "__main__":
    main()
