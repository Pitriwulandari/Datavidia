import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_ROOT = "/root/.cache/kagglehub/competitions/task-2-datavidia-11-0/Task-2"
SUB_SAMPLE = f"{DATA_ROOT}/sample_submission.csv"
BASE_FEAT_PATH = "pitri/task2/cache/features_task2.parquet"
SPATIAL_FEAT_PATH = "pitri/task2/cache/features_v2_spatial.parquet"
V9_SUB_PATH = "pitri/submissions/task2/submission_v9_pitri.csv"

CACHE_EMB_PATH = "pitri/task2/cache/siglip_so400m_emb.npz"
OUT_CSV = "pitri/submissions/task2/submission_v10_pitri.csv"
LOG_PATH = "pitri/task2/results/LOG.md"

MODEL_ID = "google/siglip-so400m-patch14-384"
CLASSES = ["Hurricane", "Earthquake", "Volcano", "Fire", "Tsunami", "Explosion"]

CLASS_PROMPTS = {
    "Hurricane": [
        "satellite image of severe hurricane storm damage, flooded coastal neighborhoods and wind destruction",
        "aerial satellite view after tropical cyclone hurricane with widespread roof damage and storm surge",
    ],
    "Earthquake": [
        "satellite image after major earthquake showing collapsed urban buildings, rubble and structural damage",
        "aerial view of city blocks destroyed by earthquake with collapsed concrete roofs and debris",
    ],
    "Volcano": [
        "satellite image of volcanic eruption with grey volcanic ash plume, crater and dark lava flow",
        "aerial view of volcano eruption covering landscape with thick grey ash and pyroclastic deposits",
    ],
    "Fire": [
        "satellite image of active wildfire forest fire with smoke plumes and dark charred burned land",
        "aerial view of massive fire burning buildings and vegetation leaving blackened burn scars",
    ],
    "Tsunami": [
        "satellite image after tsunami wave inundation with muddy coastal flooding and washed away shoreline",
        "aerial view of coastal town devastated by tsunami water surge and shoreline erosion",
    ],
    "Explosion": [
        "satellite image of localized industrial blast explosion crater and concentrated debris",
        "aerial view of port or factory explosion crater with localized blast damage",
    ],
}


def resolve_image_paths(sample_ids: pd.Series):
    all_imgs = glob.glob(f"{DATA_ROOT}/**/*.*", recursive=True)
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    all_imgs = [p for p in all_imgs if os.path.splitext(p)[1].lower() in img_exts]

    pre_map, post_map = {}, {}
    for p in all_imgs:
        bn = os.path.basename(p)
        low = p.lower()
        stem = os.path.splitext(bn)[0]
        clean_id = stem.replace("_pre_disaster", "").replace("_post_disaster", "").replace("_pre", "").replace("_post", "")
        if "pre" in low:
            pre_map[clean_id] = p
            pre_map[stem] = p
        elif "post" in low:
            post_map[clean_id] = p
            post_map[stem] = p

    pre_paths, post_paths = [], []
    for sid in sample_ids.astype(str):
        p_pre = pre_map.get(sid)
        p_post = post_map.get(sid)
        if p_pre is None or p_post is None:
            cand_pre = [p for p in all_imgs if sid in os.path.basename(p) and "pre" in p.lower()]
            cand_post = [p for p in all_imgs if sid in os.path.basename(p) and "post" in p.lower()]
            p_pre = cand_pre[0] if cand_pre else None
            p_post = cand_post[0] if cand_post else None
        assert p_pre is not None and p_post is not None, f"Pasangan gambar pre/post tidak ditemukan untuk Id={sid}"
        pre_paths.append(p_pre)
        post_paths.append(p_post)
    return pre_paths, post_paths


@torch.no_grad()
def extract_or_load_siglip(sample_ids: pd.Series, batch_size: int = 32):
    if os.path.exists(CACHE_EMB_PATH):
        print(f"Memuat cache embedding SigLIP dari {CACHE_EMB_PATH} ...")
        data = np.load(CACHE_EMB_PATH)
        return data["pre_emb"], data["post_emb"], data["zs_probs"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ekstraksi SigLIP ({MODEL_ID}) di device={device} ...")
    pre_paths, post_paths = resolve_image_paths(sample_ids)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Encode text prompts untuk zero-shot similarity
    text_embeds_per_class = []
    for cls in CLASSES:
        t_inputs = processor(text=CLASS_PROMPTS[cls], padding="max_length", return_tensors="pt").to(device)
        t_feat = model.get_text_features(**t_inputs)
        t_feat = torch.nn.functional.normalize(t_feat.float(), dim=-1).mean(dim=0, keepdim=True)
        t_feat = torch.nn.functional.normalize(t_feat, dim=-1)
        text_embeds_per_class.append(t_feat)
    text_bank = torch.cat(text_embeds_per_class, dim=0)  # (6, D)

    def encode_list(paths, desc):
        embs = []
        for i in tqdm(range(0, len(paths), batch_size), desc=desc):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            if device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            feat = model.get_image_features(**inputs).float()
            feat = torch.nn.functional.normalize(feat, dim=-1)
            embs.append(feat.cpu().numpy().astype(np.float32))
        return np.vstack(embs)

    pre_emb = encode_list(pre_paths, "SigLIP Pre")
    post_emb = encode_list(post_paths, "SigLIP Post")

    # Zero-shot logits fokus pada perubahan post relatif terhadap pre
    post_t = torch.from_numpy(post_emb).to(device)
    pre_t = torch.from_numpy(pre_emb).to(device)
    logits = 15.0 * ((post_t @ text_bank.T) - 0.35 * (pre_t @ text_bank.T))
    zs_probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(CACHE_EMB_PATH), exist_ok=True)
    np.savez_compressed(CACHE_EMB_PATH, pre_emb=pre_emb, post_emb=post_emb, zs_probs=zs_probs)
    print(f"Cache SigLIP tersimpan di {CACHE_EMB_PATH}")
    return pre_emb, post_emb, zs_probs


def main():
    sample = pd.read_csv(SUB_SAMPLE)[["Id"]]
    sample_ids = sample["Id"]

    pre_emb, post_emb, zs_probs = extract_or_load_siglip(sample_ids)
    diff_emb = post_emb - pre_emb
    cos_sim = np.sum(pre_emb * post_emb, axis=1, keepdims=True)
    l2_diff = np.linalg.norm(diff_emb, axis=1, keepdims=True)

    # Gabung fitur fisik/spasial yang sudah ada
    df_tab = (
        pd.DataFrame({"Id": sample_ids})
        .merge(pd.read_parquet(BASE_FEAT_PATH), on="Id", how="left")
        .merge(pd.read_parquet(SPATIAL_FEAT_PATH), on="Id", how="left")
    )
    tab_cols = [c for c in df_tab.columns if c != "Id"]
    X_tab = np.nan_to_num(df_tab[tab_cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_tab = np.clip(RobustScaler().fit_transform(X_tab), -8.0, 8.0)

    X_all = np.hstack([post_emb, diff_emb, cos_sim, l2_diff, zs_probs, X_tab]).astype(np.float32)

    # Susun pseudo-label konsensus antara zero-shot SigLIP dan prediksi v9 (jika tersedia)
    zs_pred = np.argmax(zs_probs, axis=1)
    zs_conf = np.max(zs_probs, axis=1)

    if os.path.exists(V9_SUB_PATH):
        v9_df = pd.read_csv(V9_SUB_PATH)
        v9_idx = (
            pd.DataFrame({"Id": sample_ids})
            .merge(v9_df[["Id", "Phenomenon"]], on="Id", how="left")["Phenomenon"]
            .map({c: i for i, c in enumerate(CLASSES)})
            .to_numpy()
        )
        agree_mask = (zs_pred == v9_idx) | (zs_conf >= np.quantile(zs_conf, 0.55))
    else:
        v9_idx = None
        agree_mask = zs_conf >= np.quantile(zs_conf, 0.45)

    y_pseudo = zs_pred.copy()
    # Pastikan tiap kelas terwakili minimal 25 sampel dari skor zero-shot tertinggi
    for c in range(len(CLASSES)):
        top_c = np.argsort(-zs_probs[:, c])[:25]
        y_pseudo[top_c] = c
        agree_mask[top_c] = True

    X_tr = X_all[agree_mask]
    y_tr = y_pseudo[agree_mask]
    print(f"Sampel latih CatBoost (konsensus + high-conf SigLIP): {len(X_tr)}/{len(X_all)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(X_tr), len(CLASSES)), dtype=np.float32)
    test_probs = np.zeros((len(X_all), len(CLASSES)), dtype=np.float32)

    task_type = "GPU" if torch.cuda.is_available() else "CPU"
    for fold, (tr_i, va_i) in enumerate(skf.split(X_tr, y_tr), 1):
        cb = CatBoostClassifier(
            iterations=900,
            learning_rate=0.04,
            depth=6,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=SEED + fold,
            task_type=task_type,
            verbose=200,
        )
        cb.fit(X_tr[tr_i], y_tr[tr_i], eval_set=(X_tr[va_i], y_tr[va_i]), early_stopping_rounds=80, verbose=False)
        oof_probs[va_i] = cb.predict_proba(X_tr[va_i])
        test_probs += cb.predict_proba(X_all) / skf.n_splits

    # Blend probabilitas CatBoost (SigLIP + Tabular) dengan zero-shot SigLIP & v9
    blended = 0.72 * test_probs + 0.28 * zs_probs
    if v9_idx is not None:
        v9_onehot = np.eye(len(CLASSES), dtype=np.float32)[v9_idx]
        blended = 0.82 * blended + 0.18 * v9_onehot

    # Kalibrasi prior halus agar ke-6 komponen Macro F1 tetap seimbang
    target_priors = np.array([0.30, 0.22, 0.16, 0.15, 0.14, 0.03], dtype=np.float32)
    emp_priors = np.clip(blended.mean(axis=0), 1e-4, 1.0)
    cal_probs = blended * (target_priors / emp_priors) ** 0.65
    pred_idx = np.argmax(cal_probs, axis=1)

    out_df = pd.DataFrame({"Id": sample_ids, "Phenomenon": [CLASSES[i] for i in pred_idx]})
    assert len(out_df) == 3180 and (out_df["Id"] == sample_ids).all()
    assert out_df["Phenomenon"].isin(CLASSES).all()
    assert out_df.isna().sum().sum() == 0

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    vc = out_df["Phenomenon"].value_counts()
    print("\nSaved:", OUT_CSV)
    print("Distribusi v10 (SigLIP-SO400M + CatBoost Balanced):")
    for k in CLASSES:
        cnt = int(vc.get(k, 0))
        print(f"- {k:10s}: {cnt:4d} ({cnt/len(out_df)*100:5.1f}%)")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n## v10 SigLIP-SO400M + CatBoost Balanced (2026-11-28)\n")
        f.write(f"- Cache: `{CACHE_EMB_PATH}`; Train samples: {len(X_tr)}/{len(X_all)}; Output: `{OUT_CSV}`\n")
        f.write("- Distribution: " + ", ".join([f"{k}={int(vc.get(k, 0))}" for k in CLASSES]) + "\n")


if __name__ == "__main__":
    main()
