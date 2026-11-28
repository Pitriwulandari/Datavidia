# 📂 Workspace — Farhat | Task-2 (Datavidia 11.0)

Folder kerja pribadi untuk Task 2: klasifikasi 6 jenis fenomena (Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion) dari pasangan citra satelit pre/post — **tanpa label ground truth**. Metrik: Macro-F1.

---

## Struktur folder (per versi)

```
C:/Datavidia/Datavidia/farhat/TASK-2/
├── v1/                     ← baseline v1 (DINOv2 CLS + k-means + probe LR/kNN)
│   ├── models/             ← cluster_mapping.json, feature_state_*.joblib, probe_*.joblib, cv_report_*.json
│   ├── notebooks/          ← Datavidia_Task2_Notebook.ipynb (inference-only)
│   ├── src/                ← config.py, utils.py, prepare.py, embed.py, features.py, cluster.py, train.py, predict.py, run_pipeline.py, make_notebook.py
│   └── requirements.txt    ← torch, timm, scikit-learn, pandas, pillow, tqdm
│
├── v2/                     ← graph clustering (Leiden) + change stats
│   ├── models/             ← Leiden clustering artifacts
│   ├── notebooks/          ← Datavidia_Task2_Notebook_v2.ipynb
│   ├── src/                ← v2 pipeline scripts
│   └── requirements.txt    ← v1 + opencv-python-headless, igraph, leidenalg, matplotlib
│
├── v3/                     ← **final submission** (Patch-Chamfer @336px + Hurricane anchor + ensemble)
│   ├── models/             ← 16 file artefak kompetisi (mlp_v3.onnx, lr_v3.onnx, knn_v3.joblib, scaler_v3.joblib, label_encoder_v3.joblib, model_config_v3.json, hurricane_mask_v3.npy, cluster_mapping_v3.json, *_evidence.csv)
│   ├── notebooks/          ← model_v3.ipynb (training), submission_v3.ipynb (inference-only)
│   ├── src/                ← patch_chamfer_v3.py, cluster_v3.py, train_v3.py, config.py, utils.py
│   └── requirements.txt    ← full v3 deps (onnx, onnxruntime, skl2onnx, jupyter_client, nbconvert, nbformat, dll.)
│
├── requirements.txt        ← v3 pipeline deps (pinned dari venv)
└── README.md               ← file ini
```

---

## Ringkasan pendekatan per versi

### v1 (baseline)
- **Encoder**: DINOv2 ViT-S/14 @224px (CLS token, 196-d) + MobileNetV3-Large alternatif
- **Fitur**: embedding L2-normalized + 31 statistik perubahan piksel (RGB/HSV/gray, persentil, tekstur, Canny edge)
- **Clustering**: balanced k-means (k=6) + consensus subset-blok
- **Pseudo-label**: visual montage (`cache/figures/cluster_*_prototypes.jpg`) + profil statistik
- **Supervisi lemah**: Logistic Regression + kNN di atas pseudo-label
- **Balancing**: Sinkhorn (karena Macro-F1 bobot tiap kelas sama)
- **Skor LB**: 0.14510 (public leaderboard)
- **Artefak**: `models/v1/`, `notebooks/v1/`, `submissions/v1/`

### v2 (perbaikan clustering)
- **Perubahan utama**: ganti k-means → **Leiden clustering** pada graph k-NN dari embedding DINOv2
- **Fitur tambahan**: change stats 31-d lebih dikembangkan
- **Notebook**: `Datavidia_Task2_Notebook_v2.ipynb`
- **Status**: eksperimen perbaikan pemetaan klaster→kelas

### v3 (submission terakhir)
- **Encoder**: DINOv2 ViT-S/14 @336px (frozen) → 24×24 patch = 576 token/citra
- **Fitur**: **Patch-Chamfer** (alignment-free) + pooled mean
  - `pmean_pre` (384) + `pmean_post` (384) + `topchg_delta` (384) + `chamfer_stats` (12) = **1152 dim**
- **Clustering**: Leiden pada `topchg_delta` residual + **Hurricane anchor** (precision 1.0 dari v3 KMeans)
- **Pseudo-label**: **v3 KMeans(7) centroid matching** (greedy) + majority vote → 32 klaster → 6 kelas
- **Supervisi lemah**: **Ensemble weighted probability** = 0.45·MLP + 0.35·LR + 0.20·kNN (cosine)
- **Override**: Hurricane mask (482 sampel) → paksa kelas Hurricane
- **CV 5-fold (seed 42)**: MLP 0.6460, LR 0.5709, kNN 0.6153, **Ensemble 0.6586 ± 0.0251**
- **Artefak kompetisi**: `.onnx` (MLP, LR) + `.joblib` (kNN, scaler, label_encoder)
- **Notebook submission**: inference-only, **exact match 0 mismatch** vs referensi (3180 baris)
- **Estimasi LB realistis**: 0.50–0.60

---

## Kepatuhan aturan kompetisi (v3)

| Aturan | Status | Bukti |
|---|---|---|
| Model artifact: `.pt` / `.onnx` / `.keras` / dll | ✅ | `mlp_v3.onnx`, `lr_v3.onnx`, `*.joblib` |
| Notebook submission jalan **tanpa retrain** | ✅ | `submission_v3.ipynb` → exact match 0 mismatch |
| Notebook model + notebook submission (maks 2) | ✅ | `model_v3.ipynb`, `submission_v3.ipynb` |
| Reproducible (seed, deterministik) | ✅ | Seed 42 semua tahap; frozen mask + mapping |
| Dilarang VLM / CLIP / prompt / fine-tune | ✅ | Encoder pure-vision frozen (`requires_grad=False`) |
| Dilarang data eksternal berlabel | ✅ | Hanya citra kompetisi + bobot pretrained publik |

---

## Cara menjalankan (v3)

```bash
cd C:/Datavidia/Datavidia/farhat/TASK-2/v3

# Submission notebook (wajib jalan tanpa retrain)
.venv/Scripts/jupyter nbconvert --to notebook --execute notebooks/submission_v3.ipynb \
    --output submission_v3_executed.ipynb --output-dir notebooks

# Training notebook (dari nol, reproduktif)
.venv/Scripts/jupyter nbconvert --to notebook --execute notebooks/model_v3.ipynb \
    --output model_v3_executed.ipynb --output-dir notebooks

# CLI per tahap
.venv/Scripts/python.exe -m src.patch_chamfer_v3   # ekstraksi fitur (~jam)
.venv/Scripts/python.exe -m src.cluster_v3         # clustering + mapping
.venv/Scripts/python.exe -m src.train_v3           # training + export + submission
```

---

## Install dependencies

```bash
cd C:/Datavidia/Datavidia/farhat/TASK-2
.venv/Scripts/python.exe -m pip install -r requirements.txt

# per versi
cd v1 && .venv/Scripts/python.exe -m pip install -r requirements.txt
cd v2 && .venv/Scripts/python.exe -m pip install -r requirements.txt
cd v3 && .venv/Scripts/python.exe -m pip install -r requirements.txt
```

---

## Catatan teknis (v3)

- **2 core CPU** → DINOv2-S @336px ≈ 5.3 detik/sampel (2 forward pass). Ekstraksi total ~4–5 jam.
- **xFormers tidak tersedia** → warning SwiGLU/Attention/Block, tapi fungsional.
- **Input ONNX**: `float32 [N, 1152]` name `input` → output `label` (int64) + `probabilities` (float32 [N,6]).
- **kNN pakai input UNSCALED** (cosine); MLP & LR pakai `scaler.transform()`.
- **Seed 42** di semua tahap: numpy, torch, sklearn, Leiden (`random_state=SEED`).

---

## Referensi

- DINOv2: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023.
- Patch-Chamfer: adaptasi alignment-free chamfer distance untuk patch-token pre/post.
- v4deep CV 0.71 pada 4 klaster (Hurricane terpisah jelas) → dasar anchor.
- Ghani v3: single-backbone Macro-F1 0.37895 (baseline tim).
