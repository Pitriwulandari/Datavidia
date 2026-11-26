# 📂 Workspace - Farhat | Task-2

Folder kerja pribadi untuk Task 2: klasifikasi 6 jenis fenomena dari pasangan citra satelit pre/post.

Prediksi multikelas `Phenomenon` (6 kelas: Earthquake, Fire, Hurricane, Tsunami, Volcano, Explosion) dengan metrik Macro F1.

---

# Datavidia 11.0 — Task 2: Pipeline Klasifikasi Fenomena (Tanpa Label Latih)

## Ringkasan pendekatan

Dataset **hanya menyediakan pasangan citra `pre.jpg` / `post.jpg` dan
`sample_submission.csv` yang seluruhnya berisi `Earthquake`**. Tidak ada satu pun
label ground truth yang dapat dipakai untuk supervised learning, dan aturan
kompetisi melarang model berbasis prompt (VLM), data eksternal berlabel, serta
pelatihan dari awal. Oleh karena itu pipeline ini memakai:

- encoder **pure-vision beku** dari `timm` (DINOv2 ViT-S/14, `lvd142m`; CNN ringan
  sebagai alternatif) — citra saja, tanpa teks/prompt, bobot publik;
- representasi pasangan: `pre`, `post`, `post-pre`, dan 31 statistik perubahan
  piksel (RGB/HSV/gray, persentil, tekstur, CANNY/edge change);
- embedding DINOv2 mentah ternormalisasi L2 (citra `post`, opsional `pre`/`delta`)
  + 31 statistik perubahan piksel yang distandardisasi. Diagnostik menunjukkan
    PCA-whitening merusak struktur (ARI ~0.0), sedangkan representasi mentah
    stabil (ARI ~0.73 antar-seed);
- k-means dengan banyak inisialisasi + konsensus subset-blok (k=6 sesuai aturan);
- interpretasi klaster -> nama fenomena berdasarkan montage bukti visual
  (`cache/figures/cluster_*_prototypes.jpg`) dan profil statistik;
- **probe supervisi-lemah** (logistic regression + kNN) di atas pseudo-label,
  lalu penugasan akhir diseimbangkan dengan Sinkhorn karena metrik resmi adalah
  Macro-F1 (bobot tiap kelas sama).

> Catatan jujur: metrik 5-fold CV pada `models/cv_report_*.json` mengukur
> **konsistensi terhadap pseudo-label**, bukan akurasi terhadap ground truth
> yang tidak tersedia. Angka publik belum diukur karena tidak ada akses ke
> leaderboard saat pipeline dibuat.

## Kepatuhan aturan

| Aturan | Status | Bukti di kode |
|---|---|---|
| Dilarang prompt / VLM / gambar lain sebagai input model | Dipatuhi | `src/embed.py` hanya `timm.create_model(...).eval()` + transform citra; tidak ada tokenizer/teks |
| Dilarang training dari awal / fine-tune bobot | Dipatuhi | encoder `pretrained=True`, `requires_grad=False`, tidak ada optimizer/scheduler |
| Dilarang data eksternal berlabel ditambahkan ke data latih | Dipatuhi | Hanya pasangan citra kompetisi yang dipakai; bobot pretrained publik diizinkan aturan (baris 27) dan dicantumkan sumber/versinya |
| Dilarang menggunakan test set untuk training | Dipatuhi | tidak ada target/label sama sekali; unsupervised + pseudo-label dari fitur citra itu sendiri |
| Ukuran gambar/format | Dipatuhi | thumbnail 256x256 RGB dari `pre.jpg`/`post.jpg`, prediksi 3.180 baris |
| Nama file submission | Dipatuhi | `submissions/submission_final.csv` dengan header `Id,Phenomenon` |
| Notebook inference-only | Dipatuhi | `notebooks/Datavidia_Task2_Notebook.ipynb` memuat model & state, tanpa melatih ulang |

Sumber bobot:
- DINOv2 ViT-S/14 — Oquab et al., *DINOv2: Learning Robust Visual Features
  without Supervision*, 2023; melalui `timm`
  (`vit_small_patch14_dinov2.lvd142m`, lisensi DINOv2).
- MobileNetV3-Large 100 — Howard et al., 2019; `mobilenetv3_large_100.ra_in1k`
  melalui `timm` (bobot ImageNet-1k).

## Struktur repo

```
src/
  config.py        konfigurasi path, kelas, hyperparameters
  utils.py         logging, seed, IO, normalisasi
  prepare.py       manifest, cache thumbnail, 31 statistik perubahan
  embed.py         embedding encoder beku (tanpa prompt)
  features.py      gabung fitur + balanced k-means + consensus clustering
  cluster.py       montage prototipe & profil klaster (bukti visual)
  train.py         probe supervisi-lemah + CV (bukan supervised GT)
  predict.py       inferensi, Sinkhorn balance, submission + validasi
  run_pipeline.py  orkestrasi seluruh tahap
  make_notebook.py pembuat notebook inference-only
  validate.py      validasi artefak sebelum pengumpulan
models/v1/         cluster_mapping.json, cv_report_main.json,
                   feature_state_main.joblib, probe_main.joblib
notebooks/v1/      Datavidia_Task2_Notebook.ipynb (inference-only)
requirements.txt   dependencies (torch, timm, sklearn, numpy, pandas, pillow)
```

**Catatan:** Data mentah (`data/`), cache (`cache/`), dan submission results tidak disertakan di repo (download dataset dari Kaggle, run pipeline untuk generate).

## Cara menjalankan

```bash
# dari C:/Datavidia/TASK-2
.venv/Scripts/python.exe -m src.run_pipeline --stage all

# atau tahap per tahap
.venv/Scripts/python.exe -m src.prepare
.venv/Scripts/python.exe -m src.embed --encoder dinov2_s --size 196
.venv/Scripts/python.exe -m src.features --encoders dinov2_s --tag main --runs 16
.venv/Scripts/python.exe -m src.cluster --clusters main
# isi models/cluster_mapping.json berdasarkan montage, lalu:
.venv/Scripts/python.exe -m src.train --clusters main --tag main
.venv/Scripts/python.exe -m src.predict --clusters main --probe main --tag final
```

Notebook: buka `notebooks/Datavidia_Task2_Notebook.ipynb` dan jalankan sel
berurutan; ia memakai encoder beku + model tersimpan, tanpa pelatihan ulang.

## Hasil

**v1 (models/v1/, notebooks/v1/):**
- Skor Kaggle: 0.14510 (public leaderboard, rank jauh dari top 0.75732)
- CV lokal vs pseudo-label: Macro-F1 0.908 (tidak mencerminkan skor GT)
- Distribusi prediksi: Earthquake 534, Fire 582, Hurricane 537, Tsunami 314, Volcano 683, Explosion 530
- Penyebab skor rendah: kemungkinan pemetaan klaster→kelas tidak cocok dengan ground truth
- Artefak tersimpan di `models/v1/` dan `notebooks/v1/`

**v2:** dalam perbaikan untuk meningkatkan skor
