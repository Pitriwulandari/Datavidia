# Datavidia 11.0 Task-2 — Panduan Pengumpulan & Reproduksi

## Artefak yang Dikumpulkan

Sesuai ketentuan kompetisi, yang dikumpulkan:

1. **Model terlatih** (`models/probe_main.joblib`, 5.1 MB)
   - Berisi logistic regression + kNN yang sudah dilatih
   - Format: joblib (kompatibel scikit-learn 1.5+)
   - Tidak perlu file .pt/.onnx/.keras karena encoder DINOv2 beku (bobot publik dari timm/HF Hub, tidak di-train ulang)

2. **Notebook submission** (`notebooks/Datavidia_Task2_Notebook.ipynb`)
   - Inference-only: load encoder beku + state + probe, prediksi 3.180 pasangan, tulis submission
   - **BISA dijalankan tanpa train ulang** — semua state sudah tersimpan di `models/`
   - Runtime estimasi: ~185 menit embedding (CPU 2 core) + ~1 menit prediksi = ~3 jam total

3. **Submission** (`submissions/submission_final.csv`)
   - 3.180 baris, 6 kelas terpakai, format valid
   - Distribusi: Earthquake 534, Fire 582, Hurricane 537, Tsunami 314, Volcano 683, Explosion 530

4. **State/artefak pendukung** (semua di `models/`):
   - `feature_state_main.joblib` — normalisasi fitur (mean/std per dimensi)
   - `cluster_mapping.json` — pemetaan interpretatif klaster→kelas (hanya dokumentasi)
   - `notebook_config.json` — konfigurasi encoder/tag untuk notebook

## Reproduksibilitas

Pipeline sudah reproducible:

- **Seed tetap** (`config.SEED = 42`) untuk KMeans, train/test split, probe
- **Encoder deterministik** (DINOv2 ViT-S/14 beku, `torch.set_num_threads(2)`)
- **Data tidak diubah** — 3.180 pasangan di `data/raw/Fenomena/` persis seperti yang diberikan
- **Dependencies terpasang** — `requirements.txt` dengan versi eksplisit

Jalankan ulang dari awal:

```bash
cd C:/Datavidia/TASK-2
.venv/Scripts/python.exe -m src.run_pipeline
```

Atau jalankan hanya notebook (tanpa train ulang):

```bash
jupyter notebook notebooks/Datavidia_Task2_Notebook.ipynb
```

## Verifikasi

```bash
.venv/Scripts/python.exe -m src.validate
```

Output menunjukkan semua artefak tersedia dan submission valid.

## Catatan Penting

1. **Tidak ada label ground truth** — metrik CV (Macro-F1 0.908 vs pseudo-label) mengukur konsistensi internal, **bukan** skor kompetisi sebenarnya.

2. **Pemetaan klaster→kelas interpretatif** — dilakukan manual berdasarkan profil statistik + inspeksi visual montase; **tidak** diverifikasi dengan label asli.

3. **Encoder tidak di-train ulang** — DINOv2 dipakai beku (pure-vision, tanpa prompt teks/VLM), sesuai aturan kompetisi yang memperkenankan bobot pretrained publik.

4. **Waktu komputasi** — embedding 3.180 pasangan butuh ~185 menit di CPU 2 core (proses paling lama); clustering + training probe ~15 menit.

5. **Sumber encoder** — `timm` model zoo (vit_small_patch14_dinov2.lvd142m), via Hugging Face Hub, lisensi Apache-2.0, tidak dimodifikasi.

## Struktur File yang Dikumpulkan

```
models/
  probe_main.joblib              # model terlatih (wajib)
  feature_state_main.joblib      # state normalisasi (wajib)
  cluster_mapping.json           # dokumentasi pemetaan (opsional)
  notebook_config.json           # config notebook (opsional)
notebooks/
  Datavidia_Task2_Notebook.ipynb # notebook submission (wajib)
submissions/
  submission_final.csv           # hasil prediksi (wajib)
requirements.txt                 # dependencies
docs/
  WRITEUP.md                     # dokumentasi pengembangan
  README.md                      # panduan ini
```

Semua artefak sudah divalidasi dan siap dikumpulkan.
