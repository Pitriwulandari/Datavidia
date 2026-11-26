# 📦 Panduan Pengumpulan Berkas Task-1 ke Google Form Panitia

- **Link Form Panitia:** https://forms.gle/Eqvg6aryJJDCFvUr9
- **Deadline:** **26 November 2026 pukul 23.59 WIB**
- **Tim:** `p<0.5%` — Universitas Pamulang (Pitri Wulandari, Farhat, Ghani)
- **Skor Kaggle Public LB Submission Terpilih (`v8`):** **`0.60757`** (Rank #16 / 213)

---

## Daftar File yang Sudah Disiapkan Siap Kumpul di Folder Ini (`ghani/task1/pengumpulan_panitia/`)

Sesuai 5 aturan resmi panitia (*maksimal 2 Notebook: Notebook Model & Notebook Submission, wajib bisa dijalankan tanpa train ulang, dan wajib reproducible*):

1. **Notebook Model (Pelatihan Reproducible):**
   - File: **`1_notebook_model_task1_p_lt_0_5pct.ipynb`**
   - Berisi 100% kode mandiri (*standalone*, tanpa import skrip luar) untuk EDA, *Feature Engineering* 9 tabel, pelatihan 5-Fold Stage-1 LightGBM, Scale-Normalized Joint GLMM, 5-Fold Stage-2 (`LGB_res`, `LGB_full`, `XGBoost`, PyTorch `TabularResNetV8` `.pt`, PyTorch `RankGaussResNetV8` `.pt`), dan *Hybrid Meta-Stacker* dengan *seed* terkunci (`SEED = 42`).

2. **Notebook Submission (Prediksi *Zero-Retraining*):**
   - File: **`2_notebook_submission_task1_p_lt_0_5pct.ipynb`**
   - Berisi 100% kode mandiri (*standalone*, tanpa import skrip luar & tanpa butuh `train/*.csv` maupun `cache/`) yang langsung membaca CSV mentah `test/*.csv` + memuat bobot model dari `models_v8/` tanpa melatih ulang satu model pun, lalu menghasilkan **`submission_v8_task1.csv`** (`362.245` baris, terverifikasi **100,0000% identik** dengan `submission_v8_ghani.csv` di Kaggle!).

3. **Berkas Model Hasil Train:**
   - Paket ZIP tunggal siap upload: **`models_v8_task1_p_lt_0_5pct.zip`** (`~160 MB`, berisi folder `models_v8/` lengkap dengan bobot PyTorch `.pt`, LightGBM `.txt`, XGBoost `.ubj`, Meta-Stacker & Scaler `.joblib`, GLMM `.npy`/`.npz`, dan spesifikasi `.json`).
   - *(Catatan: Jika Google Form memiliki batas ukuran upload file < 100 MB atau meminta link Google Drive untuk model, upload `models_v8_task1_p_lt_0_5pct.zip` ke Google Drive tim dan set akses ke **"Anyone with the link / Siapa saja yang memiliki link dapat melihat"**! Jika form menerima upload langsung hingga 1 GB, bisa langsung upload file `.zip`-nya.)*
