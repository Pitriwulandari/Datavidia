# 📂 Workspace - Farhat | Task-1

Prediksi multikelas `status_penanganan` (7 kelas) dengan metrik Macro F1.

---

## Pengerjaan

Task-1 menggunakan dataset tabular 362.245 shipment dengan 8 tabel relasional (shipments, scan_events, route_segments, invoices, customers, couriers, hubs, service_packages). Target: prediksi status akhir shipment dari 7 kemungkinan (Delivered Successfully, Returned, Cancelled, Lost, Damaged, In Transit, Pending).

**Approach:**
- **Pipeline baru (final):** 2-stage ensemble (LightGBM 5-fold + GLMM calibration)
- **Pipeline lama (arsip):** LightGBM dengan 140 fitur + target encoding OOF

---

## Struktur

```
TASK-1/
├── README.md                           # dokumentasi ini
├── requirements.txt                    # dependencies (Python 3.10)
│
├── notebooks/
│   ├── 01_model_training.ipynb         # pipeline baru: training lengkap Stage-1 & Stage-2
│   ├── 02_submission_inference.ipynb   # pipeline baru: inference-only (tanpa train ulang)
│   ├── 01_eda.ipynb                    # pipeline lama (arsip): exploratory data analysis
│   ├── 02_preprocessing_features.ipynb # pipeline lama (arsip): feature engineering
│   ├── 03_baseline_model.ipynb         # pipeline lama (arsip): baseline model
│   └── 04_advanced_modeling.ipynb      # pipeline lama (arsip): tuning & ensemble
│
├── src/
│   ├── build_features.py               # feature engineering (190 fitur dari 8 tabel)
│   ├── run_stage1.py                   # Stage-1: LightGBM 5-fold CV
│   ├── run_stage2_glmm.py              # Stage-2: GLMM calibration per entitas
│   └── ensemble_final.py               # ensemble weighted logit-space
│
├── models/                             # 17 artefak model final (315 MB total)
│   ├── lgb_f0.txt ... lgb_f4.txt       # 5 model LightGBM (1 per fold)
│   ├── feature_spec.json               # daftar 190 fitur + tipe
│   ├── meta_stage1.json                # metadata Stage-1 (CV score, params)
│   ├── model_summary.json              # ringkasan pipeline
│   ├── folds.npy                       # indeks 5-fold split
│   ├── logits_stage1_oof.npy           # prediksi Stage-1 OOF (train)
│   ├── logits_stage1_test.npy          # prediksi Stage-1 (test)
│   ├── logits_stage2_oof.npy           # prediksi Stage-2 OOF (train)
│   ├── logits_stage2_test.npy          # prediksi Stage-2 (test)
│   ├── hub_corr.npy                    # koreksi efek acak per hub
│   ├── courier_corr.npy                # koreksi efek acak per courier
│   ├── customer_corr.npy               # koreksi efek acak per customer
│   └── ensemble_weight.npy             # bobot ensemble Stage-1/Stage-2
│
└── data/                               # (tidak disertakan di repo, .gitignore)
    └── raw/
        ├── train/                      # 8 CSV training
        ├── test/                       # 8 CSV test
        └── sample_submission.csv
```

---

## Penjelasan Folder

**`notebooks/`**
- **01_model_training.ipynb:** Notebook training lengkap pipeline baru. Default `RUN_STAGE1 = False` (memakai artefak tersimpan). Ubah ke `True` untuk train ulang dari nol.
- **02_submission_inference.ipynb:** Notebook submission inference-only. Load model dari `models/`, prediksi test set, generate `submissions/submission_final.csv`. Tidak ada training ulang.
- **01-04 (arsip):** Pipeline lama dengan 140 fitur, target encoding OOF, class weights. Tidak dipakai di submission final, disimpan untuk referensi.

**`src/`**
- **build_features.py:** Feature engineering 190 fitur dari 8 tabel (agregasi scan events, route segments, invoices, entity profiles LOO).
- **run_stage1.py:** Training LightGBM 5-fold CV, simpan model ke `models/lgb_f*.txt` dan logits OOF/test.
- **run_stage2_glmm.py:** GLMM calibration — fit random effects per hub/courier/customer pada logits Stage-1, simpan koreksi ke `models/*_corr.npy`.
- **ensemble_final.py:** Weighted ensemble Stage-1 & Stage-2 di logit-space, generate submission.

**`models/`**
- 17 artefak model final (315 MB total): 5 LightGBM `.txt`, metadata JSON, fold split, logits numpy, 3 tabel koreksi GLMM, bobot ensemble.
- **Catatan:** `.gitignore` repo mengabaikan `*.npy` dan `*.csv`, jadi 9 artefak `.npy` dan `submission_final.csv` tidak ikut commit kecuali ditambahkan dengan `git add -f`. Lebih aman dibagikan sebagai arsip.

**`data/raw/`**
- Dataset mentah kompetisi (tidak disertakan di repo). Download dari Kaggle lalu letakkan:
  - `train/` dan `test/`: masing-masing 8 CSV (shipments, scan_events, route_segments, invoices, customers, couriers, hubs, service_packages)
  - `sample_submission.csv` langsung di `data/raw/`

---

## Versi & Perubahan

**Pipeline Baru (Final — untuk submission):**
- **Stage-1:** LightGBM 5-fold, 190 fitur engineered dari 9 tabel
- **Stage-2:** GLMM calibration — koreksi random effects per hub/courier/customer
- **Ensemble:** Murni Stage-2 (`ensemble_weight = 0`)
- **Skor CV:**
  - Stage-1 OOF: 0.589473 (5-fold, jujur)
  - Stage-2 cross-fit: 0.591851 (efek di-fit per fold, jujur)
  - Stage-2 in-sample: 0.605382 (optimistis, hanya diagnosa)
- **Leaderboard:** Belum disubmit
- **Artefak:** `models/` (17 file, 315 MB), `notebooks/01_model_training.ipynb`, `notebooks/02_submission_inference.ipynb`

**Pipeline Lama (Arsip — referensi):**
- LightGBM multiclass dengan 140 fitur, target encoding OOF (5-fold), class weights, tuning pengali kelas
- **Skor:**
  - Validasi internal (hold-out 12%): 0.57514
  - Leaderboard public: 0.56426
- **Artefak:** `notebooks/01_eda.ipynb` ... `04_advanced_modeling.ipynb`

**Perubahan utama:**
1. Feature engineering lebih kaya: 140 → 190 fitur (tambah entity profiles LOO, interaksi temporal)
2. 2-stage ensemble (LightGBM + GLMM) vs single-stage LightGBM
3. GLMM calibration per entitas meningkatkan Macro-F1 ~0.002
4. CV lebih jujur: 5-fold stratified vs hold-out 12%

---

## Cara Menjalankan

**Setup:**
1. Python 3.10: `pip install -r requirements.txt`
2. Letakkan data mentah di `TASK-1/data/raw/train/`, `TASK-1/data/raw/test/`, dan `sample_submission.csv` di `data/raw/`

**Inference-only (tanpa train ulang):**
```bash
cd TASK-1
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/02_submission_inference.ipynb
```
Hasil: `submissions/submission_final.csv` (362.245 baris)

**Training ulang (opsional):**
```bash
cd TASK-1
# Edit notebooks/01_model_training.ipynb: set RUN_STAGE1 = True
jupyter notebook notebooks/01_model_training.ipynb
# Jalankan semua sel, artefak tersimpan di models/
```

**Verifikasi:**
Notebook submission dijalankan ulang tanpa cache: `max|Δ|` Stage-1 `0`, Stage-2 `4.77e-07`, submission identik 100% (362.245/362.245), 24/24 pemeriksaan lolos.

---

**Tim:** Farhat, Pitri Wulandari, Ghani  
**Kompetisi:** Arkavidia 9.0 — Datavidia 11.0
