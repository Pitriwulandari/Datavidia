# TASK-1 — Farhat (Datavidia 11.0)

Prediksi multikelas `status_penanganan` (7 kelas) dengan metrik Macro F1.
Folder ini berisi paket final Task-1 sekaligus arsip eksplorasi pipeline lama.

## Isi folder

| Path | Keterangan |
|---|---|
| `notebooks/01_model_training.ipynb` | Notebook model final: feature engineering, Stage-1 LightGBM 5-fold, Stage-2 GLMM, ensemble |
| `notebooks/02_submission_inference.ipynb` | Notebook submission final: muat model → prediksi → `submissions/submission_final.csv` |
| `notebooks/01_eda.ipynb` … `04_advanced_modeling.ipynb` | Arsip pipeline lama (140 fitur, target encoding OOF), bukan bagian submission final |
| `models/` | 17 artefak model final: 5 `lgb_f*.txt`, `feature_spec.json`, `meta_stage1.json`, `model_summary.json`, `folds.npy`, logits Stage-1/Stage-2, 3 tabel koreksi, `ensemble_weight.npy` |
| `src/` | Skrip pipeline asli: `build_features.py`, `run_stage1.py`, `run_stage2_glmm.py`, `ensemble_final.py`, dll. |
| `requirements.txt` | Environment teruji (Python 3.10) |
| `README.md` | Dokumen ini |

Data mentah dan submission tidak disertakan. Letakkan data di `TASK-1/data/raw/`:
`train/` dan `test/` masing-masing berisi 8 CSV (`shipments.csv`, `scan_events.csv`,
`route_segments.csv`, `invoices.csv`, `customers.csv`, `couriers.csv`, `hubs.csv`,
`service_packages.csv`), plus `sample_submission.csv` langsung di `data/raw/`.

## Pipeline final

- **Stage-1:** LightGBM 5-fold, 190 fitur dari 9 tabel (fitur entitas, agregasi, interaksi).
- **Stage-2:** koreksi efek acak per entitas — GLMM dengan regularisasi `hub = 10`,
  `courier = 5`, `customer = 20`, di-fit pada logits OOF.
- **Prediksi akhir:** murni Stage-2 (`ensemble_weight = 0`), 7 kelas.

## Skor

| Tahap | Macro F1 | Catatan |
|---|---|---|
| Stage-1 OOF | `0.589473` | 5-fold, jujur |
| Stage-2 cross-fit | `0.591851` | efek di-fit per fold, jujur |
| Stage-2 in-sample | `0.605382` | optimistis, hanya diagnosa |

Leaderboard public belum diisi untuk paket ini. Sebagai pembanding, pipeline lama
mencatat public `0.56426`.

## Menjalankan notebook submission (tanpa latih ulang)

1. Python 3.10, lalu `pip install -r requirements.txt`.
2. Letakkan data mentah di `TASK-1/data/raw/train/` dan `TASK-1/data/raw/test/`
   (masing-masing 8 CSV seperti daftar di atas) serta `sample_submission.csv` di `data/raw/`.
3. Jalankan dari folder `TASK-1`:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 \
  notebooks/02_submission_inference.ipynb
```

Notebook mencari folder `src/` ke atas dari lokasi kerjanya, sehingga `ROOT` menjadi
folder `TASK-1`. Model dimuat dari `models/`, hasil ditulis ke
`submissions/submission_final.csv` (362.245 baris).

`01_model_training.ipynb` default `RUN_STAGE1 = False` (memakai artefak tersimpan).
Ubah ke `True` hanya bila ingin melatih ulang 5 fold dari nol.

## Verifikasi

Notebook submission dijalankan ulang tanpa cache: `max|Δ|` Stage-1 `0`, Stage-2 `4.77e-07`,
submission identik 100% (362.245/362.245), 24/24 pemeriksaan lolos.

## Catatan Git

`.gitignore` repo mengabaikan `*.npy` dan `*.csv`, jadi 9 artefak `.npy` dan
`submission_final.csv` tidak ikut ter-commit kecuali ditambahkan dengan `git add -f`.
Total `models/` 315 MB — lebih aman dibagikan sebagai arsip daripada di-commit.
