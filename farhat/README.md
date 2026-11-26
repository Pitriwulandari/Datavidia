# 📂 Workspace - Farhat

Folder kerja pribadi untuk eksplorasi, analisis, dan eksperimen model oleh **Farhat**.

Prediksi multikelas `status_penanganan` (7 kelas) dengan metrik Macro F1.

## Struktur
- `data/raw/` — dataset kompetisi (tidak disertakan di repo sesuai Competition Rules)
- `notebooks/` — 01_eda, 02_preprocessing_features, 03_baseline_model, 04_advanced_modeling
- `src/` — data_loader, features, models, utils

## Cara menjalankan
1. Letakkan dataset mentah di `farhat/data/raw/` (subfolder `train/` dan `test/`, plus `sample_submission.csv`). Folder ini di-ignore git.
2. Instal dependensi dari `farhat/requirements.txt`.
3. Buka Jupyter dari folder `farhat/`, lalu jalankan notebook berurutan: 01 → 02 → 03 → 04.

## Model
LightGBM multiclass dengan 140 fitur, target encoding out-of-fold (5-fold),
bobot kelas, dan tuning pengali kelas.

## Skor
- Macro F1 validasi internal (hold-out 12%): 0.57514
- Leaderboard public: 0.56426

## Output
`farhat/submissions/submission.csv` — 362.245 baris, kolom `shipment_id,status_penanganan`.
