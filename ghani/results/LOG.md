# 🧪 Log Eksperimen - Ghani

Skor validasi v1–v4 = hold-out lokal 15%; v5–v6 = **5-Fold Stratified Cross-Validation penuh (1.243.617 baris OOF, leak-free)**. Kolom "LB public" diisi setelah submit di Kaggle.

| # | Submission | Model | Ringkasan | Data latih | OOF LogLoss | Valid Macro F1 (tuned) | LB public |
|---|---|---|---|---|---|---|---|
| v1 | `submission_v1_ghani.csv` | `lgbm_v1.txt` | Agregasi scan/rute/invoice + tabel master, LightGBM | subsample 450 rb | - | 0.4941 | - |
| v2 | `submission_v2_ghani.csv` | `lgbm_v2.txt` | v1 + fitur urutan scan (first/last, bigram, hub change) | subsample 450 rb | - | 0.5018 | - |
| v3 | `submission_v3_ghani.csv` | `lgbm_v3.txt` | v2 + target encoding OOF (customer, courier, hub, package, promo, rute) + count encoding | subsample 450 rb | - | 0.5732 | - |
| v4 | `submission_v4_ghani.csv` | `lgbm_v4.txt` | Sama dengan v3, data penuh, lr 0.08 | 1,05 jt (85% train) | - | 0.5785 | - |
| Stage-1 | *(internal)* | 5-Fold LightGBM (`stage1/lgb_f0..4.txt`) | Feature Engineering v5: buang ~130 fitur noise, moda-normalized route delay, invoice base vs klaim, profil LOO entitas unsupervised 1,6 jt baris | 5-Fold penuh (1,24 jt) | 0.82372 | 0.60238 | - |
| v5 | `submission_v5_ghani.csv` | Stage-1 + Joint Hierarchical Multinomial GLMM (`stage2/glmm_state_full.npy`) | Model Bayesian Mixed-Effects simultan pada `hub`, `hub×peak`, `courier`, `customer` dengan prior precision per kelas | 5-Fold penuh (1,24 jt) | 0.79543 | **0.61911** | - |
| Stage-2a | *(ensemble)* | 5-Fold `LGB_res` (`stage2/lgb_res/`) | Residual-boosted LightGBM (`init_score=glmm_logit`, Fortran order) + 49 fitur GLMM + 19 interaksi domain | 5-Fold penuh (1,24 jt) | 0.78081 | 0.62406 | - |
| Stage-2b | *(ensemble)* | 5-Fold `LGB_full` (`stage2/lgb_full/`) | Full-tree LightGBM dari awal melihat semua fitur + random effects GLMM (topologi pohon berbeda) | 5-Fold penuh (1,24 jt) | 0.78176 | 0.62339 | - |
| Stage-2c | *(ensemble)* | 5-Fold `XGBoost` (`stage2/xgb/`) | Depth-wise `hist` XGBoost (`base_margin=glmm_logit`, `max_depth=6`) | 5-Fold penuh (1,24 jt) | 0.78102 | 0.62387 | - |
| Stage-2d | *(ensemble)* | 5-Fold `TabularResNet` (`stage2/nn/`) | PyTorch 256×2 SiLU ResNet + signed-log1p + residual skip dari GLMM logits | 5-Fold penuh (1,24 jt) | 0.78123 | 0.62431 | - |
| Stage-2e | *(ensemble)* | 5-Fold `RankGaussResNet` (`stage2/nn2/`) | PyTorch 256×2 SiLU ResNet + QuantileTransformer (RankGauss) + dual skip (`glmm` + `s1`) + Top-3 SWA | 5-Fold penuh (1,24 jt) | 0.78239 | 0.62395 | - |
| **v6** | **`submission_v6_ensemble_ghani.csv`** | **Hybrid 50% Prob Blend + 50% 5-Fold L2 Meta-Stacker** (`meta_v6_ensemble.json`) | Gabungan 5 arsitektur beragam (3 GBDT + 2 Neural Net PyTorch) + optimasi 7 class multipliers Macro F1 (CV leak-free) | 5-Fold penuh (1,24 jt) | **0.77786** | **0.62561** | *(menunggu hasil Kaggle)* |

## Temuan Utama (EDA & Reverse-Engineering DGP)
- **Struktur Entitas & ICC:** Terdapat efek laten kuat di level `courier_id` (`tertunda_operasional` ICC 0.359, `gagal_antar_awal` 0.118), `customer_id` (`konfirmasi_alamat` 0.347, `retur_pengirim` 0.238), dan `hub_id` (`tertunda_operasional` 0.198, `klaim_hilang` 0.063).
- **Profil Unsupervised LOO (1,6 jt baris train+test, bebas bocor target):** Rata-rata LOO `scan_gap` kurir berkorelasi `r = 0.867` dengan `tertunda_operasional`; rata-rata LOO `scan_quality` kurir berkorelasi `r = -0.783` dengan `gagal_antar_awal` (jauh melampaui kolom kasar `co_courier_score`).
- **Normalisasi Moda Rute:** Rasio keterlambatan `durasi_jam / (jarak_segmen_km / median_speed_moda)` memisahkan `normal`, `tertunda_operasional`, dan kelas insiden lain jauh lebih bersih daripada durasi mentah.
- **Invoice Terpisah:** Setiap pengiriman memiliki 1 invoice dasar (`pengiriman`/`langganan`) + 0/1 invoice `klaim`; keberadaan `has_klaim_inv` hampir melipatgandakan peluang `retur_pengirim` (5.4% → 14.6%) dan `klaim_hilang`.
- **Pembersihan Noise:** Kolom `kota_asal`, `kota_tujuan`, `promo_id`/`promotions.csv`, `device_type`, `payment_method`, bigram scan, dan 6 indeks distraktor terbukti 0 sinyal dan dibuang agar model tidak overfit.
- **Keragaman Arsitektur:** Korelasi prediksi `LGB_res` vs `XGBoost` mencapai `0.9989`, sedangkan `TabularResNet` vs GBDT hanya `0.9918` — sehingga kombinasi GBDT + PyTorch Neural Networks memberikan penurunan LogLoss terbesar (`-0.0030`) dan peningkatan Macro F1 yang konsisten.
- **Verifikasi Pra-Submit (`v6`):** 362.245 baris tepat, 0 NaN, urutan `shipment_id` 100% cocok dengan `sample_submission.csv`, adversarial AUC train vs test `0.538` (stabil), dan simulasi bootstrap Public LB (30% test) berada pada rentang `0.6229 – 0.6290` (rata-rata `0.6260`).

## Deliverable Akhir & Inferensi Tanpa Retraining
- Seluruh model tersimpan rapi di `ghani/models/` (~320 MB total, tiap file < 35 MB).
- Inferensi ulang 100% **tanpa melatih ulang** tersedia melalui:
  - Script standalone: `ghani/src/infer_v6.py`
  - Notebook akhir kompetisi: `ghani/notebooks/final_datavidia_task1_ghani.ipynb`
- Dua kandidat final untuk Private LB:
  1. `submissions/submission_v6_ensemble_ghani.csv` (utama — CV Macro F1 `0.62561`)
  2. `submissions/submission_v5_ghani.csv` (cadangan paling stabil/sederhana — CV Macro F1 `0.61911`)
