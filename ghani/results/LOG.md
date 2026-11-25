# 🧪 Log Eksperimen Lengkap (v1 – v7) — Ghani

Seluruh kode, model, cache, notebook, dan submission kini telah dirapikan per versi (`v1` s/d `v7`) di setiap folder agar mudah ditelusuri.

## Ringkasan Progresi Versi (`v1` → `v7`)

| Versi | Folder Kode | Folder Model | File Submission | Skema Validasi | OOF LogLoss | Valid Macro F1 (Tuned) | LB Public | Ringkasan Perubahan Utama |
|---|---|---|---|---|---|---|---|---|
| **v1** | [src/v1/](../src/v1/) | [models/v1/](../models/v1/) | `submission_v1_ghani.csv` | Hold-out 15% (sub 450rb) | - | `0.4941` | - | Baseline agregasi scan/rute/invoice + tabel master, LightGBM tunggal |
| **v2** | [src/v2/](../src/v2/) | [models/v2/](../models/v2/) | `submission_v2_ghani.csv` | Hold-out 15% (sub 450rb) | - | `0.5018` | - | v1 + fitur sekuensial scan (event awal/akhir, bigram, perubahan hub) |
| **v3** | [src/v3/](../src/v3/) | [models/v3/](../models/v3/) | `submission_v3_ghani.csv` | Hold-out 15% (sub 450rb) | - | `0.5732` | - | v2 + OOF Target Encoding (`customer`, `courier`, `hub`, `package`, `promo`, `rute`) |
| **v4** | [src/v4/](../src/v4/) | [models/v4/](../models/v4/) | `submission_v4_ghani.csv` | Hold-out 15% (full 1,05jt) | - | `0.5785` | - | Sama dengan v3 pada data penuh (`lr=0.08`) |
| **v5** | [src/v5/](../src/v5/) | [models/v5/](../models/v5/) | `submission_v5_ghani.csv` | 5-Fold Stratified (1,24jt) | `0.79543` | `0.61911` | - | Feature Engineering v5 (buang ~130 noise, moda-normalized delay, base vs klaim invoice, profil entitas LOO) + Stage-1 LGB (`0.60238`) + Joint GLMM |
| **v6** | [src/v6/](../src/v6/) | [models/v6/](../models/v6/) | `submission_v6_ghani.csv` | 5-Fold Stratified (1,24jt) | `0.77786` | `0.62561` | **`0.60163`** (#18) | Ensemble 6 model Stage-2 di atas Joint GLMM (`LGB_res`, `LGB_full`, `XGBoost`, `ResNet`, `RGResNet`). **Temuan audit LB:** `u_cust` & `cu_loo` gabungan `train+test` bocor di StratifiedKFold namun 0 korelasi di `test` (`-0.024` gap)! |
| **v7** *(Final)* | [src/v7/](../src/v7/) | [models/v7/](../models/v7/) | **`submission_v7_ghani.csv`** | **5-Fold `GroupKFold(customer_id)`** (0% overlap!) | **`0.77551`** *(sim `0.78652`)* | **`0.62949`** *(TEST-SIM **`0.62866`**)* | *(siap submit!)* | **Perbaikan fundamental DGP:** 122 fitur *pure within-split* `cu_sp_*` + deviasi kurir bebas drift + hapus polusi `u_cust` + Ensemble 5 model (`LGB_unw`, `LGB_tew`, `HC_GLMM`, `XGBoost`, `PyTorch ResNetV7+SWA`) + Meta-Stacker `TEST-SIM` |

---

## Rincian Komponen di Dalam `v7` (5-Fold `GroupKFold(customer_id)` — 100% Bebas Kebocoran)

| Komponen Model `v7` | Keluarga | OOF LogLoss | `TEST-SIM LogLoss` | `GroupKFold F1` | **`TEST-SIM Macro F1`** |
|---|---|---|---|---|---|
| `v7_TabularResNetV7` *(320×3 SiLU + Top-3 SWA)* | PyTorch NN | `0.78869` | `0.79935` | `0.62267` | `0.62251` |
| `v7_LGB_tew` *(sqrt test-regime weighted)* | LightGBM | `0.78818` | `0.80036` | `0.62435` | `0.62234` |
| `v7_LGB_unw` *(unweighted)* | LightGBM | `0.78505` | `0.79585` | `0.62582` | `0.62499` |
| `v7_XGBoost` *(hist depth=7)* | XGBoost | `0.78350` | `0.79438` | `0.62663` | `0.62559` |
| `v7_LGB + HC_GLMM` *(Hub + Hub×Peak + Courier)* | GBDT + Bayes | `0.78296` | `0.79478` | `0.62689` | `0.62567` |
| **`v7 FINAL (L2 Meta-Stacker + Multipliers)`** | **Super-Ensemble** | **`0.77551`** | **`0.78652`** | **`0.62949`** | **`0.62866`** |

---

## Struktur Folder Per-Versi (`ghani/`)

```text
ghani/
├── src/
│   ├── common/     # glmm_lib.py, verify_submission.py, verify_shift.py, estimate_lb.py
│   ├── v1/         # baseline_lgbm.py
│   ├── v2/         # features_v2.py, train_v2.py
│   ├── v3/         # run_v3.py
│   ├── v4/         # run_v4.py
│   ├── v5/         # build_features_v5.py, run_stage1_glmm.py
│   ├── v6/         # build_stage2_glmm.py, stage2_common.py, train_stage2_*.py, ensemble_final.py, infer_v6.py
│   └── v7/         # build_features_v7.py, train_v7_{lgb,xgb,nn}.py, ensemble_v7.py, infer_v7.py, build_final_notebook_v7.py
├── models/
│   ├── v1/ .. v4/  # lgbm_v1..4.txt + meta_v1..4.json
│   ├── v5/         # stage1/lgb_f0..4.txt + meta_v5_glmm.json
│   ├── v6/         # glmm_state_full.npy, lgb_res/, lgb_full/, xgb/, nn/, nn2/, meta_stacker.joblib
│   └── v7/         # feature_spec_v7.json, lgb_unw/, lgb_tew/, glmm_hc_full.npy, xgb/, nn/, meta_stacker_v7.joblib, meta_v7_ensemble.json
├── cache/
│   └── v1/ .. v7/  # Cache fitur & OOF logits terpisah rapi per versi
├── notebooks/
│   ├── v6/         # datavidia_task1_v6_ghani.ipynb
│   └── v7/         # final_datavidia_task1_v7_ghani.ipynb (Notebook deliverable utama — zero-retraining verified!)
└── results/
    └── LOG.md      # Dokumen ini
```
