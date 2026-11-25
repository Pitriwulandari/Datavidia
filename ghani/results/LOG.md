# 🧪 Log Eksperimen Lengkap (v1 – v8) — Ghani

Seluruh kode, model, cache, notebook, dan submission telah dirapikan per versi (`v1` s/d `v8`) di setiap folder agar mudah ditelusuri dan direproduksi tanpa *retraining*.

## Ringkasan Progresi Versi (`v1` → `v8`)

| Versi | Folder Kode | Folder Model | File Submission | Skema Validasi | OOF LogLoss | Valid Macro F1 (Tuned) | LB Public | Ringkasan Perubahan Utama |
|---|---|---|---|---|---|---|---|---|
| **v1** | [src/v1/](../src/v1/) | [models/v1/](../models/v1/) | `submission_v1_ghani.csv` | Hold-out 15% (sub 450rb) | - | `0.4941` | `0.47761` | Baseline agregasi scan/rute/invoice + tabel master, LightGBM tunggal (tanpa memori customer) |
| **v2** | [src/v2/](../src/v2/) | [models/v2/](../models/v2/) | `submission_v2_ghani.csv` | Hold-out 15% (sub 450rb) | - | `0.5018` | *(Farhat v2: `0.56426`)* | v1 + fitur sekuensial scan (event awal/akhir, bigram, perubahan hub) |
| **v3** | [src/v3/](../src/v3/) | [models/v3/](../models/v3/) | `submission_v3_ghani.csv` | Hold-out 15% (sub 450rb) | - | `0.5732` | - | v2 + OOF Target Encoding (`customer`, `courier`, `hub`, `package`, `promo`, `rute`) |
| **v4** | [src/v4/](../src/v4/) | [models/v4/](../models/v4/) | `submission_v4_ghani.csv` | Hold-out 15% (full 1,05jt) | - | `0.5785` | - | Sama dengan v3 pada data penuh (`lr=0.08`) |
| **v5** | [src/v5/](../src/v5/) | [models/v5/](../models/v5/) | `submission_v5_ghani.csv` | 5-Fold Stratified (1,24jt) | `0.79543` | `0.61911` | - | Feature Engineering v5 (buang ~130 noise, moda-normalized delay, base vs klaim invoice, profil entitas LOO) + Stage-1 LGB + Joint GLMM |
| **v6** | [src/v6/](../src/v6/) | [models/v6/](../models/v6/) | `submission_v6_ghani.csv` | 5-Fold Stratified (1,24jt) | `0.77786` | `0.62561` | **`0.60163`** | Ensemble 6 model Stage-2 di atas Joint GLMM. Tertahan di `0.60163` akibat `cu_loo_*` mencampur `train+test` (polusi 46,7% baris test + *self-cancellation* `r = -0.1055` di `u_cust`) & pergeseran skala `+67%` (3-fold vs 5-fold) di Stage-2 |
| **v7** | [src/v7/](../src/v7/) | [models/v7/](../models/v7/) | `submission_v7_ghani.csv` | 5-Fold `GroupKFold(customer_id)` | `0.77551` | `0.62949` | `0.48847` | Eksperimen diagnostik tanpa `u_cust` & hanya `cu_sp_*` di dalam `test` (53,3% `n_te = 1`). **Membuktikan 100% bahwa sinyal customer di `test` berasal dari riwayat historis periode `train` (88,58% *seen customers*), bukan sesama `test`!** |
| **v8** *(Final)* | [src/v8/](../src/v8/) | [models/v8/](../models/v8/) | **`submission_v8_ghani.csv`** | **5-Fold Stratified (Zero Test Pollution & 0% Scale Shift)** | **`0.77855`** | **`0.62912`** | *(siap submit!)* | **Solusi Puncak:** 51 fitur *Pure Train-Only Customer Behavioral Profiles* (`cu_tr_*`, 0% polusi `test`, `corr(u_te) > 0` nol *self-cancellation*) + Normalisasi Skala Poisson Eksak (`0.000` shift 3-fold/4-fold/5-fold) + Deviasi Kurir/Hub×Bulan *Within-Split* + Hybrid Super-Ensemble 6 keluarga model (`GLMM`, `LGB_res`, `LGB_full`, `XGBoost`, `TabularResNetV8`, `RankGaussResNetV8`) |

---

## Rincian Komponen di Dalam `v8` (5-Fold Stratified CV — 0% Polusi Test & 0,000 Scale Shift)

| Komponen Model `v8` | Keluarga | OOF LogLoss | `argmax_F1` | **`CV_tuned Macro F1`** |
|---|---|---|---|---|
| `v8_Stage1_LGB` *(Pure Train `cu_tr_*`, tanpa ID target)* | LightGBM Stage-1 | `0.81996` | `0.59607` | `0.60694` |
| `v8_GLMM` *(Scale-Normalized Joint Hierarchical GLMM)* | Empirical Bayes | `0.79232` | `0.61326` | `0.62294` |
| `v8_LGB_full` *(Full-Tree GBDT `num_leaves=63`)* | LightGBM | `0.78307` | `0.61901` | `0.62615` |
| `v8_RGResNet` *(RankGauss Dual-Skip Gated ResNet + Top-3 SWA)* | PyTorch NN | `0.78513` | `0.61914` | `0.62683` |
| `v8_XGBoost` *(Depth-6 Hist Residual GBDT)* | XGBoost | `0.78322` | `0.61932` | `0.62688` |
| `v8_LGB_res` *(Residual GBDT di atas `glmm_logit`)* | LightGBM | `0.78147` | `0.61940` | `0.62740` |
| `v8_TabularResNet` *(Signed-Log1p Residual ResNet + Top-3 SWA)* | PyTorch NN | `0.78308` | `0.62064` | **`0.62845`** |
| **`v8 FINAL (Hybrid 50/50 Prob Blend + 5-Fold L2 Meta-Stacker)`** | **Super-Ensemble** | **`0.77855`** | **`0.62039`** | **`0.62912`** |

- **Verifikasi Konsistensi vs `v6` (`0.60163` LB):** Kesepakatan label keseluruhan = **`91,17%`** (`macro_F1(v6, v8) = 0.8781`), dengan distribusi kelas uji yang sangat terkalibrasi (`normal: 37.64%`, `konfirmasi_alamat: 21.60%`, `tertunda_operasional: 15.20%`, `gagal_antar_awal: 10.07%`, `retur_pengirim: 7.12%`, `rusak_perjalanan: 5.26%`, `klaim_hilang: 3.10%`).
- **Verifikasi *Scale Shift* & *Self-Cancellation* pada `test`:** `h_cour_norm` identik (`16.15` inner 3-fold → `16.11` OOF 4-fold → `16.06` test 5-fold); `corr(u_te_konf, scan_cnt_upaya) = +0.0479`, `corr(u_te_ret, has_klaim_inv) = +0.0007` (bias negatif `-0.1055` di `v6` hilang 100%).

---

## Struktur Folder Per-Versi (`ghani/`)

```text
ghani/
├── src/
│   ├── common/     # glmm_lib.py, verify_submission.py, verify_shift.py, estimate_lb.py
│   ├── v1/ .. v5/  # Skrip eksperimen v1 s/d v5
│   ├── v6/         # Pipeline v6 (LB 0.60163) + infer_v6.py
│   ├── v7/         # Pipeline diagnostik GroupKFold v7 + infer_v7.py
│   └── v8/         # v8_common.py, build_features_v8.py, run_stage1_v8.py, build_glmm_v8.py,
│                   # train_stage2_trees_v8.py, train_stage2_nn_v8.py, ensemble_v8.py, infer_v8.py, build_final_notebook_v8.py
├── models/
│   └── v1/ .. v8/  # Bobot model, tabel lookup entitas & metadata zero-retraining per versi
├── cache/
│   └── v1/ .. v8/  # Cache fitur & OOF logits terpisah rapi per versi
├── notebooks/
│   ├── v6/         # datavidia_task1_v6_ghani.ipynb
│   ├── v7/         # final_datavidia_task1_v7_ghani.ipynb
│   └── v8/         # final_datavidia_task1_v8_ghani.ipynb (Notebook deliverable utama — 100% zero-retraining verified!)
└── results/
    └── LOG.md      # Dokumen ini
```
