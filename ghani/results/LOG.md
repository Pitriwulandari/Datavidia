# 🧪 Log Eksperimen - Ghani

Skor validasi = hold-out lokal dari train (bukan leaderboard). Kolom "LB public" diisi setelah submit.

| # | Submission | Model | Ringkasan | Data latih | Valid Macro F1 (argmax) | Valid Macro F1 (tuned) | LB public |
|---|---|---|---|---|---|---|---|
| v1 | `submission_v1_ghani.csv` | `lgbm_v1.txt` | Agregasi scan/rute/invoice + tabel master, LightGBM | subsample 450 rb | 0.4748 | 0.4941 | - |
| v2 | `submission_v2_ghani.csv` | `lgbm_v2.txt` | v1 + fitur urutan scan (first/last, bigram, hub change) | subsample 450 rb | 0.4824 | 0.5018 | - |
| v3 | `submission_v3_ghani.csv` | `lgbm_v3.txt` | v2 + target encoding OOF (customer, courier, hub, package, promo, rute) + count encoding | subsample 450 rb | 0.5590 | 0.5732 | - |
| v4 | `submission_v4_ghani.csv` | `lgbm_v4.txt` | Sama dengan v3, data penuh, lr 0.08 | 1,05 jt (85% train) | 0.5667 | **0.5785** | - |

## Temuan
- Fitur urutan scan hampir tidak berpengaruh (v1 → v2 hanya +0.008).
- **Target encoding per entitas adalah lompatan terbesar** (+0.07). Paling penting: `te_courier_id_*`, `te_customer_id_*`, `cu_address_quality`, `co_hub_id`, `cu_complaint_index`.
- Papan atas public LB (snapshot): 0.60247.

## Ide berikutnya
1. Target encoding kombinasi (hub x kota tujuan, courier x kategori barang) dan statistik numerik per entitas.
2. Latih ulang di seluruh train (tanpa hold-out) dengan iterasi terbaik.
3. Ensemble LightGBM + XGBoost/CatBoost, multi-seed.
4. Tuning bobot kelas dari probabilitas out-of-fold.
5. Fitur kecocokan hub kurir vs hub pengiriman, dan `jarak_domisili_hub_km`.
