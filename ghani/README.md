# 📂 Workspace - Ghani

Folder kerja pribadi **Ghani**. Semua kode, model, cache, dan catatan hasil buatan Ghani/agen ada di sini supaya root tetap bersih dan jelas pembuatnya.

```text
ghani/
├── src/        # Skrip pipeline (jalankan dari ROOT repo, bukan dari dalam ghani/)
│   ├── baseline_lgbm.py   # v1: agregasi 9 tabel + LightGBM
│   ├── features_v2.py     # fitur urutan scan -> cache fitur
│   └── train_v2.py        # target encoding + LightGBM + tuning bobot kelas
├── models/     # lgbm_v*.txt, meta_v*.json (di-ignore git)
├── cache/      # features_*.pkl, probabilitas, importance (di-ignore git)
├── notebooks/  # Notebook eksperimen
└── results/    # LOG.md (riwayat eksperimen), screenshot, catatan
```

## Cara menjalankan (dari root repo)
```bash
python ghani/src/features_v2.py train
python ghani/src/features_v2.py test
TAG=v5 python ghani/src/train_v2.py        # PowerShell: $env:TAG='v5'; python ghani/src/train_v2.py
```

## Konvensi penamaan
- **Submission** (folder bersama `submissions/`): `submission_v{n}_ghani.csv`, n = urutan eksperimen Ghani.
- **Model & cache**: `lgbm_v{n}.txt`, `meta_v{n}.json`, `proba_v{n}_*.npy` memakai nomor `n` yang sama dengan submission.
- Setiap eksperimen baru dicatat di [results/LOG.md](results/LOG.md).

## Aturan
- Jangan commit data, model, cache, atau submission (sudah di-ignore).
- Notebook final untuk panitia (pemrosesan data, fitur, training, inferensi tanpa training ulang) disusun di `notebooks/` folder ini dan menyertakan file model dari `models/`.
