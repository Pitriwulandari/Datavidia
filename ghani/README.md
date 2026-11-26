# 📂 Workspace - Ghani (`task1/` & `task2/`)

Folder kerja pribadi **Ghani** yang dipisahkan secara simetris dan rapi per babak/tugas kompetisi (**Task-1** & **Task-2**) serta per versi eksperimen (`v1`, `v2`, ...).

```text
ghani/
├── task1/                        # 📦 Task-1: Shipment Logistic Dataset (7-class Macro F1 — LB Public: 0.60757 #16/213)
│   ├── src/                      # common/ + v1/ s/d v8/ (skrip fitur, training, ensemble, zero-retrain inference)
│   ├── models/                   # v1/ s/d v8/ (bobot model, tabel lookup entitas & metadata zero-retraining)
│   ├── cache/                    # v1/ s/d v8/ (cache fitur & OOF logits - di-ignore git)
│   ├── notebooks/                # v6/, v7/, & v8/final_datavidia_task1_v8_ghani.ipynb
│   └── results/LOG.md            # Catatan lengkap eksperimen v1 s/d v8 & skor
│
├── task2/                        # 🌍 Task-2: World Phenomena (3.180 pasang citra pre/post, 6-class Macro F1)
│   ├── src/                      # common/ + v1/ .. (skrip ekstraksi fitur visi murni, klasterisasi, self-training, inferensi)
│   ├── models/                   # v1/ .. (checkpoint model & artefak zero-retraining)
│   ├── cache/                    # v1/ .. (cache embedding citra & fitur perubahan - di-ignore git)
│   ├── notebooks/                # v1/ .. (notebook deliverable siap jalankan tanpa training ulang)
│   └── results/LOG.md            # Catatan lengkap eksperimen Task-2 per versi
│
└── README.md                     # Dokumen ini
```

## File Submission Bersama (`submissions/`)
- **Task-1:** `submissions/task1/submission_v1_ghani.csv` s/d `submissions/task1/submission_v8_ghani.csv`
- **Task-2:** `submissions/task2/submission_v{n}_ghani.csv`

## Cara Menjalankan Inferensi Tanpa Melatih Ulang (*Zero-Retraining*, dari Root Repo)
- **Task-1 (`v8` Final):**
  ```bash
  python -u -W ignore ghani/task1/src/v8/infer_v8.py
  ```
- **Task-2 (unduh/ekstrak data awal):**
  ```bash
  python ghani/task2/src/common/download_task2.py
  ```
