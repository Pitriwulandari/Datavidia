# 🧪 Log Eksperimen Task-2 (World Phenomena) — Ghani

- **Kompetisi:** [Task-2 Datavidia 11.0 — World Phenomena](https://www.kaggle.com/competitions/task-2-datavidia-11-0)
- **Tugas:** Klasifikasi tanpa label latih (*Zero-Label Bi-Temporal Change Classification*) pada **3.180 pasang citra udara/satelit** (`pre.jpg` & `post.jpg`) ke dalam 6 kelas: `Earthquake`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`, `Explosion`.
- **Metrik Evaluasi:** **Macro F1-Score** (6 kelas berbobot setara).
- **Kepatuhan Aturan:** Murni menggunakan *Pure-Vision Models* tanpa prompt teks/titik/kotak (tanpa VLM/LLM/CLIP/SAM/AutoML/data berlabel eksternal/hand-labeling), lengkap dengan *zero-retraining inference*.

## Ringkasan Progresi Versi (`v1` → ...)

| Versi | Folder Kode | Folder Model | File Submission | Distribusi Prediksi | LB Public | Ringkasan Pendekatan |
|---|---|---|---|---|---|---|
| **v1** | [src/v1/](../src/v1/) | [cache/v1/](../cache/v1/) | `submissions/task2/submission_v1_ghani.csv` | EQ 20.3%, Fire 17.7%, Volc 17.8%, Expl 17.6%, Tsun 15.9%, Hurr 10.7% | **0.11158** | **Pure-Vision Bi-Temporal Physical Spectral Change Signatures:** Ekstraksi fitur perubahan spektral (ExG drop, Burn scar charcoal index, Ash desaturation, NDWI water inundation, Radial blast concentration, Edge destruction). |
| **v2** | [src/v2/](../src/v2/) | [cache/v2/](../cache/v2/) | `submissions/task2/submission_v2_ghani.csv` | EQ 20.4%, Fire 23.6%, Volc 15.4%, Expl 15.8%, Tsun 13.7%, Hurr 11.0% | *(siap disubmit)* | **DINOv2 (ViT-S/14) Foundation Model + Calibrated Physical Fusion:** Menggabungkan representasi deep visual semantic embedding DINOv2 dengan anchor prototype fisik via dual multi-modal decision boundary. 859 sampel ambigu disempurnakan. |
