# 📂 Workspace Farhat — Datavidia 11.0

Folder kerja pribadi **Farhat** untuk kompetisi Arkavidia 9.0 Datavidia.
Tim: **Farhat**, Pitri Wulandari, Ghani

---

## Struktur

```
farhat/
├── TASK-1/        # Prediksi status penanganan shipment (7 kelas, tabular)
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   ├── notebooks/
│   └── models/
│
└── TASK-2/        # Klasifikasi fenomena citra satelit (6 kelas, vision)
    ├── README.md
    ├── requirements.txt
    ├── v1/        # baseline DINOv2 CLS + k-means + LR/kNN (LB 0.14510)
    │   ├── models/
    │   ├── notebooks/
    │   ├── src/
    │   └── requirements.txt
    ├── v2/        # graph clustering (Leiden) + change stats
    │   ├── models/
    │   ├── notebooks/
    │   ├── src/
    │   └── requirements.txt
    ├── v3/        # final submission: Patch-Chamfer @336px + Hurricane anchor + ensemble
        ├── models/
        ├── notebooks/
        ├── src/
        └── requirements.txt

---

**Kompetisi:** Arkavidia — Datavidia 11.0  
**Organisasi:** Himpunan Mahasiswa Informatika ITB
