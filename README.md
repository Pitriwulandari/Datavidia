# 📊 Datavidia - Arkavidia | Data Science Competition

Repository ini berisi kode sumber, analisis data, eksperimen pemodelan *Machine Learning*, dan dokumentasi tim untuk kompetisi **Datavidia** pada rangkaian acara **Arkavidia** yang diselenggarakan oleh Himpunan Mahasiswa Informatika (HMIF) Institut Teknologi Bandung (ITB).

---

## 👥 Profil Tim

| No | Nama | Peran / Fokus | GitHub / Kontak |
|---|---|---|---|
| 1 | **Pitri Wulandari** | Data Preprocessing & Modeling | [@Pitriwulandari](https://github.com/Pitriwulandari) |
| 2 | **Farhat** | Exploratory Data Analysis (EDA) & Feature Engineering | - |
| 3 | **Ghani** | Model Evaluation, Hyperparameter Tuning & Pipeline | - |

*(Catatan: Anda dapat menyesuaikan peran spesifik masing-masing anggota di atas sesuai pembagian tugas tim.)*

---

## 📌 Tentang Kompetisi & Task-1

**Datavidia** merupakan salah satu cabang kompetisi analitik data dan *data science* nasional dalam rangkaian acara **Arkavidia**.

### 🎯 Objektif Task-1 (Status Penanganan Paket)
- **Tantangan:** Memprediksi status akhir penanganan paket pengiriman (**lancar**, **telat**, **rusak**, atau **retur**) sedini mungkin dari data awal.
- **Tujuan Bisnis:** Mengidentifikasi potensi anomali/kegagalan pengiriman secara preventif sebelum kendala terjadi di lapangan.
- **Tipe Tugas:** *Multi-class Classification*.
- **Detail & Tautan:** Lihat [TASK-1.md](docs/TASK-1.md).

---

## 📂 Struktur Direktori

```text
datavidia/
├── docs/                           # Dokumentasi & catatan kompetisi
│   ├── TASK-1.md                   # Pengumuman, link Kaggle, kontak panitia
│   └── INFO-KOMPETISI.md           # Overview, data, leaderboard, aturan lengkap
├── pitri/                          # Workspace pribadi Pitri
│   ├── notebooks/
│   └── results/
├── farhat/                         # Workspace pribadi Farhat
│   └── notebooks/
├── ghani/                          # Workspace pribadi Ghani (terorganisasi rapi per versi v1..v7)
│   ├── src/                        # common/ + v1/ s/d v7/ (kode fitur, training, ensemble, inferensi)
│   ├── models/                     # v1/ s/d v7/ (file bobot model & metadata zero-retraining)
│   ├── cache/                      # v1/ s/d v7/ (cache fitur & OOF logits - di-ignore git)
│   ├── notebooks/                  # v6/ & v7/final_datavidia_task1_v7_ghani.ipynb
│   └── results/LOG.md              # Riwayat lengkap eksperimen v1 s/d v7 & skor
├── submissions/                    # submission_v1_ghani.csv s/d submission_v7_ghani.csv (di-ignore git)
├── data/archive/                   # Zip asli dari Kaggle (di-ignore git)
├── Task-1-Datavidia-11.0/          # Data hasil ekstrak (train/, test/) - di-ignore git
├── requirements.txt
├── .gitignore
└── README.md
```

> **Catatan:** `Task-1-Datavidia-11.0/` masih di root karena path-nya dipakai skrip di `ghani/src/`. Setelah pipeline stabil, bisa dipindah ke `data/raw/` dengan mengubah `ROOT` di `ghani/src/baseline_lgbm.py`.

**Penamaan submission:** `submissions/submission_v{n}_{nama}.csv` (contoh: `submission_v1_ghani.csv`). Nomor n = urutan eksperimen orang tersebut, dicatat di `{nama}/results/LOG.md`.

**Aturan kerja:** eksperimen pribadi di folder masing-masing; notebook final untuk dikumpulkan ke panitia disiapkan dari folder anggota yang skornya terbaik. Dataset, model, dan submission tidak di-commit.

---

## 🛠️ Tech Stack & Alat

- **Bahasa Pemrograman:** Python 3.9+
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualisasi Data:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Eksperimen & Notebook:** Jupyter Notebook / Google Colab / VS Code

---

## 🚀 Alur Kerja & Metodologi

```mermaid
flowchart LR
    A["Raw Data"] --> B["EDA & Cleansing"]
    B --> C["Feature Engineering"]
    C --> D["Stratified / K-Fold CV"]
    D --> E["Model Training & Tuning"]
    E --> F["Ensemble / Stacking"]
    F --> G["Final Submission"]
```

1. **Exploratory Data Analysis (EDA):** Mengidentifikasi distribusi target, *missing values*, *outliers*, dan interaksi antar variabel.
2. **Data Preprocessing:** Imputasi nilai kosong, *encoding* data kategorikal, normalisasi/standardisasi fitur numerik, serta penanganan *imbalanced data*.
3. **Feature Engineering:** Pembentukan fitur agregasi, interaksi fitur, ekstraksi tanggal/teks (jika relevan), dan seleksi fitur penting.
4. **Validation Strategy:** Menggunakan *K-Fold Cross Validation* (misal: 5-Fold Stratified CV) untuk mencegah *data leakage* dan *overfitting*.
5. **Model Exploration & Blending:** Eksperimen algoritma berbasis *tree-based models* (LightGBM, XGBoost, CatBoost) serta teknik *blending/weighted ensemble* untuk memaksimalkan performa pada metrik evaluasi kompetisi.

---

## 💻 Panduan Instalasi & Penggunaan

### 1. Kloning Repository
```bash
git clone https://github.com/Pitriwulandari/Datavidia.git
cd Datavidia
```

### 2. Buat Lingkungan Virtual (Virtual Environment)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 4. Menjalankan Notebook
```bash
jupyter notebook ghani/notebooks/
```

---

## 📈 Eksperimen & Riwayat Skor

| Model ID | Pendekatan / Fitur | CV Macro F1 | Leaderboard Public | Leaderboard Private | Keterangan |
|---|---|---|---|---|---|
| `v1` | Baseline LightGBM + agregasi dasar 9 tabel | `0.4941` | - | - | Benchmark awal |
| `v2` | LightGBM + fitur sekuensial scan | `0.5018` | - | - | Eksplorasi urutan event |
| `v3` | LightGBM + OOF Target Encoding (sub 450rb) | `0.5732` | - | - | Lompatan sinyal entitas |
| `v4` | LightGBM + OOF Target Encoding (full 1,05jt) | `0.5785` | - | - | Skala penuh |
| `v5` | Feature Eng v5 + Stage-1 LGB + Joint GLMM | `0.61911` | - | - | 5-Fold Stratified CV |
| `v6` | Stage-2 Super-Ensemble 6 model + Joint GLMM | `0.62561` | `0.60163` (#18) | - | Audit DGP menemukan gap `u_cust` |
| **`v7`** | **Pure Within-Split (`cu_sp_*`) + 5-Fold `GroupKFold` Super-Ensemble** | **`0.62949`** *(sim `0.62866`)* | *(siap submit)* | - | **Kandidat utama #1 (bebas leakage)** |

---

## 📄 Lisensi & Ucapan Terima Kasih

Proyek ini dikembangkan khusus untuk perlombaan **Datavidia - Arkavidia**.
Terima kasih kepada panitia penyelenggara **HMIF ITB** dan seluruh pihak yang mendukung tim kami.
