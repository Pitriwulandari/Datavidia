# 🌍 TASK-2 DATAVIDIA 11.0 — World Phenomena

Dokumentasi lengkap **Task-2 Datavidia 11.0** (Overview, Struktur Data, Metrik Evaluasi, dan Aturan Penting).

---

## 🔗 Tautan & Jadwal
- **Kaggle Task-2:** [https://www.kaggle.com/competitions/task-2-datavidia-11-0](https://www.kaggle.com/competitions/task-2-datavidia-11-0)
- **Durasi:** 3 hari (dibuka 25 Nov 2026 malam WIB).
- **Batas Submit Harian:** Maksimal **3 submission / hari / tim** (*melebihi batas dengan alasan apa pun = diskualifikasi*).
- **Pembagian Leaderboard:** Public LB ~30% sampel acak, Private LB ~70% sampel sisanya (penentu utama pemenang).

---

## 🎯 Deskripsi Masalah & Karakteristik Unik Dataset
- **Total Data:** **3.180 pasang citra udara/satelit** (`6.361` file, `1,69 GB`, lisensi `CC BY-NC-SA 4.0`).
- **TANPA DATA LATIH BERLABEL (*Zero-Label / Unsupervised Change Classification*):**
  - **Seluruh 3.180 sampel berstatus uji** (tidak ada `train.csv` berlabel).
  - Setiap sampel `<Id>` (`12` karakter acak, tanpa makna urutan/lokasi/waktu) berisi sepasang citra lokasi & sudut pandang yang sama:
    ```text
    Fenomena/
    ├── 000c166d1af6/
    │   ├── pre.jpg   # citra sebelum kejadian
    │   └── post.jpg  # citra sesudah kejadian
    └── ... (3.180 folder)
    sample_submission.csv
    ```
- **6 Kelas Target (`Phenomenon` — huruf kapital wajib persis):**
  1. `Earthquake` — Guncangan kerak bumi; meninggalkan pola kerusakan bangunan/struktur yang membentang luas. *(Catatan panitia: menebak semua `Earthquake` memberi Macro F1 ~`0.06` $\implies$ proporsi `Earthquake` sekitar ~20–22%, kelas lain tidak seimbang / ada kelas langka!)*
  2. `Fire` — Kebakaran vegetasi, lahan, atau bangunan; meninggalkan area terbakar yang gelap/hitam dan gersang.
  3. `Hurricane` — Badai tropis berputar dengan angin kencang; merontokkan vegetasi dan merusak struktur secara tersebar.
  4. `Tsunami` — Gelombang laut besar yang menggenangi daratan; mengubah garis pantai dan menyeret material.
  5. `Volcano` — Aktivitas letusan gunung api; material vulkanik menutupi permukiman dan lahan sekitar.
  6. `Explosion` — Ledakan besar yang merusak area di sekitar titik kejadian secara radiatif/terpusat.

- **Metrik Evaluasi:** **Macro F1-Score** (rata-rata F1-score 6 kelas dengan bobot setara; kemampuan mengenali kelas langka menjadi penentu utama).
- **Format Submission (`3.180` baris):**
  ```csv
  Id,Phenomenon
  000c166d1af6,Earthquake
  0047c432f3c2,Fire
  ```

---

## ⚠️ Aturan Metode & Model (SANGAT KRUSIAL — WAJIB PATUH 100%!)

### ❌ DILARANG KERAS:
1. **Hand-labeling / manual labeling / hardcoding** prediksi pada sampel uji.
2. **Probing leaderboard** untuk menebak/merekonstruksi kunci jawaban.
3. **LLM, AutoML, AI Generatif, dan Visual Language Model (VLM)** seperti GPT-4o, Gemma, Gemini, LLaVA, Qwen-VL, dll.
4. **Model berbasis prompt teks, titik, kotak, atau input interaktif lainnya** — **TERMASUK `SAM` dan `CLIP` (OpenCLIP/SigLIP) dengan prompt teks/visual!**
5. **Data eksternal berlabel** (*external labeled training data*) tidak boleh ditambahkan ke pelatihan.

### ✅ DIPERBOLEHKAN SECARA EKSPLISIT:
1. **Pure-Vision Image Foundation Models & Pretrained Backbones Tanpa Prompt:**
   - Model visi murni yang hanya menerima piksel citra tanpa prompt teks/titik/kotak (contoh: **DINOv2**, **ConvNeXt**, **Swin**, **ResNet**, **EVA-02 vision**, **Places365 / ImageNet-1k / ImageNet-21k / Remote-Sensing pure-vision models** dengan bobot publik).
   - Sumber, versi, dan modifikasi bobot *pretrained* wajib dicantumkan di notebook.
2. **Self-Supervised Learning, Unsupervised Change Detection, Deep Clustering, & Physical/Spectral Change Features** yang dilatih/dijalankan sendiri pada 3.180 pasang citra `pre.jpg` & `post.jpg`.
3. **Deliverable Wajib:**
   - Notebook yang menjelaskan pemrosesan data, pengembangan model, pelatihan, dan prediksi.
   - **Wajib bisa menjalankan inferensi tanpa melatih ulang model dari awal** (*zero-retraining inference*) menggunakan bobot/checkpoint tersimpan (`.pt`, `.pth`, `.pkl`, `.joblib`, dll.) serta menyertakan output kode asli.

---

## 👥 Checklist Tim Sebelum Mulai Submit Task-2
- [ ] **Ketiga anggota tim (Pitri, Farhat, Ghani) klik `Join Competition` & `I Understand and Accept` di halaman Task-2 Kaggle**, lalu bentuk & konfirmasi tim yang berisi lengkap 3 anggota (tim kurang dari anggota terdaftar = diskualifikasi!).
- [ ] Unduh dataset ke folder lokal `Task-2-Datavidia-11.0/` (atau `data/archive/` — sudah masuk `.gitignore` agar aman tidak ter-push ke GitHub).
