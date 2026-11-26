# Datavidia 11.0 Task-2 — Catatan Pengembangan

## 1. Apa yang diminta

Berdasarkan `data/raw/Overview.txt`, `Dataset Description.txt`, `Competition Rules.txt`,
`FAQ Datavidia 11.0.xlsx`, dan `rumus.png`:

- 3.180 pasangan citra udara `pre.jpg` / `post.jpg`, satu Id per folder di
  `data/raw/Fenomena`; tiap pasangan tepat satu kelas.
- 6 kelas: `Earthquake`, `Fire`, `Hurricane`, `Tsunami`, `Volcano`, `Explosion`.
- Metrik: **Macro-F1** = `(1/6) Σ F1_k`, dengan
  `precision_k = TP_k/(TP_k+FP_k)`, `recall_k = TP_k/(TP_k+FN_k)`,
  `F1_k = 2·precision_k·recall_k/(precision_k+recall_k)`.
  Baseline semua-`Earthquake` ≈ 0,06, jadi prediksi seimbang penting.
- Format: `Id,Phenomenon`, 3.180 baris, Id unik, label persis sesuai ejaan.
- Aturan: dilarang LLM / AutoML / generative AI / VLM sebagai pembuat prediksi,
  dilarang data eksternal berlabel ditambahkan ke data latih; bobot pretrained
  publik diperkenankan jika sumber, versi, dan modifikasi dicantumkan; wajib
  mengembangkan dan menjalankan proses pelatihan.

Karena tidak ada label latih (hanya placeholder `Earthquake` di
`sample_submission.csv`), pipeline memakai **pseudo-label unsupervised** dan
**probe supervisi-lemah**. Tidak ada ground truth yang dipakai dalam pelatihan
maupun evaluasi lokal; angka CV di bawah mengukur konsistensi terhadap
pseudo-label, **bukan** skor kompetisi.

## 2. Struktur folder yang dipakai

```
data/raw/          dokumentasi, sample_submission.csv, Fenomena/<Id>/{pre,post}.jpg
data/processed/    manifest.csv, change_stats.csv
src/               config, utils, prepare, embed, features, cluster, train,
                   predict, run_pipeline, make_notebook, validate
cache/thumbs/      thumbnail RGB 256px (6.360 berkas)
cache/features/    embedding, matriks klaster, OOF probabilitas
cache/figures/     montase prototipe dan atlas klaster
models/            feature_state, cluster_mapping, probe, cv_report
submissions/       submission_final.csv
notebooks/         Datavidia_Task2_Notebook.ipynb
```

## 3. Pipeline

1. **Inventarisasi (`src/prepare.py`)** — memverifikasi 3.180 folder sama persis
   dengan Id di `sample_submission.csv`, menyimpan `manifest.csv`, membuat cache
   thumbnail 256px, dan menghitung 31 statistik perubahan piksel (RGB/HSV/gray,
   persentil, tekstur, edge/Canny).
2. **Embedding (`src/embed.py`)** — encoder **pure-vision beku** dari `timm`,
   tanpa prompt teks:
   - `vit_small_patch14_dinov2.lvd142m` (DINOv2 ViT-S/14, self-supervised,
     21,6 M parameter), input 196px;
   - `mobilenetv3_large_100.ra_in1k` (ImageNet-1k) sebagai alternatif.
   Sumber: `timm` model zoo / Hugging Face Hub, versi sesuai nama weight di atas.
   Tidak ada fine-tuning encoder, tidak ada data tambahan.
3. **Fitur + klaster (`src/features.py`)** — representasi mentah ternormalisasi
   L2: embedding `post` (bobot 1,0) + statistik perubahan terstandardisasi
   (bobot 0,25; opsi `pre`/`delta`). KMeans `n_init=20` pada gabungan fitur,
   konsensus 12 run pada subset blok. `k=6` berasal dari aturan kompetisi.
4. **Diagnosis (`src/cluster.py`)** — profil statistik dan montase 16 anggota
   paling representatif per klaster, dipakai untuk menetapkan pemetaan
   klaster→kelas secara transparan (`models/cluster_mapping.json`).
5. **Probe (`src/train.py`)** — logistic regression + kNN cosine dilatih pada
   pseudo-label klaster, dievaluasi 5-fold terhadap pseudo-label yang sama.
6. **Prediksi (`src/predict.py`)** — rata-rata bobot `0,7·LR + 0,3·kNN`, lalu
   Sinkhorn untuk menyebar prediksi ke enam kelas (mendukung Macro-F1).
7. **Notebook (`notebooks/Datavidia_Task2_Notebook.ipynb`)** — inference-only,
   memakai encoder beku + state + probe yang sudah disimpan; tanpa label/GT dan
   tanpa pelatihan ulang.

## 4. Diagnostik yang mempengaruhi desain

Pada subset 400 pasangan, PCA-whitening pada embedding DINOv2 menghasilkan
stabilitas ARI ≈ 0,0 antar-seed (klaster acak), sedangkan embedding mentah
ternormalisasi L2 menghasilkan ARI ≈ 0,73. Karena itu PCA dibuang. Konfigurasi
yang dipakai (`post + stats`) memberi ARI ≈ 0,66 dan silhouette cosine ≈ 0,15
pada subset tersebut. Angka final untuk 3.180 pasangan dicatat di bawah.

Montase subset (`cache/figures/cluster_*_prototypes.jpg`) memberi bukti visual
yang berbeda per klaster: bekas bakar hitam, genangan luas, material vulkanik
abu-abu, keruntuhan bangunan, kawah ledakan, dan longsor. Interpretasi ini
dilakukan **di luar model prediksi**; model akhir tidak memakai LLM/VLM.

## 5. Hasil final

*(diisi setelah pipeline penuh selesai)*

## 6. Keterbatasan

- Tidak ada label ground truth; pemetaan klaster→kelas bersifat interpretatif
  dan tidak dapat diverifikasi skornya secara lokal.
- `pre`/`post` tidak sepenuhnya sejajar (korelasi piksel rendah, pergeseran
  hingga >12 px pada thumbnail), sehingga statistik perubahan bersifat noise
  tapi tetap berguna sebagai pembeda.
- Encoder berjalan di CPU 2 core; resolusi dibatasi 196px untuk menyesuaikan
  waktu komputasi, sehingga detail halus (mis. retakan kecil) mungkin hilang.
