# 🌍 TASK-2 DATAVIDIA 11.0 — World Phenomena

Dokumentasi resmi dan catatan teknis internal **Task-2 Datavidia 11.0**.

---

## 🔗 Informasi Resmi Kompetisi (dari Kaggle)
- **Tautan Kaggle:** [https://www.kaggle.com/competitions/task-2-datavidia-11-0](https://www.kaggle.com/competitions/task-2-datavidia-11-0)
- **Penyelenggara:** Datavidia (HMIF ITB / Arkavidia)
- **Tipe:** Community Prediction Competition (Private)
- **Batas Waktu:** Dibuka ~25 Nov 2026 malam, durasi 3 hari.
- **Batas Submisi Harian:** Maksimal **3 submission / hari / tim**.
- **Pembagian Leaderboard:** Public LB ~30% sampel acak, Private LB ~70% sampel sisanya.

---

## 📖 Deskripsi Masalah & Narasi Kasus (Verbatim dari Kaggle Overview)

> *Citra udara dan satelit merekam permukaan bumi dari waktu ke waktu. Ketika sebuah fenomena terjadi, citra sesudah kejadian dapat memperlihatkan perubahan yang khas: tutupan vegetasi mengering, bekas terbakar menghitam, garis pantai mundur beberapa meter, atau hamparan material vulkanik perlahan menelan permukiman. Bagi mata yang terlatih, perbedaan itulah yang membisikkan jenis bencana yang baru saja lewat.*
>
> *Philipp, Technical Director dari Introspeksi Fenomena Institut Tanpa Bencana (IF ITB), tertidur di kursi kantornya. Sejak pagi, meja kerjanya bertumpukan dengan berkas-berkas laporan dari berbagai belahan dunia. Tim dokumentasi drone telah berhasil mengembalikan citra dari berbagai wilayah terdampak bencana ke meja Philipp. Setiap berkas berisi sepasang gambar: satu diambil sebelum kejadian, dan satunya diambil lagi setelah kejadian.*
>
> *Pada musim ketika bencana datang beruntun, tumpukan pasangan citra tumbuh setinggi layar monitor, sementara regu penyelamat, konvoi bantuan, dan petugas evakuasi menunggu arahan. Membuka dua gambar satu per satu untuk membandingkan perubahan yang terlihat menjadi pekerjaan yang tak pernah selesai. Namun, sebuah ide muncul ketika Philipp sedang doomscrolling: bagaimana jika sebuah model dapat membantu mengidentifikasi jenis bencana berdasarkan pasangan citra sebelum dan sesudah kejadian? Model tersebut dapat berfungsi sebagai penyaring awal untuk menentukan laporan mana yang perlu ditinjau lebih dahulu, sehingga tim analis memiliki lebih banyak waktu untuk menentukan langkah penanganan yang tepat.*
>
> *Namun, mewujudkan ide tersebut tidaklah sederhana. Pasangan citra yang masuk diambil dari ketinggian dan sudut yang berbeda, kadang tertutup awan, dan pencahayaannya bergeser antarwaktu. Beberapa fenomena meninggalkan pola yang mirip satu sama lain, seolah sengaja mengaburkan batas antara gempa, ledakan, dan badai. Model perlu tetap berguna pada citra-citra yang tidak sempurna itu, cukup andal untuk membantu analisis, cukup jujur untuk mengakui ketidakpastian. Keputusan akhir tetap di tangan analis; mesin hanya membantu mereka sampai di depan gambar yang tepat.*

---

## 🎯 6 Kelas Target & Karakteristik (Tabel Resmi)

Setiap pasangan citra termasuk ke dalam tepat satu dari enam jenis fenomena:

| Kelas (`Phenomenon`) | Deskripsi Resmi Panitia | Karakteristik Visual Kunci |
|---|---|---|
| **`Earthquake`** | Guncangan akibat pelepasan energi di kerak bumi; sering meninggalkan pola kerusakan yang membentang luas. | Kerusakan/keruntuhan struktur atap & dinding bangunan, puing (*rubble*), vegetasi/warna tanah relatif tidak hangus/terendam. *(Catatan panitia: tebak semua Earthquake F1 ~0.06)* |
| **`Fire`** | Kebakaran pada vegetasi, lahan, atau bangunan; meninggalkan area terbakar yang gelap dan gersang. | Jejak arang hitam pekat (*burn scar*), penurunan drastis indeks kehijauan (*Excess Green / vegetation drop*). |
| **`Hurricane`** | Badai tropis berputar dengan angin kencang; merontokkan vegetasi dan merusak struktur secara tersebar. | Defoliasi meluas (pohon meranggas/kuning), atap rumah terkelupas tersebar, genangan air sporadis tanpa jelaga api/abu vulkanik. |
| **`Tsunami`** | Gelombang laut besar yang menggenangi daratan; mengubah garis pantai dan menyeret material. | Inundasi air laut ke daratan, garis pantai bergeser/tererosi, kenaikan indeks air (*NDWI / blue-green water shift*). |
| **`Volcano`** | Aktivitas letusan gunung api; material vulkanik menutupi permukiman dan lahan sekitar. | Hamparan abu vulkanik abu-abu/putih seragam (*desaturation* luas), aliran lava/piroklastik. |
| **`Explosion`** | Ledakan besar yang merusak area di sekitar titik kejadian secara radiatif. | Kerusakan terpusat radial di episentrum (*high peak-to-mean change*, radius sebaran perubahan sempit konsentris). |

---

## 📐 Metrik Evaluasi & Format Pengumpulan

### 1. Metrik Evaluasi: Macro F1-Score
Submission dinilai dengan **Macro F1-score**, yaitu rata-rata F1-score dari enam kelas dengan bobot yang sama untuk tiap kelas:

$$precision_k = \frac{TP_k}{TP_k + FP_k}, \qquad recall_k = \frac{TP_k}{TP_k + FN_k}$$

$$F1_k = 2 \cdot \frac{precision_k \cdot recall_k}{precision_k + recall_k}, \qquad Macro\ F1 = \frac{1}{6} \sum_{k=1}^6 F1_k$$

> **Petunjuk Penting dari Panitia:**
> *"Karena tiap kelas berbobot sama, pemenang ditentukan oleh kemampuan mengenali kelas yang jarang, bukan sekadar menebak kelas mayoritas. Sebagai gambaran, sample_submission.csv yang seluruhnya berisi Earthquake hanya memperoleh skor sekitar 0,06. Nilai yang lebih tinggi lebih baik."*

### 2. Format File Submission (`sample_submission.csv`)
- Memuat seluruh **3.180 baris** tepat.
- Header: `Id,Phenomenon`
- Format contoh:
  ```csv
  Id,Phenomenon
  000c166d1af6,Earthquake
  0047c432f3c2,Fire
  006b070b2644,Hurricane
  ```
- Nilai `Phenomenon` wajib diawali huruf kapital persis seperti nama kelas.
- Setiap `Id` wajib muncul tepat satu kali tanpa duplikasi.

---

## 📦 Deskripsi Dataset (Verbatim dari Kaggle Data)

- **Total Sampel:** 3.180 pasang citra udara (`6.361` file total, ukuran `1,69 GB`, lisensi `CC BY-NC-SA 4.0`).
- **Status Data:** Seluruh sampel berstatus uji (tidak ada `train.csv` berlabel).
- **Struktur Folder:**
  ```text
  Task-2/
  ├── Fenomena/
  │   ├── 000c166d1af6/
  │   │   ├── pre.jpg    # citra sebelum kejadian (1024x1024 RGB)
  │   │   └── post.jpg   # citra sesudah kejadian (1024x1024 RGB)
  │   ├── 0047c432f3c2/
  │   │   ├── pre.jpg
  │   │   └── post.jpg
  │   └── ... (3.180 folder)
  └── sample_submission.csv
  ```
- **Keterangan Id:** Setiap folder diidentifikasi oleh string acak 12 karakter. *Id tidak membawa urutan, wilayah, waktu, atau makna tersembunyi lainnya. Semua informasi yang dibutuhkan ada di piksel citra.*

---

## ⚖️ Penjelasan Aturan Kompetisi vs Catatan Internal Tim

### A. Aturan Resmi Kompetisi (dari Rules & Guidebook Datavidia):
1. **Larangan Manual / Hand-Labeling:** Prediksi wajib dihasilkan oleh kode/model otomatis. Dilarang melabeli manual sampel uji.
2. **Larangan Probing Leaderboard:** Dilarang melakukan probing sistematis untuk merekonstruksi ground truth.
3. **Larangan LLM / Generative AI:** Prediksi tidak boleh dihasilkan oleh prompt generative AI / LLM teks.
4. **Deliverable Akhir:** Wajib mengumpulkan notebook yang reproducible dan bisa menjalankan inferensi tanpa train ulang (*zero-retraining inference*).

### B. Catatan & Strategi Teknis Internal Tim (Kenapa kita pakai DINOv2?):
- **Pertanyaan:** *Apakah panitia di web Kaggle menyebut nama "DINOv2 (ViT-S/14)"?*
  - **Jawab:** **Tidak.** Teks panitia hanya sampai pada deskripsi masalah dan data di atas.
  - Nama **DINOv2 (ViT-S/14)** adalah **pilihan arsitektur machine learning tim kita**, yang diturunkan agar 100% patuh aturan lomba:
    1. DINOv2 adalah model **Vision murni (*Pure-Vision Self-Supervised*)**, bukan LLM, bukan VLM generatif, dan tidak memakai prompt teks (seperti CLIP).
    2. Model ini menerima input piksel gambar mentah (`pre` dan `post`), mengekstrak representasi semantik visual 384-dimensi, lalu dikelompokkan dengan *domain-physical signatures* (arang, abu, air, radius ledakan) yang sudah kita rancang.
