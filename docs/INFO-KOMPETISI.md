# ðŸ“š Info Kompetisi: Task-1 Datavidia 11.0 (Shipment Logistic Dataset)

Catatan terkumpul dari halaman Kaggle (Overview, Data, Leaderboard, Rules).
Link: https://www.kaggle.com/competitions/task-1-datavidia-11-0
Penyelenggara: [Datavidia](https://www.kaggle.com/datavidia) (HMIF ITB, bagian dari Arkavidia). Tipe: Community Prediction Competition (Private).

> Snapshot halaman diambil sekitar 23 Nov 2026 22:30 WIB, tertulis **"2 days to go"** (cek tenggat pasti di Kaggle: Overview â†’ Timeline).

---

## 1. Overview
- Setiap paket meninggalkan jejak: scan pickup, sortasi hub, pindah moda, upaya antar, tagihan, promo, catatan alamat.
- **Skenario:** Rian (kepala QC ekspedisi antarpulau). Status penanganan selama ini diisi manual oleh petugas (telat/lupa/asal pilih), sehingga laporan bulanan tidak bisa dipercaya. Tim klaim tidak tahu paket yang benar-benar hilang, tim operasional tidak tahu hub mana yang paling sering menahan barang.
- **Tujuan:** model yang membaca jejak operasional dan menandai `status_penanganan` tiap pengiriman secara otomatis. Keputusan akhir tetap di petugas; model membantu memprioritaskan pengecekan.
- **Tantangan yang disebut panitia:**
  - 9 tabel relasional; satu pengiriman bisa punya belasan baris scan dan beberapa kaki rute.
  - Kelas penting (rusak, klaim hilang) justru paling jarang (imbalanced).
  - Sebagian catatan tidak lengkap (missing values).
  - Tidak semua kolom berguna (ada noise / fitur tidak relevan).
  - Harus jeli di kelas langka tanpa mengorbankan kelas umum.

## 2. Task & Metrik
- **Tugas:** klasifikasi multi-kelas, 1 label `status_penanganan` per `shipment_id` di `test/shipments.csv`.
- **Metrik: Macro F1-score** (semua kelas berbobot sama, jadi kelas langka sangat menentukan).
- Format submission: `shipment_id,status_penanganan` (lihat `sample_submission.csv`, 362.245 baris).

| Kelas | Deskripsi |
|---|---|
| `normal` | Selesai tanpa insiden |
| `konfirmasi_alamat` | Perlu konfirmasi/perbaikan alamat |
| `tertunda_operasional` | Tertunda (penumpukan hub, telat transit) |
| `gagal_antar_awal` | Gagal diantar di percobaan pertama |
| `retur_pengirim` | Dikembalikan ke pengirim |
| `rusak_perjalanan` | Rusak selama perjalanan |
| `klaim_hilang` | Hilang, berakhir klaim |

## 3. Data
- Total **1.605.862 pengiriman** (lokal: train 1.243.617 + test 362.245). Ukuran 1,47 GB, 19 file CSV.
- **File:**
  - `train/` & `test/`: `shipments.csv`, `scan_events.csv`, `route_segments.csv`, `invoices.csv` (train punya kolom target `status_penanganan` di shipments).
  - Bersama (di folder train dan test): `customers.csv`, `couriers.csv`, `hubs.csv`, `service_packages.csv`, `promotions.csv`.
  - `sample_submission.csv`.
- Lisensi data: MIT.
- Kolom per tabel (dari cek lokal) dan distribusi target: lihat [pitri/](../pitri/) / catatan EDA berikutnya.

**Distribusi target (train):** normal 38,08% Â· konfirmasi_alamat 21,94% Â· tertunda_operasional 14,92% Â· gagal_antar_awal 10,00% Â· retur_pengirim 7,06% Â· rusak_perjalanan 4,98% Â· klaim_hilang 3,03%.

## 4. Leaderboard (snapshot)
- Public LB memakai ~30% test; private memakai 70% sisanya (**bisa berubah drastis, hindari overfit ke public**).
- 85 tim, 147 submission, 237 peserta (saat snapshot).
- Skor teratas: #1 **0.60247**, #2 0.60126, #3 0.59895, #10 0.59783, #20 0.59480, #30 0.59252, #40 0.58699, #49 0.57671.
- Selisih top-30 sangat tipis (~0.01), jadi sinyal sangat ber-noise. Skor "60-an" dari Pitri = ~0.60 macro F1 (setara papan atas).
- Baseline lokal Pitri (sampel 40.000): accuracy 0.489, macro F1 0.316, `retur_pengirim` F1 0.000.

## 5. Aturan Penting (ringkasan)
**Tim & submission**
- 1 tim = **3 peserta**, semua dari **perguruan tinggi yang sama**.
- Maks **3 submission per hari**.
- Pilih submission terbaik untuk evaluasi private; skor private = skor tertinggi dari **dua submission terpilih**.
- Memakai ground truth sebagai submission = diskualifikasi.

**Larangan metode**
- Prediksi **wajib dihasilkan model**; dilarang isi label manual / hardcode.
- **LLM, AutoML, dan AI generatif tidak boleh** dipakai menghasilkan prediksi atau konten keluaran submission.
- Tabular Foundation Model (mis. TabPFN) hanya boleh jika bobot publik dan dijalankan sendiri; cantumkan sumber, versi, modifikasi di notebook.
- Tim harus menjalankan sendiri training dan inferensi.

**Deliverable notebook & model**
- Notebook menjelaskan: pemrosesan data, penggabungan tabel, pembuatan fitur, pengembangan model, training, cara prediksi.
- Notebook harus bisa **inferensi tanpa melatih ulang** â†’ simpan file model (`.pkl`, `.joblib`, `.json`, `.txt`, `.cbm`, `.pt`, dll).
- Output yang ditampilkan harus dihasilkan kode (bukan dimasukkan manual).
- Jelaskan dependensi & langkah load model.

**Data & kode**
- Data kompetisi **tidak boleh dibagikan/dipublikasikan** di luar kebutuhan lomba (jangan push CSV ke GitHub; `.gitignore` sudah mengabaikan `*.csv`/`*.zip`, tapi cek folder `Task-1-Datavidia-11.0/`).
- Dilarang berbagi kode/data secara privat antar tim berbeda. Boleh publik lewat forum Kaggle.
- Hanya 1 akun Kaggle per orang, 1 tim per orang; anggota harus konfirmasi tim di Kaggle.
- Aturan Kaggle Foundational Rules berlaku dan mengalahkan aturan khusus jika bentrok. Kudos only (tidak ada poin/medali).

## 6. Catatan Untuk Tim (Pitri, Farhat, Ghani)
- [ ] Ketiga anggota klik **Join Competition** dan setujui rules, lalu bentuk/konfirmasi tim (3 orang) di Kaggle.
- [ ] Catat tenggat final di Overview â†’ Timeline.
- [ ] Rencana submission: maks 3/hari, sisakan jatah; tandai 2 submission final.
- [ ] Gunakan **CV lokal (stratified)** sebagai acuan utama, jangan mengejar public LB saja.
- [ ] Siapkan notebook akhir + file model + penjelasan dependensi.
- [ ] Jangan pakai LLM/AutoML untuk menghasilkan prediksi (boleh dipakai untuk bantu menulis kode analisis, tapi konfirmasi interpretasi ke CP bila ragu).
- Kontak panitia (CP): Majiid 081360758309, Agatha 081322254056.

---

## 7. Tambahan dari Kaggle Foundational Rules (yang belum tercatat di atas)
Aturan Kaggle ini **mengalahkan** aturan khusus Datavidia jika bentrok.

**Pendaftaran & tim**
- Peserta harus punya akun Kaggle, berusia minimal 18 tahun (atau usia dewasa di wilayahnya), dan bukan dari wilayah/entitas yang kena sanksi (Crimea, DNR/LNR, Kuba, Iran, Korea Utara).
- Hanya boleh submit lewat **1 akun Kaggle unik** per orang. Submit lewat lebih dari satu akun = diskualifikasi.
- Hanya boleh bergabung/membentuk **1 tim**. Tiap anggota daftar **sendiri-sendiri dulu**, lalu **konfirmasi keanggotaan tim** lewat notifikasi di akun Kaggle masing-masing agar resmi.
- Tim boleh merger asalkan: total tidak melebihi ukuran maks tim (3), jumlah submission gabungan tidak melebihi jatah 1 tim pada tanggal merger, dan merger selesai sebelum deadline merger/kompetisi.
- Jika memakai nama kampus/instansi, aturan mengikat individu dan institusinya.

**Submission & kode**
- Dilarang memakai **hand labeling / prediksi manusia** pada data validasi atau test.
- Submission dianggap batal jika tidak lengkap, rusak, diubah, hasil kecurangan, atau terlambat.
- **Dilarang berbagi kode atau data secara privat di luar tim** (termasuk antar tim berbeda). Boleh dibagikan jika **publik** lewat forum/notebook Kaggle kompetisi ini.
- Kode yang dibagikan publik otomatis dianggap berlisensi open source (OSI-approved) yang tidak membatasi penggunaan komersial.
- Library open source yang dipakai harus berlisensi **OSI-approved yang tidak membatasi komersial** (LightGBM, XGBoost, CatBoost, scikit-learn aman).
- Segala bentuk upaya merusak sistem/kompetisi dilarang dan bisa dituntut.

**Penilaian & pemenang**
- Pemenang ditentukan **hanya dari private leaderboard**, selama patuh aturan.
- Seri: submission yang **masuk lebih dulu** menang. Jika pemenang didiskualifikasi, peringkat berikutnya naik.
- Panitia berhak memverifikasi kepatuhan; jika ada pelanggaran bisa diminta perbaikan (remediasi) dalam 1 minggu atau didiskualifikasi.
- Pemenang dinotifikasi lewat **email** (cek email akun Kaggle!). Tidak merespons dalam **1 minggu** = hangus. Dokumen hadiah harus dikirim dalam **2 minggu**.
- Hadiah dibagi **rata** antar anggota tim kecuali tim sepakat membagi lain dan memberi tahu Kaggle sebelum hadiah dikeluarkan. Pajak hadiah ditanggung pemenang.
- Kompetisi ini hanya memberi **Kudos** (tanpa Points/Medali).
- Panitia boleh memakai nama dan kemiripan peserta untuk publikasi; data pribadi (nama, email, dll.) dapat dibagikan Kaggle ke panitia.

**Waktu**
- Timeline bisa berubah dan panitia dapat menambah deadline antara; **pantau halaman Kaggle secara rutin**. Peserta bertanggung jawab menyesuaikan zona waktu (WIB).

## 8. Checklist Tambahan Berdasarkan Aturan
- [ ] Pastikan ketiga anggota satu perguruan tinggi yang sama (syarat tim).
- [ ] Masing-masing klik **I Understand and Accept**, lalu satu orang kirim undangan tim dan dua lainnya konfirmasi.
- [ ] Jangan berbagi kode/data ke tim lain secara privat (grup chat lintas tim termasuk).
- [ ] Hanya pakai library berlisensi open source komersial-friendly; cantumkan versi di notebook.
- [ ] Jika pakai TabPFN: bobot publik, dijalankan sendiri, cantumkan sumber/versi/modifikasi.
- [ ] Cek email akun Kaggle setelah pengumuman (notifikasi pemenang batas 1 minggu).
- [ ] Hitung deadline dalam WIB; jangan menunggu menit terakhir (jatah 3 submission/hari).

