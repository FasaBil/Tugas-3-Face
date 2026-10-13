# CLAUDE.md — Tugas 3 Face Recognition (Kelompok 15)

Konteks proyek ini untuk Claude Code. Taruh file ini di root folder `Tugas 3 Face/` (sejajar dengan `common.py`, `train.py`, dll). File ini isinya latar belakang proyek, bukan daftar tugas — instruksi "apa yang harus dikerjakan sekarang" dikasih terpisah lewat prompt tiap sesi.

## Tentang proyek

Tugas mata kuliah Teknologi Berkembang (Computer Vision), week 6, ITS Sistem Informasi. Face recognition pakai OpenCV (Haar Cascade untuk deteksi + LBPH untuk recognition), beda dari materi kelas yang cuma demo single-person: proyek ini harus bisa bedain wajah 2-3 anggota kelompok sekaligus secara real-time.

Anggota kelompok:
- Ary Ratna Aida Safa — 5026241029
- Tazkia Oktaviana Mardhotillah — 5026241078
- Adhima Anfasa Bilqi (Fasa) — 5026241167

## Ketentuan tugas (verbatim dari instruksi dosen)

- Kelompok minimal 2, maksimal 3 orang
- Program harus bisa mendeteksi dan mengenali wajah masing-masing anggota kelompok, serta membedakan wajah satu anggota dengan anggota lainnya
- Setiap anggota harus ditandai dengan warna yang berbeda agar hasil deteksi mudah dibedakan
- Saat proses prediction, program harus menampilkan nama anggota dan nilai accuracy/confidence score
- Output tugas berupa laporan PDF yang menjelaskan metode dan tahapan pengerjaan, serta dilengkapi hasil pengujian program

## Struktur kode

- `common.py` — fungsi bersama: load cascade, deteksi wajah, crop+resize, CLAHE preprocessing, load/save model, hitung jarak LBPH per identitas, penetapan identitas per wajah dalam satu frame
- `capture_dataset.py` — rekam dataset dari webcam (`--name <Nama> --split train|test`)
- `train.py` — train LBPH dari `dataset/train/<Nama>/`, simpan model
- `evaluate.py` — ukur akurasi dari `dataset/test/`, nyari threshold optimal (FAR orang asing ≤5%)
- `recognize.py` — real-time recognition dengan kotak warna per anggota

Alur: `capture_dataset.py` (train) → `capture_dataset.py` (test, sesi beda) → `train.py` → `evaluate.py --save-threshold` → `evaluate.py` → `recognize.py`

## Keputusan yang sudah diambil tim

- Sistem menampilkan confidence score (bukan jarak LBPH mentah) saat prediction, sesuai ketentuan tugas. Belum diimplementasikan ke kode — ini keputusan arah, bukan spesifikasi teknis.

## Status saat ini / gap yang diketahui (per 2026-10-13)

- Dataset train+test untuk ketiga anggota (Aida, Tazkia, Adhima) baru selesai direkam ulang (sebelumnya sempat dihapus karena ada kesalahan di pemisahan train/test).
- `train.py` dan `evaluate.py` belum pernah dijalankan sampai akhir di kondisi dataset yang sekarang — belum ada model dan belum ada angka akurasi real. Jangan asumsikan keduanya udah ada, cek dulu.
- Tabel hasil pengujian di README masih placeholder kosong.
- Tampilan real-time saat ini masih nunjukkin jarak LBPH mentah, bukan confidence score — lihat bagian "Keputusan yang sudah diambil tim" di atas.
- Ada folder duplikat `face_recognition/` yang isinya identik sama file-file di root. Belum diputuskan tim apakah mau dihapus.
- Observasi dari percobaan real-time: anggota asli (bukan orang asing) kadang kedetect Unknown kalau berdiri agak jauh dari kamera. Belum didiagnosis penyebabnya — bisa dari ukuran crop wajah yang mengecil pas jauh, variasi jarak yang kurang di dataset latih, atau threshold yang ketat. Butuh data real (dataset + hasil evaluate.py) buat didiagnosis, belum bisa disimpulkan sekarang.

## Cara kerja yang diharapkan

- Jangan rewrite logic yang udah jalan dan dijelasin di README/docstring tanpa alasan kuat — proyek ini sengaja didesain dengan pertimbangan tertentu (preprocessing, cara penetapan identitas per wajah, cara nyari threshold), bukan kode asal jalan.
- Komentar/docstring di kode ini pakai Bahasa Indonesia, pertahankan gaya itu kalau nambah kode baru.
- Kalau nemu keputusan desain yang ambigu atau berisiko nggak sesuai ketentuan tugas, laporkan dan tanya dulu, jangan langsung diubah sendiri.
- requirements.txt: `opencv-contrib-python` (bukan `opencv-python` biasa — dua paket ini konflik kalau dipasang bareng).
