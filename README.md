# Face Recognition 3 Anggota (OpenCV + LBPH)

Cakupan: (A) kenali 3 anggota & multi-wajah, (B) warna berbeda per anggota, (C) nama + jarak/confidence dari model.

## Instalasi
```

pip install -r requirements.txt
```
- Perlu `opencv-contrib-python` (LBPH ada di contrib). Jangan pasang bersama `opencv-python`.
- Taruh `haarcascade_frontalface_default.xml` (URL di slide) di folder proyek.
- Jika OpenCV bermasalah di Python terbaru, pakai Python 3.12 di virtual env.

## Alur
```
python capture_dataset.py --name Tazkia                 # latih (80 foto); ulangi tiap anggota (nama tanpa spasi)
python capture_dataset.py --name Tazkia --split test    # uji (40 foto), SESI BERBEDA dari latih
python capture_dataset.py --name Unknown --split test   # foto beberapa orang ASING (bukan anggota)
python train.py
python evaluate.py --save-threshold
python evaluate.py                                      # ulangi agar ringkasan memakai threshold baru
python recognize.py
```

## Tampilan
`[Kotak biru] Nama | Jarak: xx.x`, `[hijau]`, `[merah]` untuk 3 anggota; `[kuning] Unknown | Jarak: ...` untuk wajah tidak dikenali. Pojok kiri atas: akurasi uji (dibaca dari `evaluate.py`), threshold, FPS, jumlah wajah. Legenda warna di kiri bawah.

## Nilai Jarak (confidence)
LBPH mengembalikan **jarak** wajah ke sampel latih paling mirip: makin kecil makin mirip. Bukan persentase dan bukan peluang benar. Nilai diambil langsung dari `predict_collect` model. Jarak lebih besar dari threshold berarti Unknown. **Akurasi** hanya dari `evaluate.py`: foto uji yang diprediksi benar / total foto uji x 100%, per orang dan keseluruhan.

## Menjaga nama, warna, dan jarak tetap pada wajah yang benar
1. Tiap wajah dibuat satu record (kotak, nama, jarak) sekaligus; penggambaran hanya memakai record itu.
2. Warna dicari dari nama hasil prediksi, bukan dari urutan/posisi wajah.
3. Satu nama hanya boleh dipakai satu wajah per frame (jarak terkecil menang); sisanya Unknown.
4. Uji: minta anggota bertukar tempat dan pastikan nama/warna mengikuti orangnya; uji satu orang asing bersama dua anggota.

## Tabel hasil (isi dari pengujian nyata)
| Orang | Benar / total foto uji | Akurasi |
|---|---|---|
| (nama 1) | | |
| (nama 2) | | |
| (nama 3) | | |
| Keseluruhan | | |

| Skenario real-time | Hasil pengamatan (nama/warna benar? tertukar?) |
|---|---|
| 1 orang | |
| 2 orang | |
| 3 orang bersamaan | |
| 2 anggota + 1 asing | |

## Keterbatasan
Haar frontal sulit mendeteksi wajah miring/tertutup. LBPH sensitif pada cahaya dan pose: variasikan data. Label dihitung per frame tanpa smoothing sehingga bisa sesekali berkedip.
