# -*- coding: utf-8 -*-
"""
recognize.py - pengenalan 3 anggota secara real-time: kotak berwarna per anggota, nama, dan jarak LBPH.

Jalankan:  python recognize.py        (tekan q untuk keluar, p untuk simpan screenshot)

Tampilan per wajah:  [kotak berwarna]  Nama | Jarak: xx.x
  - Warna anggota tetap (biru, hijau, merah); Unknown = kuning.
  - "Jarak" = nilai LBPH dari model: makin KECIL makin mirip. Ini bukan persentase dan bukan akurasi.
  - Akurasi (%) hanya dari evaluate.py dan ditampilkan sebagai teks terpisah di pojok kiri atas.
"""
import json
import os
import time

import cv2

from common import (BASE_DIR, EVAL_PATH, UNKNOWN, crop_face, detect_faces, get_threshold, load_cascade,
                    load_model, preprocess_face, identity_distances, assign_from_distances)

# Warna BGR. Anggota diberi warna menurut urutan id (id 0 = biru, 1 = hijau, 2 = merah).
MEMBER_COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
UNKNOWN_COLOR = (0, 255, 255)   # kuning, tidak dipakai anggota


def recognize_frame(frame, cascade, recognizer, id2name, threshold):
    """
    Mengembalikan list record, satu per wajah: {"box", "name", "dist"}.
    Kotak, nama, dan jarak dibuat BERSAMA dalam satu record, jadi tidak mungkin tertukar antar wajah.
    """
    boxes = detect_faces(cascade, frame)
    if not boxes:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    per_face = [identity_distances(recognizer, preprocess_face(crop_face(gray, b))) for b in boxes]
    results = assign_from_distances(per_face, id2name, threshold)   # nama unik per frame
    return [{"box": b, "name": n, "dist": d} for b, (n, d) in zip(boxes, results)]


def put(frame, text, org, color, scale=0.6, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw(frame, records, colors):
    for r in records:
        x, y, w, h = r["box"]
        color = colors.get(r["name"], UNKNOWN_COLOR)   # warna dicari dari NAMA hasil prediksi, bukan dari posisi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if r["name"] == UNKNOWN:
            text = "Unknown | Jarak: %.1f" % r["dist"]
        else:
            text = "%s | Jarak: %.1f" % (r["name"], r["dist"])
        put(frame, text, (x, y - 8 if y > 25 else y + h + 22), color)


def main():
    cascade = load_cascade()
    recognizer, id2name = load_model()
    names = [id2name[k] for k in sorted(id2name)]
    colors = dict((n, MEMBER_COLORS[i % len(MEMBER_COLORS)]) for i, n in enumerate(names))   # tetap selama program jalan
    threshold = get_threshold()

    info = "Akurasi uji: belum dievaluasi (jalankan evaluate.py)"
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, encoding="utf-8") as f:
            ev = json.load(f)
        if ev.get("overall_pct") is not None:
            info = "Akurasi uji model: %.1f%% (dari evaluate.py)" % ev["overall_pct"]
    print("Anggota:", names, "| threshold:", threshold)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")
    shots, fps, last = 0, 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        records = recognize_frame(frame, cascade, recognizer, id2name, threshold)
        draw(frame, records, colors)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(1e-6, now - last) if fps else 1.0 / max(1e-6, now - last)
        last = now
        put(frame, info, (10, 22), (255, 255, 255), 0.55, 1)
        put(frame, "Threshold jarak: %.0f | FPS %.0f | Wajah: %d" % (threshold, fps, len(records)),
            (10, 44), (255, 255, 255), 0.55, 1)
        y = frame.shape[0] - 12 - 22 * len(names)
        for n in names + [UNKNOWN]:   # legenda warna
            put(frame, n, (10, y + 22), colors.get(n, UNKNOWN_COLOR), 0.55, 1)
            y += 22
        cv2.imshow("Face Recognition (q = keluar, p = screenshot)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            shots += 1
            out = os.path.join(BASE_DIR, "screenshot_%03d.jpg" % shots)
            cv2.imwrite(out, frame)
            print("Screenshot:", out)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
