# -*- coding: utf-8 -*-
"""
recognize.py - pengenalan 3 anggota real-time: kotak berwarna per anggota + label "Nama (X%)".

Jalankan:  python recognize.py        (q atau Esc = keluar, p = screenshot)

  - X% = kemiripan = 100 - jarak LBPH (rumus yang sama dengan slide dosen), dari prediksi wajah itu sendiri.
    Ini skor kemiripan kasar, BUKAN akurasi hasil pengujian (akurasi ada di evaluate.py).
  - Jarak LBPH makin KECIL = makin mirip. Jarak > threshold (common.py) = Unknown (kotak kuning).
  - Jika ada Unknown, jarak asli tercetak di terminal tiap ~1 detik untuk membantu menyetel threshold.
"""
import os

import cv2

from common import (BASE_DIR, UNKNOWN, assign_from_distances, crop_face, detect_faces, get_threshold,
                    identity_distances, load_cascade, load_model, preprocess_face)

# Warna BGR. Anggota diberi warna menurut urutan id; Unknown = kuning (tidak dipakai anggota).
MEMBER_COLORS = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)]
UNKNOWN_COLOR = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def similarity(dist):
    """Skor kemiripan 0-100 = 100 - jarak LBPH (seperti slide dosen). Bukan akurasi."""
    return max(0, min(100, int(round(100 - dist))))


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
    results = assign_from_distances(per_face, id2name, threshold)   # satu nama hanya untuk satu wajah
    return [{"box": b, "name": n, "dist": d} for b, (n, d) in zip(boxes, results)]


def put(frame, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(frame, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw(frame, records, colors):
    for r in records:
        x, y, w, h = r["box"]
        color = colors.get(r["name"], UNKNOWN_COLOR)   # warna dicari dari NAMA hasil prediksi, bukan dari posisi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%d%%)" % (r["name"], similarity(r["dist"]))
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
        tag_h = th + 10
        y0 = y - tag_h if y - tag_h >= 0 else y + h          # label di atas kotak (atau di bawah jika mepet atas)
        cv2.rectangle(frame, (x, y0), (x + tw + 10, y0 + tag_h), color, -1)
        b, g, rr = color
        text_color = (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * rr > 150 else (255, 255, 255)
        cv2.putText(frame, text, (x + 5, y0 + th + 4), FONT, 0.6, text_color, 2, cv2.LINE_AA)


def header_text(records):
    ids = sorted(r["name"] for r in records if r["name"] != UNKNOWN)
    text = "Terdeteksi: %d wajah | Teridentifikasi: %d" % (len(records), len(ids))
    return text + (" (%s)" % ", ".join(ids) if ids else "")


def main():
    cascade = load_cascade()
    recognizer, id2name = load_model()
    names = [id2name[k] for k in sorted(id2name)]
    colors = dict((n, MEMBER_COLORS[i % len(MEMBER_COLORS)]) for i, n in enumerate(names))   # tetap selama program jalan
    threshold = get_threshold()
    print("Anggota:", names, "| threshold jarak:", threshold)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")
    shots, frame_no = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        records = recognize_frame(frame, cascade, recognizer, id2name, threshold)
        draw(frame, records, colors)
        put(frame, header_text(records), (10, 28))
        if records and frame_no % 30 == 0 and any(r["name"] == UNKNOWN for r in records):
            print("[info] " + " | ".join("%s jarak=%.0f" % (r["name"], r["dist"]) for r in records))
        cv2.imshow("Pengenalan Wajah 3 Orang (q / Esc = keluar, p = screenshot)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
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
