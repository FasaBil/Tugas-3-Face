# -*- coding: utf-8 -*-
"""
capture_dataset.py - merekam dataset wajah lewat webcam.

    python capture_dataset.py --name Tazkia                 # data latih (train)
    python capture_dataset.py --name Tazkia --split test    # data uji (sesi berbeda!)
    python capture_dataset.py --name Unknown --split test   # foto ORANG ASING untuk uji Unknown
"""
import argparse
import os
import time

import cv2

from common import DATASET_DIR, crop_face, detect_faces, load_cascade

MIN_WIDTH = 80   # wajah lebih kecil dari ini tidak disimpan
INTERVAL = 0.3   # jeda antar foto (detik)


def put(frame, text, y, color=(0, 255, 255)):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="Nama tanpa spasi")
    ap.add_argument("--split", choices=("train", "test"), default=None)
    ap.add_argument("--count", type=int, default=None, help="Jumlah foto (default 80 train, 40 test)")
    args = ap.parse_args()

    name = (args.name or input("Nama (mis. Tazkia, atau Unknown untuk orang asing): ")).strip()
    split = args.split or input("Split [train/test] (Enter = train): ").strip().lower() or "train"
    if split not in ("train", "test"):
        raise SystemExit("Split harus train atau test.")
    count = args.count or (80 if split == "train" else 40)
    folder = os.path.join(DATASET_DIR, split, name)
    os.makedirs(folder, exist_ok=True)
    idx = len(os.listdir(folder)) + 1

    cascade = load_cascade()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera.")

    start, saved, last = time.time(), 0, 0.0
    while saved < count:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detect_faces(cascade, frame)
        shown = frame.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(shown, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if time.time() - start < 3:
            put(shown, "Bersiap... (%s | %s)" % (name, split), 30)
        else:
            put(shown, "%d/%d | Variasikan pose, ekspresi, jarak" % (saved, count), 30)
            if len(boxes) != 1:
                put(shown, "Harus tepat 1 wajah (terdeteksi: %d)" % len(boxes), 60, (0, 0, 255))
            elif boxes[0][2] < MIN_WIDTH:
                put(shown, "Dekatkan wajah ke kamera", 60, (0, 0, 255))
            elif time.time() - last >= INTERVAL:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(folder, "%s_%04d.jpg" % (name, idx)), crop_face(gray, boxes[0]))
                idx += 1
                saved += 1
                last = time.time()
        cv2.imshow("Capture dataset (q = batal)", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Selesai. %d foto tersimpan di %s" % (saved, folder))


if __name__ == "__main__":
    main()
