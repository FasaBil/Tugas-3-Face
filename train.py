# -*- coding: utf-8 -*-
"""train.py - melatih LBPH dari dataset/train/<Nama>/ (3 orang). Jalankan: python train.py"""
import json
import os

import numpy as np

from common import (LABELS_PATH, MODEL_DIR, MODEL_PATH, UNKNOWN, create_recognizer, list_split_images,
                    load_face_image, preprocess_face)


def main():
    data = list_split_images("train")
    names = [n for n in sorted(data) if n.lower() != UNKNOWN.lower()]
    if len(names) < 2:
        raise SystemExit("Butuh minimal 2 orang (idealnya 3) di dataset/train/. Ditemukan: %s" % names)

    faces, ids, id2name = [], [], {}
    for label, name in enumerate(names):
        id2name[label] = name
        n = 0
        for path in data[name]:
            img = load_face_image(path)
            if img is not None:
                faces.append(preprocess_face(img))
                ids.append(label)
                n += 1
        print("  %-14s id=%d  %d foto latih" % (name, label, n))
        if n < 30:
            print("  [PERINGATAN] %s: disarankan >= 60 foto dengan variasi pose/cahaya." % name)

    recognizer = create_recognizer()
    recognizer.train(faces, np.array(ids, dtype=np.int32))
    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer.write(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2name.items()}, f, indent=2, ensure_ascii=False)
    print("Model disimpan. Langkah berikut: python evaluate.py --save-threshold")


if __name__ == "__main__":
    main()
