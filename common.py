# -*- coding: utf-8 -*-
"""common.py - fungsi bersama: deteksi, crop, preprocessing, model, dan penetapan identitas."""
import json
import os

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")      # dataset/<train|test>/<Nama>/*.jpg
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "face-model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
EVAL_PATH = os.path.join(MODEL_DIR, "eval_summary.json")

FACE_SIZE = (160, 160)
DEFAULT_THRESHOLD = 80.0   # jarak LBPH maksimum agar dianggap dikenali (kecil = mirip)
UNKNOWN = "Unknown"


def load_cascade():
    """Haar Cascade dibaca dari file XML di folder proyek (URL unduhan ada di slide kuliah)."""
    cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        raise SystemExit("File haarcascade_frontalface_default.xml tidak ditemukan di folder proyek.")
    return cascade


def detect_faces(cascade, frame):
    """Daftar kotak (x, y, w, h), urut dari kiri ke kanan."""
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return sorted((tuple(int(v) for v in b) for b in boxes), key=lambda b: b[0])


def crop_face(gray, box):
    """Potong area wajah dari gambar grayscale lalu samakan ukurannya."""
    x, y, w, h = box
    return cv2.resize(gray[max(0, y):y + h, max(0, x):x + w], FACE_SIZE, interpolation=cv2.INTER_AREA)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_face(face):
    """Normalisasi kontras (CLAHE). Dipakai sama persis saat training, evaluasi, dan real-time."""
    return _clahe.apply(face)


def list_split_images(split):
    """{nama: [path gambar]} untuk dataset/<split>/."""
    root = os.path.join(DATASET_DIR, split)
    data = {}
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if os.path.isdir(folder):
                files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if files:
                    data[name] = files
    return data


def load_face_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None and img.shape[:2] != (FACE_SIZE[1], FACE_SIZE[0]):
        img = cv2.resize(img, FACE_SIZE, interpolation=cv2.INTER_AREA)
    return img


def create_recognizer():
    if not hasattr(cv2, "face"):
        raise SystemExit("cv2.face tidak ada. Pasang opencv-contrib-python (dan hapus opencv-python).")
    return cv2.face.LBPHFaceRecognizer_create()


def load_model():
    """Mengembalikan (recognizer, {id: nama})."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        raise SystemExit("Model belum ada. Jalankan dulu: python train.py")
    recognizer = create_recognizer()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, encoding="utf-8") as f:
        return recognizer, {int(k): v for k, v in json.load(f).items()}


def get_threshold():
    """Dari model/threshold.txt jika ada, jika tidak DEFAULT_THRESHOLD."""
    try:
        with open(THRESHOLD_PATH) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return DEFAULT_THRESHOLD


def identity_distances(recognizer, face):
    """Jarak LBPH terkecil untuk SETIAP identitas: {id: jarak}. Kecil = mirip."""
    collector = cv2.face.StandardCollector_create()
    recognizer.predict_collect(face, collector)
    best = {}
    for label, dist in collector.getResults():
        label = int(label)
        if label not in best or dist < best[label]:
            best[label] = float(dist)
    return best


def assign_from_distances(per_face, id2name, threshold):
    """
    per_face: list {id: jarak}, satu per wajah dalam frame yang sama.
    Pasangan (wajah, identitas) dengan jarak <= threshold diproses dari jarak terkecil;
    satu identitas hanya untuk SATU wajah. Sisanya Unknown.
    Mengembalikan list (nama, jarak) sejajar dengan per_face.
    """
    out = [(UNKNOWN, min(d.values()) if d else float("inf")) for d in per_face]
    pairs = sorted((d, i, lab) for i, dists in enumerate(per_face) for lab, d in dists.items() if d <= threshold)
    used_faces, used_ids = set(), set()
    for d, i, lab in pairs:
        if i not in used_faces and lab not in used_ids:
            out[i] = (id2name.get(lab, UNKNOWN), d)
            used_faces.add(i)
            used_ids.add(lab)
    return out
