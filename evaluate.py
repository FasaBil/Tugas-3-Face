# -*- coding: utf-8 -*-
"""
evaluate.py - akurasi model pada dataset/test (foto TIDAK dipakai untuk melatih; ambil di sesi berbeda).

  dataset/test/<Nama>/    foto uji tiap anggota
  dataset/test/Unknown/   foto ORANG ASING (opsional, untuk uji penolakan dan threshold)

Rumus:
  Akurasi per orang   = foto orang itu yang diprediksi dengan namanya sendiri / jumlah foto orang itu x 100%
  Akurasi keseluruhan = total benar semua anggota / total foto uji anggota x 100%
  Penolakan asing     = foto Unknown yang diprediksi Unknown / jumlah foto Unknown x 100%  (FAR = 100% - ini)
Jarak LBPH (confidence di layar) BUKAN akurasi. Akurasi hanya dari pengujian ini.

    python evaluate.py                    # pakai threshold tersimpan/default
    python evaluate.py --save-threshold   # simpan saran threshold (FAR <= 5%)
Hasil ringkas disimpan di model/eval_summary.json (dibaca recognize.py untuk ditampilkan di layar).
"""
import argparse
import json

from common import (EVAL_PATH, THRESHOLD_PATH, UNKNOWN, get_threshold, identity_distances, list_split_images,
                    load_face_image, load_model, preprocess_face)


def predict(dists, id2name, thr):
    lab, d = min(dists.items(), key=lambda kv: kv[1])
    return id2name[lab] if d <= thr else UNKNOWN


def pct(a, b):
    return 100.0 * a / b if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-threshold", action="store_true")
    args = ap.parse_args()

    recognizer, id2name = load_model()
    names = [id2name[k] for k in sorted(id2name)]
    data = list_split_images("test")
    if not data:
        raise SystemExit("dataset/test/ kosong. Ambil: python capture_dataset.py --name <Nama> --split test")

    records = []   # (nama sebenarnya, {id: jarak})
    for folder, paths in data.items():
        truth = folder if folder in names else UNKNOWN
        for p in paths:
            img = load_face_image(p)
            if img is not None:
                records.append((truth, identity_distances(recognizer, preprocess_face(img))))

    def summarize(thr):
        conf = {t: {c: 0 for c in names + [UNKNOWN]} for t in names + [UNKNOWN]}
        for truth, dists in records:
            conf[truth][predict(dists, id2name, thr)] += 1
        return conf

    thr = get_threshold()
    conf = summarize(thr)
    print("=== EVALUASI | threshold=%.0f | total foto uji=%d ===" % (thr, len(records)))
    per_person, total, correct = {}, 0, 0
    for n in names:
        t = sum(conf[n].values())
        per_person[n] = pct(conf[n][n], t)
        total, correct = total + t, correct + conf[n][n]
        if t:
            print("  %-14s %5.1f%% (%d/%d)  salah nama: %d, ditolak sbg Unknown: %d"
                  % (n, per_person[n], conf[n][n], t,
                     sum(v for c, v in conf[n].items() if c not in (n, UNKNOWN)), conf[n][UNKNOWN]))
        else:
            print("  %-14s tidak ada foto uji" % n)
    overall = pct(correct, total)
    print("Akurasi keseluruhan anggota: %s" % ("%.1f%% (%d/%d)" % (overall, correct, total) if total else "n/a"))
    unk_total = sum(conf[UNKNOWN].values())
    if unk_total:
        rej = conf[UNKNOWN][UNKNOWN]
        print("Orang asing ditolak: %.1f%% (%d/%d) | FAR: %.1f%%" % (pct(rej, unk_total), rej, unk_total,
                                                                      pct(unk_total - rej, unk_total)))
    else:
        print("Tidak ada foto orang asing (dataset/test/Unknown/): FAR dan threshold tidak bisa dievaluasi.")

    cols = names + [UNKNOWN]
    print("\nMatriks kebingungan (baris = sebenarnya, kolom = prediksi):")
    print("  %-14s" % "" + "".join("%-14s" % c[:13] for c in cols))
    for t in cols:
        if sum(conf[t].values()):
            print("  %-14s" % t[:13] + "".join("%-14d" % conf[t][c] for c in cols))

    if unk_total:
        print("\nSapuan threshold:\n  thr   akurasi_anggota   FAR_asing")
        best = None
        for t in range(30, 151, 5):
            c = summarize(t)
            acc = pct(sum(c[n][n] for n in names), total)
            far = pct(unk_total - c[UNKNOWN][UNKNOWN], unk_total)
            print("  %3d   %13.1f%%   %8.1f%%" % (t, acc, far))
            if far <= 5.0 and (best is None or acc > best[1]):
                best = (t, acc)
        if best:
            print("\nSaran threshold (FAR <= 5%%, akurasi anggota tertinggi): %d" % best[0])
            if args.save_threshold:
                with open(THRESHOLD_PATH, "w") as f:
                    f.write(str(best[0]))
                print("Disimpan. Jalankan evaluate.py lagi agar ringkasan memakai threshold baru.")
        else:
            print("\nTidak ada threshold dengan FAR <= 5%. Tambah variasi data latih.")
    print("\nCatatan: threshold disetel dan akurasi dilaporkan pada data uji yang sama, jadi sedikit optimistis.")

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": thr, "overall_pct": overall, "per_person_pct": per_person,
                   "total_test": total}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
