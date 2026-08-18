"""
Script 07: Fine-Tuning YOLO11n dengan Dataset Balanced
=======================================================
Masalah pada training sebelumnya (100 epoch):
- lubang     : 32.065 instances (dominan 91%)
- tangga     : 1.516 instances
- orang      :   301 instances
- got_terbuka:     0 instances (hanya sedikit dari rf-pothole-intel-unnati)
- motor      :     0 instances (data SafeWalkBD belum di-merge!)
- tiang      :     0 instances (data SafeWalkBD belum di-merge!)

Fix yang dilakukan script ini — semua dataset yang tersedia di GPU dipakai:
─────────────────────────────────────────────────────────────────────────────
Dataset              | Kelas yang diambil           | Mapping ke GUIDIO
─────────────────────────────────────────────────────────────────────────────
dataset_master_yolo  | lubang, tangga, orang        | identity (0,2,3)
SafeWalkBD           | Person, Pole, Vehicle,       | orang(3), tiang(5),
                     | Stairs, Pothole              | motor(4), tangga(2), lubang(0)
rf-stairs-data       | escalera (stairs), persona   | tangga(2), orang(3)
rf-stairs-updown     | stairsdown, stairsup         | tangga(2)
rf-pothole-intel     | Drain Hole, Sewer Cover,     | got_terbuka(1),
  -unnati            | Pothole                      | lubang(0)
─────────────────────────────────────────────────────────────────────────────

Strategi lain:
- Oversample kelas minority (tangga, orang, motor, tiang) agar rasio seimbang
- Fine-tune dari best.pt yang sudah ada (bukan dari nol) → lebih efisien
- Export .pt (Backend FastAPI) + .tflite INT8 (Flutter Mobile)
"""

import argparse
import collections
import os
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


# ─── Path Default (GPU) ───────────────────────────────────────────────────────
GPU_ROOT      = Path("/root/datasets")
OUTPUT_DS     = GPU_ROOT / "dataset_balanced_yolo"
OUTPUT_YAML   = Path("/workspace/guidio-cv-training-/configs/custom_navigasi_balanced.yaml")
BEST_PT_PATH  = Path(
    "/workspace/guidio-cv-training-/runs/yolo"
    "/navigasi_yolo11n_e100_20260818_041523/weights/best.pt"
)

# ─── Daftar Dataset Sumber ───────────────────────────────────────────────────
# Format: (folder, sub_splits, label_subdir, img_subdir, remap, tag)
#
# remap : {src_class_id: dst_class_id_guidio}
# GUIDIO: {0:lubang, 1:got_terbuka, 2:tangga, 3:orang, 4:motor, 5:tiang}
#
# sub_splits: list of (src_split_name, dst_split_name) agar bisa normalize
#   train→train dan valid/val/test→valid
# Path khusus stairs dataset (Pascal VOC XML → perlu konversi dulu)
STAIRS_VOC_DIR   = GPU_ROOT / "stairs" / "kaggle-stairs-dataclusterlabs"
STAIRS_YOLO_DIR  = GPU_ROOT / "stairs_yolo_converted"  # output konversi

DATASET_CONFIGS = [
    # 1. Master YOLO (sudah pakai format GUIDIO, identity remap)
    {
        "path":   GPU_ROOT / "dataset_master_yolo",
        "splits": [("train", "train"), ("valid", "valid")],
        "labels": "labels",
        "images": "images",
        "remap":  {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "tag":    "master",
    },
    # 2. SafeWalkBD — Person(4)→orang, Pole(5)→tiang, Vehicle(15)→motor,
    #                 Stairs(10)→tangga, Pothole(6)→lubang
    {
        "path":   GPU_ROOT / "SafeWalkBD.v1i.yolov11",
        "splits": [("train", "train"), ("valid", "valid")],
        "labels": "labels",
        "images": "images",
        "remap":  {4: 3, 5: 5, 15: 4, 10: 2, 6: 0},
        "tag":    "swbd",
    },
    # 3. rf-stairs-data — escalera(2)→tangga, persona(6)→orang
    #    (Label Spanyol; hanya ada split train)
    {
        "path":   GPU_ROOT / "rf-stairs-data",
        "splits": [("train", "train"), ("valid", "valid"), ("test", "valid")],
        "labels": "labels",
        "images": "images",
        "remap":  {2: 2, 6: 3},   # escalera→tangga, persona→orang
        "tag":    "rfstairs",
    },
    # 4. rf-stairs-updown — stairsdown(0)+stairsup(1) → tangga
    {
        "path":   GPU_ROOT / "rf-stairs-updown",
        "splits": [("train", "train"), ("valid", "valid"), ("test", "valid")],
        "labels": "labels",
        "images": "images",
        "remap":  {0: 2, 1: 2},   # both → tangga
        "tag":    "rfupdown",
    },
    # 5. rf-pothole-intel-unnati — DrainHole(0)→got_terbuka,
    #                              Pothole(1)→lubang, SewerCover(2)→got_terbuka
    {
        "path":   GPU_ROOT / "rf-pothole-intel-unnati",
        "splits": [("train", "train"), ("valid", "valid"), ("test", "valid")],
        "labels": "labels",
        "images": "images",
        "remap":  {0: 1, 1: 0, 2: 1},
        "tag":    "rfdrain",
    },
    # 6. stairs (Kaggle DataClusterLabs) — format Pascal VOC XML
    #    Dikonversi dulu ke YOLO sebelum dipakai.
    #    class 'stair' → tangga (2)
    #    (ditangani oleh convert_voc_stairs(), bukan add_dataset())
    {
        "path":   STAIRS_YOLO_DIR,          # hasil konversi
        "splits": [("train", "train")],      # semua masuk train
        "labels": "labels",
        "images": "images",
        "remap":  {0: 2},                   # cls_0 → tangga
        "tag":    "kgstairs",
    },
]

# Kelas minority yang perlu di-oversample (repeat_factor x)
# Lebih tinggi factor = makin banyak salinan di dataset train
OVERSAMPLE = {
    1: 6,   # got_terbuka — data sangat sedikit
    2: 3,   # tangga      — masih minoritas vs lubang
    3: 5,   # orang       — perlu balance
    4: 2,   # motor       — SafeWalkBD cukup banyak
    5: 2,   # tiang       — SafeWalkBD cukup banyak
}

CLASS_NAMES = {0:"lubang", 1:"got_terbuka", 2:"tangga", 3:"orang", 4:"motor", 5:"tiang"}


# ─── Pascal VOC XML → YOLO Converter ────────────────────────────────────────

def convert_voc_stairs(voc_root: Path, out_yolo_dir: Path) -> int:
    """
    Konversi anotasi Pascal VOC XML dari dataset stairs (Kaggle DataClusterLabs)
    ke format YOLO txt, lalu hardlink gambarnya.

    Struktur input:
      voc_root/
        annotation/stairs annoted/*.xml   ← anotasi XML
        stairs_sample_100/**/*.jpg         ← gambar (dicari rekursif by filename)

    Struktur output (YOLO):
      out_yolo_dir/
        images/train/*.jpg
        labels/train/*.txt   ← format YOLO: cls cx cy w h (normalized)

    Return: jumlah file yang berhasil dikonversi.
    """
    import xml.etree.ElementTree as ET

    if out_yolo_dir.exists():
        existing = list((out_yolo_dir / "labels" / "train").glob("*.txt"))
        if existing:
            print(f"  [kgstairs] Sudah ada {len(existing)} file konversi — skip konversi ulang")
            return len(existing)

    ann_dir = voc_root / "annotation" / "stairs annoted"
    if not ann_dir.exists():
        print(f"  [kgstairs] Folder anotasi tidak ditemukan: {ann_dir}")
        return 0

    # Build index: filename → path (untuk cari gambar dengan cepat)
    print("  [kgstairs] Indexing gambar...")
    img_index: dict[str, Path] = {}
    for img_path in voc_root.rglob("*.jpg"):
        # Nama file di XML kadang berakhiran .jpg.jpg, normalize
        clean = img_path.name.replace(".jpg.jpg", ".jpg")
        img_index[clean] = img_path
        img_index[img_path.name] = img_path  # juga simpan nama asli

    out_img_dir   = out_yolo_dir / "images" / "train"
    out_label_dir = out_yolo_dir / "labels" / "train"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    converted, skipped = 0, 0
    for xml_file in ann_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Dimensi gambar
            size  = root.find("size")
            img_w = int(size.find("width").text)
            img_h = int(size.find("height").text)
            if img_w == 0 or img_h == 0:
                skipped += 1
                continue

            # Nama file gambar (bisa berakhiran .jpg.jpg)
            filename = root.find("filename").text
            clean_fn = filename.replace(".jpg.jpg", ".jpg").replace(".jpg.jpeg", ".jpg")

            img_src = img_index.get(clean_fn) or img_index.get(filename)
            if img_src is None:
                skipped += 1
                continue

            # Konversi semua objek 'stair' ke YOLO
            yolo_lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().lower()
                if name not in ("stair", "stairs", "escalera", "steps"):
                    continue  # skip kelas lain
                bbox  = obj.find("bndbox")
                xmin  = float(bbox.find("xmin").text)
                ymin  = float(bbox.find("ymin").text)
                xmax  = float(bbox.find("xmax").text)
                ymax  = float(bbox.find("ymax").text)

                # Normalize ke YOLO format (cx, cy, w, h)
                cx = ((xmin + xmax) / 2) / img_w
                cy = ((ymin + ymax) / 2) / img_h
                bw = (xmax - xmin) / img_w
                bh = (ymax - ymin) / img_h

                # Kelas 0 di output (remap ke tangga(2) dilakukan oleh add_dataset)
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not yolo_lines:
                skipped += 1
                continue

            # Tulis label
            stem = xml_file.stem.replace(".jpg", "")  # hapus .jpg dari nama
            (out_label_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n")

            # Hardlink / copy gambar
            dst_img = out_img_dir / f"{stem}.jpg"
            hardlink_or_copy(img_src, dst_img)
            converted += 1

        except Exception as e:
            skipped += 1
            continue

    print(f"  [kgstairs] Konversi selesai: {converted} OK, {skipped} skip")
    return converted


# ─── Utilities ────────────────────────────────────────────────────────────────

def hardlink_or_copy(src: Path, dst: Path):
    """Hardlink (hemat disk 0 byte) atau copy sebagai fallback."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_image(label_path: Path, img_dir: Path) -> Path | None:
    """Cari file gambar yang berkoresponden dengan label file."""
    stem = label_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = img_dir / (stem + ext)
        if cand.exists():
            return cand
    return None


def remap_label_file(src: Path, dst: Path, remap: dict) -> bool:
    """
    Baca label YOLO, remap class ID, tulis ke dst.
    Return False jika tidak ada baris yang lolos remap (semua di-skip).
    """
    lines_out = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        if cls_id in remap:
            parts[0] = str(remap[cls_id])
            lines_out.append(" ".join(parts))
    if not lines_out:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + "\n")
    return True


# ─── Build Dataset ────────────────────────────────────────────────────────────

def build_balanced_dataset(out_dir: Path) -> dict:
    """
    Gabungkan semua dataset sumber, remap kelas, tulis ke out_dir.
    Returns dict: class_id → list of label paths di split train (untuk oversample)
    """
    print("\n" + "=" * 65)
    print("  GUIDIO — Build Dataset Balanced")
    print("=" * 65)

    # Konversi Pascal VOC stairs terlebih dahulu sebelum loop utama
    if STAIRS_VOC_DIR.exists():
        print("\n[→] Konversi Pascal VOC XML → YOLO untuk dataset stairs...")
        convert_voc_stairs(STAIRS_VOC_DIR, STAIRS_YOLO_DIR)
    else:
        print(f"\n[SKIP] stairs VOC tidak ditemukan: {STAIRS_VOC_DIR}")

    by_class: dict[int, list[Path]] = collections.defaultdict(list)

    for cfg in DATASET_CONFIGS:
        ds_path = cfg["path"]
        if not ds_path.exists():
            print(f"\n[SKIP] Dataset tidak ditemukan: {ds_path}")
            continue

        print(f"\n[+] Dataset: {cfg['tag']} ({ds_path.name})")
        for src_split, dst_split in cfg["splits"]:
            label_dir = ds_path / src_split / cfg["labels"]
            img_dir   = ds_path / src_split / cfg["images"]

            if not label_dir.exists():
                # Coba flat structure: labels/{split}
                label_dir = ds_path / cfg["labels"] / src_split
                img_dir   = ds_path / cfg["images"]  / src_split

            if not label_dir.exists():
                continue

            added, skipped = 0, 0
            class_hits: dict[int, int] = collections.defaultdict(int)

            for lf in label_dir.glob("*.txt"):
                new_stem  = f"{cfg['tag']}_{lf.stem}"
                dst_label = out_dir / "labels" / dst_split / f"{new_stem}.txt"

                if not remap_label_file(lf, dst_label, cfg["remap"]):
                    skipped += 1
                    continue

                img = find_image(lf, img_dir)
                if img is None:
                    dst_label.unlink(missing_ok=True)
                    skipped += 1
                    continue

                dst_img = out_dir / "images" / dst_split / f"{new_stem}{img.suffix}"
                hardlink_or_copy(img, dst_img)
                added += 1

                # Catat kelas
                for line in dst_label.read_text().splitlines():
                    if line.strip():
                        cls_id = int(line.split()[0])
                        class_hits[cls_id] += 1
                        if dst_split == "train":
                            by_class[cls_id].append(dst_label)

            hit_str = ", ".join(
                f"{CLASS_NAMES.get(k,k)}:{v}" for k, v in sorted(class_hits.items())
            )
            print(f"    [{src_split}→{dst_split}] +{added} files | skip {skipped} | kelas: {hit_str or '-'}")

    return by_class


def apply_oversample(by_class: dict, out_dir: Path):
    """Duplikasi label+gambar untuk kelas minority di split train."""
    print("\n[OVERSAMPLE] Menggandakan kelas minority di split train...")
    img_dir   = out_dir / "images" / "train"
    label_dir = out_dir / "labels" / "train"

    for cls_id, repeat in OVERSAMPLE.items():
        # Ambil hanya label unik (bukan hasil oversample sebelumnya)
        paths = [p for p in by_class.get(cls_id, []) if "_os" not in p.stem]
        if not paths:
            print(f"  [{CLASS_NAMES[cls_id]:<12}] 0 gambar — skip")
            continue

        n_added = 0
        for i in range(1, repeat):
            for lp in paths:
                new_stem  = f"{lp.stem}_os{i}"
                dst_label = label_dir / f"{new_stem}.txt"
                if dst_label.exists():
                    continue
                shutil.copy2(lp, dst_label)
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    src_img = img_dir / f"{lp.stem}{ext}"
                    if src_img.exists():
                        hardlink_or_copy(src_img, img_dir / f"{new_stem}{ext}")
                        n_added += 1
                        break

        total = len(paths) * repeat
        print(f"  [{CLASS_NAMES[cls_id]:<12}] {len(paths)} unik × {repeat} = ~{total} (+{n_added} file baru)")


def print_summary(out_dir: Path):
    """Cetak distribusi kelas akhir setelah merge + oversample."""
    print("\n[SUMMARY] Distribusi akhir split train:")
    counts = collections.Counter()
    for lf in (out_dir / "labels" / "train").glob("*.txt"):
        seen = set()
        for line in lf.read_text().splitlines():
            if line.strip():
                seen.add(int(line.split()[0]))
        for c in seen:
            counts[c] += 1
    total = sum(counts.values())
    for cid, name in CLASS_NAMES.items():
        n = counts.get(cid, 0)
        bar = "█" * int(n / max(counts.values()) * 20) if total else ""
        pct = n / total * 100 if total else 0
        print(f"  {name:<14}: {n:>6} images ({pct:.1f}%) {bar}")
    print(f"  {'TOTAL':<14}: {total:>6} images")


def write_yaml(out_dir: Path, yaml_path: Path):
    cfg = {
        "path":  str(out_dir),
        "train": "images/train",
        "val":   "images/valid",
        "nc":    6,
        "names": {i: n for i, n in CLASS_NAMES.items()},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"\n[OK] Data YAML → {yaml_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning YOLO11n balanced — merge semua dataset + oversample"
    )
    parser.add_argument("--epochs",       type=int, default=80)
    parser.add_argument("--imgsz",        type=int, default=640)
    parser.add_argument("--batch",        type=int, default=16)
    parser.add_argument("--weights",      default=str(BEST_PT_PATH))
    parser.add_argument("--project",      default="/workspace/guidio-cv-training-/runs/yolo")
    parser.add_argument("--name",         default=None)
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Skip rebuild dataset (pakai folder yang sudah ada)")
    args = parser.parse_args()

    if not args.name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"navigasi_yolo11n_balanced_e{args.epochs}_{ts}"

    print("=" * 65)
    print("  GUIDIO — Fine-Tuning YOLO11n Balanced")
    print(f"  Weights  : {args.weights}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Run Name : {args.name}")
    print("=" * 65)

    # ── Step 1: Build dataset ─────────────────────────────────────────────────
    if not args.skip_rebuild or not OUTPUT_DS.exists():
        if OUTPUT_DS.exists():
            print(f"\n[!] Hapus dataset lama: {OUTPUT_DS}")
            shutil.rmtree(OUTPUT_DS)
        by_class = build_balanced_dataset(OUTPUT_DS)
        apply_oversample(by_class, OUTPUT_DS)
    else:
        print(f"\n[SKIP] Gunakan dataset yang sudah ada: {OUTPUT_DS}")

    print_summary(OUTPUT_DS)
    write_yaml(OUTPUT_DS, OUTPUT_YAML)

    # ── Step 2: Fine-tune dari best.pt ───────────────────────────────────────
    print(f"\n[→] Fine-tuning dari: {args.weights}")
    model = YOLO(args.weights)
    model.train(
        data=str(OUTPUT_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=20,
        device=0,

        # LR kecil untuk fine-tuning (jangan "lupa" bobot lama)
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=2.0,

        # Augmentasi kontekstual trotoar
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,      # JANGAN flip vertikal (tangga terbalik!)
        mosaic=1.0,
        copy_paste=0.3,  # Copy-paste augmentation untuk instance minority
        erasing=0.1,
        close_mosaic=10,
    )

    # ── Step 3: Validasi ─────────────────────────────────────────────────────
    metrics = model.val(data=str(OUTPUT_YAML), imgsz=args.imgsz)
    print("\n=== Validasi Final ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    if hasattr(metrics.box, "ap_class_index") and hasattr(metrics.box, "ap50"):
        print("\nPer-kelas mAP50:")
        for cls_id, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            print(f"  {model.names[int(cls_id)]:<14}: {ap:.4f}")

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[OK] Backend model (.pt) → {best_pt}")

    # ── Step 4: Export TFLite untuk Mobile ───────────────────────────────────
    print("\n[→] Export LiteRT INT8 untuk Flutter mobile...")
    try:
        exp = YOLO(str(best_pt))
        tflite_path = exp.export(
            format="litert",
            quantize=True,
            data=str(OUTPUT_YAML),
            imgsz=args.imgsz,
        )
        print(f"[OK] Mobile model (.tflite) → {tflite_path}")
    except Exception as e:
        print(f"[!] Export TFLite gagal: {e}")
        tflite_path = None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  [FINISHED] Fine-Tuning Balanced YOLO11n Selesai!")
    print(f"  Backend  (.pt)     : {best_pt}")
    weights_dir = Path(args.project) / args.name / "weights"
    print(f"  Mobile   (.tflite) : {weights_dir}/best_int8.tflite")
    print("=" * 65)


if __name__ == "__main__":
    main()
