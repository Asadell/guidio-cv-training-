"""
Script 03: Gabungkan 3 sumber YOLO → dataset_master_yolo/

Sumber:
  1. SafeWalkBD.v1i.yolov11  — remap 16 kelas → 6 kelas target
  2. pothole_1_yolo           — hasil script 01
  3. pothole_2_yolo           — hasil script 02 (semua di train, diacak ulang)

Output:
  dataset_sidewalk/dataset_master_yolo/images/{train,valid,test}/
  dataset_sidewalk/dataset_master_yolo/labels/{train,valid,test}/
  dataset_sidewalk/dataset_master_yolo/NEED_MANUAL_REVIEW_STAIRS.txt
    → daftar gambar dari kelas Stairs SafeWalkBD (default 'tangga')
      yang perlu dicek manual jika ingin bedakan naik/turun di masa depan.
"""
import argparse
import random
import shutil
import yaml
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.class_mapping import SAFEWALKBD_TO_TARGET, YOLO_NAME_TO_ID

RANDOM_SEED = 42
SPLIT_MAP   = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def copy_pair(img_path, lbl_path, out_root, split, prefix):
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    new_stem = f"{prefix}_{img_path.stem}"
    shutil.copy2(img_path, img_out / f"{new_stem}{img_path.suffix}")
    if lbl_path.exists():
        shutil.copy2(lbl_path, lbl_out / f"{new_stem}.txt")
    else:
        (lbl_out / f"{new_stem}.txt").write_text("")
    return new_stem


def process_safewalkbd(root, out_root, review_log):
    src_root  = root / "SafeWalkBD.v1i.yolov11"
    data_yaml = src_root / "data.yaml"
    if not data_yaml.exists():
        print(f"[!] {data_yaml} tidak ditemukan, lewati SafeWalkBD.")
        return 0

    names = yaml.safe_load(data_yaml.read_text())["names"]
    id_to_name = {int(k): v for k, v in names.items()} if isinstance(names, dict) \
                 else {i: n for i, n in enumerate(names)}

    total = 0
    for split_dir in ("train", "valid", "test"):
        img_dir = src_root / split_dir / "images"
        lbl_dir = src_root / split_dir / "labels"
        if not img_dir.exists():
            continue
        split = SPLIT_MAP[split_dir]

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_path  = lbl_dir / f"{img_path.stem}.txt"
            new_lines = []
            has_stairs = False

            if lbl_path.exists():
                for line in lbl_path.read_text().strip().splitlines():
                    if not line.strip():
                        continue
                    parts     = line.split()
                    orig_name = id_to_name.get(int(parts[0]))
                    if orig_name not in SAFEWALKBD_TO_TARGET:
                        continue
                    if orig_name == "Stairs":
                        has_stairs = True
                    new_id = YOLO_NAME_TO_ID[SAFEWALKBD_TO_TARGET[orig_name]]
                    new_lines.append(" ".join([str(new_id)] + parts[1:]))

            if not new_lines:
                continue

            img_out_dir = out_root / "images" / split
            lbl_out_dir = out_root / "labels" / split
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            new_stem = f"safewalkbd_{img_path.stem}"
            shutil.copy2(img_path, img_out_dir / f"{new_stem}{img_path.suffix}")
            (lbl_out_dir / f"{new_stem}.txt").write_text("\n".join(new_lines))
            total += 1
            if has_stairs:
                review_log.append(f"{split}/{new_stem}{img_path.suffix}")

    return total


def process_pothole_yolo(folder_name, root, out_root, prefix, force_random_split):
    src_root = root / folder_name
    if not src_root.exists():
        print(f"[!] {src_root} tidak ditemukan, lewati.")
        return 0

    pairs = []
    for split_dir in ("train", "valid", "test"):
        img_dir = src_root / "images" / split_dir
        lbl_dir = src_root / "labels" / split_dir
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            pairs.append((img_path, lbl_dir / f"{img_path.stem}.txt", SPLIT_MAP[split_dir]))

    if force_random_split:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)
        for i, (img_path, lbl_path, _) in enumerate(pairs):
            split = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
            copy_pair(img_path, lbl_path, out_root, split, prefix)
    else:
        for img_path, lbl_path, split in pairs:
            copy_pair(img_path, lbl_path, out_root, split, prefix)

    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Gabung dataset YOLO ke dataset_master_yolo/")
    parser.add_argument("--dataset-root", default="/home/asadel/kuliah/lomba/smstr6/guido/dataset_sidewalk")
    args   = parser.parse_args()
    root   = Path(args.dataset_root)
    out_root = root / "dataset_master_yolo"

    review_log  = []
    n_safewalk  = process_safewalkbd(root, out_root, review_log)
    n_pothole1  = process_pothole_yolo("pothole_1_yolo", root, out_root, "pothole1", False)
    n_pothole2  = process_pothole_yolo("pothole_2_yolo", root, out_root, "pothole2", True)

    if review_log:
        (out_root / "NEED_MANUAL_REVIEW_STAIRS.txt").write_text("\n".join(review_log))

    total = n_safewalk + n_pothole1 + n_pothole2
    print(f"[OK] Dataset master YOLO siap di: {out_root}")
    print(f"     SafeWalkBD  : {n_safewalk} gambar")
    print(f"     pothole_1   : {n_pothole1} gambar")
    print(f"     pothole_2   : {n_pothole2} gambar")
    print(f"     Total       : {total} gambar")
    if review_log:
        print(f"     [!] {len(review_log)} gambar 'Stairs' perlu review manual "
              f"-> {out_root / 'NEED_MANUAL_REVIEW_STAIRS.txt'}")


if __name__ == "__main__":
    main()
