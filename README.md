# GUIDIO CV Training — Mode Navigasi

Folder ini berisi pipeline lengkap untuk melatih 2 model CV yang dipakai
di **Mode Navigasi** aplikasi GUIDIO (pemandu tunanetra):

| Model | Task | Format Output |
|-------|------|--------------|
| **YOLO11n** | Deteksi 6 kelas rintangan | `.pt` → langsung ke backend |
| **PIDNet-S** | Segmentasi jalur 3-zona | `.pth` → export `.onnx` → backend |

---

## Struktur Folder

```
guido_cv_training/
├── .venv/                          ← symlink ke project/backend/venv (CUDA ready)
├── configs/
│   └── custom_navigasi.yaml        ← data.yaml untuk YOLO training
├── scripts/
│   ├── 01_convert_voc_to_yolo.py   ← pothole_1 (VOC) → YOLO TXT
│   ├── 02_convert_coco_pothole_to_yolo.py  ← pothole_2 (COCO) → YOLO TXT
│   ├── 03_merge_yolo_datasets.py   ← gabung SafeWalkBD + pothole_1 + pothole_2
│   ├── 04_convert_coco_segmentation_to_mask.py  ← Sidewalk Seg → PNG mask
│   ├── 05_train_yolo.py            ← training YOLO11n
│   ├── 06_train_pidnet.py          ← training PIDNet-S
│   ├── 07_export_onnx.py           ← export PIDNet best.pth → .onnx
│   └── run_all.sh                  ← jalankan seluruh pipeline
├── src/
│   ├── class_mapping.py            ← mapping kelas semua dataset → target
│   ├── pidnet/
│   │   ├── model.py                ← arsitektur PIDNet-S (3-kelas, ringan)
│   │   └── dataset.py              ← DataLoader image+mask PNG
│   └── geometry/
│       └── distance_estimator.py   ← estimasi jarak tanpa model depth (IMU + trigonometri)
├── runs/                           ← output training (dibuat otomatis)
└── requirements.txt
```

---

## Cara Pakai

### 1. Aktifkan venv

```bash
source .venv/bin/activate
# atau langsung pakai path absolut:
# /home/asadel/kuliah/lomba/smstr6/guido/project/backend/venv/bin/python
```

### 2. Jalankan pipeline satu per satu (rekomendasi)

```bash
# Pre-processing
python scripts/01_convert_voc_to_yolo.py
python scripts/02_convert_coco_pothole_to_yolo.py
python scripts/03_merge_yolo_datasets.py
python scripts/04_convert_coco_segmentation_to_mask.py

# Training (~2-3 jam YOLO, ~3-4 jam PIDNet di RTX 3050 6GB)
python scripts/05_train_yolo.py --epochs 100
python scripts/06_train_pidnet.py --epochs 80

# Export ONNX (khusus PIDNet)
python scripts/07_export_onnx.py --weights runs/pidnet/best.pth
```

### 3. Atau jalankan semuanya sekaligus

```bash
bash scripts/run_all.sh
```

---

## Setelah Training Selesai

### YOLO11n
```bash
# Copy best.pt ke backend
cp runs/yolo/navigasi_v1/weights/best.pt \
   /home/asadel/kuliah/lomba/smstr6/guido/project/backend/models/yolo_navigasi.pt

# Tambahkan ke backend/.env
echo "YOLO_NAVIGASI_MODEL=models/yolo_navigasi.pt" >> ../../../project/backend/.env
```

### PIDNet-S
Script `07_export_onnx.py` sudah otomatis copy ke `backend/models/pidnet_s_navigasi.onnx`.
Pastikan `backend/.env` atau `segmentation_service.py` mengarah ke file ini.

---

## Kelas Target

### YOLO (6 kelas deteksi)
| ID | Nama | Sumber Data |
|----|------|-------------|
| 0 | `lubang` | SafeWalkBD (Pothole), pothole_1, pothole_2 |
| 1 | `got_terbuka` | pothole_2 (manhole) |
| 2 | `tangga` | SafeWalkBD (Stairs, naik+turun digabung) |
| 3 | `orang` | SafeWalkBD (Person) ← **ya, orang terdeteksi** |
| 4 | `motor` | SafeWalkBD (Vehicle) |
| 5 | `tiang` | SafeWalkBD (Pole, Obstacle) |

### PIDNet-S (3 zona segmentasi)
| Nilai Piksel | Nama | Warna TTS |
|---|---|---|
| 0 | `non_walkable` | — |
| 1 | `walkable` | "Jalur aman" |
| 2 | `hazard` | "Hati-hati" |

---

## Catatan

- **"Orang" sudah terdeteksi** dari dataset SafeWalkBD yang punya label `Person`
- **Tidak pakai model depth** — estimasi jarak pakai trigonometri kamera + IMU pitch angle
- **`tangga_naik` dan `tangga_turun` digabung** jadi `tangga` karena SafeWalkBD tidak membedakannya
- Backend sudah diupdate untuk dual-model: pakai YOLO navigasi custom jika tersedia, fallback ke COCO
