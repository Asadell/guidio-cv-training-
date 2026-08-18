# 🦮 GUIDIO Computer Vision Training Pipeline

Pipeline training otomatis untuk dua model vision utama pada aplikasi **Vinara (GUIDIO)**:
1. **YOLO11n (Deteksi Rintangan 6-Kelas)**: Deteksi *lubang, got_terbuka, tangga, orang, motor, tiang*. Output disiapkan untuk **Backend (.pt)** dan **Mobile On-Device (.tflite INT8)**.
2. **PIDNet-S (Segmentasi Jalur Trotoar 3-Zona)**: Mewarnai piksel foto secara real-time menjadi *non_walkable (0), walkable (1), hazard (2)*. Output disiapkan untuk **Backend (.onnx)** dan **Mobile On-Device (.tflite FP16)**.

---

## 🛠️ Step-by-Step Alur & Tahapan Training

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               GUIDIO VISION PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────────────┘
   │
   ├─► STEP 1: Consolidate & Standardize Datasets (03_merge_all_datasets.py)
   │   └─ Convert VOC XML / COCO JSON / YOLO format -> Master Dataset (6 classes)
   │
   ├─► STEP 2: Configure Physical Context Augmentation (05_train_yolo.py)
   │   └─ flipud=0.0 (strictly no upside down), degrees=10.0 (chest camera wobble)
   │
   ├─► STEP 3: Train Ultralytics YOLO11n on NVIDIA GPU (100 Epochs)
   │   └─ Real-time loss logging, validation, & best weight saving (`best.pt`)
   │
   ├─► STEP 4: Dual Export Strategy YOLO11n (Backend & Mobile)
   │   ├─► PyTorch Model (`.pt`)  --> FastAPI Backend Server (High Precision Cloud)
   │   └─► TFLite Model (`.tflite INT8 ~4-8MB`) --> Flutter App (On-Device Offline)
   │
   └─► STEP 5: Train PIDNet-S Sidewalk Segmentation (06_train_pidnet.py - 80 Epochs)
       ├─► Class Weighting `[1.0, 1.0, 3.0]` (Hazard awareness 3x)
       ├─► ONNX Model (`pidnet_s.onnx`) --> FastAPI Backend Server
       └─► TFLite Model (`pidnet_s.tflite FP16 ~7-12MB`) --> Flutter App (On-Device Offline)
```

### Detail Penjelasan Tiap Step:

1. **Step 1 — Konsolidasi & Merger Dataset**:
   - Membaca puluhan folder dataset dari Kaggle & Roboflow.
   - Melakukan parsing otomatis untuk bounding box format Pascal VOC XML (`.xml`), YOLO text (`.txt`), maupun COCO JSON (`.json`).
   - Normalisasi dan konsolidasi kelas menjadi 6 kelas baku: `0: lubang`, `1: got_terbuka`, `2: tangga`, `3: orang`, `4: motor`, `5: tiang`.
   - Menggunakan hardlink/symlink OS untuk menghemat ruang disk GPU.

2. **Step 2 — Augmentasi Khusus Kamera Dada/Kepala (Walking Context)**:
   - **`flipud = 0.0` (Dilarang Vertical Flip)**: Tangga dan jalan memiliki orientasi vertikal sakral. Tangga terbalik akan membingungkan model.
   - **`degrees = 10.0`**: Rotasi dibatasi ±10° untuk meniru guncangan badan pejalan kaki.
   - **`hsv_h = 0.015, hsv_s = 0.5, hsv_v = 0.4`**: Mensimulasikan kondisi pencahayaan Indonesia (terik, mendung, dan bayangan pohon).
   - **`scale = 0.3` & `mosaic = 1.0`**: Mensimulasikan rintangan jarak jauh dan dekat secara bersamaan.

3. **Step 3 — High-Speed GPU Training**:
   - Pelatihan menggunakan PyTorch + CUDA pada GPU server (RTX 5090 / RTX 3050).
   - 100 Epochs diselesaikan untuk 18.098 data YOLO.
   - Evaluasi mAP50 dan Recall dilakukan di akhir tiap epoch menggunakan split validasi (`images/valid`).

4. **Step 4 — Strategi Dual Export Model YOLO11n**:
   - **Backend Server (`.pt`)**: File bobot PyTorch asli untuk inference berkecepatan tinggi di server cloud FastAPI.
   - **Mobile Flutter (`.tflite`)**: Model diexport dan di-quantized ke **TFLite INT8** (ukuran ~4-8 MB). Model ini bisa di-load via package `tflite_flutter` untuk deteksi **offline tanpa sinyal internet** di HP tunanetra.

5. **Step 5 — PIDNet-S Sidewalk Segmentation Training & Dual Export**:
   - **Class Weighting `[1.0, 1.0, 3.0]`**: Kelas `hazard` (tangga/lubang) diberi bobot 3x lebih besar dari trotoar biasa agar model sangat peka terhadap bahaya.
   - **Pembaruan & Perbaikan Script (`06_train_pidnet.py`)**:
     - *Auto-detect Path GPU*: Secara otomatis membaca `/root/datasets/dataset_master_seg` saat running di GPU Vast.ai, dengan fallback ke folder lokal.
     - *Fix ONNX Export Trace Bug*: Menggunakan `_OnnxWrapper` khusus agar model mengembalikan tensor tunggal `logits` saat di-trace (menghindari error tuple `(logits, aux)`).
     - *Dual Export (Backend + Mobile)*: Menghasilkan `pidnet_s.onnx` untuk backend dan `pidnet_s.tflite` (FP16 ~7-12MB) untuk Flutter on-device.
     - *Validation Clean Architecture*: Refactoring `run_epoch` dengan flag `is_train` yang jelas dan penanganan `vl_iou` aman tanpa `NameError`.

---

## 📚 Dokumentasi Lengkap Dataset yang Digunakan

Seluruh dataset dikumpulkan dan digabungkan dari 4 fase pengumpulan dataset (sesuai dokumentasi proyek `initial_dataset.md`, `new_dataset.md`, `new_dataset2.md`, dan `new_dataset_sidewalk.md`).

---

### 1. Fase 1 — Dataset Awal (Initial Datasets)

| Nama Dataset | Sumber / Link | Jml Gambar | Ekstensi Gambar & Anotasi Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Poth Hole Image Dataset** | [Kaggle - Lokesh97Jain](https://www.kaggle.com/datasets/lokesh97jain/poth-hole-image-dataset) | 1.751 | `.jpg` / Custom `.txt` | YOLO11n | Menyediakan variasi gambar lubang aspal & trotoar dasar. |
| **Annotated Potholes Image Dataset** | [Kaggle - AmeyPatil_07](https://www.kaggle.com/code/ameypatil07/pothole-detection/input) | 665 | `.png`, `.jpg` / Pascal VOC (`.xml`) | YOLO11n | Dataset benchmark lubang beranotasi bounding box akurat. |
| **Sidewalk Segmentation** | [Roboflow Universe](https://universe.roboflow.com/sidewalk/sidewalk-segmentation) | 1.928 | `.jpg` / COCO Polygon (`.json`) | PIDNet-S | Baseline awal segmentasi trotoar aman, jalan raya, dan tangga. |
| **SafeWalkBD Dataset** | [Roboflow Universe](https://universe.roboflow.com/safewalkbd/safewalkbd-l8jbn) | 10.241 | `.jpg` / YOLO (`.txt`) | YOLO11n | Dataset komprehensif rintangan pejalan kaki (orang, kendaraan, tiang, tangga, lubang). |

---

### 2. Fase 2 — Dataset Tambahan Lubang & Tangga (New Potholes & Stairs)

| Nama Dataset | Sumber / Link | Jml Gambar | Ekstensi Gambar & Anotasi Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Potholes-Detection-YOLOv8** | [Kaggle - Angga DwiSunarto](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8) | 1.977 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Menambah variasi lubang kondisi terik, mendung, dan malam hari di trotoar/jalan. |
| **Pothole Dataset v8** | [Kaggle - DenisG04](https://www.kaggle.com/datasets/denisg04/pothle-detect) | 7.751 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Menambah sampel lubang dan permukaan retak dalam skala besar. |
| **Pothole Detection Larxel** | [Kaggle - andrewmvd](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) | 665 | `.png` / Pascal VOC (`.xml`) | YOLO11n | Memperkuat variasi lubang trotoar perkotaan. |
| **Pothole Detection Dataset** | [Kaggle - abhinavkulshreshth](https://www.kaggle.com/datasets/abhinavkulshreshth/pothole-detection-dataset) | 665 | `.jpg` / Pascal VOC (`.xml`) | YOLO11n | Suplementasi data lubang skala sedang-besar. |
| **Annotated Potholes Dataset** | [Kaggle - chitholian](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset) | 665 | `.jpg`, `.png` / Pascal VOC (`.xml`) | YOLO11n | Suplementasi anotasi lubang presisi tinggi. |
| **Pothole Intel Unnati** | [Roboflow - intel-unnati](https://universe.roboflow.com/intel-unnati-training-program/pothole-detection-bqu6s) | 3.753 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Menambah dataset lubang dengan kondisi sudut pandang kamera bervariasi. |
| **Pothole Detection YOLOv5** | [Roboflow - projects-hjaax](https://universe.roboflow.com/projects-hjaax/pothole-detection-using-yolov5) | 665 | `.jpg` / YOLOv5 (`.txt`) | YOLO11n | Sampel tambahan lubang permukaan jalan. |
| **Road Damage Dataset** | [Roboflow - roaddamage-ak8w6](https://universe.roboflow.com/roaddamage-ak8w6/road-damage-uyvns) | 1.234 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Dataset RDD2022 (pothole, alligator crack, longitudinal damage). |
| **Stairs Detection** | [Roboflow - stair-eyhvv](https://universe.roboflow.com/stair-eyhvv/stairs-detection-6cq2a) | 114 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Menambah sampel tangga outdoor & indoor. |
| **Stairs Data** | [Roboflow - tesisusbbog](https://universe.roboflow.com/tesisusbbog/stairs-data) | 960 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Menambah sampel anak tangga elevasi tinggi. |
| **Stair Detect** | [Roboflow - group10textdetect](https://universe.roboflow.com/group10textdetect/stair-detect) | 223 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Menambah variasi tangga gedung & akses pejalan kaki. |

---

### 3. Fase 3 — Dataset Tambahan Tangga Spesifik (Extra Stairs)

| Nama Dataset | Sumber / Link | Jml Gambar | Ekstensi Gambar & Anotasi Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Stairs Image Dataset** | [Kaggle - DataCluster Labs](https://www.kaggle.com/datasets/dataclusterlabs/stairs-image-dataset) | 3.000+ | `.jpg` / Pascal VOC (`.xml`) | YOLO11n | Dataset tangga urban & rural beresolusi tinggi (captured via mobile camera). |
| **Stairs Dataset** | [Kaggle - Samuel Ayman](https://www.kaggle.com/datasets/samuelayman/stairs) | 1.000 | `.jpg` / YOLO (`.txt`) | YOLO11n | Dataset tangga outdoor & indoor dengan berbagai sudut pandang kamera. |

---

### 4. Fase 4 — Dataset Tambahan Trotoar/Sidewalk (Extra Sidewalk)

| Nama Dataset | Sumber / Link | Jml Gambar | Ekstensi Gambar & Anotasi Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Sidewalk Segmentation v4gpn** | [Roboflow - project-nlr2u](https://universe.roboflow.com/project-nlr2u/sidewalk-segmentation-v4gpn) | 1.066 | `.jpg` / COCO JSON (`.json`) | PIDNet-S | Menambah sampel segmentasi trotoar area perumahan & taman. |
| **Sidewalk Semantic dz4ug** | [Roboflow - school-stpl7](https://universe.roboflow.com/school-stpl7/sidewalk-dz4ug) | 1.356 | `.png` / Semantic Mask (`.png`) | PIDNet-S | Menambah mask trotoar aman (walkable) vs area bukan trotoar. |
| **Sidewalk Object Detection 1smxs** | [Roboflow - project-ii3cz](https://universe.roboflow.com/project-ii3cz/sidewalk-1smxs) | 1.311 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Deteksi koridor jalur trotoar aman sebagai bounding box. |
| **Sidewalk Object Detection eqnxe** | [Roboflow - happy-jmswg](https://universe.roboflow.com/happy-jmswg/sidewalk-eqnxe) | 241 | `.jpg` / YOLOv8 (`.txt`) | YOLO11n | Suplementasi deteksi batas trotoar perkotaan. |
| **Sidewalk Semantic zhrul** | [Roboflow - dika-biyq4](https://universe.roboflow.com/dika-biyq4/sidewalk-zhrul) | 1.102 | `.png` / Semantic Mask (`.png`) | PIDNet-S | Menambah sampel trotoar Indonesia dengan kondisi permukaan semen & paving block. |

---

## 📊 Rekapitulasi Real-Time Master Dataset di GPU

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Master Dataset YOLO (dataset_master_yolo/)                               │
│    - Total Gambar : 18.098 Foto & Label (100% Terverifikasi)               │
│    - Rincian Split:                                                         │
│      * Train : 14.475 gambar (.jpg / .png) & labels (.txt)                   │
│      * Valid :  1.806 gambar (.jpg / .png) & labels (.txt)                   │
│      * Test  :  1.817 gambar (.jpg / .png) & labels (.txt)                   │
│    - Kelas (6)    : 0: lubang, 1: got_terbuka, 2: tangga,                   │
│                     3: orang, 4: motor, 5: tiang                            │
│                                                                             │
│ 2. Master Dataset PIDNet Segmentasi (dataset_master_seg/)                   │
│    - Total Gambar : 5.452 Pasangan Foto (.jpg) + Mask 8-bit (.png)          │
│    - Kelas (3)    : 0: non_walkable, 1: walkable, 2: hazard                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Cara Menjalankan & Memantau Pipeline di GPU Server

```bash
cd /workspace/guidio-cv-training-

# 1. Pull update kode & dokumentasi terbaru
git pull origin main

# 2. Jalankan master pipeline script (atau 05_train_yolo.py secara independen)
python3 scripts/05_train_yolo.py --epochs 100

# 3. Pantau log training real-time
tail -f train_yolo_v3.log
```
