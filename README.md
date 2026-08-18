# 🦮 GUIDIO Computer Vision Training Pipeline

Pipeline training otomatis untuk dua model vision utama pada aplikasi **Vinara (GUIDIO)**:
1. **YOLO11n (Deteksi Rintangan 6-Kelas)**: Deteksi *lubang, got_terbuka, tangga, orang, motor, tiang*. Output disiapkan untuk **Backend (.pt)** dan **Mobile On-Device (.tflite INT8)**.
2. **PIDNet-S (Segmentasi Jalur Trotoar 3-Zona)**: Mewarnai piksel foto secara real-time menjadi *non_walkable (0), walkable (1), hazard (2)*. Output disiapkan untuk **Backend (.onnx)**.

---

## 📚 Dokumentasi Lengkap Dataset yang Digunakan

Seluruh dataset dikumpulkan dan digabungkan dari 4 fase pengumpulan dataset (sesuai dokumentasi proyek `initial_dataset.md`, `new_dataset.md`, `new_dataset2.md`, dan `new_dataset_sidewalk.md`).

---

### 1. Fase 1 — Dataset Awal (Initial Datasets)

| Nama Dataset | Sumber / Link | Jml Gambar | Format Format Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Poth Hole Image Dataset** | [Kaggle - Lokesh97Jain](https://www.kaggle.com/datasets/lokesh97jain/poth-hole-image-dataset) | 1.751 | Custom TXT | YOLO11n | Menyediakan variasi gambar lubang aspal & trotoar dasar. |
| **Annotated Potholes Image Dataset** | [Kaggle - AmeyPatil_07](https://www.kaggle.com/code/ameypatil07/pothole-detection/input) | 665 | Pascal VOC (XML) | YOLO11n | Dataset benchmark lubang beranotasi bounding box akurat. |
| **Sidewalk Segmentation** | [Roboflow Universe](https://universe.roboflow.com/sidewalk/sidewalk-segmentation) | 1.928 | COCO JSON Polygon | PIDNet-S | Baseline awal segmentasi trotoar aman, jalan raya, dan tangga. |
| **SafeWalkBD Dataset** | [Roboflow Universe](https://universe.roboflow.com/safewalkbd/safewalkbd-l8jbn) | 10.241 | YOLOv11 | YOLO11n | Dataset komprehensif rintangan pejalan kaki (orang, kendaraan, tiang, tangga, lubang). |

---

### 2. Fase 2 — Dataset Tambahan Lubang & Tangga (New Potholes & Stairs)

| Nama Dataset | Sumber / Link | Jml Gambar | Format Format Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Potholes-Detection-YOLOv8** | [Kaggle - Angga DwiSunarto](https://www.kaggle.com/datasets/anggadwisunarto/potholes-detection-yolov8) | 1.977 | YOLOv8 | YOLO11n | Menambah variasi lubang kondisi terik, mendung, dan malam hari di trotoar/jalan. |
| **Pothole Dataset v8** | [Kaggle - DenisG04](https://www.kaggle.com/datasets/denisg04/pothle-detect) | 7.751 | YOLOv8 | YOLO11n | Menambah sampel lubang dan permukaan retak dalam skala besar. |
| **Pothole Detection Larxel** | [Kaggle - andrewmvd](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) | 665 | Pascal VOC (XML) | YOLO11n | Memperkuat variasi lubang trotoar perkotaan. |
| **Pothole Detection Dataset** | [Kaggle - abhinavkulshreshth](https://www.kaggle.com/datasets/abhinavkulshreshth/pothole-detection-dataset) | 665 | Pascal VOC (XML) | YOLO11n | Suplementasi data lubang skala sedang-besar. |
| **Annotated Potholes Dataset** | [Kaggle - chitholian](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset) | 665 | Pascal VOC (XML) | YOLO11n | Suplementasi anotasi lubang presisi tinggi. |
| **Pothole Intel Unnati** | [Roboflow - intel-unnati](https://universe.roboflow.com/intel-unnati-training-program/pothole-detection-bqu6s) | 3.753 | YOLOv8 | YOLO11n | Menambah dataset lubang dengan kondisi sudut pandang kamera bervariasi. |
| **Pothole Detection YOLOv5** | [Roboflow - projects-hjaax](https://universe.roboflow.com/projects-hjaax/pothole-detection-using-yolov5) | 665 | YOLOv5 | YOLO11n | Sampel tambahan lubang permukaan jalan. |
| **Road Damage Dataset** | [Roboflow - roaddamage-ak8w6](https://universe.roboflow.com/roaddamage-ak8w6/road-damage-uyvns) | 1.234 | YOLOv8 | YOLO11n | Dataset RDD2022 (pothole, alligator crack, longitudinal damage). |
| **Stairs Detection** | [Roboflow - stair-eyhvv](https://universe.roboflow.com/stair-eyhvv/stairs-detection-6cq2a) | 114 | YOLOv8 | YOLO11n | Menambah sampel tangga outdoor & indoor. |
| **Stairs Data** | [Roboflow - tesisusbbog](https://universe.roboflow.com/tesisusbbog/stairs-data) | 960 | YOLOv8 | YOLO11n | Menambah sampel anak tangga elevasi tinggi. |
| **Stair Detect** | [Roboflow - group10textdetect](https://universe.roboflow.com/group10textdetect/stair-detect) | 223 | YOLOv8 | YOLO11n | Menambah variasi tangga gedung & akses pejalan kaki. |

---

### 3. Fase 3 — Dataset Tambahan Tangga Spesifik (Extra Stairs)

| Nama Dataset | Sumber / Link | Jml Gambar | Format Format Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Stairs Image Dataset** | [Kaggle - DataCluster Labs](https://www.kaggle.com/datasets/dataclusterlabs/stairs-image-dataset) | 3.000+ | Pascal VOC (XML) | YOLO11n | Dataset tangga urban & rural beresolusi tinggi (captured via mobile camera). |
| **Stairs Dataset** | [Kaggle - Samuel Ayman](https://www.kaggle.com/datasets/samuelayman/stairs) | 1.000 | YOLO TXT | YOLO11n | Dataset tangga outdoor & indoor dengan berbagai sudut pandang kamera. |

---

### 4. Fase 4 — Dataset Tambahan Trotoar/Sidewalk (Extra Sidewalk)

| Nama Dataset | Sumber / Link | Jml Gambar | Format Format Asli | Target Model | Alasan Pemilihan |
|---|---|---|---|---|---|
| **Sidewalk Segmentation v4gpn** | [Roboflow - project-nlr2u](https://universe.roboflow.com/project-nlr2u/sidewalk-segmentation-v4gpn) | 1.066 | COCO JSON Polygon | PIDNet-S | Menambah sampel segmentasi trotoar area perumahan & taman. |
| **Sidewalk Semantic dz4ug** | [Roboflow - school-stpl7](https://universe.roboflow.com/school-stpl7/sidewalk-dz4ug) | 1.356 | PNG Mask Semantic | PIDNet-S | Menambah mask trotoar aman (walkable) vs area bukan trotoar. |
| **Sidewalk Object Detection 1smxs** | [Roboflow - project-ii3cz](https://universe.roboflow.com/project-ii3cz/sidewalk-1smxs) | 1.311 | YOLOv8 | YOLO11n | Deteksi koridor jalur trotoar aman sebagai bounding box. |
| **Sidewalk Object Detection eqnxe** | [Roboflow - happy-jmswg](https://universe.roboflow.com/happy-jmswg/sidewalk-eqnxe) | 241 | YOLOv8 | YOLO11n | Suplementasi deteksi batas trotoar perkotaan. |
| **Sidewalk Semantic zhrul** | [Roboflow - dika-biyq4](https://universe.roboflow.com/dika-biyq4/sidewalk-zhrul) | 1.102 | PNG Mask Semantic | PIDNet-S | Menambah sampel trotoar Indonesia dengan kondisi permukaan semen & paving block. |

---

## 📊 Rekapitulasi Total Master Dataset

Setelah dijalankan melalui script penggabungan otomatis (`03_merge_all_datasets.py` & `04_convert_extra_seg.py`), data berhasil terkonsolidasi menjadi dua master dataset:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Master Dataset YOLO (dataset_master_yolo/)                               │
│    - Total Gambar : 19.650 Foto Beranotasi                                  │
│    - Kelas (6)    : 0: lubang, 1: got_terbuka, 2: tangga,                   │
│                     3: orang, 4: motor, 5: tiang                            │
│                                                                             │
│ 2. Master Dataset PIDNet Segmentasi (dataset_master_seg/)                   │
│    - Total Gambar : 5.452 Pasangan Foto (.jpg) + Mask 8-bit (.png)          │
│    - Kelas (3)    : 0: non_walkable, 1: walkable, 2: hazard                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Analisis Kode & Strategi Penanganan Model

### 1. Augmentasi Kontekstual Berjalan Kaki (YOLO11n)
- **`flipud = 0.0` (Wajib)**: Vertical flip dimatikan 100% karena membalikkan gambar tangga secara vertikal dapat merusak makna semantik (tangga naik terlihat seperti kebalik/tangga turun).
- **`degrees = 10.0`**: Rotasi dibatasi ±10° untuk meniru goyangan tubuh saat pengguna berjalan dengan kamera di dada/kepala.
- **`hsv_s = 0.5, hsv_v = 0.4`**: Variasi saturasi dan brightness untuk menangani cuaca terik, mendung, dan bayangan pohon di trotoar Indonesia.

### 2. Penanganan Imbalance & Hazard Sensitivity (PIDNet-S)
- **Class Weighting `[1.0, 1.0, 3.0]`**: Di `06_train_pidnet.py`, kelas `hazard` (tangga/lubang) diberi bobot loss **3x lebih besar** daripada trotoar polos agar model sangat sensitif dalam memperingatkan bahaya.
- **OHEM (Online Hard Example Mining)**: Memaksa model fokus mempelajari piksel-piksel batas antara trotoar dan jalan raya/selokan yang paling sering membingungkan.

---

## 🚀 Cara Menjalankan Pipeline di GPU Server

```bash
cd ~/guidio-cv-training-

# 1. Pull update kode terbaru
git pull origin main

# 2. Jalankan seluruh pipeline (Merger + YOLO Training + PIDNet Training + Export)
./scripts/run_vast.sh 100 80
```
