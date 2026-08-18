"""
Semua mapping kelas dari dataset publik -> skema kelas target GUIDIO.

PERUBAHAN dari draft awal:
  - tangga_naik + tangga_turun DIGABUNG jadi satu kelas 'tangga' (ID 2)
  - Alasan: SafeWalkBD tidak membedakan naik/turun pada kelas 'Stairs',
    sehingga 'tangga_turun' akan punya 0 sampel training — model tidak akan
    bisa belajar kelas itu. Lebih baik 1 kelas solid daripada 2 kelas di mana
    1-nya kosong. TTS cukup bilang: "Ada tangga di depan."

Dua target terpisah:
  1. TARGET_YOLO_CLASSES  -> untuk YOLO11n (deteksi rintangan spesifik, 6 kelas)
  2. TARGET_SEG_CLASSES   -> untuk PIDNet-S (segmentasi jalur 3-zona)
"""

# ---------------------------------------------------------------------------
# 1) SKEMA KELAS TARGET — MODEL DETEKSI (YOLO11n) — 6 KELAS
# ---------------------------------------------------------------------------
TARGET_YOLO_CLASSES = [
    "lubang",       # 0 — pothole / lubang jalan / trotoar rusak
    "got_terbuka",  # 1 — manhole tanpa tutup / selokan terbuka
    "tangga",       # 2 — tangga (naik atau turun, tidak dibedakan)
    "orang",        # 3 — person / pedestrian
    "motor",        # 4 — kendaraan bermotor (motor, mobil, bus, truk)
    "tiang",        # 5 — pole / obstacle / tiang listrik / gerobak
]
YOLO_NAME_TO_ID = {name: idx for idx, name in enumerate(TARGET_YOLO_CLASSES)}

# --- Mapping dari SafeWalkBD.v1i.yolov11 (16 kelas asli) ------------------
SAFEWALKBD_TO_TARGET = {
    "Pothole":  "lubang",
    "Stairs":   "tangga",   # naik/turun digabung — lihat komentar di atas
    "Person":   "orang",
    "Pole":     "tiang",
    "Obstacle": "tiang",
    "Vehicle":  "motor",
}

# --- Mapping dari dataset Pothole publik (Roboflow & Kaggle) ---------------
POTHOLE_CLASS_MAP = {
    "pothole": "lubang",
    "potholes": "lubang",
    "hole": "lubang",
    "manhole": "got_terbuka",
    "drain": "got_terbuka",
    "got_terbuka": "got_terbuka",
    "open_manhole": "got_terbuka",
    "crack": "lubang",
    "D40": "lubang",         # RDD2022 pothole code
    "D00": "lubang",         # Longitudinal crack / hazard
    "D10": "lubang",         # Transverse crack / hazard
    "D20": "lubang",         # Alligator crack / hazard
}

# --- Mapping dari dataset Tangga / Stairs publik --------------------------
STAIRS_CLASS_MAP = {
    "stair": "tangga",
    "stairs": "tangga",
    "stairs_up": "tangga",
    "stairs_down": "tangga",
    "stair_up": "tangga",
    "stair_down": "tangga",
    "upstairs": "tangga",
    "downstairs": "tangga",
    "step": "tangga",
    "steps": "tangga",
    "stairway": "tangga",
    "ladder": "tangga",
}

# ---------------------------------------------------------------------------
# 2) SKEMA KELAS TARGET — MODEL SEGMENTASI (PIDNet-S) — 3 KELAS
# ---------------------------------------------------------------------------
SEG_NON_WALKABLE = 0  # Jalan raya, rumput, dinding, selokan
SEG_WALKABLE = 1      # Trotoar, tactile block, lantai aman
SEG_HAZARD = 2        # Tangga, lubang (area bertekstur berbahaya)

TARGET_SEG_CLASSES = {
    SEG_NON_WALKABLE: "non_walkable",
    SEG_WALKABLE:     "walkable",
    SEG_HAZARD:       "hazard",
}

# --- Mapping dari "Sidewalk Segmentation.v1i.coco-segmentation" -----------
# Case-insensitive karena Roboflow suka variasi kapitalisasi antar split.
SIDEWALK_SEG_TO_TARGET = {
    "Roadway":    SEG_NON_WALKABLE,
    "roadway":    SEG_NON_WALKABLE,
    "Sidewalks":  SEG_WALKABLE,
    "sidewalks":  SEG_WALKABLE,
    "Sidewalk":   SEG_WALKABLE,
    "sidewalk":   SEG_WALKABLE,
    "path":       SEG_WALKABLE,
    "Path":       SEG_WALKABLE,
    "upstairs":   SEG_HAZARD,
    "Upstairs":   SEG_HAZARD,
    "downstairs": SEG_HAZARD,
    "Downstairs": SEG_HAZARD,
}
