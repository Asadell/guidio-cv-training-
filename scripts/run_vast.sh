#!/usr/bin/env bash
# ==============================================================================
# GUIDIO — Complete GPU Pipeline (YOLO11n + PIDNet-S Sidewalk Segmentation)
# ==============================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="/venv/main/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "  GUIDIO Complete Pipeline (YOLO11n Obstacles + PIDNet-S Sidewalk)"
echo "  Python Executable: $PYTHON_BIN"
echo "======================================================================"

# 1. Merge YOLO datasets (Potholes + Stairs + Sidewalk BBoxes) -> dataset_master_yolo
echo -e "\n[1/4] Merging all YOLO Datasets..."
"$PYTHON_BIN" scripts/03_merge_all_datasets.py --base-dir /root/datasets

# 2. Merge PIDNet Segmentation Datasets -> dataset_master_seg
echo -e "\n[2/4] Converting & Merging Sidewalk Segmentation Datasets..."
"$PYTHON_BIN" scripts/04_convert_extra_seg.py --base-dir /root/datasets/sidewalk-extra

# 3. Train YOLO11n (100 Epochs) -> Output .pt + .tflite
EPOCHS_YOLO="${1:-100}"
echo -e "\n[3/4] Training YOLO11n Obstacle Detection (${EPOCHS_YOLO} epochs)..."
"$PYTHON_BIN" scripts/05_train_yolo.py --epochs "$EPOCHS_YOLO"

# 4. Train PIDNet-S (80 Epochs) -> Output .pth + .onnx
EPOCHS_SEG="${2:-80}"
echo -e "\n[4/4] Training PIDNet-S Sidewalk 3-Zone Segmentation (${EPOCHS_SEG} epochs)..."
"$PYTHON_BIN" scripts/06_train_pidnet.py --epochs "$EPOCHS_SEG" --dataset-root /root/datasets/dataset_master_seg

echo -e "\n======================================================================"
echo "  [SUCCESS] All Training & Export Tasks Finished Successfully!"
echo "======================================================================"
