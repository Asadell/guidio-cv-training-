#!/usr/bin/env bash
# ==============================================================================
# GUIDIO — One-Command Pipeline Script for Vast.ai GPU Training
# ==============================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "======================================================================"
echo "  GUIDIO CV Pipeline — Dataset Merge & Training (GPU Vast.ai)"
echo "======================================================================"

# 1. Pull latest code from GitHub
echo -e "\n[1/3] Pulling latest code from GitHub..."
git pull origin main || echo "[!] Warning: Git pull failed or offline"

# 2. Merge all datasets from ~/datasets/ into dataset_master_yolo/
echo -e "\n[2/3] Merging all Kaggle + Roboflow + SafeWalkBD datasets..."
python3 scripts/03_merge_all_datasets.py --base-dir /root/datasets

# 3. Run YOLO11n training with timestamped outputs
EPOCHS="${1:-100}"
echo -e "\n[3/3] Starting YOLO11n Training (${EPOCHS} epochs)..."
python3 scripts/05_train_yolo.py --epochs "$EPOCHS"

echo -e "\n======================================================================"
echo "  [SUCCESS] All pipeline steps completed successfully!"
echo "======================================================================"
