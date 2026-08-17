#!/usr/bin/env bash
# run_all.sh — Jalankan seluruh pipeline training GUIDIO navigation model
# Pastikan .venv sudah aktif sebelum menjalankan script ini:
#   source .venv/bin/activate

set -e  # hentikan jika ada error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_ROOT="/home/asadel/kuliah/lomba/smstr6/guido/dataset_sidewalk"
BACKEND_DIR="/home/asadel/kuliah/lomba/smstr6/guido/project/backend"

echo "=================================================="
echo " GUIDIO CV Training Pipeline"
echo " Dataset root : $DATASET_ROOT"
echo " Project dir  : $PROJECT_DIR"
echo "=================================================="
echo ""

# Pastikan venv aktif
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "[!] Virtual environment belum aktif."
    echo "    Jalankan: source $PROJECT_DIR/.venv/bin/activate"
    exit 1
fi

echo "[1/7] Konversi pothole_1 (VOC XML -> YOLO TXT)..."
python "$SCRIPT_DIR/01_convert_voc_to_yolo.py" --dataset-root "$DATASET_ROOT"
echo ""

echo "[2/7] Konversi pothole_2 (COCO JSON -> YOLO TXT)..."
python "$SCRIPT_DIR/02_convert_coco_pothole_to_yolo.py" --dataset-root "$DATASET_ROOT"
echo ""

echo "[3/7] Merge SafeWalkBD + pothole_1 + pothole_2 -> dataset_master_yolo/..."
python "$SCRIPT_DIR/03_merge_yolo_datasets.py" --dataset-root "$DATASET_ROOT"
echo ""

echo "[4/7] Konversi Sidewalk Segmentation COCO -> PNG mask 3-kelas..."
python "$SCRIPT_DIR/04_convert_coco_segmentation_to_mask.py" --dataset-root "$DATASET_ROOT"
echo ""

echo "[5/7] Training YOLO11n (deteksi rintangan)..."
echo "      Estimasi: ~2-3 jam di RTX 3050 6GB"
python "$SCRIPT_DIR/05_train_yolo.py" \
    --epochs 100 --imgsz 640 --batch 16
echo ""

echo "[6/7] Training PIDNet-S (segmentasi jalur 3-zona)..."
echo "      Estimasi: ~3-4 jam di RTX 3050 6GB"
python "$SCRIPT_DIR/06_train_pidnet.py" \
    --dataset-root "$DATASET_ROOT/dataset_master_seg" \
    --epochs 80 --batch-size 8
echo ""

echo "[7/7] Export PIDNet-S -> ONNX -> copy ke backend/models/..."
BEST_PTH="$PROJECT_DIR/runs/pidnet/best.pth"
if [[ -f "$BEST_PTH" ]]; then
    python "$SCRIPT_DIR/07_export_onnx.py" \
        --weights "$BEST_PTH" \
        --out "$PROJECT_DIR/runs/pidnet_s_navigasi.onnx"
else
    echo "[!] best.pth tidak ditemukan di $BEST_PTH — skip export ONNX."
fi
echo ""

echo "=================================================="
echo " Selesai! Langkah selanjutnya:"
echo ""
echo " 1. Copy YOLO best.pt ke backend:"
echo "    cp $PROJECT_DIR/runs/yolo/navigasi_v1/weights/best.pt \\"
echo "       $BACKEND_DIR/models/yolo_navigasi.pt"
echo ""
echo " 2. Update $BACKEND_DIR/.env:"
echo "    YOLO_NAVIGASI_MODEL=models/yolo_navigasi.pt"
echo "    SEGMENTATION_MODEL=models/pidnet_s_navigasi.onnx"
echo ""
echo " 3. Restart backend:"
echo "    cd $BACKEND_DIR && uvicorn main:app --reload"
echo "=================================================="
