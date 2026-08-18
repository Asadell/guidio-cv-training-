"""
Script 05 (Balanced Fine-Tuning): Training YOLO dengan Class Loss Weighting & Multi-Model (Backend & Mobile)

Meningkatkan akurasi kelas yang lebih langka (tangga, tiang, orang, motor)
dengan meningkatkan bobot loss klasifikasi (cls=2.0) serta mendukung:
- Fine-tuning model Nano (yolo11n) untuk Mobile Flutter
- Training model Small (yolo11s) untuk FastAPI Backend Server
"""
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Training YOLO11 Balanced (Backend & Mobile)")
    parser.add_argument("--data",    default=str(Path(__file__).resolve().parents[1] / "configs" / "custom_navigasi.yaml"))
    parser.add_argument("--model",   default="yolo11s.pt", help="Base model (yolo11s.pt untuk Backend, atau path ke best.pt untuk fine-tune Mobile)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=32, help="Batch size (RTX 5090 32GB muat batch 32-64)")
    parser.add_argument("--cls-weight", type=float, default=2.0, help="Bobot loss klasifikasi untuk mendorong pembelajaran kelas langka")
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1] / "runs" / "yolo"))
    parser.add_argument("--name",    default=None, help="Nama eksperimen")
    parser.add_argument("--export-tflite", action="store_true", help="Export LiteRT INT8 setelah training")
    args = parser.parse_args()

    # Auto name
    if not args.name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_stem = Path(args.model).stem
        args.name = f"navigasi_balanced_{model_stem}_e{args.epochs}_{timestamp}"

    print("=" * 65)
    print("  GUIDIO — Balanced YOLO Training (Fine-Tuned Class Weights)")
    print(f"  Data Config  : {args.data}")
    print(f"  Base Model   : {args.model}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch Size   : {args.batch}")
    print(f"  Cls Weight   : {args.cls_weight}")
    print(f"  Run Name     : {args.name}")
    print("=" * 65)

    model = YOLO(args.model)

    # Train dengan bobot klasifikasi lebih tinggi & augmentasi seimbang
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=30,
        device=0,
        
        # ── Penyesuaian Loss Function ───────────────────────────────────────
        cls=args.cls_weight, # Tingkatkan bobot loss klasifikasi (default 0.5 -> 2.0)
        box=7.5,             # Bobot loss bounding box
        dfl=1.5,             # Distribution Focal Loss
        lr0=0.003,           # Learning rate awal yang stabil
        lrf=0.01,
        
        # ── Augmentasi Khusus Trotoar & Tangga ─────────────────────────────
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.4,           # Naikkan skala variasi objek dekat/jauh
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,          # JANGAN vertical flip (tangga tidak boleh balik)
        mosaic=1.0,
        mixup=0.1,           # Tambahkan mixup ringan untuk melatih fitur campuran
        copy_paste=0.1,      # Copy paste objek antar gambar
        erasing=0.1,
    )

    # Validasi
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print("\n=== Hasil Validasi Final ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap50'):
        print("\nPer-kelas mAP50:")
        for cls_id, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            print(f"  {model.names[int(cls_id)]:<14}: {ap:.4f}")

    best_pt = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[OK] Model PyTorch terbaik disimpan di: {best_pt}")

    # Export LiteRT jika diminta
    if args.export_tflite:
        print("\n[→] Meng-export model ke LiteRT INT8...")
        try:
            litert_model = YOLO(str(best_pt))
            litert_path  = litert_model.export(
                format="litert",
                quantize=True,
                data=args.data,
                imgsz=args.imgsz,
            )
            print(f"[OK] LiteRT INT8 berhasil diexport ke: {litert_path}")
        except Exception as e:
            print(f"[!] Gagal export LiteRT: {e}")

    print("\n" + "=" * 65)
    print(f"  [FINISHED] Training selesai! Weight disimpan di {best_pt}")
    print("=" * 65)


if __name__ == "__main__":
    main()
