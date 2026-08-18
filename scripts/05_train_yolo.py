"""
Script 05: Training YOLO11n — Deteksi Rintangan (6 Kelas)

Menjalankan training Ultralytics YOLO11n dengan augmentasi khusus
untuk skenario kamera dada/kepala pengguna tunanetra.

Aturan Augmentasi Penting:
1. flipud = 0.0 (TIDAK BOLEH vertical flip — merusak pola/arti tangga)
2. degrees = 10.0 (Rotasi halus ±10° untuk simulasi langkah kaki, bukan 90°/180°)
3. fliplr = 0.5 (Horizontal flip aman untuk pothole & tangga)
4. hsv_s = 0.5, hsv_v = 0.4 (Simulasi cuaca terik vs mendung vs sore di Indonesia)
5. erasing = 0.1 (Cutout ringan untuk objek sebagian tertutup)

Output:
- PyTorch .pt (untuk FastAPI Backend)
- TFLite INT8 (untuk Flutter On-Device Offline Inference)
- Run name otomatis menggunakan timestamp & total epochs agar tidak pernah menimpa hasil sebelumnya.
"""
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Training YOLO11n deteksi rintangan navigasi")
    parser.add_argument("--data",    default=str(Path(__file__).resolve().parents[1] / "configs" / "custom_navigasi.yaml"))
    parser.add_argument("--model",   default="yolo11n.pt", help="Base model pretrained COCO")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1] / "runs" / "yolo"))
    parser.add_argument("--name",    default=None, help="Nama eksperimen (default: navigasi_yolo11n_e{epochs}_{timestamp})")
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    # Generate timestamped name if not provided
    if not args.name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"navigasi_yolo11n_e{args.epochs}_{timestamp}"

    print("=" * 65)
    print("  GUIDIO — Training YOLO11n Navigasi (Backend & On-Device)")
    print(f"  Data Config  : {args.data}")
    print(f"  Base Model   : {args.model}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Run Name     : {args.name}")
    print("=" * 65)

    model = YOLO(args.model)

    # Train dengan augmentasi kontekstual
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=30,
        device=0,
        
        # ── Augmentasi Khusus Trotoar & Tangga ─────────────────────────────
        hsv_h=0.015,     # Sensitivitas warna ringan
        hsv_s=0.5,       # Variasi saturasi (pencahayaan)
        hsv_v=0.4,       # Variasi brightness (terik/sore/bayangan pohon)
        degrees=10.0,    # Rotasi halus ±10° (goyangan jalan kaki)
        translate=0.1,   # Geser ringan
        scale=0.3,       # Scale objek
        perspective=0.0005, # Sudut elevasi pandang
        fliplr=0.5,      # Horizontal flip (50%)
        flipud=0.0,      # CRITICAL: 0.0 (JANGAN vertical flip agar tangga tidak kebalik!)
        mosaic=1.0,      # Mosaic augmentasi
        erasing=0.1,     # Cutout/erasing ringan (simulasi daun/bayangan menutupi lubang)
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

    # Export ke LiteRT (TFLite) INT8 untuk On-Device Mobile (Flutter)
    #
    # FIX dari run sebelumnya:
    # - format="tflite" → deprecated sejak Ultralytics 8.4.83, ganti ke format="litert"
    # - int8=True       → deprecated, ganti ke quantize=True
    # - data=args.data  → pakai dataset kustom kita (bukan coco8.yaml default)
    #   agar kalibrasi INT8 representatif (minimal 300+ gambar dari domain trotoar)
    print("\n[→] Meng-export model ke LiteRT INT8 (untuk Flutter mobile offline)...")
    try:
        litert_model = YOLO(str(best_pt))
        litert_path  = litert_model.export(
            format="litert",       # menggantikan format="tflite" yang deprecated
            quantize=True,         # menggantikan int8=True yang deprecated
            data=args.data,        # dataset kustom kita untuk kalibrasi INT8 (bukan coco8)
            imgsz=args.imgsz,
        )
        print(f"[OK] Model LiteRT INT8 berhasil diexport ke: {litert_path}")
    except Exception as e:
        print(f"[!] Gagal export LiteRT otomatis: {e}")
        print("    Kamu bisa jalankan manual:")
        print(f"    yolo export model={best_pt} format=litert quantize=True data={args.data}")

    print("\n" + "=" * 65)
    print("  [FINISHED] Model siap digunakan:")
    print(f"  Backend (.pt)      : {best_pt}")
    print(f"  Mobile  (.tflite)  : hasil export litert di folder weights/")
    print("=" * 65)


if __name__ == "__main__":
    main()
