"""
Script 05: Training YOLO11n — Deteksi Rintangan (6 Kelas)

Menggunakan ultralytics YOLO API dengan .pt pretrained COCO sebagai base.
Setelah selesai, simpan best.pt ke backend/models/ (TIDAK perlu export ONNX
untuk YOLO — backend sudah pakai Ultralytics langsung).

Target mAP50:
  lubang       >= 0.80
  got_terbuka  >= 0.75
  tangga       >= 0.80
  orang        >= 0.90
  motor        >= 0.85
  tiang        >= 0.80
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Training YOLO11n deteksi rintangan navigasi")
    parser.add_argument("--data",    default=str(Path(__file__).resolve().parents[1] / "configs" / "custom_navigasi.yaml"))
    parser.add_argument("--model",   default="yolo11n.pt", help="Base model pretrained COCO")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1] / "runs" / "yolo"))
    parser.add_argument("--name",    default="navigasi_v1")
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    print(f"[i] Training YOLO11n | data={args.data} | epochs={args.epochs} | batch={args.batch}")
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=30,
        device=0,          # GPU RTX 3050
        # Augmentasi untuk skenario outdoor trotoar
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print("\n=== Hasil Validasi Final ===")
    print(f"mAP50    : {metrics.box.map50:.4f}  (target semua kelas > 0.80)")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print("\nPer-kelas mAP50:")
    for cls_id, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {model.names[int(cls_id)]:<14}: {ap:.4f}")

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[OK] Model terbaik: {best_path}")
    print(f"[→]  Setelah selesai, copy ke backend:")
    print(f"     cp {best_path} /home/asadel/kuliah/lomba/smstr6/guido/project/backend/models/yolo_navigasi.pt")
    print(f"     Lalu update YOLO_NAVIGASI_MODEL di backend/.env")


if __name__ == "__main__":
    main()
