"""
Script 06: Training PIDNet-S — Segmentasi Jalur 3-Zona (Trotoar, Non-Walkable, Hazard)

Pembaruan Penting:
1. Class Weighting [1.0, 1.0, 3.0] pada Loss Function agar model 3x lebih sensitif
   mendeteksi area hazard (tangga/lubang) daripada sekadar trotoar polos.
2. Synchronized Albumentations (Flip, Lighting Jitter, Motion Blur) untuk memperbanyak
   variasi kondisi trotoar Indonesia secara otomatis di setiap epoch.
3. Dual Export otomatis setelah training:
   - ONNX  → untuk FastAPI Backend Server
   - TFLite → untuk Flutter On-Device Offline Inference

Target metrik:
  mIoU walkable      >= 0.78
  walkable-safety error (non_walkable salah jadi walkable) <= 5%
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pidnet.model   import PIDNetS
from src.pidnet.dataset import SidewalkSegDataset

NUM_CLASSES = 3
CLASS_NAMES = ["non_walkable", "walkable", "hazard"]

# Path dataset default:
# - GPU (Vast.ai / training server) : /root/datasets/dataset_master_seg
# - Lokal (fallback)                : dataset_sidewalk/dataset_master_seg
_GPU_DATASET_ROOT   = Path("/root/datasets/dataset_master_seg")
_LOCAL_DATASET_ROOT = Path(__file__).resolve().parents[2] / "dataset_master_seg"
DEFAULT_DATASET_ROOT = str(
    _GPU_DATASET_ROOT if _GPU_DATASET_ROOT.exists() else _LOCAL_DATASET_ROOT
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion(pred, target, nc):
    mask = (target >= 0) & (target < nc)
    idx  = nc * target[mask] + pred[mask]
    return torch.bincount(idx, minlength=nc * nc).reshape(nc, nc)


def iou_from_conf(conf):
    inter = torch.diag(conf).float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    return inter / union.clamp(min=1)


def walkable_safety_error(conf):
    """% piksel non_walkable (kelas 0) yang salah prediksi sebagai walkable (kelas 1)."""
    total_non_walk = conf[0].sum().float()
    wrong_as_walk  = conf[0, 1].float()
    if total_non_walk == 0:
        return 0.0
    return float(wrong_as_walk / total_non_walk * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Training / validation epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """
    FIX: optimizer hanya dipakai saat is_train=True.
    Tanda tangan dipisah agar tidak ada parameter yang misleading.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    conf_total = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, desc="train" if is_train else "valid", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)

            if is_train:
                optimizer.zero_grad()

            # model.forward() mengembalikan (logits, aux) saat return_aux=True
            logits, aux = model(imgs, return_aux=True)

            loss_main = criterion(logits, masks)

            # Auxiliary boundary loss — target: piksel kelas hazard (2) sebagai boundary
            bnd_target = (masks == 2).float().unsqueeze(1)
            loss_aux   = nn.functional.binary_cross_entropy_with_logits(aux, bnd_target)

            loss = loss_main + 0.4 * loss_aux

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            conf_total += compute_confusion(preds.cpu(), masks.cpu(), NUM_CLASSES)

    avg_loss = total_loss / len(loader.dataset)
    iou      = iou_from_conf(conf_total)
    return avg_loss, iou.mean().item(), iou, walkable_safety_error(conf_total)


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────

class _OnnxWrapper(nn.Module):
    """
    FIX: ONNX tracer tidak bisa handle conditional return (return_aux).
    Wrapper ini memastikan model selalu return tensor tunggal saat di-trace.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # return hanya logits, bukan tuple (logits, aux)
        return self.model(x, return_aux=False)


def export_onnx(model, img_size: int, out_path: Path) -> bool:
    """Export model PyTorch → ONNX untuk Backend FastAPI."""
    try:
        wrapper = _OnnxWrapper(model).cpu().eval()
        dummy   = torch.randn(1, 3, img_size, img_size)
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy,
                str(out_path),
                input_names=["input"],
                output_names=["output"],
                opset_version=12,
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        print(f"[OK] ONNX berhasil diexport ke: {out_path}")
        return True
    except Exception as e:
        print(f"[!] Gagal export ONNX: {e}")
        return False


def export_tflite(onnx_path: Path, tflite_path: Path) -> bool:
    """
    Export ONNX → TFLite FP16 menggunakan onnxruntime + tensorflow.
    Metode: ONNX → SavedModel → TFLite.
    onnx-tf sudah deprecated, jadi pakai pipeline tf.lite.TFLiteConverter.from_saved_model.
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        print("[→] Mengonversi ONNX → SavedModel...")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        saved_model_dir = str(tflite_path.parent / "pidnet_savedmodel")
        tf_rep.export_graph(saved_model_dir)

        print("[→] Mengonversi SavedModel → TFLite FP16...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        with open(str(tflite_path), "wb") as f:
            f.write(tflite_model)

        size_mb = tflite_path.stat().st_size / 1024 / 1024
        print(f"[OK] TFLite berhasil diexport ke: {tflite_path} ({size_mb:.1f} MB)")
        return True

    except ImportError:
        print("[!] onnx-tf / tensorflow tidak terinstall.")
        print("    Install: pip install onnx-tf tensorflow")
        print(f"    Lalu jalankan manual: python3 scripts/07_export_tflite.py --onnx {onnx_path} --out {tflite_path}")
        return False
    except Exception as e:
        print(f"[!] Gagal export TFLite: {e}")
        print(f"    Jalankan manual: python3 scripts/07_export_tflite.py --onnx {onnx_path} --out {tflite_path}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Training PIDNet-S segmentasi jalur")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help="Path ke folder dataset_master_seg/ hasil script 04")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--img-size",   type=int,   default=512)
    parser.add_argument("--out-dir",    default=str(Path(__file__).resolve().parents[1] / "runs" / "pidnet"))
    parser.add_argument("--name",       default=None)
    args = parser.parse_args()

    if not args.name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"pidnet_s_e{args.epochs}_{timestamp}"

    # Validasi path dataset sebelum mulai training
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset tidak ditemukan di: {dataset_root}")
        print(f"        Jalankan scripts/04_convert_coco_segmentation_to_mask.py dulu.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  GUIDIO — Training PIDNet-S Segmentasi Jalur (Trotoar 3-Zona)")
    print(f"  Dataset  : {dataset_root}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Run Name : {args.name}")
    print("=" * 65)

    train_ds = SidewalkSegDataset(args.dataset_root, "train", args.img_size, augment=True)
    valid_ds = SidewalkSegDataset(args.dataset_root, "valid", args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train samples: {len(train_ds)} | Valid samples: {len(valid_ds)}")
    print("=" * 65)

    model = PIDNetS(num_classes=NUM_CLASSES).to(device)

    # Class weighting: hazard (kelas 2) diberi bobot 3x
    # agar model tidak mengabaikan area tangga/lubang yang jumlahnya lebih sedikit
    class_weights = torch.tensor([1.0, 1.0, 3.0], device=device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = 0.0
    vl_iou    = None  # simpan untuk print di akhir

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_miou, _,       _          = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        vl_loss, vl_miou, vl_iou, safety_err  = run_epoch(model, valid_loader, criterion, optimizer, device, is_train=False)
        scheduler.step()

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train loss={tr_loss:.4f} mIoU={tr_miou:.4f} | "
              f"valid loss={vl_loss:.4f} mIoU={vl_miou:.4f} | "
              f"safety_err={safety_err:.1f}%")

        if vl_miou > best_miou:
            best_miou = vl_miou
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  → Checkpoint terbaik disimpan (mIoU={best_miou:.4f}, safety_err={safety_err:.1f}%)")

    print("\n=== Selesai Training PIDNet-S ===")
    print(f"Best valid mIoU : {best_miou:.4f}  (target >= 0.78)")
    if vl_iou is not None:
        print("IoU per kelas (epoch terakhir):")
        for name, val in zip(CLASS_NAMES, vl_iou.tolist()):
            print(f"  {name:<16}: {val:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # DUAL EXPORT — Backend (.onnx) + Mobile (.tflite)
    # ─────────────────────────────────────────────────────────────────────────
    best_pth   = out_dir / "best.pth"
    onnx_path  = out_dir / "pidnet_s.onnx"
    tflite_path = out_dir / "pidnet_s.tflite"

    # Muat ulang bobot terbaik sebelum export
    model.load_state_dict(torch.load(best_pth, map_location=device))
    model.eval()

    # 1. Export ONNX — untuk FastAPI Backend
    print("\n[→] Export ke ONNX (Backend FastAPI)...")
    onnx_ok = export_onnx(model, args.img_size, onnx_path)

    # 2. Export TFLite — untuk Flutter On-Device (hanya kalau ONNX berhasil)
    if onnx_ok:
        print("\n[→] Export ke TFLite FP16 (Flutter mobile offline)...")
        export_tflite(onnx_path, tflite_path)

    print("\n" + "=" * 65)
    print("  [FINISHED] Model siap digunakan:")
    print(f"  PyTorch  (.pth)    : {best_pth}")
    print(f"  Backend  (.onnx)   : {onnx_path}")
    print(f"  Mobile   (.tflite) : {tflite_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
