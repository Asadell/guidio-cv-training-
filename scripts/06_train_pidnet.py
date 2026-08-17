"""
Script 06: Training PIDNet-S — Segmentasi Jalur 3-Zona

Target metrik:
  mIoU walkable      >= 0.78
  walkable-safety error (non_walkable salah jadi walkable) <= 5%

Output: runs/pidnet/best.pth
        → Script 07 akan export ini ke ONNX untuk backend
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pidnet.model   import PIDNetS
from src.pidnet.dataset import SidewalkSegDataset

NUM_CLASSES  = 3
CLASS_NAMES  = ["non_walkable", "walkable", "hazard"]


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


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    conf_total = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, desc="train" if train else "valid", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            if train:
                optimizer.zero_grad()
            logits, aux  = model(imgs, return_aux=True)
            loss_main    = criterion(logits, masks)
            bnd_target   = (masks == 2).float().unsqueeze(1)
            loss_aux     = nn.functional.binary_cross_entropy_with_logits(aux, bnd_target)
            loss         = loss_main + 0.4 * loss_aux
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            conf_total += compute_confusion(preds.cpu(), masks.cpu(), NUM_CLASSES)
    avg_loss = total_loss / len(loader.dataset)
    iou      = iou_from_conf(conf_total)
    return avg_loss, iou.mean().item(), iou, walkable_safety_error(conf_total)


def main():
    parser = argparse.ArgumentParser(description="Training PIDNet-S segmentasi jalur")
    parser.add_argument("--dataset-root", default=str(Path(__file__).resolve().parents[2] / "dataset_master_seg"))
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--img-size",   type=int,   default=512)
    parser.add_argument("--out-dir",    default=str(Path(__file__).resolve().parents[1] / "runs" / "pidnet"))
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Device: {device}")

    train_ds = SidewalkSegDataset(args.dataset_root, "train", args.img_size, augment=True)
    valid_ds = SidewalkSegDataset(args.dataset_root, "valid", args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model     = PIDNetS(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_miou, _,         _       = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_miou, vl_iou, safety_err = run_epoch(model, valid_loader, criterion, optimizer, device, False)
        scheduler.step()

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train loss={tr_loss:.4f} mIoU={tr_miou:.4f} | "
              f"valid loss={vl_loss:.4f} mIoU={vl_miou:.4f} | "
              f"safety_err={safety_err:.1f}%")

        if vl_miou > best_miou:
            best_miou = vl_miou
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  → model terbaik disimpan (mIoU={best_miou:.4f}, safety_err={safety_err:.1f}%)")

    print("\n=== Selesai Training PIDNet-S ===")
    print(f"Best valid mIoU: {best_miou:.4f}  (target >= 0.78)")
    print("IoU per kelas (epoch terakhir):")
    for name, val in zip(CLASS_NAMES, vl_iou.tolist()):
        print(f"  {name:<16}: {val:.4f}")
    print(f"\n[→] Lanjut: python scripts/07_export_onnx.py --weights {out_dir}/best.pth")


if __name__ == "__main__":
    main()
