"""
Script 07: Export PIDNet-S (.pth) → ONNX untuk backend FastAPI

Output ONNX langsung di-copy ke:
  /home/asadel/kuliah/lomba/smstr6/guido/project/backend/models/pidnet_s_navigasi.onnx

Backend (segmentation_service.py) sudah siap membaca ONNX dengan path di atas.
Tidak perlu ubah kode backend — cukup pastikan env var SEGMENTATION_MODEL
diarahkan ke file ini, atau update path default di segmentation_service.py.
"""
import argparse
import shutil
from pathlib import Path

import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pidnet.model import PIDNetS

BACKEND_MODELS_DIR = Path("/home/asadel/kuliah/lomba/smstr6/guido/project/backend/models")


def export(weights: Path, out: Path, img_size: int = 512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = PIDNetS(num_classes=3).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"[OK] ONNX tersimpan: {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    # Verifikasi dengan onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        dummy_np = dummy.cpu().numpy()
        logits = sess.run(None, {sess.get_inputs()[0].name: dummy_np})[0]
        assert logits.shape == (1, 3, img_size, img_size), f"Shape salah: {logits.shape}"
        print(f"[OK] ONNX Runtime verifikasi berhasil. Output shape: {logits.shape}")
    except Exception as e:
        print(f"[!] Verifikasi ONNX Runtime gagal: {e}")

    # Copy ke backend
    BACKEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = BACKEND_MODELS_DIR / "pidnet_s_navigasi.onnx"
    shutil.copy2(out, dst)
    print(f"[→] Di-copy ke backend: {dst}")
    print(f"    Pastikan env var SEGMENTATION_MODEL atau path di segmentation_service.py")
    print(f"    sudah mengarah ke: {dst}")


def main():
    parser = argparse.ArgumentParser(description="Export PIDNet-S .pth -> .onnx")
    parser.add_argument("--weights",  required=True, help="Path ke best.pth hasil training")
    parser.add_argument("--out",      default=str(Path(__file__).resolve().parents[1] / "runs" / "pidnet_s_navigasi.onnx"))
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"[!] File weights tidak ditemukan: {weights}")
        return

    export(weights, Path(args.out), args.img_size)


if __name__ == "__main__":
    main()
