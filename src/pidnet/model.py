"""
PIDNet-S (implementasi ramping, 3-kelas) untuk segmentasi jalur navigasi tunanetra.

3 Cabang mengikuti ide PIDNet:
  P (Pixel/Detail)   — resolusi tinggi, jaga tepi trotoar
  I (Instance/Context) — resolusi rendah, konteks area luas
  D (Difference/Boundary) — batas antar-zona

Dibuat ringan agar trainable dari nol dengan dataset ~1.900 gambar.
Untuk performa produksi lebih tinggi: ganti stem dengan ResNet-18/MobileNetV3
pretrained ImageNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(ch, ch),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class PIDNetS(nn.Module):
    """
    Input : (B, 3, H, W)  — default 512×512
    Output: (B, num_classes, H, W)  — logits per piksel
    """
    def __init__(self, num_classes: int = 3, base_ch: int = 32):
        super().__init__()
        c = base_ch

        self.stem = nn.Sequential(
            conv_bn_relu(3, c, k=3, s=2, p=1),   # /2
            conv_bn_relu(c, c, k=3, s=2, p=1),   # /4
        )
        self.branch_i = nn.Sequential(
            conv_bn_relu(c, c * 2, k=3, s=2, p=1),  # /8
            ResBlock(c * 2),
            conv_bn_relu(c * 2, c * 4, k=3, s=2, p=1),  # /16
            ResBlock(c * 4),
        )
        self.branch_p = nn.Sequential(ResBlock(c), conv_bn_relu(c, c))
        self.branch_d = nn.Sequential(ResBlock(c), conv_bn_relu(c, c))
        self.i_proj   = conv_bn_relu(c * 4, c, k=1, s=1, p=0)
        self.fusion   = nn.Sequential(conv_bn_relu(c * 3, c * 2), ResBlock(c * 2))
        self.cls_head = nn.Conv2d(c * 2, num_classes, 1)
        self.aux_head = nn.Conv2d(c, 1, 1)  # boundary auxiliary

    def forward(self, x, return_aux: bool = False):
        h, w = x.shape[2:]
        s   = self.stem(x)
        fp  = self.branch_p(s)
        fd  = self.branch_d(s)
        fi  = self.branch_i(s)
        fi  = F.interpolate(self.i_proj(fi), size=fp.shape[2:], mode="bilinear", align_corners=False)
        out = self.cls_head(self.fusion(torch.cat([fp, fi, fd], dim=1)))
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        if return_aux:
            aux = F.interpolate(self.aux_head(fd), size=(h, w), mode="bilinear", align_corners=False)
            return out, aux
        return out


if __name__ == "__main__":
    m   = PIDNetS(num_classes=3)
    x   = torch.randn(2, 3, 512, 512)
    out = m(x)
    print("Output shape:", out.shape)  # (2, 3, 512, 512)
    print(f"Parameters  : {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
