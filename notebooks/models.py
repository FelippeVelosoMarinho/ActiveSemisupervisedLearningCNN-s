import torch
import torch.nn as nn
import torch.nn.functional as F

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class SmallConvEncoder(nn.Module):
    """Encoder leve e estável em CPU."""
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x):
        h = self.net(x)                 # [B,128,1,1]
        h = h.view(h.size(0), -1)       # [B,128]
        z = self.fc(h)                  # [B,feat_dim]
        return z

class LinearHead(nn.Module):
    def __init__(self, feat_dim=128, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)
    def forward(self, z): return self.fc(z)

class Classifier(nn.Module):
    """Encoder + cabeça linear (para supervised/linear probe/ASSL)."""
    def __init__(self, encoder: SmallConvEncoder, head: LinearHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, x):
        z = self.encoder(x)
        logits = self.head(z)
        return logits

# pre texto (multitask)

class RotHead(nn.Module):
    """Cabeça para tarefa de rotação (0,90,180,270)."""
    def __init__(self, feat_dim=128, n_classes=4):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)
    def forward(self, z): return self.fc(z)

class JigsawHead(nn.Module):
    """Cabeça para Jigsaw 2x2 com 24 permutações."""
    def __init__(self, feat_dim=128, n_perm=24):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_perm)
    def forward(self, z): return self.fc(z)

class ColorizationDecoder(nn.Module):
    """Decoder para colorização (reconstrução RGB)."""
    def __init__(self, feat_dim=128, out_ch=3, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden * 4 * 4),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, out_size):
        b = z.size(0)
        x = self.fc(z).view(b, -1, 4, 4)
        x = self.deconv(x)
        return F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)