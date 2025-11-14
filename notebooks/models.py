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
