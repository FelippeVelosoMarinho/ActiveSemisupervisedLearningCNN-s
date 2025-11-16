# ------------------------- Lightweight SSL pretext tasks + linear probe -------------------------
import argparse, random, time
from typing import Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms
from PIL import Image

# ===================== utils =====================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _ensure_pil(x):
    # Converte para PIL se vier como Tensor/ndarray
    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        return transforms.ToPILImage()(x)
    if isinstance(x, np.ndarray):
        return Image.fromarray(x)
    return x  # assume PIL

# ===================== datasets =====================
class RotationDataset(Dataset):
    """Gera rótulo 0..3 para rotação 0/90/180/270 e imagem rotacionada."""
    def __init__(self, base: Dataset, img_size: int = 96):
        self.base = base
        self.resize = transforms.Resize((img_size, img_size))

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        x = _ensure_pil(x)
        x = self.resize(x).convert("RGB") 
        k = random.randint(0, 3)
        x = x.rotate(90*k)
        x = transforms.ToTensor()(x)
        return x, k

class ColorizationDataset(Dataset):
    """Entrada: cinza (repetida p/ 3 canais); Alvo: RGB original; perda: MSE."""
    def __init__(self, base: Dataset, img_size: int = 96):
        self.base = base
        self.resize = transforms.Resize((img_size, img_size))

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        x = _ensure_pil(x)
        x = self.resize(x).convert("RGB")
        x_rgb  = transforms.ToTensor()(x)
        x_gray = transforms.functional.rgb_to_grayscale(x)
        x_gray = transforms.ToTensor()(x_gray)          # [1,H,W] in [0,1]
        x_gray3 = x_gray.repeat(3,1,1)                  # [3,H,W]
        return x_gray3, x_rgb

# Jigsaw 2x2 com 24 permutações
JIGSAW_PERMS = [
    (0,1,2,3),(0,1,3,2),(0,2,1,3),(0,2,3,1),(0,3,1,2),(0,3,2,1),
    (1,0,2,3),(1,0,3,2),(1,2,0,3),(1,2,3,0),(1,3,0,2),(1,3,2,0),
    (2,0,1,3),(2,0,3,1),(2,1,0,3),(2,1,3,0),(2,3,0,1),(2,3,1,0),
    (3,0,1,2),(3,0,2,1),(3,1,0,2),(3,1,2,0),(3,2,0,1),(3,2,1,0)
]
class JigsawDataset(Dataset):
    def __init__(self, base: Dataset, img_size: int = 96, pad: int = 2):
        self.base = base
        self.resize = transforms.Resize((img_size, img_size))
        self.pad = pad

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        x = _ensure_pil(x)
        x = self.resize(x).convert("RGB")
        x = transforms.ToTensor()(x)  # [3,H,W]
        # pad pra não cortar borda
        x = F.pad(x.unsqueeze(0), (self.pad, self.pad, self.pad, self.pad), mode='reflect').squeeze(0)
        C,H,W = x.shape; h2,w2 = H//2, W//2
        p0, p1, p2, p3 = x[:, :h2, :w2], x[:, :h2, w2:], x[:, h2:, :w2], x[:, h2:, w2:]
        patches = [p0,p1,p2,p3]
        idx_perm = random.randint(0, len(JIGSAW_PERMS)-1)
        perm = JIGSAW_PERMS[idx_perm]
        shuf = [patches[i] for i in perm]
        top = torch.cat([shuf[0], shuf[1]], dim=2)
        bot = torch.cat([shuf[2], shuf[3]], dim=2)
        img_shuf = torch.cat([top, bot], dim=1)         # [3,H,W]
        return img_shuf, idx_perm

# ===================== modelos =====================
class SmallConvEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x):
        h = self.net(x)            # [B,128,1,1]
        h = h.view(h.size(0), -1)  # [B,128]
        z = self.fc(h)             # [B,feat]
        return z

class RotHead(nn.Module):
    def __init__(self, feat_dim=128, n_classes=4):
        super().__init__(); self.fc = nn.Linear(feat_dim, n_classes)
    def forward(self, z): return self.fc(z)

class JigsawHead(nn.Module):
    def __init__(self, feat_dim=128, n_perm=24):
        super().__init__(); self.fc = nn.Linear(feat_dim, n_perm)
    def forward(self, z): return self.fc(z)

class ColorizationDecoder(nn.Module):
    def __init__(self, feat_dim=128, out_ch=3, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(feat_dim, hidden*4*4), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),
            nn.Sigmoid()
        )
    def forward(self, z, out_size: Tuple[int,int]=(32,32)):
        b = z.size(0)
        x = self.fc(z).view(b, -1, 4, 4)
        x = self.deconv(x)
        return F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)

class LinearProbe(nn.Module):
    def __init__(self, feat_dim=128, n_classes=10):
        super().__init__(); self.fc = nn.Linear(feat_dim, n_classes)
    def forward(self, z): return self.fc(z)

# ===================== loops de treino =====================
def train_rotation(encoder, head, loader, device, epochs=5, lr=1e-3, log_every=100):
    encoder.train(); head.train()
    opt = optim.Adam(list(encoder.parameters())+list(head.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        t0 = time.time(); loss_sum=acc_sum=n=0
        for i,(x,y) in enumerate(loader):
            x,y = x.to(device), y.to(device)
            z = encoder(x); logits = head(z); loss = ce(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = x.size(0); loss_sum += loss.item()*bs; acc_sum += (logits.argmax(1)==y).float().sum().item(); n+=bs
            if (i+1)%log_every==0: print(f"[rot ep{ep+1}] {i+1}/{len(loader)}  loss={loss_sum/n:.4f} acc={acc_sum/n:.4f}")
        print(f"[rot ep{ep+1}] {time.time()-t0:.1f}s  loss={loss_sum/n:.4f} acc={acc_sum/n:.4f}")

def train_jigsaw(encoder, head, loader, device, epochs=5, lr=1e-3, log_every=100):
    encoder.train(); head.train()
    opt = optim.Adam(list(encoder.parameters())+list(head.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        t0 = time.time(); loss_sum=acc_sum=n=0
        for i,(x,y) in enumerate(loader):
            x,y = x.to(device), y.to(device)
            z = encoder(x); logits = head(z); loss = ce(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = x.size(0); loss_sum += loss.item()*bs; acc_sum += (logits.argmax(1)==y).float().sum().item(); n+=bs
            if (i+1)%log_every==0: print(f"[jig ep{ep+1}] {i+1}/{len(loader)}  loss={loss_sum/n:.4f} acc={acc_sum/n:.4f}")
        print(f"[jig ep{ep+1}] {time.time()-t0:.1f}s  loss={loss_sum/n:.4f} acc={acc_sum/n:.4f}")

def train_colorization(encoder, dec, loader, device, epochs=5, lr=1e-3, log_every=100):
    encoder.train(); dec.train()
    opt = optim.Adam(list(encoder.parameters())+list(dec.parameters()), lr=lr)
    mse = nn.MSELoss()
    for ep in range(epochs):
        t0 = time.time(); loss_sum=n=0
        for i,(xg, xr) in enumerate(loader):
            xg, xr = xg.to(device), xr.to(device)
            z = encoder(xg)
            _,_,H,W = xr.shape
            xhat = dec(z, out_size=(H,W))
            loss = mse(xhat, xr)
            opt.zero_grad(); loss.backward(); opt.step()
            bs = xg.size(0); loss_sum += loss.item()*bs; n+=bs
            if (i+1)%log_every==0: print(f"[col ep{ep+1}] {i+1}/{len(loader)}  mse={loss_sum/n:.4f}")
        print(f"[col ep{ep+1}] {time.time()-t0:.1f}s  mse={loss_sum/n:.4f}")

@torch.no_grad()
def eval_linear_probe(encoder, clf, loader, device):
    encoder.eval(); clf.eval()
    ce = nn.CrossEntropyLoss(); loss_sum=acc_sum=n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        z = encoder(x); logits = clf(z); loss = ce(logits,y)
        bs = x.size(0); loss_sum += loss.item()*bs; acc_sum += (logits.argmax(1)==y).float().sum().item(); n+=bs
    return loss_sum/max(1,n), acc_sum/max(1,n)

def train_linear_probe(encoder, tr_loader, te_loader, n_classes, device, epochs=5, lr=5e-3, log_every=100):
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad=False
    feat_dim = encoder.fc.out_features if hasattr(encoder,"fc") else 128
    clf = LinearProbe(feat_dim, n_classes).to(device)
    opt = optim.Adam(clf.parameters(), lr=lr); ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        clf.train(); t0=time.time(); loss_sum=acc_sum=n=0
        for i,(x,y) in enumerate(tr_loader):
            x,y = x.to(device), y.to(device)
            with torch.no_grad(): z = encoder(x)
            logits = clf(z); loss = ce(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            bs=x.size(0); loss_sum+=loss.item()*bs; acc_sum += (logits.argmax(1)==y).float().sum().item(); n+=bs
            if (i+1)%log_every==0: print(f"[lin ep{ep+1}] {i+1}/{len(tr_loader)} loss={loss_sum/n:.4f} acc={acc_sum/n:.4f}")
        tr_loss, tr_acc = loss_sum/max(1,n), acc_sum/max(1,n)
        te_loss, te_acc = eval_linear_probe(encoder, clf, te_loader, device)
        print(f"[lin ep{ep+1}] {time.time()-t0:.1f}s train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")
    return clf

# ===================== loaders =====================
def build_loaders(args):
    img_size, data_dir = args.img_size, args.data_dir

    if args.dataset.lower()=="mnist":
        # Para PRETEXT: base sem transform (PIL)
        base_pretext = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
        rot_ds = RotationDataset(base_pretext, img_size)
        jig_ds = JigsawDataset(base_pretext, img_size)
        col_ds = ColorizationDataset(base_pretext, img_size)
        # Para supervised: ToTensor (3 canais)
        common_sup = transforms.Compose([transforms.Resize((img_size,img_size)),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor()])
        sup_tr = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=common_sup)
        sup_te = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=common_sup)
        n_classes=10

    elif args.dataset.lower()=="stl10":
        # STL já entrega PIL; para pretext, usar train (ou concatenar com unlabeled se quiser)
        sup_tr = torchvision.datasets.STL10(data_dir, split="train", download=True,
                    transform=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()]))
        sup_te = torchvision.datasets.STL10(data_dir, split="test", download=True,
                    transform=transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor()]))
        rot_ds = RotationDataset(torchvision.datasets.STL10(data_dir, split="train", download=True, transform=None), img_size)
        jig_ds = JigsawDataset(torchvision.datasets.STL10(data_dir, split="train", download=True, transform=None), img_size)
        col_ds = ColorizationDataset(torchvision.datasets.STL10(data_dir, split="train", download=True, transform=None), img_size)
        n_classes=10
    else:
        raise ValueError("dataset inválido")

    kw_pre = dict(batch_size=args.batch_pretext, num_workers=args.num_workers, pin_memory=True)
    kw_probe= dict(batch_size=args.batch_probe,  num_workers=args.num_workers, pin_memory=True)

    rot_loader = DataLoader(rot_ds, shuffle=True, **kw_pre)
    jig_loader = DataLoader(jig_ds, shuffle=True, **kw_pre)
    col_loader = DataLoader(col_ds, shuffle=True, **kw_pre)
    tr_loader  = DataLoader(sup_tr, shuffle=True, **kw_probe)
    te_loader  = DataLoader(sup_te, shuffle=False, **kw_probe)
    return rot_loader, jig_loader, col_loader, tr_loader, te_loader, n_classes

def main():
    ap = argparse.ArgumentParser("Lightweight SSL (rotation/colorization/jigsaw) + linear probe")
    ap.add_argument("--dataset", choices=["mnist","stl10"], default="mnist")
    ap.add_argument("--task", choices=["rotation","colorization","jigsaw"], default="rotation")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--img-size", type=int, default=96)
    # >>> SEPARAMOS OS BATCHES
    ap.add_argument("--batch-pretext", type=int, default=128)
    ap.add_argument("--batch-probe",   type=int, default=256)
    ap.add_argument("--epochs-pretext", type=int, default=5)
    ap.add_argument("--epochs-linear",  type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=0, help="usa N amostras p/ teste rápido (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    ap.add_argument("--save-encoder", action="store_true",
                    help="Se setado, salva o encoder treinado para uso no ASSL")
    ap.add_argument("--encoder-out", type=str, default=None,
                    help="Caminho do arquivo .pt para salvar o encoder (state_dict)")
    
    args = ap.parse_args()

    set_seed(args.seed); device = torch.device(args.device)
    print("== Config =="); [print(f"{k}: {v}") for k,v in vars(args).items()]
    rot_loader, jig_loader, col_loader, tr_loader, te_loader, n_classes = build_loaders(args)

    # subset rápido
    if args.subset>0:
        def subset_loader(loader, bs):
            ds = loader.dataset
            idx = list(range(min(args.subset, len(ds))))
            sub = Subset(ds, idx)
            return DataLoader(sub, batch_size=bs, shuffle=True,
                              num_workers=loader.num_workers, pin_memory=True)
        if args.task=="rotation": rot_loader=subset_loader(rot_loader, args.batch_pretext)
        elif args.task=="jigsaw": jig_loader=subset_loader(jig_loader, args.batch_pretext)
        else: col_loader=subset_loader(col_loader, args.batch_pretext)
        tr_loader=subset_loader(tr_loader, args.batch_probe); te_loader=subset_loader(te_loader, args.batch_probe)

    encoder = SmallConvEncoder(in_ch=3, feat_dim=128).to(device)

    # -------- PRETEXT --------
    pretext_t0 = time.time()
    if args.task=="rotation":
        head = RotHead(128,4).to(device)
        print(f"[model] params: {count_params(encoder)+count_params(head):,}")
        train_rotation(encoder, head, rot_loader, device, epochs=args.epochs_pretext, lr=args.lr)
        encoder.head=head
    elif args.task=="jigsaw":
        head = JigsawHead(128, len(JIGSAW_PERMS)).to(device)
        print(f"[model] params: {count_params(encoder)+count_params(head):,}")
        train_jigsaw(encoder, head, jig_loader, device, epochs=args.epochs_pretext, lr=args.lr)
        encoder.head=head
    else:
        dec = ColorizationDecoder(128,3).to(device)
        print(f"[model] params: {count_params(encoder)+count_params(dec):,}")
        train_colorization(encoder, dec, col_loader, device, epochs=args.epochs_pretext, lr=args.lr)
        encoder.dec=dec
    pretext_secs = time.time() - pretext_t0
    print(f"[pretext] secs: {pretext_secs:.3f}")

    # -------- LINEAR PROBE --------
    print("== Linear probe ==")
    probe_t0 = time.time()
    clf = train_linear_probe(encoder, tr_loader, te_loader, n_classes, device,
                             epochs=args.epochs_linear, lr=5e-3)
    probe_secs = time.time() - probe_t0
    print(f"[probe] secs: {probe_secs:.3f}")

    te_loss, te_acc = eval_linear_probe(encoder, clf, te_loader, device)
    print(f"[final] test_acc: {te_acc:.4f}")
    
    # >>> NOVO: salvar encoder para uso no ASSL <<<
    if args.save_encoder:
        import os
        ckpt_dir = "./checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        if args.encoder_out is not None:
            out_path = args.encoder_out
        else:
            out_path = os.path.join(
                ckpt_dir,
                f"encoder_{args.dataset}_{args.task}_img{args.img_size}.pt"
            )
        torch.save(encoder.state_dict(), out_path)
        print(f"[ckpt] encoder salvo em: {out_path}")


if __name__ == "__main__": main()
