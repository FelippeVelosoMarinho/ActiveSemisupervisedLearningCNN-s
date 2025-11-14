from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms

# ---- Augmentations (FixMatch-like minimal) ----
def make_weak_aug(img_size: int, is_gray_ok: bool = False):
    # Mantém ToTensor (0..1) e 3 canais para estabilidade
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1 if is_gray_ok else 3),
        transforms.ToTensor(),
    ]
    aug = [
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    return transforms.Compose(base + aug)

def make_strong_aug(img_size: int, is_gray_ok: bool = False):
    # policy simples: jitter + crop
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1 if is_gray_ok else 3),
        transforms.ToTensor(),
    ]
    aug = [
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    return transforms.Compose(aug + base)  # forte antes, normaliza depois

class IndexOnlyDataset(Dataset):
    """
    Retorna (x, sid) para aquisição ativa.
    Usa a mesma base (root ou Subset) e preserva o índice global.
    """
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        if isinstance(self.base, Subset) and hasattr(self.base, "indices"):
            sid = int(self.base.indices[idx])
            x, _ = self.base.dataset[sid]
        else:
            sid = int(idx)
            x, _ = self.base[idx]
        # Garante 3 canais
        if isinstance(x, torch.Tensor) and x.ndim == 3 and x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x, sid

class LabeledDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        # base pode ser Subset(...) ou dataset raiz
        if isinstance(self.base, Subset) and hasattr(self.base, "indices"):
            sid = int(self.base.indices[idx])
            x, y = self.base.dataset[sid]
        else:
            sid = int(idx)
            x, y = self.base[idx]
        # Garante 3 canais
        if isinstance(x, torch.Tensor) and x.ndim == 3 and x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        return x, int(y), sid

class UnlabeledDataset(Dataset):
    def __init__(self, base: Dataset, weak_tf, strong_tf):
        self.base = base
        self.weak_tf = weak_tf
        self.strong_tf = strong_tf
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        if isinstance(self.base, Subset) and hasattr(self.base, "indices"):
            sid = int(self.base.indices[idx])
            x, _ = self.base.dataset[sid]
        else:
            sid = int(idx)
            x, _ = self.base[idx]

        # Converte p/ PIL e aplica aug
        if isinstance(x, torch.Tensor):
            if x.size(0) == 1:
                x_img = transforms.ToPILImage()(x.expand(3, *x.shape[1:]).clamp(0,1))
            else:
                x_img = transforms.ToPILImage()(x.clamp(0,1))
        else:
            x_img = x

        x_w = self.weak_tf(x_img)
        x_s = self.strong_tf(x_img)
        if x_w.size(0) == 1: x_w = x_w.repeat(3,1,1)
        if x_s.size(0) == 1: x_s = x_s.repeat(3,1,1)
        return x_w, x_s, sid

def get_base_torchvision(dataset: str, split: str, img_size: int, data_dir: str):
    dataset = dataset.lower()
    if dataset == "mnist":
        # Aplica resize + grayscale->3 + ToTensor diretamente aqui
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        is_train = split in ["train", "train+unlabeled"]
        return torchvision.datasets.MNIST(root=data_dir, train=is_train, download=True, transform=tf)
    elif dataset == "stl10":
        tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        if split == "train":
            return torchvision.datasets.STL10(root=data_dir, split="train", download=True, transform=tf)
        elif split in ["unlabeled", "train+unlabeled"]:
            tr = torchvision.datasets.STL10(root=data_dir, split="train", download=True, transform=tf)
            ul = torchvision.datasets.STL10(root=data_dir, split="unlabeled", download=True, transform=tf)
            return torch.utils.data.ConcatDataset([tr, ul])
        elif split == "test":
            return torchvision.datasets.STL10(root=data_dir, split="test", download=True, transform=tf)
        else:
            raise ValueError("split inválido p/ STL10")
    else:
        raise ValueError("dataset deve ser mnist ou stl10")

def build_assl_splits(
    dataset: str,
    img_size: int,
    data_dir: str,
    labeled_init: int,
    subset_total: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    batch_l: int = 128,
    batch_u: int = 128,
    batch_t: int = 256,
    prefetch_factor: int = 2,
):
    """
    Retorna:
      - L_indices (lista de idx rotulados)
      - U_indices (lista de idx não-rotulados)
      - loaders de treino/val (rotulados) e U_loader
    """
    from torch.utils.data import DataLoader

    # Base de treino para rótulos + U
    if dataset.lower() == "mnist":
        base_LU = get_base_torchvision("mnist", "train", img_size, data_dir)
        n_classes = 10
        is_gray_ok = True
    else:
        base_LU = get_base_torchvision("stl10", "train+unlabeled", img_size, data_dir)
        n_classes = 10
        is_gray_ok = False

    N = len(base_LU)
    idx_all = list(range(N))
    if subset_total > 0:
        idx_all = idx_all[:subset_total]

    # Split inicial: primeiros labeled_init como L0, resto U0 (poderia ser estratificado)
    L_indices = idx_all[:min(labeled_init, len(idx_all))]
    U_indices = idx_all[len(L_indices):]

    # Datasets rotulados (para supervised & probe)
    L_ds = LabeledDataset(Subset(base_LU, L_indices))

    # Val/test set
    base_test = get_base_torchvision(dataset, "test", img_size, data_dir)
    T_ds = LabeledDataset(base_test)

    # Unlabeled dataset com aug fraco/forte
    weak_tf = make_weak_aug(img_size, is_gray_ok=is_gray_ok)
    strong_tf = make_strong_aug(img_size, is_gray_ok=is_gray_ok)
    U_ds = UnlabeledDataset(Subset(base_LU, U_indices), weak_tf=weak_tf, strong_tf=strong_tf)

    # Loaders
    L_loader = DataLoader(
        L_ds, batch_size=batch_l, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
    )
    T_loader = DataLoader(
        T_ds, batch_size=batch_t, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
    )
    U_loader = DataLoader(
        U_ds, batch_size=batch_u, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
    )

    return L_indices, U_indices, L_loader, T_loader, U_loader, base_LU, n_classes
