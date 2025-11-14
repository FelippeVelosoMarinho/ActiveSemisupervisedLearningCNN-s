from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

from models import SmallConvEncoder, LinearHead, Classifier
from assl_strategies import select_topk_by_entropy

@dataclass
class ASSLConfig:
    dataset: str = "mnist"
    data_dir: str = "./data"
    img_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # active learning
    labeled_init: int = 500
    rounds: int = 5
    budget_per_round: int = 100

    # training
    epochs_per_round: int = 2
    lr: float = 1e-3
    lambda_u_max: float = 1.0
    tau: float = 0.7              # pseudo-label confidence
    temp_T: float = 0.5           # temperature afina softmax
    subset_total: int = 0         # 0=usar todo dataset

    # encoder
    feat_dim: int = 128
    n_classes: int = 10
    freeze_encoder_during_head: bool = False
    encoder_ckpt: Optional[str] = None   # opcional: pesos de pretext
    
    num_workers: int = 0
    pin_memory: bool = False
    batch_l: int = 128
    batch_u: int = 128
    batch_t: int = 256
    prefetch_factor: int = 2   # ignorado quando num_workers=0

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def adaptive_tau_from_conf(conf: torch.Tensor, base: float = 0.6, hi: float = 0.95) -> float:
    """
    conf: tensor [B] de confidências (0..1).
    Retorna tau_eff = clip(p90(conf), [base, hi]).
    """
    if conf.numel() == 0:
        return base
    p90 = float(np.percentile(conf.detach().cpu().numpy(), 90))
    return max(base, min(hi, p90))

class SemiSupTrainer:
    """FixMatch-like loss: supervised + consistency nos não rotulados."""
    def __init__(self, model: Classifier, cfg: ASSLConfig):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.ce = nn.CrossEntropyLoss()

    def train_epoch(
        self,
        L_loader: DataLoader,
        U_loader: DataLoader,
        epoch_idx: int,
    ) -> Tuple[float, float, float]:
        self.model.train()
        loss_sup_sum = 0.0; loss_unsup_sum = 0.0; n_sup = 0; n_uns = 0

        # warmup de lambda_u
        lam = self.cfg.lambda_u_max * min(1.0, (epoch_idx+1)/max(1, self.cfg.epochs_per_round))

        # Itera usando o menor número de batches entre L e U
        it_L = iter(L_loader)
        it_U = iter(U_loader)
        steps = min(len(L_loader), len(U_loader))
        for _ in range(steps):
            # --- batch rotulado
            x_l, y_l, _ = next(it_L)
            x_l, y_l = x_l.to(self.device), y_l.to(self.device)

            # --- batch não rotulado (weak/strong, mas U_loader já traz x_w, x_s)
            x_w, x_s, _ = next(it_U)
            x_w, x_s = x_w.to(self.device), x_s.to(self.device)

            # supervised
            logits_l = self.model(x_l)
            loss_sup = self.ce(logits_l, y_l)

            # unsupervised consistency com pseudo-label (mask por confiança)
            with torch.no_grad():
                logits_w = self.model(x_w)
                probs_w = F.softmax(logits_w / self.cfg.temp_T, dim=-1)
                conf, yhat = probs_w.max(dim=-1)
                # τ adaptativo: p90 das confs, limitado por [base=tau, hi=0.95]
                tau_eff = adaptive_tau_from_conf(conf, base=self.cfg.tau, hi=0.95)
                mask = conf >= tau_eff

            if mask.any():
                logits_s = self.model(x_s[mask])
                loss_uns = self.ce(logits_s, yhat[mask])
            else:
                loss_uns = torch.tensor(0.0, device=self.device)

            loss = loss_sup + lam * loss_uns

            self.opt.zero_grad()
            loss.backward()
            # filtra None implicitamente; clipping opcional
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.opt.step()

            bs_sup = x_l.size(0); bs_uns = int(mask.sum().item())
            loss_sup_sum += loss_sup.item() * bs_sup; n_sup += bs_sup
            loss_unsup_sum += loss_uns.item() * max(1, bs_uns); n_uns += max(1, bs_uns)

        return loss_sup_sum/max(1,n_sup), loss_unsup_sum/max(1,n_uns), lam

    @torch.no_grad()
    def eval_acc(self, loader: DataLoader) -> float:
        self.model.eval()
        n=0; acc=0.0
        for x,y, _ in loader:
            x,y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            acc += (logits.argmax(1)==y).float().sum().item()
            n += x.size(0)
        return acc/max(1,n)

class ActiveLoop:
    def __init__(self, cfg: ASSLConfig, base_LU, L_indices, U_indices,
                 L_loader, T_loader, U_loader):
        self.cfg = cfg
        self.base_LU = base_LU
        self.L_indices = list(L_indices)
        self.U_indices = list(U_indices)
        self.L_loader = L_loader
        self.T_loader = T_loader
        self.U_loader = U_loader

        # modelo
        encoder = SmallConvEncoder(in_ch=3, feat_dim=cfg.feat_dim)
        if cfg.encoder_ckpt:
            sd = torch.load(cfg.encoder_ckpt, map_location="cpu")
            encoder.load_state_dict(sd, strict=False)
        head = LinearHead(cfg.feat_dim, cfg.n_classes)
        self.model = Classifier(encoder, head)

        if cfg.freeze_encoder_during_head:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        self.trainer = SemiSupTrainer(self.model, cfg)

    def _make_loaders_from_indices(self):
        """Reconstrói DataLoaders após mover índices L/U."""
        from datasets_assl import LabeledDataset, UnlabeledDataset, make_weak_aug, make_strong_aug
        # L
        L_ds = LabeledDataset(Subset(self.base_LU, self.L_indices))
        self.L_loader = torch.utils.data.DataLoader(L_ds, batch_size=128, shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=(self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None),
            persistent_workers=(self.cfg.num_workers > 0),
        )
        # U
        is_gray_ok = (self.cfg.dataset.lower() == "mnist")
        weak_tf = make_weak_aug(self.cfg.img_size, is_gray_ok=is_gray_ok)
        strong_tf = make_strong_aug(self.cfg.img_size, is_gray_ok=is_gray_ok)
        U_ds = UnlabeledDataset(Subset(self.base_LU, self.U_indices), weak_tf=weak_tf, strong_tf=strong_tf)
        self.U_loader = torch.utils.data.DataLoader(U_ds, batch_size=128, shuffle=False, 
            num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                prefetch_factor=(self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None),
                persistent_workers=(self.cfg.num_workers > 0),                                         
        )

    def active_select(self, k: int) -> List[int]:
        """Seleciona k amostras da pool U pelo critério de entropia."""
        from datasets_assl import IndexOnlyDataset
        # Criar um Subset(U) que retorne (x, idx) para pontuar
        # Reutilizamos LabeledDataset para obter (x, y, idx) e ignorar y
        ds_U_for_score = IndexOnlyDataset(Subset(self.base_LU, self.U_indices))
        # Loader simples
        loader = torch.utils.data.DataLoader(ds_U_for_score, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
        # Precisamos de um dataset que retorne (x, idx); adaptamos on-the-fly:
        # xs, idxs = [], []
        # for x, y in loader:
        #     xs.append(x);  
        sel = select_topk_by_entropy(self.trainer.model, loader, k, device=torch.device(self.cfg.device))
        return sel

    def run(self):
        log = []
        for r in range(self.cfg.rounds):
            print(f"\n== Round {r+1}/{self.cfg.rounds} ==")
            t0 = time.time()

            # 1) Treino semi-supervisionado nesta rodada
            for ep in range(self.cfg.epochs_per_round):
                ls, lu, lam = self.trainer.train_epoch(self.L_loader, self.U_loader, epoch_idx=ep)
                print(f"[ep {ep+1}] L_sup={ls:.4f}  L_uns={lu:.4f}  λ_u={lam:.2f}")

            test_acc = self.trainer.eval_acc(self.T_loader)
            print(f"[eval] test_acc={test_acc:.4f}  (round time {time.time()-t0:.1f}s)")

            # 2) Seleção ativa de k amostras e mover U->L
            if r < self.cfg.rounds - 1 and len(self.U_indices) > 0:
                k = min(self.cfg.budget_per_round, len(self.U_indices))
                chosen = self.active_select(k)
                chosen_set = set(chosen)
                # mover
                new_L = []
                new_U = []
                for idx in self.U_indices:
                    if idx in chosen_set: new_L.append(idx)
                    else: new_U.append(idx)
                self.L_indices.extend(new_L)
                self.U_indices = new_U
                print(f"[active] moved {len(new_L)} from U->L  |  |L|={len(self.L_indices)}  |U|={len(self.U_indices)}")
                # rebuild loaders
                self._make_loaders_from_indices()

            log.append(dict(round=r, test_acc=float(test_acc), L=len(self.L_indices), U=len(self.U_indices)))
        return log
