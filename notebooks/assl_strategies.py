from typing import List, Tuple
import torch
import torch.nn.functional as F

@torch.no_grad()
def score_uncertainty_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1)

@torch.no_grad()
def select_topk_by_entropy(
    model,
    loader,                 # DataLoader que retorna (x, sid)
    k: int,
    device: torch.device,
) -> List[int]:
    model.eval()
    all_logits = []
    all_sids: List[int] = []
    for x, sid in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_sids.extend([int(s) for s in sid])
    if not all_logits:
        return []
    logits = torch.cat(all_logits, dim=0)
    ent = score_uncertainty_entropy(logits)
    topk = torch.topk(ent, k=min(k, ent.numel())).indices.tolist()
    return [all_sids[i] for i in topk]

def diversify_by_farthest(features: torch.Tensor, m: int) -> List[int]:
    """
    Seleção gulosa por pontos mais distantes no espaço de features (CPU friendly).
    features: [N,D]
    """
    N = features.size(0)
    if N == 0: return []
    m = min(m, N)
    chosen = []
    # inicia com ponto de maior norma
    norms = (features**2).sum(dim=1)
    i0 = norms.argmax().item()
    chosen.append(i0)
    dists = torch.cdist(features[i0:i0+1], features)[0]  # [N]
    for _ in range(1, m):
        # distância mínima ao conjunto escolhido
        # (mantemos dists = min_dist para cada ponto)
        if len(chosen) == 1:
            min_dist = dists.clone()
        else:
            min_dist = torch.minimum(min_dist, dists)
        i = min_dist.argmax().item()
        chosen.append(i)
        dists = torch.cdist(features[i:i+1], features)[0]
    return chosen
