import argparse, json, time
import torch
from datasets_assl import build_assl_splits
from assl_core import ASSLConfig, set_seed, ActiveLoop

def main():
    ap = argparse.ArgumentParser("ASSL leve com FixMatch-like + Active Learning (entropy)")
    ap.add_argument("--dataset", choices=["mnist","stl10"], default="mnist")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--img-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--labeled-init", type=int, default=1000) #500
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--budget-per-round", type=int, default=300) #100

    ap.add_argument("--epochs-per-round", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-u-max", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--temp-T", type=float, default=0.7)
    ap.add_argument("--subset-total", type=int, default=0)

    ap.add_argument("--feat-dim", type=int, default=128)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--encoder-ckpt", default=None, help="opcional: pesos de encoder pré-treinado via pretext")
    
    ap.add_argument("--num-workers", type=int, default=0)          # seguro no Windows: 0
    ap.add_argument("--pin-memory", action="store_true")           # default False
    ap.add_argument("--no-pin-memory", action="store_true")        # atalho para desligar
    ap.add_argument("--batch-l", type=int, default=128)            # L batch size
    ap.add_argument("--batch-u", type=int, default=128)            # U batch size
    ap.add_argument("--batch-t", type=int, default=256)            # Test batch size
    ap.add_argument("--prefetch-factor", type=int, default=2)      # ignorado se workers==0

    args = ap.parse_args()
    set_seed(args.seed)

    cfg = ASSLConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        img_size=args.img_size,
        device=args.device,
        seed=args.seed,
        labeled_init=args.labeled_init,
        rounds=args.rounds,
        budget_per_round=args.budget_per_round,
        epochs_per_round=args.epochs_per_round,
        lr=args.lr,
        lambda_u_max=args.lambda_u_max,
        tau=args.tau,
        temp_T=args.temp_T,
        subset_total=args.subset_total,
        feat_dim=args.feat_dim,
        freeze_encoder_during_head=args.freeze_encoder,
        encoder_ckpt=args.encoder_ckpt,
        n_classes=10,  # MNIST/STL-10
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        batch_l=args.batch_l,
        batch_u=args.batch_u,
        batch_t=args.batch_t,
        prefetch_factor=args.prefetch_factor,
    )
    
    if args.no_pin_memory:
        args.pin_memory = False
    # Em CPU/Windows é comum preferir 0 workers:
    if args.device == "cpu" and args.num_workers < 0:
        args.num_workers = 0

    print("== Config ==")
    for k,v in vars(cfg).items():
        print(f"{k}: {v}")

    # Builds splits/loaders
    L_idx, U_idx, L_loader, T_loader, U_loader, base_LU, n_classes = build_assl_splits(
        dataset=cfg.dataset, img_size=cfg.img_size, data_dir=cfg.data_dir,
        labeled_init=cfg.labeled_init, subset_total=cfg.subset_total,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        batch_l=cfg.batch_l, batch_u=cfg.batch_u, batch_t=cfg.batch_t,
        prefetch_factor=cfg.prefetch_factor
    )
    cfg.n_classes = n_classes

    loop = ActiveLoop(cfg, base_LU, L_idx, U_idx, L_loader, T_loader, U_loader)
    t0 = time.time()
    log = loop.run()
    print(f"[done] total secs: {time.time()-t0:.1f}")
    print(json.dumps(log, indent=2))

if __name__ == "__main__":
    main()
