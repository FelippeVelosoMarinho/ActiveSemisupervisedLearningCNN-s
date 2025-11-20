# run_assl_stl10.py
from assl_core import ASSLConfig, ActiveLoop, set_seed
from datasets_assl import build_assl_splits
import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("./results_stl10")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def run_single(cfg: ASSLConfig):
    print(f"\nRodando {cfg.dataset} | seed={cfg.seed} | pretext={cfg.pretext_task} | "
          f"lambda_u_max={cfg.lambda_u_max} | lambda_pretext={cfg.lambda_pretext} | "
          f"L_init={cfg.labeled_init}")

    set_seed(cfg.seed)

    # monta splits
    L_idx, U_idx, L_loader, T_loader, U_loader, base_LU, n_classes = build_assl_splits(
        dataset=cfg.dataset,
        img_size=cfg.img_size,
        data_dir=cfg.data_dir,
        labeled_init=cfg.labeled_init,
        subset_total=cfg.subset_total,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        batch_l=cfg.batch_l,
        batch_u=cfg.batch_u,
        batch_t=cfg.batch_t,
        prefetch_factor=cfg.prefetch_factor,
    )
    cfg.n_classes = n_classes

    loop = ActiveLoop(cfg, base_LU, L_idx, U_idx, L_loader, T_loader, U_loader)
    log_rounds = loop.run()  # lista de dicts: {round, test_acc, L, U}

    # salva metadata + log em JSON
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = {
        "config": cfg.__dict__,
        "log_rounds": log_rounds,
    }
    out_path = RESULTS_DIR / f"stl10_{stamp}_seed{cfg.seed}_{cfg.pretext_task}_Lu{cfg.lambda_u_max}_Lp{cfg.lambda_pretext}_Linit{cfg.labeled_init}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Salvo em: {out_path}")


def main():
    base_cfg = dict(
        dataset="stl10",
        data_dir="./data",
        img_size=64,               # 64 é um bom compromisso
        rounds=8,                  # 8 rodadas de AL
        budget_per_round=300,      # 300 novos rótulos/rodada
        epochs_per_round=8,
        lr=1e-3,
        lambda_u_max=1.0,          # default (pode mudar por cenário)
        tau=0.7,
        temp_T=0.5,
        subset_total=0,            # 0 = usa todo STL-10 train+unlabeled
        feat_dim=128,
        n_classes=10,
        num_workers=0,             # CPU + Windows: manter 0
        pin_memory=False,
        batch_l=128,
        batch_u=128,
        batch_t=256,
        prefetch_factor=2,
        freeze_encoder_during_head=False,
        encoder_ckpt=None,
        pretext_task=None,
        lambda_pretext=0.0,
    )

    # ========= CENÁRIOS =========
    scenarios = []

    # 1) Baseline "quase supervisionado" (sem SSL, sem pretext)
    #    Faz 1 rodada só, sem mover amostras (budget_per_round=0)
    for L_init in [500, 1000]:
        scenarios.append(dict(
            name=f"sup_only_L{L_init}",
            overrides=dict(
                labeled_init=L_init,
                rounds=1,
                budget_per_round=0,
                lambda_u_max=0.0,
                pretext_task="none",
                lambda_pretext=0.0,
            ),
        ))

    # 2) ASSL puro (SSL + AL, sem pretext)
    for L_init in [500, 1000]:
        for seed in [13, 42]:
            scenarios.append(dict(
                name=f"assl_no_pretext_L{L_init}_seed{seed}",
                overrides=dict(
                    labeled_init=L_init,
                    seed=seed,
                    lambda_u_max=1.0,
                    pretext_task="none",
                    lambda_pretext=0.0,
                ),
            ))

    # 3) ASSL + Rotation
    for L_init in [500]:
        for seed in [13, 42]:
            scenarios.append(dict(
                name=f"assl_rot_L{L_init}_seed{seed}",
                overrides=dict(
                    labeled_init=L_init,
                    seed=seed,
                    lambda_u_max=1.0,
                    pretext_task="rotation",
                    lambda_pretext=0.5,
                ),
            ))

    # 4) ASSL + Colorization
    for L_init in [500]:
        for seed in [13, 42]:
            scenarios.append(dict(
                name=f"assl_color_L{L_init}_seed{seed}",
                overrides=dict(
                    labeled_init=L_init,
                    seed=seed,
                    lambda_u_max=1.0,
                    pretext_task="colorization",
                    lambda_pretext=0.5,
                ),
            ))

    # ========= EXECUÇÃO DA BATERIA =========
    for sc in scenarios:
        cfg_dict = {**base_cfg, **sc["overrides"]}
        cfg = ASSLConfig(**cfg_dict)
        run_single(cfg)


if __name__ == "__main__":
    main()
