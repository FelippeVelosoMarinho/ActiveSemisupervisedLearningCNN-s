import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime

# Pasta onde está o run_assl.py (ajuste se necessário)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ASSL = os.path.join(THIS_DIR, "run_assl.py")

RESULTS_DIR = os.path.join(THIS_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "assl_multitask_results.csv")


def parse_json_log(stdout: str):
    """
    Seu run_assl.py imprime várias linhas e no final um json.dumps(log, indent=2).
    Aqui pegamos o ÚLTIMO bloco que parece JSON (começa em '[' e termina em ']').
    """
    # procura o último '[' até o final
    match = re.findall(r"(\[\s*{.*}\s*\])", stdout, flags=re.S | re.M)
    if not match:
        raise ValueError("Não foi possível encontrar JSON no stdout.\nTrecho:\n" + stdout[-1000:])
    json_str = match[-1]
    log = json.loads(json_str)
    return log


def run_single_experiment(exp_cfg: dict):
    """
    exp_cfg: dict com chaves como dataset, img_size, pretext_task, etc.
    Retorna lista de linhas (dicts) para salvar no CSV.
    """
    # Monta o comando para chamar run_assl.py
    cmd = [
        sys.executable,
        RUN_ASSL,
        "--dataset", exp_cfg["dataset"],
        "--img-size", str(exp_cfg["img_size"]),
        "--device", exp_cfg["device"],
        "--labeled-init", str(exp_cfg["labeled_init"]),
        "--rounds", str(exp_cfg["rounds"]),
        "--budget-per-round", str(exp_cfg["budget_per_round"]),
        "--epochs-per-round", str(exp_cfg["epochs_per_round"]),
        "--lr", str(exp_cfg["lr"]),
        "--lambda-u-max", str(exp_cfg["lambda_u_max"]),
        "--tau", str(exp_cfg["tau"]),
        "--temp-T", str(exp_cfg["temp_T"]),
        "--feat-dim", str(exp_cfg["feat_dim"]),
        "--seed", str(exp_cfg["seed"]),
        "--batch-l", str(exp_cfg["batch_l"]),
        "--batch-u", str(exp_cfg["batch_u"]),
        "--batch-t", str(exp_cfg["batch_t"]),
        "--num-workers", str(exp_cfg["num_workers"]),
        "--prefetch-factor", str(exp_cfg["prefetch_factor"]),
        "--data-dir", exp_cfg.get("data_dir", "./data"),
    ]

    # pin_memory flags
    if exp_cfg.get("pin_memory", False):
        cmd.append("--pin-memory")
    else:
        cmd.append("--no-pin-memory")

    # pretext
    pretext_task = exp_cfg.get("pretext_task", "none")
    cmd += [
        "--pretext-task", pretext_task,
        "--lambda-pretext", str(exp_cfg["lambda_pretext"]),
    ]

    print("\n======================================")
    print("Rodando experimento:")
    print(exp_cfg)
    print("Comando:", " ".join(cmd))
    print("======================================\n")

    p = subprocess.run(cmd, capture_output=True, text=True)

    stdout = p.stdout
    stderr = p.stderr

    if p.returncode != 0:
        print("ERRO no experimento!", file=sys.stderr)
        print("STDOUT:\n", stdout, file=sys.stderr)
        print("STDERR:\n", stderr, file=sys.stderr)
        raise RuntimeError("run_assl.py retornou código != 0")

    # tenta achar a linha [done] total secs: X
    total_secs = None
    for line in stdout.splitlines():
        if "[done] total secs:" in line:
            try:
                total_secs = float(line.strip().split()[-1])
            except Exception:
                pass

    log = parse_json_log(stdout)  # lista de dicts: {round, test_acc, L, U}

    rows = []
    dt_str = datetime.now().isoformat(timespec="seconds")

    for item in log:
        row = {
            "timestamp": dt_str,
            "dataset": exp_cfg["dataset"],
            "img_size": exp_cfg["img_size"],
            "pretext_task": pretext_task,
            "lambda_pretext": exp_cfg["lambda_pretext"],
            "lambda_u_max": exp_cfg["lambda_u_max"],
            "seed": exp_cfg["seed"],
            "labeled_init": exp_cfg["labeled_init"],
            "rounds": exp_cfg["rounds"],
            "budget_per_round": exp_cfg["budget_per_round"],
            "epochs_per_round": exp_cfg["epochs_per_round"],
            "tau": exp_cfg["tau"],
            "temp_T": exp_cfg["temp_T"],
            "feat_dim": exp_cfg["feat_dim"],
            "batch_l": exp_cfg["batch_l"],
            "batch_u": exp_cfg["batch_u"],
            "batch_t": exp_cfg["batch_t"],
            "num_workers": exp_cfg["num_workers"],
            "prefetch_factor": exp_cfg["prefetch_factor"],
            "data_dir": exp_cfg.get("data_dir", "./data"),
            "round": item.get("round", -1),
            "L_size": item.get("L", -1),
            "U_size": item.get("U", -1),
            "test_acc": item.get("test_acc", -1.0),
            "total_secs": total_secs if total_secs is not None else "",
        }
        rows.append(row)
    return rows


def init_csv_if_needed():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "dataset",
                "img_size",
                "pretext_task",
                "lambda_pretext",
                "lambda_u_max",
                "seed",
                "labeled_init",
                "rounds",
                "budget_per_round",
                "epochs_per_round",
                "tau",
                "temp_T",
                "feat_dim",
                "batch_l",
                "batch_u",
                "batch_t",
                "num_workers",
                "prefetch_factor",
                "data_dir",
                "round",
                "L_size",
                "U_size",
                "test_acc",
                "total_secs",
            ])


def append_rows(rows):
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([
                r["timestamp"],
                r["dataset"],
                r["img_size"],
                r["pretext_task"],
                r["lambda_pretext"],
                r["lambda_u_max"],
                r["seed"],
                r["labeled_init"],
                r["rounds"],
                r["budget_per_round"],
                r["epochs_per_round"],
                r["tau"],
                r["temp_T"],
                r["feat_dim"],
                r["batch_l"],
                r["batch_u"],
                r["batch_t"],
                r["num_workers"],
                r["prefetch_factor"],
                r["data_dir"],
                r["round"],
                r["L_size"],
                r["U_size"],
                r["test_acc"],
                r["total_secs"],
            ])


def main():
    init_csv_if_needed()

    # --------- Definição da bateria de testes ---------
    seeds = [13, 42, 123]

    experiments = []

    # MNIST: baselines + pré-texto (rotation, colorization)
    for seed in seeds:
        # Baseline: supervisionado fraco (sem SSL, sem pretext)
        experiments.append(dict(
            dataset="mnist",
            img_size=32,
            device="cpu",
            labeled_init=500,
            rounds=3,
            budget_per_round=100,
            epochs_per_round=2,
            lr=1e-3,
            lambda_u_max=0.0,   # SSL desligado
            tau=0.7,
            temp_T=0.5,
            feat_dim=128,
            seed=seed,
            pretext_task="none",
            lambda_pretext=0.0,
            batch_l=128,
            batch_u=128,
            batch_t=256,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            data_dir="./data",
        ))

        # Baseline: FixMatch-like puro (SSL, sem pré-texto)
        experiments.append(dict(
            dataset="mnist",
            img_size=32,
            device="cpu",
            labeled_init=500,
            rounds=8,
            budget_per_round=300,
            epochs_per_round=8,
            lr=1e-3,
            lambda_u_max=1.0,
            tau=0.7,
            temp_T=0.5,
            feat_dim=128,
            seed=seed,
            pretext_task="none",
            lambda_pretext=0.0,
            batch_l=128,
            batch_u=128,
            batch_t=256,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            data_dir="./data",
        ))

        # Proposta: SSL + Rotation
        experiments.append(dict(
            dataset="mnist",
            img_size=32,
            device="cpu",
            labeled_init=500,
            rounds=8,
            budget_per_round=300,
            epochs_per_round=8,
            lr=1e-3,
            lambda_u_max=1.0,
            tau=0.7,
            temp_T=0.5,
            feat_dim=128,
            seed=seed,
            pretext_task="rotation",
            lambda_pretext=0.5,
            batch_l=128,
            batch_u=128,
            batch_t=256,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            data_dir="./data",
        ))

        # Proposta: SSL + Colorization
        experiments.append(dict(
            dataset="mnist",
            img_size=32,
            device="cpu",
            labeled_init=500,
            rounds=8,
            budget_per_round=300,
            epochs_per_round=8,
            lr=1e-3,
            lambda_u_max=1.0,
            tau=0.7,
            temp_T=0.5,
            feat_dim=128,
            seed=seed,
            pretext_task="colorization",
            lambda_pretext=0.5,
            batch_l=128,
            batch_u=128,
            batch_t=256,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=2,
            data_dir="./data",
        ))

    # ASSL multi-task + pré-texto (Rotation, por ex.)
    experiments.append(dict(
        dataset="mnist",
        img_size=32,
        device="cpu",
        labeled_init=500,
        rounds=8,
        budget_per_round=300,
        epochs_per_round=8,
        lr=1e-3,
        lambda_u_max=1.0,         # SSL ON
        tau=0.7,
        temp_T=0.5,
        feat_dim=128,
        seed=seed,
        pretext_task="rotation",  # usa pré-texto
        lambda_pretext=0.5,       # peso da loss de pretext
        batch_l=128,
        batch_u=128,
        batch_t=256,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        data_dir="./data",
    ))

    # STL-10: pode adicionar depois se quiser (igual estrutura)

    # --------- Execução ---------
    for exp in experiments:
        rows = run_single_experiment(exp)
        append_rows(rows)
        print(f"Experimento concluído. Salvo {len(rows)} linhas em {RESULTS_CSV}")


if __name__ == "__main__":
    main()
