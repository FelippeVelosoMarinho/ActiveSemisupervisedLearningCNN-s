#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, re, subprocess, sys, time
from pathlib import Path

METRICS_REGEX = {
    "test_acc":     re.compile(r"\[final\].*test_acc[:=]\s*([0-9.]+)", re.I),
    "pretext_secs": re.compile(r"\[pretext\].*secs[:=]\s*([0-9.]+)", re.I),
    "probe_secs":   re.compile(r"\[probe\].*secs[:=]\s*([0-9.]+)", re.I),
    "params":       re.compile(r"\[model\].*params[:=]\s*([0-9,]+)", re.I),
}

ARG_ALIAS = {
    "img_size":       "img-size",
    "epochs_pretext": "epochs-pretext",
    "epochs_probe":   "epochs-linear",
    "batch_pretext":  "batch-pretext",
    "batch_probe":    "batch-probe",
}

def parse_stdout(stdout: str) -> dict:
    out = {}
    for key, rx in METRICS_REGEX.items():
        m = rx.search(stdout)
        if not m:
            continue
        val = m.group(1)
        if key == "params":
            out[key] = int(val.replace(",", ""))
        else:
            out[key] = float(val)
    return out

def run_one(ssl_script: Path, cfg: dict, dry=False, verbose=True) -> dict:
    cmd = [sys.executable, str(ssl_script)]

    # Monte os argumentos usando o alias correto (sem duplicar nada)
    for k, v in cfg.items():
        if v is None:
            continue
        arg_name = ARG_ALIAS.get(k, k)
        if isinstance(v, bool):
            cmd.append(f"--{arg_name}" if v else f"--no-{arg_name}")
        else:
            cmd.append(f"--{arg_name}={v}")

    if verbose:
        print("==> Running:", " ".join(cmd))
    if dry:
        return {"dry": True, "cmd": " ".join(cmd)}

    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    stdout = p.stdout or ""
    stderr = p.stderr or ""

    if verbose:
        print("---- stdout (tail) ----")
        print(stdout[-2000:])
        if p.returncode != 0:
            print("---- stderr (tail) ----")
            print(stderr[-2000:])

    parsed = parse_stdout(stdout)
    parsed.update({
        "returncode": p.returncode,
        "elapsed_wall": dt,
        "stdout_tail": stdout[-500:].replace("\n", " / "),
    })
    return parsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl_script", default="ssl_pretext.py", help="./ssl_pretext.py")
    ap.add_argument("--config", default="experiments_config.json", help="./experiments_config.json")
    ap.add_argument("--out_csv", default="experiments_results.csv")
    ap.add_argument("--dry", action="store_true", help="Apenas mostrar os comandos, sem executar")
    args = ap.parse_args()

    ssl_script = Path(args.ssl_script)
    if not ssl_script.exists():
        print(f"[error] ssl_script não encontrado: {ssl_script}", file=sys.stderr)
        sys.exit(2)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[error] config não encontrado: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    runs = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(runs, list) or not runs:
        print("[error] config deve ser uma lista não vazia de dicionários", file=sys.stderr)
        sys.exit(2)

    fieldnames = [
        "dataset","task","img_size","subset",
        "epochs_pretext","epochs_probe",
        "batch_pretext","batch_probe",
        "seed","params","pretext_secs","probe_secs",
        "test_acc","returncode","elapsed_wall","stdout_tail"
    ]

    out_rows = []
    for i, cfg in enumerate(runs, 1):
        print(f"\n=== Run {i}/{len(runs)}: {cfg.get('dataset')} | {cfg.get('task')} ===")
        res = run_one(ssl_script, cfg, dry=args.dry, verbose=True)

        row = {k: cfg.get(k) for k in [
            "dataset","task","img_size","subset",
            "epochs_pretext","epochs_probe",
            "batch_pretext","batch_probe","seed"
        ]}
        row.update({k: res.get(k) for k in [
            "params","pretext_secs","probe_secs",
            "test_acc","returncode","elapsed_wall","stdout_tail"
        ]})
        out_rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n[done] Resultados salvos em {args.out_csv}")

if __name__ == "__main__":
    main()
