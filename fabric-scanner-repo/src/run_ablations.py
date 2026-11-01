"""
run_ablations.py -- trains and evaluates every ablation variant with ONE command and writes the
results table for the report.

All variants use the same fixed-seed 70/15/15 split and the same class-weighted loss, so the
numbers are directly comparable.

Usage (from the src folder):
    python run_ablations.py                      # all variants (several hours on CPU)
    python run_ablations.py --variants spatial_only spatial_dwt     # the two most important ones
    python run_ablations.py --epochs 10          # faster, slightly weaker models
    python run_ablations.py --dry_run            # just print the commands

Checkpoints that already exist are NOT retrained (so you can stop and resume); use --force to retrain.
The existing model_full.pt (DWT + sheen) is reused as the "full" variant.
Outputs: ablation_results.csv, ablation_results.md, logs/<variant>_eval.txt
"""

import argparse
import csv
import os
import re
import subprocess
import sys

CLASSES = ["Cotton", "Denim", "Silk", "Wool", "Polyester", "Leather"]

VARIANTS = {
    "spatial_only":  dict(label="Spatial only (ResNet18)",         flags=["--no_frequency", "--no_sheen"], ckpt="model_spatial_only.pt"),
    "spatial_dwt":   dict(label="Spatial + DWT",                    flags=["--no_sheen"],                    ckpt="model_spatial_dwt.pt"),
    "spatial_sheen": dict(label="Spatial + Sheen",                  flags=["--no_frequency"],                ckpt="model_spatial_sheen.pt"),
    "full_dwt":      dict(label="Full: Spatial + DWT + Sheen",      flags=[],                                ckpt="model_full.pt"),
    "full_gabor":    dict(label="Full: Spatial + Gabor + Sheen",    flags=["--freq_type", "gabor"],          ckpt="model_full_gabor.pt"),
}
DEFAULT_ORDER = ["spatial_only", "spatial_dwt", "spatial_sheen", "full_dwt", "full_gabor"]


def parse_report(text):
    """Extracts accuracy, macro F1 and per-class recall from sklearn's classification_report text."""
    out = {}
    m = re.search(r"accuracy\s+([0-9.]+)\s+(\d+)", text)
    out["accuracy"], out["n_test"] = (float(m.group(1)), int(m.group(2))) if m else (None, None)
    m = re.search(r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text)
    out["macro_f1"] = float(m.group(3)) if m else None
    for c in CLASSES:
        m = re.search(rf"^\s*{c}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text, re.M)
        out[f"recall_{c}"] = float(m.group(2)) if m else None
    return out


def run(cmd, dry, capture=False):
    print("\n>>>", " ".join(cmd))
    if dry:
        return ""
    if capture:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout, res.stderr)
            sys.exit(f"Command failed: {' '.join(cmd)}")
        return res.stdout
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")
    return ""


def main(args):
    os.makedirs("logs", exist_ok=True)
    rows = []
    for name in args.variants:
        v = VARIANTS[name]
        print(f"\n================ {v['label']} ================")
        if os.path.exists(v["ckpt"]) and not args.force:
            print(f"Checkpoint {v['ckpt']} exists - skipping training.")
        else:
            run([sys.executable, "train_ibug.py", "--data_dir", args.data_dir, "--epochs", str(args.epochs),
                 "--batch_size", "16", "--checkpoint", v["ckpt"]] + v["flags"], args.dry_run)
        text = run([sys.executable, "evaluate_ibug.py", "--data_dir", args.data_dir,
                    "--checkpoint", v["ckpt"]] + v["flags"], args.dry_run, capture=True)
        if args.dry_run:
            continue
        open(f"logs/{name}_eval.txt", "w", encoding="utf-8").write(text)
        r = parse_report(text)
        r["variant"] = v["label"]
        rows.append(r)
        print(f"-> accuracy {r['accuracy']}, macro F1 {r['macro_f1']}")

    if not rows:
        return
    fields = ["variant", "accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASSES] + ["n_test"]
    with open("ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])

    md = ["| Variant | Accuracy | Macro F1 | Silk recall | Wool recall | Polyester recall | Leather recall |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['accuracy']:.2f} | {r['macro_f1']:.2f} | {r['recall_Silk']:.2f} | "
                  f"{r['recall_Wool']:.2f} | {r['recall_Polyester']:.2f} | {r['recall_Leather']:.2f} |")
    md.append(f"\nEvaluated on {rows[0]['n_test']} held-out test samples (same fixed-seed split for every variant).")
    open("ablation_results.md", "w", encoding="utf-8").write("\n".join(md))
    print("\n" + "\n".join(md))
    print("\nSaved ablation_results.csv, ablation_results.md and logs/. Send me ablation_results.md.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="archive")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=DEFAULT_ORDER)
    p.add_argument("--force", action="store_true", help="retrain even if the checkpoint exists")
    p.add_argument("--dry_run", action="store_true")
    main(p.parse_args())
