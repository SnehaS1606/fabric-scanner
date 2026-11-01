"""
evaluate.py
Loads a trained checkpoint and reports accuracy, precision, recall,
F1 per class, plus a confusion matrix. Use this on your held-out
test set for the final report.

Usage:
    python evaluate.py --data_dir ./data/raw --checkpoint best_model.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset_ibug import FabricIlluminationDataset, TARGET_CLASSES
from model import FusionFabricClassifier

CLASS_NAMES = TARGET_CLASSES


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_ds = FabricIlluminationDataset(args.data_dir, CLASS_NAMES)

    # Reproduce the EXACT same split train_ibug.py used, so we only
    # evaluate on samples the model never trained on.
    n = len(full_ds)
    n_val = int(0.15 * n)
    n_test = int(0.15 * n)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=generator)

    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(test_ds)} held-out test samples (never seen during training).")

    model = FusionFabricClassifier(
        num_classes=len(CLASS_NAMES),
        use_frequency=not args.no_frequency,
        use_sheen=not args.no_sheen,
        freq_type=args.freq_type,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            off_img = batch["off_image"].to(device)
            diff_img = batch["diff_image"].to(device)
            labels = batch["label"]
            logits = model(off_img, diff_img)
            preds = logits.argmax(1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="archive")
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_frequency", action="store_true", help="Disable the DWT frequency branch (ablation)")
    parser.add_argument("--no_sheen", action="store_true", help="Disable the sheen branch (ablation)")
    parser.add_argument("--freq_type", choices=["dwt", "gabor"], default="dwt", help="Frequency branch type")
    args = parser.parse_args()
    main(args)
