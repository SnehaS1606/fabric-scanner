"""
train.py
End-to-end training loop for the fusion classifier.

Usage:
    python train.py --data_dir ./data/raw --epochs 20 --batch_size 16
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from dataset_ibug import FabricIlluminationDataset, TARGET_CLASSES
from model import FusionFabricClassifier

CLASS_NAMES = TARGET_CLASSES  # ["Cotton", "Denim", "Silk", "Wool", "Polyester", "Leather"]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_ds = FabricIlluminationDataset(args.data_dir, CLASS_NAMES)
    n = len(full_ds)
    n_val = int(0.15 * n)
    n_test = int(0.15 * n)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(42)  # fixed seed so evaluate.py can reproduce this split
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = FusionFabricClassifier(
        num_classes=len(CLASS_NAMES),
        use_frequency=not args.no_frequency,
        use_sheen=not args.no_sheen,
        freq_type=args.freq_type,
    ).to(device)

    # Stage 1: freeze the pretrained backbone, train fusion + other branches
    for p in model.spatial.features.parameters():
        p.requires_grad = False

    # Class weighting: rarer classes (silk, wool, leather) get a higher loss
    # weight so the model can't just ignore them in favor of cotton/polyester.
    class_counts = torch.zeros(len(CLASS_NAMES))
    for _, label in full_ds.samples:
        class_counts[label] += 1
    class_weights = (1.0 / class_counts) * class_counts.sum() / len(CLASS_NAMES)
    class_weights = class_weights.to(device)
    print("Class counts:", dict(zip(CLASS_NAMES, class_counts.tolist())))
    print("Class weights:", dict(zip(CLASS_NAMES, class_weights.tolist())))

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Unfreeze backbone halfway through training (stage 2 fine-tuning)
        if epoch == args.epochs // 2:
            for p in model.spatial.features.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-5)
            print("Unfroze backbone for fine-tuning.")

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            off_img = batch["off_image"].to(device)
            diff_img = batch["diff_image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(off_img, diff_img)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={running_loss/total:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"  Saved new best checkpoint (val_acc={val_acc:.3f})")

    print(f"Training done. Best val acc: {best_val_acc:.3f}")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            off_img = batch["off_image"].to(device)
            diff_img = batch["diff_image"].to(device)
            labels = batch["label"].to(device)
            logits = model(off_img, diff_img)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="archive")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--no_frequency", action="store_true", help="Disable the DWT frequency branch (ablation)")
    parser.add_argument("--no_sheen", action="store_true", help="Disable the sheen branch (ablation)")
    parser.add_argument("--freq_type", choices=["dwt", "gabor"], default="dwt", help="Frequency branch type")
    args = parser.parse_args()
    train(args)
