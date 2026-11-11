#!/usr/bin/env python3
"""
train.py - Transfer-learning trainer for retinal image classification (PyTorch).

DATASET LAYOUT
--------------
Each sub-folder name becomes a class label. Two layouts are supported:

  Option A - already split:           Option B - one folder (auto stratified split):
    data/                               data/
      train/                              Normal/      img001.jpg ...
        Normal/    *.jpg                  Glaucoma/    ...
        Glaucoma/  ...                    Cataract/    ...
      val/            (or valid/ or validation/)
        Normal/    ...
        Glaucoma/  ...

USAGE
-----
  python train.py --data-dir ./data
  python train.py --data-dir ./data --arch resnet18 --epochs 30 --batch-size 16
  python train.py --help                  # see every option

OUTPUT
------
  models/best_model.pth        weights + metadata (class names, architecture, image size)
  models/training_history.json per-epoch loss / accuracy

HOW TRAINING WORKS
------------------
1. Load an ImageNet-pretrained network (ResNet / EfficientNet).
2. Replace its final layer with a new one sized for YOUR number of classes.
3. Phase 1 ("warm-up", --freeze-epochs): only the new head trains; the
   pretrained backbone is frozen so random head weights can't wreck it.
4. Phase 2: unfreeze everything and fine-tune the whole network with a small
   learning rate for the backbone and a larger one for the head.
5. After every epoch, evaluate on the validation set and save the weights
   whenever validation accuracy improves.
"""

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ImageNet statistics - pretrained models expect inputs normalised with these.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_ARCHS = ["resnet18", "resnet34", "resnet50", "efficientnet_b0"]
VAL_FOLDER_NAMES = ["val", "valid", "validation"]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Make runs repeatable (as far as GPU non-determinism allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device: NVIDIA GPU > Apple Silicon GPU > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Data loading & augmentation
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int):
    """
    Training transforms add random variation so the model generalises instead of
    memorising. The augmentations are chosen for fundus photos:
      * flips / rotations - the retina has no fixed "up", and left vs right eyes
        are mirror images, so these are realistic variations;
      * mild crop/zoom and colour jitter - mimic differences between cameras,
        illumination and patient positioning.
    Strong hue shifts are avoided because colour carries diagnostic information.

    Validation/inference transforms are deterministic: just resize + normalise.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def stratified_split(targets, val_fraction: float, seed: int):
    """
    Split sample indices into train/val while keeping each class's proportion
    the same in both sets (important when some diseases are rare).
    """
    rng = random.Random(seed)
    by_class = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for label in sorted(by_class):
        idxs = by_class[label]
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_fraction))
        # Every class with >=2 images gets at least one validation and one training image.
        n_val = min(max(n_val, 1), len(idxs) - 1) if len(idxs) >= 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def load_datasets(data_dir: str, img_size: int, val_split: float, seed: int):
    """
    Returns (train_dataset, val_dataset, class_names, train_targets).
    ImageFolder assigns label indices alphabetically by folder name.
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root.resolve()}")

    train_tf, eval_tf = build_transforms(img_size)
    train_dir = root / "train"
    val_dir = next((root / n for n in VAL_FOLDER_NAMES if (root / n).is_dir()), None)

    if train_dir.is_dir() and val_dir is not None:
        # Option A: the user already split the data.
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
        if train_ds.classes != val_ds.classes:
            raise ValueError(
                "Class folders differ between train and validation sets:\n"
                f"  train: {train_ds.classes}\n  val:   {val_ds.classes}"
            )
        print(f"Using pre-split data: {train_dir} | {val_dir}")
        return train_ds, val_ds, train_ds.classes, list(train_ds.targets)

    # Option B: a single folder of class sub-folders -> make our own split.
    # Two ImageFolder views of the same files so train and val get different transforms.
    base_train = datasets.ImageFolder(root, transform=train_tf)
    base_val = datasets.ImageFolder(root, transform=eval_tf)
    train_idx, val_idx = stratified_split(base_train.targets, val_split, seed)
    if not val_idx:
        raise ValueError("Not enough images to create a validation set.")
    print(f"No train/val folders found - created a stratified {1 - val_split:.0%}/{val_split:.0%} split of {root}")
    train_targets = [base_train.targets[i] for i in train_idx]
    return Subset(base_train, train_idx), Subset(base_val, val_idx), base_train.classes, train_targets


def compute_class_weights(train_targets, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights for the loss: rare classes count more, so the model
    can't score well by always predicting the majority class (e.g. 'Normal').
    """
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero for an empty class
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Load a pretrained backbone and swap its final layer for a new head:
        Dropout -> Linear(features, num_classes)
    NOTE: predict.py rebuilds this exact structure, so keep them in sync.
    """
    if arch.startswith("resnet"):
        ctor, weights = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }[arch]
        model = ctor(weights=weights if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED_ARCHS}")
    return model


def get_head(model: nn.Module) -> nn.Module:
    """Return the newly-added classification head."""
    return model.fc if hasattr(model, "fc") else model.classifier


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze every parameter except the head (which always trains)."""
    for p in model.parameters():
        p.requires_grad = trainable
    for p in get_head(model).parameters():
        p.requires_grad = True


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    """
    One pass over `loader`. Trains if an optimizer is given, otherwise evaluates.
    Returns (mean_loss, accuracy, all_predictions, all_targets).
    """
    is_train = optimizer is not None
    model.train(is_train)  # enables dropout & batch-norm updates only when training

    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_targets = [], []

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixed precision (CUDA only): faster, lower memory, same accuracy.
            amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                       if use_amp else contextlib.nullcontext())
            with amp_ctx:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()  # scaler is a no-op when AMP is off
                scaler.step(optimizer)
                scaler.update()

            preds = logits.argmax(dim=1)
            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (preds == labels).sum().item()
            seen += batch
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    return total_loss / max(seen, 1), correct / max(seen, 1), all_preds, all_targets


def print_report(preds, targets, class_names) -> None:
    """Per-class precision / recall / F1 and a confusion matrix (no sklearn needed)."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1

    width = max(len(c) for c in class_names) + 2
    print(f"\n{'Class':<{width}}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        precision = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        recall = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{name:<{width}}{precision:>10.3f}{recall:>10.3f}{f1:>10.3f}{cm[i, :].sum():>10d}")

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    print(" " * width + "".join(f"{i:>6d}" for i in range(n)))
    for i, name in enumerate(class_names):
        print(f"{name:<{width - 4}}[{i:>2d}] " + "".join(f"{v:>6d}" for v in cm[i]))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a pretrained CNN on retinal images.")
    p.add_argument("--data-dir", required=True, help="Dataset root (see layout in file header).")
    p.add_argument("--output-dir", default="models", help="Where to save the model and history.")
    p.add_argument("--arch", default="resnet50", choices=SUPPORTED_ARCHS)
    p.add_argument("--img-size", type=int, default=224, help="Input resolution (pixels).")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--freeze-epochs", type=int, default=3,
                   help="Warm-up epochs training only the new head (0 = fine-tune all from the start).")
    p.add_argument("--batch-size", type=int, default=32, help="Lower this if you run out of GPU memory.")
    p.add_argument("--lr-head", type=float, default=1e-3, help="Learning rate for the new head.")
    p.add_argument("--lr-backbone", type=float, default=1e-4, help="Learning rate for pretrained layers.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Validation fraction when the data isn't pre-split.")
    p.add_argument("--patience", type=int, default=7,
                   help="Stop if val accuracy hasn't improved for this many epochs (0 = off).")
    p.add_argument("--class-weights", action=argparse.BooleanOptionalAction, default=True,
                   help="Weight the loss by inverse class frequency (helps with imbalance).")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="Start from ImageNet weights (downloaded once on first run).")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Mixed-precision training on CUDA GPUs.")
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="Parallel data-loading processes (use 0 if you hit loader errors).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    use_amp = args.amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"

    print(f"Device: {device} | mixed precision: {use_amp}")

    # ---- 1. Data ---------------------------------------------------------------
    train_ds, val_ds, class_names, train_targets = load_datasets(
        args.data_dir, args.img_size, args.val_split, args.seed)
    num_classes = len(class_names)
    if num_classes < 2:
        raise SystemExit("Need at least 2 class folders to train a classifier.")

    counts = np.bincount(train_targets, minlength=num_classes)
    print(f"Classes ({num_classes}): " + ", ".join(f"{c}={n}" for c, n in zip(class_names, counts)))
    print(f"Train images: {len(train_ds)} | Validation images: {len(val_ds)}")

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=device.type == "cuda",
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ---- 2. Model --------------------------------------------------------------
    model = build_model(args.arch, num_classes, pretrained=args.pretrained, dropout=args.dropout).to(device)
    print(f"Model: {args.arch} ({'ImageNet-pretrained' if args.pretrained else 'random init'})")

    # ---- 3. Loss, optimiser, scheduler ----------------------------------------
    weight = compute_class_weights(train_targets, num_classes).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    # Two parameter groups: gentle updates for pretrained layers, faster for the new head.
    head_ids = {id(p) for p in get_head(model).parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr_backbone},
         {"params": list(get_head(model).parameters()), "lr": args.lr_head}],
        weight_decay=args.weight_decay,
    )
    # Cosine schedule: learning rate decays smoothly to ~0 by the final epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Phase 1: freeze the backbone (frozen params get no gradients, so AdamW skips them).
    if args.freeze_epochs > 0:
        set_backbone_trainable(model, False)
        print(f"Phase 1: training only the new head for {args.freeze_epochs} epoch(s).")

    # ---- 4. Training loop ------------------------------------------------------
    history = []
    best_acc, best_loss, best_epoch, epochs_without_improvement = -1.0, float("inf"), 0, 0

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            print("Phase 2: unfreezing backbone - fine-tuning the whole network.")

        start = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, device,
                                                optimizer, scaler, use_amp)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "lr_backbone": optimizer.param_groups[0]["lr"],
                        "lr_head": optimizer.param_groups[1]["lr"]})

        # ---- 5. Save on improvement (ties broken by lower validation loss) ------
        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
        marker = ""
        if improved:
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "num_classes": num_classes,
                "class_names": list(class_names),
                "img_size": args.img_size,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, checkpoint_path)
            marker = "  <- best, saved"
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.2%} | "
              f"val loss {val_loss:.4f} acc {val_acc:.2%} | "
              f"{time.time() - start:.0f}s{marker}")

        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- 6. Final report on the best checkpoint -------------------------------
    print(f"\nBest validation accuracy {best_acc:.2%} at epoch {best_epoch}. Saved to {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    _, _, preds, targets = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
    print_report(preds, targets, class_names)


if __name__ == "__main__":  # guard required for multi-worker data loading on Windows/macOS
    main()
