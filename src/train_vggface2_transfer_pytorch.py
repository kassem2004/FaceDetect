import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from PIL import Image
import numpy as np
import random

from facenet_pytorch import InceptionResnetV1, MTCNN


# Dataset with MTCNN
class FaceFolder(Dataset):
    def __init__(self, root: str, mtcnn: MTCNN, image_size: int = 160, fallback_center_crop: bool = True):
        self.base = ImageFolder(root=root)
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.samples = self.base.samples
        self.mtcnn = mtcnn
        self.image_size = image_size
        self.fallback_center_crop = fallback_center_crop

    def __len__(self):
        return len(self.samples)

    @torch.no_grad()
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # Try MTCNN alignment; returns a 3x160x160 tensor in [0,1]
        face = self.mtcnn(img, return_prob=False)
        if face is None:
            if self.fallback_center_crop:
                # Fallback: center-crop to square then resize
                w, h = img.size
                side = min(w, h)
                left = (w - side) // 2
                top = (h - side) // 2
                img = img.crop((left, top, left + side, top + side))
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                face = TF.to_tensor(img)  # [0,1], shape 3xHxW
            else:
                # If we insist on a detected face, raise
                raise RuntimeError(f"No face detected in {path}")

        # Normalize to [-1, 1] as expected by InceptionResnetV1
        face = (face - 0.5) / 0.5
        return face, label


# Model: Feature extractor + classifier head
class FaceClassifier(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.bn = nn.BatchNorm1d(embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, emb):
        x = self.bn(emb)
        return self.fc(x)


# Utilities
def compute_class_weights(labels: List[int], num_classes: int):
    # Inverse frequency weighting
    counts = np.bincount(labels, minlength=num_classes)
    counts[counts == 0] = 1
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def collate_simple(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys


@torch.no_grad()
def extract_labels_from_subset(subset: torch.utils.data.Subset) -> List[int]:
    # Subset.indices maps into the original dataset.samples labels
    return [subset.dataset.samples[i][1] for i in subset.indices]


def stratified_split_indices(labels: Sequence[int], test_frac: float, seed: int):
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        by_class.setdefault(y, []).append(idx)
    trainval_idx, test_idx = [], []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, int(round(n * test_frac))) if n > 1 else 1  # ensure at least 1 if possible
        test_idx.extend(idxs[:n_test])
        trainval_idx.extend(idxs[n_test:])
    rng.shuffle(trainval_idx)
    rng.shuffle(test_idx)
    return trainval_idx, test_idx


def stratified_kfold_indices(indices: List[int], labels_full: Sequence[int], n_folds: int, seed: int):
    rng = random.Random(seed)
    # group provided indices by class
    by_class: Dict[int, List[int]] = {}
    for i in indices:
        by_class.setdefault(labels_full[i], []).append(i)
    # shuffle within class
    for idxs in by_class.values():
        rng.shuffle(idxs)
    # distribute to folds round-robin per class
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for idxs in by_class.values():
        for j, i in enumerate(idxs):
            folds[j % n_folds].append(i)
    # yield folds
    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = [i for j, fold in enumerate(folds) if j != k for i in fold]
        yield train_idx, val_idx


# Train / Eval
def train_one_epoch(encoder, head, loader, device, optimizer, loss_fn):
    encoder.eval()  # feature extractor stays frozen (or set to eval if frozen)
    head.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.no_grad():
            emb = encoder(xb)  # [B, 512]

        logits = head(emb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum())
        total += xb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(encoder, head, loader, device, loss_fn):
    encoder.eval()
    head.eval()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        emb = encoder(xb)
        logits = head(emb)
        loss = loss_fn(logits, yb)

        total_loss += float(loss) * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum())
        total += xb.size(0)

    return total_loss / total, correct / total


# Save / Load
def save_checkpoint(out_dir: Path, head: FaceClassifier, class_to_idx: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "classifier_head.pt"
    torch.save({"state_dict": head.state_dict(), "class_to_idx": class_to_idx}, ckpt_path)
    with open(out_dir / "class_indices.json", "w", encoding="utf-8") as f:
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        json.dump(idx_to_class, f, indent=2)
    return ckpt_path


def load_checkpoint(ckpt_path: Path, num_classes: Optional[int] = None) -> Tuple[FaceClassifier, dict]:
    data = torch.load(ckpt_path, map_location="cpu")
    class_to_idx = data["class_to_idx"]
    if num_classes is None:
        num_classes = len(class_to_idx)
    head = FaceClassifier(num_classes=num_classes)
    head.load_state_dict(data["state_dict"])
    return head, class_to_idx


# Main routines
def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # MTCNN for detection/alignment
    mtcnn = MTCNN(image_size=160, margin=14, post_process=True, device=device if device.type == "cuda" else None)

    # Full dataset
    ds = FaceFolder(args.data_dir, mtcnn=mtcnn, image_size=160, fallback_center_crop=True)

    # Stratified 90/10 split into trainval/test
    all_labels = [lbl for _, lbl in ds.samples]
    trainval_idx, test_idx = stratified_split_indices(all_labels, test_frac=0.10, seed=args.seed)

    test_ds = Subset(ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=(device.type == "cuda"), collate_fn=collate_simple)

    # Encoder (feature extractor) pretrained on VGGFace2
    encoder = InceptionResnetV1(pretrained="vggface2", classify=False).to(device)
    for p in encoder.parameters():
        p.requires_grad = False  # freeze encoder

    num_classes = len(ds.classes)

    # 5-fold cross-validation over trainval
    best_global_val_acc = -1.0
    best_ckpt_path = None

    fold_id = 0
    for train_idx, val_idx in stratified_kfold_indices(trainval_idx, all_labels, n_folds=5, seed=args.seed):
        fold_id += 1

        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=(device.type == "cuda"), collate_fn=collate_simple)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=(device.type == "cuda"), collate_fn=collate_simple)

        head = FaceClassifier(num_classes=num_classes, dropout=args.dropout).to(device)

        # Class weights per fold
        train_labels = extract_labels_from_subset(train_ds)
        class_weights = compute_class_weights(train_labels, num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(2, args.patience // 2), min_lr=1e-6)

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(encoder, head, train_loader, device, optimizer, loss_fn)
            val_loss, val_acc = evaluate(encoder, head, val_loader, device, loss_fn)
            scheduler.step(val_acc)

            # (Keep original per-epoch printing behavior)
            print(f"[Fold {fold_id}] Epoch {epoch:02d}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

            improved = val_acc > best_val_acc + 1e-4
            if improved:
                best_val_acc = val_acc
                epochs_no_improve = 0
                ckpt_path = save_checkpoint(Path(args.out_dir) / f"fold_{fold_id}", head, ds.class_to_idx)
                print(f"[Fold {fold_id}] Saved best head: {ckpt_path} (val_acc={best_val_acc:.4f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(f"[Fold {fold_id}] Early stopping.")
                break

        # Track global best across folds
        if best_val_acc > best_global_val_acc:
            best_global_val_acc = best_val_acc
            best_ckpt_path = Path(args.out_dir) / f"fold_{fold_id}" / "classifier_head.pt"

    # Evaluate the best model (across folds) on the held-out 10% test set
    assert best_ckpt_path is not None, "No best checkpoint found from cross-validation."
    best_head, _ = load_checkpoint(best_ckpt_path)
    best_head = best_head.to(device).eval()

    # Use plain CE for reporting; it doesn't affect accuracy calculation
    test_loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(encoder, best_head, test_loader, device, test_loss_fn)

    # IMPORTANT: Only print the accuracy (no other stats)
    print(f"{test_acc:.4f}")


@torch.no_grad()
def run_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load encoder and head
    encoder = InceptionResnetV1(pretrained="vggface2", classify=False).to(device).eval()
    head, class_to_idx = load_checkpoint(Path(args.model_path))
    head = head.to(device).eval()

    # Reverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # MTCNN for detection
    mtcnn = MTCNN(image_size=160, margin=14, post_process=True, device=device if device.type == "cuda" else None)

    # Load image
    img = Image.open(args.image).convert("RGB")

    face = mtcnn(img, return_prob=False)
    if face is None:
        # fallback: center-crop + resize
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((160, 160), Image.BILINEAR)
        face = TF.to_tensor(img)

    face = (face - 0.5) / 0.5
    face = face.unsqueeze(0).to(device)

    emb = encoder(face)  # [1, 512]
    logits = head(emb)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = int(np.argmax(probs))
    label = idx_to_class[top_idx]
    conf = float(probs[top_idx])

    result = {
        "label": label,
        "confidence": conf,
        "probs": {idx_to_class[i]: float(p) for i, p in enumerate(probs)}
    }
    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Transfer learning with VGGFace2 encoder (PyTorch)")
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train", help="Train with 5-fold CV on 90% split, then test on held-out 10%")
    pt.add_argument("--data_dir", type=str, default="data_raw", help="Root with class subfolders")
    pt.add_argument("--out_dir", type=str, default="models_vggface2", help="Where to save checkpoints")
    pt.add_argument("--epochs", type=int, default=25)
    pt.add_argument("--batch_size", type=int, default=16)
    pt.add_argument("--seed", type=int, default=1337)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--wd", type=float, default=1e-4)
    pt.add_argument("--dropout", type=float, default=0.4)
    pt.add_argument("--patience", type=int, default=5)
    pt.add_argument("--workers", type=int, default=2)
    pt.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    pp = sub.add_parser("predict", help="Predict a single image")
    pp.add_argument("--model_path", type=str, required=True, help="Path to classifier_head.pt")
    pp.add_argument("--image", type=str, required=True, help="Path to an image to classify")
    pp.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)