"""
Strategy 1: fine-tune the last l layers of a pretrained ResNet34 from the start of training.

"Layer" here means a ResNet stage: layer1, layer2, layer3, layer4.
With l = 1 only layer4 (+ fc) is trainable; with l = 4 all four stages
(+ fc) are trainable. Everything earlier stays frozen.

Run:
    python scripts/partialtuning.py --l 1
    python scripts/partialtuning.py --l 2
    ...
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision.models import resnet34, ResNet34_Weights

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import get_dataloaders
from src.evaluation import evaluate

NUM_CLASSES = 37

# last entry is the deepest stage, i.e. the one closest to fc.
# unfreeze from the right, so STAGES[-l:] is the set of stages trainable
# for a given l.
STAGES = ["layer1", "layer2", "layer3", "layer4"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(l: int, device):
    if not 1 <= l <= len(STAGES):
        raise ValueError(f"l must be in [1, {len(STAGES)}], got {l}")

    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    # Start by freezing everything, then unfreeze
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the last l stages.
    trainable_stages = STAGES[-l:]
    for stage_name in trainable_stages:
        for p in getattr(model, stage_name).parameters():
            p.requires_grad = True
            
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # sanity print so we can see what is trainable.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"l = {l}: trainable stages = {trainable_stages} + fc")
    print(f"trainable params: {trainable:,} / {total:,} "
          f"({100.0 * trainable / total:.2f}%)")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, default=1,
                        help="Number of trailing ResNet stages to unfreeze (1-4).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # 37 breed labels
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        task="breed", batch_size=args.batch_size, image_size=224
    )

    model = build_model(args.l, device)

    # Pass only the trainable parameters to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable_params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

    ckpt_dir = Path(__file__).resolve().parent.parent / "results" / "partialtuning"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_l{args.l}.pt"

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / n

        val_acc, val_f1 = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  "
              f"time={epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, ckpt_path)

    total_time = time.time() - train_start
    print(f"Total training time: {total_time:.1f}s "
          f"({total_time / 60:.2f} min) for {args.epochs} epochs")
    print(f"Best val_acc={best_val_acc:.4f} at epoch {best_epoch}; "
          f"best model saved to {ckpt_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    val_acc, val_f1 = evaluate(model, val_loader, device)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"Final best model (l={args.l}, epoch {best_epoch}): "
          f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}  "
          f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()