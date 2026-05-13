import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

# Make src/ and scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import evaluate
from src.data import get_dataloaders 


NUM_CLASSES = 37


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_config(loaders, *, epochs, weight_decay, device):
    """PLACEHOLDER, REPLACE WITH STRATEGY"""
    train_loader, val_loader, test_loader, _, _ = loaders

    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    optim = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
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
        val_acc, _ = evaluate(model, val_loader, device)
        print(f" Epoch {epoch}: train_loss={running/n:.4f}  val_acc={val_acc:.4f}")

    test_acc, _ = evaluate(model, test_loader, device)
    return test_acc


def main():
    device = get_device()
    print(f"Device: {device}")

    fractions = [1.0, 0.1, 0.01]
    settings = [
        ("no_aug, wd=0", dict(augment=False, weight_decay=0.0)),
        ("aug, wd=0", dict(augment=True,  weight_decay=0.0)),
        ("no_aug, wd=1e-4", dict(augment=False, weight_decay=1e-4)),
        ("aug, wd=1e-4", dict(augment=True,  weight_decay=1e-4)),
    ]

    results = []
    total_start = time.time()
    for frac in fractions:
        for setting_name, params in settings:
            print(f"\n=== fraction={frac}  setting={setting_name} ===")
            loaders = get_dataloaders(
                task="breed",
                batch_size=32,
                train_fraction=frac,
                augment=params["augment"],
                seed=0,
            )
            t0 = time.time()
            test_acc = train_one_config(
                loaders,
                epochs=5,
                weight_decay=params["weight_decay"],
                device=device,
            )
            elapsed = time.time() - t0
            print(f"  -> test_acc={test_acc:.4f}  ({elapsed:.0f}s)")
            results.append({
                "fraction": frac,
                "setting": setting_name,
                "test_acc": test_acc,
                "seconds": elapsed,
            })

    print(f"\nTotal experiment time: {time.time() - total_start:.0f}s")

    # Summary table
    print("\n========= Summary =========")
    print(f"{'fraction':>10}  {'setting':>18}  {'test_acc':>10}")
    for r in results:
        print(f"{r['fraction']:>10.2f}  {r['setting']:>18}  {r['test_acc']:>10.4f}")

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "limited_data.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
