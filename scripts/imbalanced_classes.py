import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torchvision.models import resnet34, ResNet34_Weights

# Make src/ and scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation import evaluate_detailed
from src.data import cat_class_ids, get_dataloaders


NUM_CLASSES = 37


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_config(loaders, *, epochs, loss_fn, device):
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
    )

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
        print(f"Epoch {epoch}: train_loss={running/n:.4f}")

    return model


def summarize_per_group(report, cat_ids, class_names):
    cat_f1s = [report[class_names[i]]["f1-score"] for i in cat_ids]
    dog_ids = [i for i in range(len(class_names)) if i not in cat_ids]
    dog_f1s = [report[class_names[i]]["f1-score"] for i in dog_ids]
    return sum(cat_f1s) / len(cat_f1s), sum(dog_f1s) / len(dog_f1s)


def main():
    device = get_device()
    print(f"Device: {device}")

    cat_ids = cat_class_ids()

    # Same data for no_fix and weighted_ce
    base_loaders = get_dataloaders(
        task="breed", batch_size=32, imbalance="cats_20pct", seed=0,
    )
    # Different loaders for oversample
    oversample_loaders = get_dataloaders(
        task="breed", batch_size=32, imbalance="cats_20pct",
        use_weighted_sampler=True, seed=0,
    )
    class_names = base_loaders[3]
    class_weights_tensor = base_loaders[4].to(device)

    configs = [
        ("no_fix", base_loaders, nn.CrossEntropyLoss()),
        ("weighted_ce", base_loaders, nn.CrossEntropyLoss(weight=class_weights_tensor)),
        ("oversample", oversample_loaders, nn.CrossEntropyLoss()),
    ]

    results = {}
    total_start = time.time()
    for name, loaders, loss_fn in configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        model = train_one_config(
            loaders, 
            epochs=5, 
            loss_fn=loss_fn, 
            device=device)
        
        test_loader = loaders[2]
        metrics = evaluate_detailed(model, test_loader, device, class_names)

        acc = float(metrics["acc"])
        macro_f1 = float(metrics["macro_f1"])
        report = metrics["report"]
        cat_f1, dog_f1 = summarize_per_group(report, cat_ids, class_names)

        elapsed = time.time() - t0
        
        print(f"  -> test_acc={acc:.4f}  macro_f1={macro_f1:.4f}  "
              f"cat_f1={cat_f1:.4f}  dog_f1={dog_f1:.4f}  ({elapsed:.0f}s)")
        results[name] = {
            "test_acc": acc,
            "macro_f1": macro_f1,
            "cat_avg_f1": cat_f1,
            "dog_avg_f1": dog_f1,
            "per_class": report,
            "seconds": elapsed,
        }

    print(f"\nTotal experiment time: {time.time() - total_start:.0f}s")

    # Summary table
    print("\n========= Summary =========")
    print(f"{'config':>14}  {'test_acc':>10}  {'macro_f1':>10}  {'cat_f1':>8}  {'dog_f1':>8}")
    for name, r in results.items():
        print(f"{name:>14}  {r['test_acc']:>10.4f}  {r['macro_f1']:>10.4f}  "
              f"{r['cat_avg_f1']:>8.4f}  {r['dog_avg_f1']:>8.4f}")

    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "imbalanced.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
