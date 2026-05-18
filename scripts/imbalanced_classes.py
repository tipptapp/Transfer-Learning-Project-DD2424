"""Imbalanced-classes experiment.

  no_fix     — vanilla CE on the imbalanced data
  weighted_ce — CrossEntropyLoss(weight=class_weights)
  oversample — WeightedRandomSampler

    python scripts/imbalanced_classes.py --strategy partial2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import cat_class_ids, get_dataloaders
from src.evaluation import evaluate_detailed
from src.strategies import STRATEGY_NAMES, get_device, run_strategy


def cat_dog_f1(report, cat_ids, class_names):
    cat_f1s = [report[class_names[i]]["f1-score"] for i in cat_ids]
    dog_f1s = [report[c]["f1-score"] for i, c in enumerate(class_names) if i not in cat_ids]
    return sum(cat_f1s) / len(cat_f1s), sum(dog_f1s) / len(dog_f1s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="linprobe")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}  |  strategy: {args.strategy}")

    cat_ids = cat_class_ids()
    base = get_dataloaders(task="breed", batch_size=32, imbalance="cats_20pct", seed=0)
    over = get_dataloaders(task="breed", batch_size=32, imbalance="cats_20pct",
                           use_weighted_sampler=True, seed=0)
    class_names = base[3]
    cw = base[4].to(device)

    configs = [
        ("no_fix",      base, nn.CrossEntropyLoss()),
        ("weighted_ce", base, nn.CrossEntropyLoss(weight=cw)),
        ("oversample",  over, nn.CrossEntropyLoss()),
    ]

    results = {}
    t_start = time.time()
    for name, loaders, loss_fn in configs:
        print(f"\n=== {name} ===")
        t0 = time.time()
        model, _ = run_strategy(args.strategy, loaders, device, loss_fn=loss_fn)
        m = evaluate_detailed(model, loaders[2], device, class_names)
        cat_f1, dog_f1 = cat_dog_f1(m["report"], cat_ids, class_names)
        elapsed = time.time() - t0
        print(f"  -> test_acc={m['acc']:.4f}  macro_f1={m['macro_f1']:.4f}  "
              f"cat_f1={cat_f1:.4f}  dog_f1={dog_f1:.4f}  ({elapsed:.0f}s)")
        results[name] = {"test_acc": float(m["acc"]), "macro_f1": float(m["macro_f1"]),
                         "cat_avg_f1": cat_f1, "dog_avg_f1": dog_f1,
                         "per_class": m["report"], "seconds": elapsed}

    print(f"\nTotal time: {time.time() - t_start:.0f}s")

    print("\n========= Summary =========")
    print(f"{'config':>14}  {'test_acc':>10}  {'macro_f1':>10}  {'cat_f1':>8}  {'dog_f1':>8}")
    for name, r in results.items():
        print(f"{name:>14}  {r['test_acc']:>10.4f}  {r['macro_f1']:>10.4f}  "
              f"{r['cat_avg_f1']:>8.4f}  {r['dog_avg_f1']:>8.4f}")

    out = Path(__file__).resolve().parent.parent / "results" / f"imbalanced_{args.strategy}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
