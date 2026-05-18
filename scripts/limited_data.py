"""Limited-data experiment.

How does test accuracy change as the training set shrinks?
Compare with/without augmentation and L2 regularization.

    python scripts/limited_data.py --strategy partial2
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import get_dataloaders
from src.strategies import STRATEGY_NAMES, get_device, run_strategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=STRATEGY_NAMES, default="linprobe")
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.1, 0.01])
    parser.add_argument("--weight-decays", nargs="+", type=float,
                        default=[0.0, 1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--no-aug-sweep", action="store_true",
                        help="Only run without augmentation (skip the aug=True half)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}  |  strategy: {args.strategy}")

    # Full grid: {augment off, on} x {wd values}. Defaults to 10 settings per fraction.
    aug_options = [False] if args.no_aug_sweep else [False, True]
    settings = [
        (f"{'aug' if aug else 'no_aug'}, wd={wd:g}",
         dict(augment=aug, weight_decay=wd))
        for aug in aug_options
        for wd in args.weight_decays
    ]

    results = []
    t_start = time.time()
    for frac in args.fractions:
        for name, params in settings:
            print(f"\n=== fraction={frac}  setting={name} ===")
            loaders = get_dataloaders(
                task="breed", batch_size=32, train_fraction=frac,
                augment=params["augment"], seed=0,
            )
            t0 = time.time()
            _, test_acc = run_strategy(args.strategy, loaders, device,
                                       weight_decay=params["weight_decay"])
            elapsed = time.time() - t0
            print(f"  -> test_acc={test_acc:.4f}  ({elapsed:.0f}s)")
            results.append({"fraction": frac, "setting": name,
                            "test_acc": float(test_acc), "seconds": elapsed})

    print(f"\nTotal time: {time.time() - t_start:.0f}s")

    print("\n========= Summary =========")
    print(f"{'fraction':>10}  {'setting':>18}  {'test_acc':>10}")
    for r in results:
        print(f"{r['fraction']:>10.2f}  {r['setting']:>18}  {r['test_acc']:>10.4f}")

    out = Path(__file__).resolve().parent.parent / "results" / f"limited_data_{args.strategy}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
