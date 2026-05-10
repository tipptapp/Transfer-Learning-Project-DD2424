import sys
import time # need to compare training time vs fine-tuning
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from linprobing import get_device, evaluate, NUM_CLASSES

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import get_dataloaders

def unfreeze_layers(layers_to_unfreeze, model):
    """Unfreezes the last `layers_to_unfreeze` layers."""
    frozen_layers = 0
    for child in model.children():
        if frozen_layers < layers_to_unfreeze:
            frozen_layers += 1
            continue
        else:
            for param in child.parameters():
                param.requires_grad = True


def main():
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        task="breed", batch_size=32, image_size=224
    )

    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    total_start = time.time()
    epoch_global = 0

    num_layers = 0
    # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/4
    for layer_idx, child in enumerate(start=1, iterable=model.children()): # model.children() which returns it’s layers
        for param in child.parameters():
            param.requires_grad = False
        print("Layer frozen ", layer_idx)
        num_layers += 1

    for param in model.fc.parameters():
        param.requires_grad = True

    print("Number of layers: ", num_layers)

    # Gradual unfreezing
    epochs_per_stage = 3
    for i in range(1, num_layers + 1):
        # Each stage starts from a fully-frozen model
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        unfreeze_layers(num_layers - i, model)

        # pick up new unfrozen layers
        optim = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=1e-3,
        )

        stage_start = time.time()
        for epoch in range(1, epochs_per_stage + 1):
            epoch_global += 1
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

            val_acc = evaluate(model, val_loader, device)
            test_acc = evaluate(model, test_loader, device)
            print(
                f"  Epoch {epoch_global:2d} (stage {i} ep {epoch}): "
                f"train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
                f"test_acc={test_acc:.4f}"
            )
        print(f"  Stage {i} time: {time.time() - stage_start:.1f}s")

    total_time = time.time() - total_start
    final_test = evaluate(model, test_loader, device)
    print(f"\nTotal training time: {total_time:.1f}s")
    print(f"Final test accuracy:  {final_test:.4f}")


if __name__ == "__main__":
    main()
