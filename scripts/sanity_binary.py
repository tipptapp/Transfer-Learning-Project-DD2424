import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision.models import resnet34, ResNet34_Weights

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import get_dataloaders


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        all_preds.append(model(x).argmax(dim=1).cpu())
        all_labels.append(y)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    return acc, f1


def main():
    device = get_device()
    print(f"Device: {device}")

    # Cat/dog labels
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        task="binary", batch_size=32, image_size=224
    )

    # Pretrained ResNet34, freeze everything, swap in a 2-class head
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Only the new fc layer is trainable
    optim = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
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
        train_loss = running / n

        val_acc, val_f1 = evaluate(model, val_loader, device)
        test_acc, test_f1 = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")

    val_acc, val_f1 = evaluate(model, val_loader, device)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"Final: val_acc={val_acc:.4f} val_f1={val_f1:.4f}  test_acc={test_acc:.4f} test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
