import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision.models import resnet34, ResNet34_Weights

# Make src/ importable
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


def main():
    device = get_device()
    print(f"Device: {device}")

    # 37 breed labels
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(
        task="breed", batch_size=32, image_size=224
    )

    # Pretrained ResNet34, freeze everything, swap in a 37-class head
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # Only the new fc layer is trainable
    optim = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 15
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d}: train_loss={running_loss/n:.4f}  train_acc={correct/n:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

    # Final evaluation on val and test sets.
    val_acc, val_f1 = evaluate(model, val_loader, device)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"Final: val_acc={val_acc:.4f} val_f1={val_f1:.4f}  test_acc={test_acc:.4f} test_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
