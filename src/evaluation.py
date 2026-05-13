import torch
from sklearn.metrics import f1_score
import json, time
from pathlib import Path

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

from sklearn.metrics import classification_report, f1_score

@torch.no_grad()
def evaluate_detailed(model, loader, device, class_names=None):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "acc": (preds == labels).mean(),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "report": classification_report(labels, preds, target_names=class_names,
                                       output_dict=True, zero_division=0),
    }