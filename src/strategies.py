"""Reusable fine-tuning strategies for the transfer-learning experiments.

Each `train_*` function takes a 5-tuple `loaders` from src.data.get_dataloaders,
trains a fresh ResNet34, and returns (best_model, final_test_acc). They all
accept `weight_decay` (for L2) and `loss_fn` (e.g. for weighted CE) so the
limited-data and imbalanced-classes scripts can plug in whichever experiment
condition they need.
"""
import copy

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from src.evaluation import evaluate


def freeze_layers(model):
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for p in layer.parameters():
            p.requires_grad = False


def unfreeze_layer(model, idx):
    layer = [model.layer1, model.layer2, model.layer3, model.layer4][idx]
    for p in layer.parameters():
        p.requires_grad = True

NUM_CLASSES = 37
STAGES = ["layer1", "layer2", "layer3", "layer4"]

# The set of strategies experiment scripts can dispatch to.
STRATEGY_NAMES = ["linprobe", "partial1", "partial2", "partial3", "partial4", "unfreezing"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------- model builders --------

def build_linprobe(num_classes=NUM_CLASSES, device=None):
    """Frozen backbone + new fc head."""
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def build_partial(l, num_classes=NUM_CLASSES, device=None):
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
    return model.to(device)


# -------- shared training loop --------

def _train_epochs(model, train_loader, val_loader, loss_fn, optim, epochs, device, *, verbose=True):
    """Generic per-epoch loop. Keeps and returns the best-val_acc state dict."""
    best_val_acc = -1.0
    best_state = None
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
        if verbose:
            print(f"    Epoch {epoch}: train_loss={running/n:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# -------- strategy wrappers --------

def train_linprobe(loaders, device, *, epochs=5, weight_decay=0.0, loss_fn=None, lr=1e-3):
    train_loader, val_loader, test_loader, _, _ = loaders
    model = build_linprobe(device=device)
    optim = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    _train_epochs(model, train_loader, val_loader, loss_fn, optim, epochs, device)
    test_acc, _ = evaluate(model, test_loader, device)
    return model, test_acc


def train_partial(l, loaders, device, *, epochs=5, weight_decay=0.0, loss_fn=None, lr=1e-4):
    train_loader, val_loader, test_loader, _, _ = loaders
    model = build_partial(l, device=device)
    optim = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    _train_epochs(model, train_loader, val_loader, loss_fn, optim, epochs, device)
    test_acc, _ = evaluate(model, test_loader, device)
    return model, test_acc


def train_unfreezing(loaders, device, *, epochs_per_stage=3, weight_decay=0.0, loss_fn=None):
    train_loader, val_loader, test_loader, _, _ = loaders

    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    freeze_layers(model)

    loss_fn = loss_fn or nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for stage in range(4, -1, -1):
        if stage != 4:
            unfreeze_layer(model, stage)
        
        lr = 1e-3 if stage == 4 else 1e-4
        optim = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr, weight_decay=weight_decay,
        )
        print(f"  Stage {stage}")
        _, stage_best = _train_epochs(model, train_loader, val_loader, loss_fn, optim,
                                      epochs_per_stage, device)
        
        if stage_best > best_val_acc:
            best_val_acc = stage_best
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, _ = evaluate(model, test_loader, device)
    return model, test_acc


# -------- one-call dispatcher used by experiment scripts --------

def run_strategy(name, loaders, device, *, weight_decay=0.0, loss_fn=None):
    """Dispatch a strategy by name. name in {"linprobe","partial1".."partial4","unfreezing"}."""
    if name == "linprobe":
        return train_linprobe(loaders, device, weight_decay=weight_decay, loss_fn=loss_fn)
    if name.startswith("partial"):
        l = int(name[-1])
        return train_partial(l, loaders, device, weight_decay=weight_decay, loss_fn=loss_fn)
    if name == "unfreezing":
        return train_unfreezing(loaders, device, weight_decay=weight_decay, loss_fn=loss_fn)
    raise ValueError(f"unknown strategy: {name}")
