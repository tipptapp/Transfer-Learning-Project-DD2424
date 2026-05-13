import sklearn
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

# ImageNet stats, pretrained ConvNets expect inputs normalized
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def stratified_subset_indices(labels, fraction, seed):
    labels = np.asarray(labels)
    n = len(labels)
    if fraction >= 1.0:
        return np.arange(n)

    # StratifiedShuffleSplit needs train_size >= num_classes
    num_classes = len(np.unique(labels))
    target_size = max(int(fraction * n), num_classes)

    sss = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=target_size, random_state=seed
    )
    train_idx, _ = next(sss.split(np.zeros(n), labels))
    return train_idx


def imbalance_indices(labels, minority_class_ids, keep_fraction, seed):
    labels = np.asarray(labels)
    minority_indices = np.where(np.isin(labels, minority_class_ids))[0]
    majority_indices = np.where(~np.isin(labels, minority_class_ids))[0]
    keep_minority_indices = stratified_subset_indices(
        labels[minority_indices], keep_fraction, seed
    )
    return np.concatenate([majority_indices, minority_indices[keep_minority_indices]])


def cat_class_ids(data_root="data"):
    ds = datasets.OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types=["category", "binary-category"],
        download=False,
    )
    cat_ids = set()
    for breed_idx, bin_idx in zip(ds._labels, ds._bin_labels):
        if bin_idx == 0:  # 0 = cat in OxfordIIITPet
            cat_ids.add(breed_idx)
    return sorted(cat_ids)


def class_weights(labels, num_classes):
    labels = torch.as_tensor(labels)
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = torch.where(counts > 0, total / (num_classes * counts), torch.tensor(0.0))
    return weights


def make_weighted_sampler(labels):
    labels = torch.as_tensor(labels)
    class_counts = torch.bincount(labels)
    sample_weights = 1.0 / class_counts[labels].float()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def get_transforms(image_size=224, augment=False):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = test_transform

    return train_transform, test_transform


# Fixed seed for the val split so validation stays the same across experiments
_VAL_SPLIT_SEED = 0


def get_dataloaders(
    task="breed",
    batch_size=32,
    image_size=224,
    data_root="data",
    train_fraction=1.0,
    imbalance=None,
    augment=False,
    use_weighted_sampler=False,
    seed=0,
):

    if task == "breed":
        target_type = "category"
        class_names = None
        num_classes = 37
    elif task == "binary":
        target_type = "binary-category"
        class_names = ["cat", "dog"]
        num_classes = 2
    else:
        raise ValueError("task is not 'breed' or 'binary'")

    train_transform, test_transform = get_transforms(image_size, augment=augment)

    full_train_dataset = datasets.OxfordIIITPet(
        root=data_root,
        split="trainval",
        target_types=target_type,
        transform=train_transform,
        download=True
    )

    test_dataset = datasets.OxfordIIITPet(
        root=data_root,
        split="test",
        target_types=target_type,
        transform=test_transform,
        download=True
    )

    validation_dataset_size = int(0.2 * len(full_train_dataset))
    training_dataset_size = len(full_train_dataset) - validation_dataset_size

    generator = torch.Generator().manual_seed(_VAL_SPLIT_SEED)
    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [training_dataset_size, validation_dataset_size],
        generator=generator,
    )

    # Labels of the train portion so we can stratify / subsample
    all_labels = np.array(
        full_train_dataset._bin_labels if task == "binary" else full_train_dataset._labels
    )
    train_indices = np.array(train_dataset.indices)
    train_labels = all_labels[train_indices]

    # Optional: imbalance
    keep = np.arange(len(train_indices))
    if imbalance == "cats_20pct":
        if task != "breed":
            raise ValueError("imbalance='cats_20pct' only makes sense for task='breed'")
        cat_ids = cat_class_ids(data_root)
        keep = imbalance_indices(train_labels[keep], cat_ids, keep_fraction=0.2, seed=seed)
    elif imbalance is not None:
        raise ValueError(f"unknown imbalance scheme: {imbalance}")

    # Optional: stratified sub-sampling
    if train_fraction < 1.0:
        sub = stratified_subset_indices(train_labels[keep], train_fraction, seed)
        keep = keep[sub]

    final_train_indices = train_indices[keep].tolist()
    final_train_labels = all_labels[final_train_indices]
    train_dataset = Subset(full_train_dataset, final_train_indices)

    # Optional: oversampling sampler
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = make_weighted_sampler(final_train_labels)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    cw = class_weights(final_train_labels, num_classes)

    if task == "breed":
        class_names = full_train_dataset.classes

    return train_loader, validation_loader, test_loader, class_names, cw
