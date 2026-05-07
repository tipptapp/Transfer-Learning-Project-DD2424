from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    return train_transform, test_transform

def get_dataloaders(task="breed", batch_size=32, image_size=224, data_root="data"):
    
    if task == "breed":
        target_type = "category"
        class_names = None
    elif task == "binary":
        target_type = "binary-category"
        class_names = ["cat", "dog"]
    else:
        raise ValueError("task is not 'breed' or 'binary'")
    
    train_transform, test_transform = get_transforms(image_size)
    
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
    
    train_dataset, validation_dataset = random_split(
        full_train_dataset, 
        [training_dataset_size, validation_dataset_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
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
    
    if task == "breed":
        class_names = full_train_dataset.classes
    
    return train_loader, validation_loader, test_loader, class_names


