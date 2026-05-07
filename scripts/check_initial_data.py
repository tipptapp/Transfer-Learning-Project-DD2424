from src.data import get_dataloaders


def inspect_task(task):
    print("\n" + "=" * 50)
    print(f"Checking task: {task}")
    print("=" * 50)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        task=task,
        batch_size=5,
        image_size=224,
    )

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("First labels:", labels)
    print("Number of classes:", len(class_names))
    print("Class names:", class_names)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))


def main():
    inspect_task("breed")
    inspect_task("binary")


if __name__ == "__main__":
    main()