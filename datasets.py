from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Define dataloader
def build_dataloader(dataset, batch_size):
    if dataset == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        datasets.CIFAR10.url = (
            "https://scidata.sjtu.edu.cn/records/p4t8m-rbe26/"
            "files/cifar-10-python.tar.gz?download=1"
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "cifar100":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        datasets.CIFAR100.url = (
            "https://scidata.sjtu.edu.cn/records/xk2s3-v1e12/"
            "files/cifar-100-python.tar.gz?download=1"
        )
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for images, labels in train_loader:
        print(f"Shape of images: {images.shape}")
        print(f"Shape of labels: {labels.shape}")
        break

    return train_loader, test_loader
