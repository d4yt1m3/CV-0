import argparse
import numpy as np
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from matplotlib import pyplot as plt

# Set font to Times New Roman
plt.rcParams["font.sans-serif"] = ["Times New Roman"]

# Set random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Define models
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, 100)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc2 = nn.Linear(16 * 16 * 16, 128)  # Must calculate the input size
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.flatten(1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Define dataloader
def build_dataloader(batch_size):
    # Set source of CIFAR-100 dataset
    datasets.CIFAR100.url = (
        "https://scidata.sjtu.edu.cn/records/xk2s3-v1e12/"
        "files/cifar-100-python.tar.gz?download=1"
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for images, labels in train_loader:
        print(f"Shape of images: {images.shape}")
        print(f"Shape of labels: {labels.shape}")
        break

    return train_loader, test_loader


# Define plot function
def plot_loss_acc(loss_history, acc_history, epochs, backbone, optimizer):
    # Set figure size
    plt.figure(figsize=(8, 6))

    # Plot loss and acc
    plt.plot(range(epochs), loss_history, linewidth=2, color="red", label="Train Loss")
    plt.plot(range(epochs), acc_history, linewidth=2, color="blue", label="Test Acc")

    # Set title and labels
    plt.title(f"{backbone.upper()} {optimizer.upper()} Loss/Acc")
    plt.xlabel("Epoch")
    plt.xticks(range(epochs))
    plt.ylabel("Loss/Acc")
    plt.ylim(0, max(max(loss_history), max(acc_history)) + 0.1)  # Auto set ylim limit
    plt.yticks(np.arange(0, max(max(loss_history), max(acc_history)) + 0.1, 0.1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Save figure
    if not os.path.exists("./images"):
        os.makedirs("./images")
    plt.savefig(f"./images/{backbone}_{optimizer}.png", dpi=300)
    plt.close()


# Define train function
def train(model, train_loader, test_loader, optimizer, criterion, epochs, device):

    # Initialize loss and acc history
    loss_history = []
    acc_history = []

    # Train model
    for epoch in range(epochs):

        model.train()
        pbar = tqdm(train_loader)  # for images, labels in train_loader:
        epoch_loss = 0

        for images, labels in pbar:
            # Zero gradient
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            pbar.set_description(f"Epoch {epoch}, Loss: {batch_loss:.4f}")

        # Update loss history
        epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # Evaluate model on test set
        acc, _ = evaluate(model, test_loader, criterion, device=device)
        acc_history.append(acc)

        # Print test acc
        print(f"Epoch {epoch}, Test Acc: {acc:.4f}")
    return loss_history, acc_history


# Define evaluate function
@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()

    # Initialize loss and acc
    total_loss = 0.0
    correct = 0
    total = 0

    # Evaluate model on test set
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Update loss history
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate test loss and acc
    test_loss = total_loss / len(test_loader)
    acc = correct / total
    return acc, test_loss


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--backbone", type=str, default="cnn")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--criterion", type=str, default="cross_entropy")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    # Set device
    device = None
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # Build dataloader
    train_loader, test_loader = build_dataloader(args.batch_size)

    # Build model
    model = None
    if args.backbone == "mlp":
        model = MLP().to(device)
    elif args.backbone == "cnn":
        model = CNN().to(device)
    elif args.backbone == "linear":
        model = Linear().to(device)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    # Build optimizer
    optimizer = None
    if args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Build criterion
    criterion = None
    if args.criterion == "cross_entropy":
        criterion = CrossEntropyLoss()
    elif args.criterion == "mse":
        criterion = MSELoss()
    else:
        raise ValueError(f"Unknown criterion: {args.criterion}")

    # Train model
    loss_history, acc_history = train(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        epochs=args.epochs,
        device=device,
    )

    # Plot loss and acc
    plot_loss_acc(loss_history, acc_history, args.epochs, args.backbone, args.optimizer)


if __name__ == "__main__":
    main()
