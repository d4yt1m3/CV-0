import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from matplotlib import pyplot as plt

device = torch.device("mps" if torch.mps.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_dataloader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, test_loader, optimizer, criterion, epochs):
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        total_loss = 0.0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        acc = evaluate(model, test_loader, criterion)
        pbar.set_description(f"Epoch {epoch}, Loss: {train_loss:.4f}, Acc: {acc:.4f}")


@torch.no_grad()
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_loss = total_loss / len(test_loader)
    acc = correct / total
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--backbone", type=str, default="mlp")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--criterion", type=str, default="cross_entropy")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    train_loader, test_loader = build_dataloader(args.batch_size)

    model = MLP().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    train(model, train_loader, test_loader, optimizer, criterion, epochs=args.epochs)
    acc = evaluate(model, test_loader, criterion)
    print(f"Test Acc: {acc:.4f}")


if __name__ == "__main__":
    main()
