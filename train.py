import argparse

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from datasets import build_dataloader, get_num_classes
from evaluate import compute_loss, evaluate
from models import Linear, MLP, CNN
from plot import plot_loss_acc

# Set random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Define build helpers, shared by CLI and the web UI
def build_model(backbone, in_channels, size, classes):
    if backbone == "linear":
        return Linear(in_channels=in_channels, size=size, classes=classes)
    if backbone == "mlp":
        return MLP(in_channels=in_channels, size=size, classes=classes)
    if backbone == "cnn":
        return CNN(in_channels=in_channels, size=size, classes=classes)
    raise ValueError(f"Unknown backbone: {backbone}")


def build_optimizer(name, params, lr):
    if name == "sgd":
        return SGD(params, lr=lr)
    if name == "adam":
        return Adam(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def build_criterion(name):
    if name == "cross_entropy":
        return CrossEntropyLoss()
    if name == "mse":
        return MSELoss()
    raise ValueError(f"Unknown criterion: {name}")


def resolve_device(name):
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Define train function
def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    device,
    on_epoch_end=None,
    progress=True,
):

    # Initialize loss and acc history
    loss_history = []
    acc_history = []

    # Train model
    for epoch in range(epochs):

        model.train()
        iterator = tqdm(train_loader) if progress else train_loader
        epoch_loss = 0

        for images, labels in iterator:
            # Zero gradient
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            loss = compute_loss(criterion, logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            if progress:
                iterator.set_description(f"Epoch {epoch}, Loss: {batch_loss:.4f}")

        # Update loss history
        epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(epoch_loss)

        # Evaluate model on test set
        acc, _ = evaluate(model, test_loader, criterion, device=device)
        acc_history.append(acc)

        # Print test acc
        print(f"Epoch {epoch}, Test Acc: {acc:.4f}")

        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_loss, acc)

    return loss_history, acc_history


def main():

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--backbone", type=str, default="cnn")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--criterion", type=str, default="cross_entropy")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    # Set device
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # Build dataloader
    train_loader, test_loader = build_dataloader(args.dataset, args.batch_size)

    for images, labels in train_loader:
        shape_i = images.shape
        shape_l = labels.shape
        print(f"Shape of images: {shape_i}")
        print(f"Shape of labels: {shape_l}")
        break

    # Build model
    model = build_model(
        args.backbone,
        in_channels=shape_i[1],
        size=shape_i[2],
        classes=get_num_classes(args.dataset),
    ).to(device)

    # Print num of model parameters
    print(f"Num of model parameters: {sum(p.numel() for p in model.parameters())}")

    # Build optimizer
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)

    # Build criterion
    criterion = build_criterion(args.criterion)

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
    plot_loss_acc(
        loss_history,
        acc_history,
        args.dataset,
        args.backbone,
        args.optimizer,
        args.criterion,
        args.lr,
        args.batch_size,
        args.epochs,
        device,
    )


if __name__ == "__main__":
    main()
