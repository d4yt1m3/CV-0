import argparse

import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from datasets import build_dataloader
from evaluate import evaluate
from models import Linear, MLP, CNN
from plot import plot_loss_acc

# Set random seed
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

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