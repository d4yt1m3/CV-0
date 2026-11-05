import torch
import torch.nn.functional as F
from torch.nn import MSELoss


def compute_loss(criterion, logits, labels):
    """MSELoss expects probabilities and one-hot targets, not logits."""
    if isinstance(criterion, MSELoss):
        probs = torch.softmax(logits, dim=1)
        targets = F.one_hot(labels, num_classes=logits.size(1)).float()
        return criterion(probs, targets)
    return criterion(logits, labels)


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
        loss = compute_loss(criterion, logits, labels)

        # Update loss history
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate test loss and acc
    test_loss = total_loss / len(test_loader)
    acc = correct / total
    return acc, test_loss


@torch.no_grad()
def predict_all(model, loader, device):
    """Return predictions, ground-truth labels and probabilities as numpy arrays."""
    model.eval()

    preds = []
    labels_all = []
    probs_all = []
    for images, labels in loader:
        images = images.to(device)
        probs = torch.softmax(model(images), dim=1)
        preds.append(probs.argmax(dim=1).cpu())
        probs_all.append(probs.cpu())
        labels_all.append(labels)

    return (
        torch.cat(preds).numpy(),
        torch.cat(labels_all).numpy(),
        torch.cat(probs_all).numpy(),
    )
