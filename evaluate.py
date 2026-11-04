import torch

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

