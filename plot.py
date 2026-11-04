import os

import numpy as np
from matplotlib import pyplot as plt

# Set font to Times New Roman
plt.rcParams["font.sans-serif"] = ["Times New Roman"]


# Define plot function
def plot_loss_acc(
    loss_history,
    acc_history,
    dataset,
    backbone,
    optimizer,
    criterion,
    lr,
    batch_size,
    epochs,
    device,
):
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
    if not os.path.exists(f"./images/{dataset}"):
        os.makedirs(f"./images/{dataset}")
    plt.savefig(
        f"./images/{backbone}_{optimizer}_{criterion}_{lr}_{batch_size}_{epochs}_{device}.png",
        dpi=300,
    )
    plt.close()
