from torch import nn


# Define models
class Linear(nn.Module):
    def __init__(self, in_channels, size, classes):
        super().__init__()
        self.fc = nn.Linear(in_channels * size * size, classes)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, size, classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * size * size, 128)
        self.fc2 = nn.Linear(128, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, size, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc2 = nn.Linear(16 * size * size // 4, 128)  # Must calculate the input size
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.flatten(1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
