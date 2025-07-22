# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from loguru import logger


# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Train model
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save weights
    weights = {
        "W1": model.fc1.weight.detach().clone(),  # [128, 784]
        "b1": model.fc1.bias.detach().clone(),  # [128]
        "W2": model.fc2.weight.detach().clone(),  # [64, 128]
        "b2": model.fc2.bias.detach().clone(),  # [64]
        "W3": model.fc3.weight.detach().clone(),  # [10, 64]
        "b3": model.fc3.bias.detach().clone(),  # [10]
    }
    torch.save(weights, "mlp_mnist_weights.pt")
    logger.info("Weights saved to mlp_mnist_weights.pt")


if __name__ == "__main__":
    main()
