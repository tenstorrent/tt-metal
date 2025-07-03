# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # -> [B, 16, 16, 16]
        x = self.pool(torch.relu(self.conv2(x)))  # -> [B, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, trainloader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")


def test_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            if i >= 5:  # Limit to first 5 for comparison with TT-NN
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append((predicted.item(), labels.item()))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            print(f"Sample {i + 1}: Predicted = {predicted.item()}, Actual = {labels.item()}")

    print(f"PyTorch Test Accuracy (5 samples): {correct}/{total} = {100.0 * correct / total:.2f}%")
    return predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    model = SimpleCNN().to(device)
    train_model(model, trainloader, device)

    # Save weights
    torch.save(model.state_dict(), "simple_cnn_cifar10_weights.pt")
    print("Saved weights to simple_cnn_cifar10_weights.pt")

    # Load and test model
    model.load_state_dict(torch.load("simple_cnn_cifar10_weights.pt", map_location=device))
    predictions = test_model(model, testloader, device)

    # Optionally save predictions to file
    torch.save(predictions, "pytorch_predictions.pt")


if __name__ == "__main__":
    main()
