# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import ttnn
import os
from loguru import logger


def main():
    # Open Tenstorrent device
    device = ttnn.open_device(device_id=0)

    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    if os.path.exists("mlp_mnist_weights.pt"):
        # Pretrained weights
        weights = torch.load("mlp_mnist_weights.pt")
        W1 = ttnn.from_torch(weights["W1"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b1 = ttnn.from_torch(weights["b1"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        W2 = ttnn.from_torch(weights["W2"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b2 = ttnn.from_torch(weights["b2"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        W3 = ttnn.from_torch(weights["W3"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b3 = ttnn.from_torch(weights["b3"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        logger.info("Loaded pretrained weights from mlp_mnist_weights.pt")
    else:
        # Random weights for MLP - will not predict correctly
        logger.warning("mlp_mnist_weights.pt not found, using random weights")
        W1 = ttnn.rand((128, 28 * 28), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b1 = ttnn.rand((128,), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        W2 = ttnn.rand((64, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b2 = ttnn.rand((64,), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        W3 = ttnn.rand((10, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b3 = ttnn.rand((10,), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    correct = 0
    total = 0

    for i, (image, label) in enumerate(testloader):
        if i >= 5:
            break

        image = image.view(1, -1).to(torch.float32)

        # Convert to TT-NN Tensor
        # Convert the PyTorch tensor to TT-NN format with bfloat16 data type and
        # TILE\_LAYOUT. This is necessary for efficient computation on the
        # Tenstorrent device.
        image_tt = ttnn.from_torch(image, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Layer 1
        # Transposed weights are used to match TT-NN's expected shape. Bias
        # reshaped to 1x128 for broadcasting, and compute output 1.
        W1_final = ttnn.transpose(W1, -2, -1)
        b1_final = ttnn.reshape(b1, [1, -1])
        out1 = ttnn.linear(image_tt, W1_final, bias=b1_final)
        out1 = ttnn.relu(out1)

        # Layer 2
        # Same pattern as Layer 1, but with different weights and biases.
        W2_final = ttnn.transpose(W2, -2, -1)
        b2_final = ttnn.reshape(b2, [1, -1])
        out2 = ttnn.linear(out1, W2_final, bias=b2_final)
        out2 = ttnn.relu(out2)

        # Layer 3
        # Final layer with 10 output (for digits 0-9). No ReLU activation here, as
        # this is the output layer.
        W3_final = ttnn.transpose(W3, -2, -1)
        b3_final = ttnn.reshape(b3, [1, -1])
        out3 = ttnn.linear(out2, W3_final, bias=b3_final)

        # Convert result back to torch
        prediction = ttnn.to_torch(out3)
        predicted_label = torch.argmax(prediction, dim=1).item()

        correct += predicted_label == label.item()
        total += 1

        logger.info(f"Sample {i+1}: Predicted={predicted_label}, Actual={label.item()}")

    logger.info(f"\nTT-NN MLP Inference Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
