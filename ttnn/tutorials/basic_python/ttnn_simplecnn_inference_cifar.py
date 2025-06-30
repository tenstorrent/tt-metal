# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torchvision
import torchvision.transforms as transforms
import ttnn
from loguru import logger


def main():
    device = ttnn.open_device(device_id=0, l1_small_size=8192)

    try:
        logger.info("\n--- Simple CNN Inference Using TT-NN on CIFAR-10 ---")

        # Load CIFAR-10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

        # Load pretrained weights or use random weights if not found
        if os.path.exists("simple_cnn_cifar10_weights.pt"):
            weights = torch.load("simple_cnn_cifar10_weights.pt")
            logger.info("Loaded pretrained weights from simple_cnn_cifar10_weights.pt")
        else:
            logger.warning("simple_cnn_cifar10_weights.pt not found, using random weights")
            torch.manual_seed(0)
            weights = {
                "conv1.weight": torch.randn((16, 3, 3, 3), dtype=torch.float32),
                "conv1.bias": torch.randn((16,), dtype=torch.float32),
                "conv2.weight": torch.randn((32, 16, 3, 3), dtype=torch.float32),
                "conv2.bias": torch.randn((32,), dtype=torch.float32),
                "fc1.weight": torch.randn((128, 2048), dtype=torch.float32),
                "fc1.bias": torch.randn((128,), dtype=torch.float32),
                "fc2.weight": torch.randn((10, 128), dtype=torch.float32),
                "fc2.bias": torch.randn((10,), dtype=torch.float32),
            }

        correct = 0
        total = 0

        for i, (image, label) in enumerate(testloader):
            if i >= 5:
                break

            # Preprocess input
            ttnn_image = ttnn.from_torch(image, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)

            # Conv1
            W1 = weights["conv1.weight"]
            B1 = weights["conv1.bias"]
            # W1_tt = ttnn.from_torch(W1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            # W1_tt = ttnn.to_layout(W1_tt, ttnn.TILE_LAYOUT)
            W1 = W1.permute(0, 2, 3, 1)  # Convert to [out_channels, kernel_height, kernel_width, in_channels]
            B1 = B1.view(1, 1, 1, -1)  # Reshape bias to [1, 1, 1, out_channels]
            # B1_tt = ttnn.from_torch(B1.view(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            # B1_tt = ttnn.to_layout(B1_tt, ttnn.TILE_LAYOUT)

            logger.info(f"Sample {i+1}: Input shape: {ttnn.to_torch(ttnn_image).shape}")

            W1_ttnn = ttnn.from_torch(W1, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
            B1_ttnn = ttnn.from_torch(B1, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

            # Data prep: permute and reshape input
            # BCHW -> BHWC
            image_permuted = ttnn.permute(ttnn_image, (0, 2, 3, 1))
            image_B, image_H, image_W, image_C = image_permuted.shape
            image_reshaped = ttnn.reshape(image_permuted, (1, 1, image_H * image_W, image_C))

            conv_config = ttnn.Conv2dConfig(dtype=ttnn.bfloat16, weights_dtype=ttnn.bfloat16)
            print("CONV1 input_tensor shape:", image_reshaped.shape)
            print("weight_tensor shape:", W1_ttnn.shape)
            print("bias_tensor shape:", B1_ttnn.shape)
            print("in_channels:", image_C)
            print("out_channels:", 16)
            print("device:", device)
            print("kernel_size:", (3, 3))
            print("stride:", (1, 1))
            print("padding:", (1, 1))
            print("batch_size:", 1)
            print("input_height:", 1)
            print("input_width:", image_B * image_H * image_W)
            print("conv_config:", conv_config)
            print("groups:", 0)

            conv1_out = ttnn.conv2d(
                input_tensor=image_reshaped,
                weight_tensor=W1_ttnn,
                bias_tensor=B1_ttnn,
                in_channels=image_C,
                out_channels=16,
                device=device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=1,
                input_height=1,
                input_width=image_B * image_H * image_W,
                conv_config=conv_config,
                groups=0,
            )

            print("conv1_out shape:", conv1_out.shape)

            print("Input parameters to first relu:")
            print(f"  input shape: {conv1_out.shape}")

            conv1_relu = ttnn.relu(conv1_out)
            print("conv1_relu shape:", conv1_relu.shape)

            print("Input parameters to first max_pool2d:")
            print(f"  input shape: {conv1_relu.shape}")
            print(f"  batch_size: {1}")
            print(f"  input_h: {32}")
            print(f"  input_w: {32}")
            print(f"  channels: {16}")
            print(f"  kernel_size: {[2, 2]}")
            print(f"  stride: {[2, 2]}")
            print(f"  padding: {[0, 0]}")
            print(f"  dilation: {[1, 1]}")
            print(f"  ceil_mode: {False}")

            conv1_pool = ttnn.max_pool2d(
                conv1_relu,
                batch_size=1,
                input_h=32,
                input_w=32,
                channels=16,
                kernel_size=[2, 2],
                stride=[2, 2],
                padding=[0, 0],
                dilation=[1, 1],
                ceil_mode=False,
            )
            print("conv1_pool shape:", conv1_pool.shape)
            logger.info(f"Sample {i+1}: Output shape after Conv1: {ttnn.to_torch(conv1_pool).shape}")

            # Conv2
            W2 = weights["conv2.weight"]
            B2 = weights["conv2.bias"]
            # W2_tt = ttnn.from_torch(W2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            # W2_tt = ttnn.to_layout(W2_tt, ttnn.TILE_LAYOUT)
            W2 = W2.permute(0, 1, 2, 3)  # Convert to [out_channels, kernel_height, kernel_width, in_channels]
            B2 = B2.view(1, 1, 1, -1)  # Reshape bias to [1, 1, 1, out_channels]
            # B2_tt = ttnn.from_torch(B2.view(1, -1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            # B2_tt = ttnn.to_layout(B2_tt, ttnn.TILE_LAYOUT)

            W2_ttnn = ttnn.from_torch(W2, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
            B2_ttnn = ttnn.from_torch(B2, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

            conv1_B, conv1_H, conv1_W, conv1_C = conv1_pool.shape
            print(f"conv1_pool shape: {conv1_pool.shape}")
            conv1_reshaped = ttnn.reshape(conv1_pool, (1, 1, conv1_H * conv1_W, conv1_C))
            print(f"conv1_reshaped shape: {conv1_reshaped.shape}")

            print("CONV2 input_tensor shape:", conv1_reshaped.shape)
            print("weight_tensor shape:", W2_ttnn.shape)
            print("bias_tensor shape:", B2_ttnn.shape)
            print("in_channels:", conv1_C)
            print("out_channels:", 32)
            print("device:", device)
            print("kernel_size:", (3, 3))
            print("stride:", (1, 1))
            print("padding:", (1, 1))
            print("batch_size:", 1)
            print("input_height:", 1)
            print("input_width:", conv1_H * conv1_W)
            print("conv_config:", conv_config)
            print("groups:", 0)

            conv2_out = ttnn.conv2d(
                input_tensor=conv1_reshaped,
                weight_tensor=W2_ttnn,
                bias_tensor=B2_ttnn,
                in_channels=conv1_C,
                out_channels=32,
                device=device,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=1,
                input_height=1,
                input_width=conv1_H * conv1_W,
                conv_config=conv_config,
                groups=0,
            )

            print("conv2_out shape:", conv2_out.shape)

            conv2_relu = ttnn.relu(conv2_out)
            print("conv2_relu shape:", conv2_relu.shape)

            conv2_B, conv2_H, conv2_W, conv2_C = conv2_relu.shape

            print("max_pool2d input parameters:")
            print(f"  input shape: {conv2_relu.shape}")
            print(f"  batch_size: {1}")
            print(f"  input_h: {16}")
            print(f"  input_w: {16}")
            print(f"  channels: {32}")
            print(f"  kernel_size: {[2, 2]}")
            print(f"  stride: {[2, 2]}")
            print(f"  padding: {[0, 0]}")
            print(f"  dilation: {[1, 1]}")
            print(f"  ceil_mode: {False}")

            conv2_pool = ttnn.max_pool2d(
                conv2_relu,
                batch_size=1,
                input_h=16,
                input_w=16,
                channels=32,
                kernel_size=[2, 2],
                stride=[2, 2],
                padding=[0, 0],
                dilation=[1, 1],
                ceil_mode=False,
            )

            print("conv2_pool shape:", conv2_pool.shape)
            # After conv2_pool is computed and has shape: [1, 8, 8, 32]
            B, H, W, C = conv2_pool.shape
            out_flat = ttnn.to_torch(conv2_pool)
            out_flat = out_flat.permute(0, 3, 1, 2).contiguous().view(B, -1)  # [1, 32*8*8]

            # Load pretrained FC weights
            W3 = weights["fc1.weight"]  # [hidden_dim, 2048]
            B3 = weights["fc1.bias"]  # [hidden_dim]
            W4 = weights["fc2.weight"]  # [10, hidden_dim]
            B4 = weights["fc2.bias"]  # [10]

            # FC1 setup
            W3_tt = ttnn.from_torch(W3.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            W3_tt = ttnn.to_layout(W3_tt, ttnn.TILE_LAYOUT)

            B3_tt = ttnn.from_torch(B3.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            B3_tt = ttnn.to_layout(B3_tt, ttnn.TILE_LAYOUT)

            # Convert out_flat to TT
            x_tt = ttnn.from_torch(out_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            x_tt = ttnn.to_layout(x_tt, ttnn.TILE_LAYOUT)

            # FC1 + ReLU
            out = ttnn.linear(x_tt, W3_tt, bias=B3_tt)
            out = ttnn.relu(out)

            # FC2 setup
            W4_tt = ttnn.from_torch(W4.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            W4_tt = ttnn.to_layout(W4_tt, ttnn.TILE_LAYOUT)

            B4_tt = ttnn.from_torch(B4.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            B4_tt = ttnn.to_layout(B4_tt, ttnn.TILE_LAYOUT)

            # FC2
            out = ttnn.linear(out, W4_tt, bias=B4_tt)

            # Prediction
            prediction = ttnn.to_torch(out)
            predicted_label = torch.argmax(prediction, dim=1).item()
            correct += predicted_label == label.item()
            total += 1

            logger.info(f"Sample {i+1}: Predicted={predicted_label}, Actual={label.item()}")

        logger.info(f"\nTT-NN SimpleCNN Inference Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
