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
    logger.info("\n--- Simple CNN Inference Using TT-NN on CIFAR-10 ---")

    # Define input transforms: Convert to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 test data
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    if os.path.exists("simple_cnn_cifar10_weights.pt"):
        weights = torch.load("simple_cnn_cifar10_weights.pt")
        weights = {
            k: ttnn.from_torch(v, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
            for k, v in weights.items()
        }
        logger.info("Loaded pretrained weights")
    else:
        logger.warning("Weights not found, using random weights")
        torch.manual_seed(0)
        weights = {
            "conv1.weight": ttnn.rand((16, 3, 3, 3), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
            "conv1.bias": ttnn.rand((16,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
            "conv2.weight": ttnn.rand((32, 16, 3, 3), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
            "conv2.bias": ttnn.rand((32,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
            "fc1.weight": ttnn.rand((128, 2048), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
            "fc1.bias": ttnn.rand((128,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
            "fc2.weight": ttnn.rand((10, 128), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
            "fc2.bias": ttnn.rand((10,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        }

    def conv_pool_stage(
        input_tensor: ttnn.Tensor,
        input_NHWC: ttnn.Shape,
        conv_outchannels: int,
        weights: dict,
        weight_str: str,
        bias_str: str,
        activation: ttnn.UnaryWithParam,
        device: ttnn.Device,
        log_first_sample: bool = False,
    ) -> ttnn.Tensor:
        """
        Perform convolution + activation + max pooling using TT-NN.
        Args:
            input_tensor: Input TT tensor in NHWC format.
            input_NHWC: Tuple representing (Batch, Height, Width, Channels) of the input tensor.
            conv_outchannels: Number of output channels for the convolution layer.
            weights: Dictionary containing model weights and biases.
            weight_str: Key name for convolution weights in the weights dict.
            bias_str: Key name for convolution biases in the weights dict.
            activation: Activation function as UnaryWithParam to apply after conv.
            device: Target TT device to execute the operations on.
            log_first_sample: Whether to log detailed info (used for debugging first sample).
        Returns:
            Output tensor after conv + max pooling (TT format).
        """
        # Extract weight and bias tensors from weights dictionary
        W = weights[weight_str]
        B = weights[bias_str]
        B = ttnn.reshape(B, (1, 1, 1, -1))  # Ensure bias is in correct shape for TT-NN

        # Define convolution parameters
        conv_kernel_size = (3, 3)
        conv_stride = (1, 1)
        conv_padding = (1, 1)

        # Set up TT-NN convolution configuration including activation function
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=activation,
        )

        # Optional detailed logging for the first sample (shape, config, etc.)
        if log_first_sample:
            logger.info("=====================================================================")
            logger.info("Input parameters to conv2d:")
            logger.info(f"  input_tensor shape: {input_tensor.shape}")
            logger.info(f"  weight_tensor shape: {W.shape}")
            logger.info(f"  bias_tensor shape: {B.shape}")
            logger.info(f"  in_channels: {input_NHWC[3]}")
            logger.info(f"  out_channels: {conv_outchannels}")
            logger.info(f"  device: {device}")
            logger.info(f"  kernel_size: {conv_kernel_size}")
            logger.info(f"  stride: {conv_stride}")
            logger.info(f"  padding: {conv_padding}")
            logger.info(f"  batch_size: {input_NHWC[0]}")
            logger.info(f"  input_height: {input_NHWC[1]}")
            logger.info(f"  input_width: {input_NHWC[2]}")
            logger.info(f"  conv_config: {conv_config}")
            logger.info(f"  groups: {0}")

        # Perform convolution
        conv1_out = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=W,
            bias_tensor=B,
            in_channels=input_NHWC[3],
            out_channels=conv_outchannels,
            device=device,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=conv_padding,
            batch_size=input_NHWC[0],
            input_height=input_NHWC[1],
            input_width=input_NHWC[2],
            conv_config=conv_config,
            groups=0,
        )

        # Define max pooling parameters
        max_pool2d_kernel_size = [2, 2]
        max_pool2d_stride = [2, 2]
        max_pool2d_padding = [0, 0]
        max_pool2d_dilation = [1, 1]

        # Optional logging for max pooling input and parameters
        if log_first_sample:
            logger.info("Input parameters to max_pool2d:")
            logger.info(f"  input shape: {conv1_out.shape}")
            logger.info(f"  batch_size: {input_NHWC[0]}")
            logger.info(f"  input_h: {input_NHWC[1]}")
            logger.info(f"  input_w: {input_NHWC[2]}")
            logger.info(f"  channels: {conv_outchannels}")
            logger.info(f"  kernel_size: {max_pool2d_kernel_size}")
            logger.info(f"  stride: {max_pool2d_stride}")
            logger.info(f"  padding: {max_pool2d_padding}")
            logger.info(f"  dilation: {max_pool2d_dilation}")
            logger.info(f"  ceil_mode: {False}")

        # Perform max pooling
        max_pool2d_out = ttnn.max_pool2d(
            conv1_out,
            batch_size=input_NHWC[0],
            input_h=input_NHWC[1],
            input_w=input_NHWC[2],
            channels=conv_outchannels,
            kernel_size=max_pool2d_kernel_size,
            stride=max_pool2d_stride,
            padding=max_pool2d_padding,
            dilation=max_pool2d_dilation,
            ceil_mode=False,
        )

        # Log output shape after pooling
        if log_first_sample:
            logger.info(f"max_pool2d output shape: {max_pool2d_out.shape}")
            logger.info("=====================================================================")

        return max_pool2d_out

    correct = 0
    total = 0

    # Run inference on a few test samples
    for i, (image, label) in enumerate(testloader):
        if i >= 5:
            break

        # Convert image to TT tensor
        ttnn_image = ttnn.from_torch(image, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        ttnn_image_permuated = ttnn.permute(ttnn_image, (0, 2, 3, 1))  # NCHW -> NHWC

        # Only log details for first sample
        log_this = i == 0

        # Apply first conv + pool stage
        conv1_pool = conv_pool_stage(
            ttnn_image_permuated,
            ttnn_image_permuated.shape,
            16,
            weights,
            "conv1.weight",
            "conv1.bias",
            ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            device,
            log_first_sample=log_this,
        )

        # Apply second conv + pool stage
        conv2_pool = conv_pool_stage(
            conv1_pool,
            (1, 16, 16, 16),
            32,
            weights,
            "conv2.weight",
            "conv2.bias",
            ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            device,
            log_first_sample=log_this,
        )

        # Flatten for FC layers
        B, H, W, C = conv2_pool.shape
        out_flat = ttnn.to_torch(conv2_pool)  # Convert back to torch
        out_flat = out_flat.permute(0, 3, 1, 2).contiguous().view(B, -1)  # NHWC -> NCHW -> Flatten

        # Prepare fully connected layers
        W3 = weights["fc1.weight"]
        B3 = weights["fc1.bias"].reshape((1, -1))  # Reshape bias for broadcast compatibility
        W4 = weights["fc2.weight"]
        B4 = weights["fc2.bias"]

        # Convert to TT format for FC1
        W3_tt = ttnn.to_layout(ttnn.transpose(W3, 0, 1), ttnn.TILE_LAYOUT)
        B3_tt = ttnn.to_layout(B3.reshape((1, -1)), ttnn.TILE_LAYOUT)

        # Convert input to TT format
        x_tt = ttnn.from_torch(out_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Apply FC1 + ReLU
        out = ttnn.linear(x_tt, W3_tt, bias=B3_tt)
        out = ttnn.relu(out)

        # Convert to TT format for FC2
        W4_tt = ttnn.to_layout(ttnn.transpose(W4, 0, 1), ttnn.TILE_LAYOUT)
        B4_tt = ttnn.to_layout(B4.reshape((1, -1)), ttnn.TILE_LAYOUT)

        # Apply FC2 (output logits)
        out = ttnn.linear(out, W4_tt, bias=B4_tt)

        # Convert prediction back to torch
        prediction = ttnn.to_torch(out)
        predicted_label = torch.argmax(prediction, dim=1).item()
        correct += predicted_label == label.item()
        total += 1

        logger.info(f"Sample {i+1}: Predicted={predicted_label}, Actual={label.item()}")

    logger.info(f"\nTT-NN SimpleCNN Inference Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
