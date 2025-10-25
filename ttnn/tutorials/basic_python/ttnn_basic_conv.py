# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from loguru import logger


def main():
    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0, l1_small_size=8192)

    def forward(
        input_tensor: ttnn.Tensor,
        weight_tensor: ttnn.Tensor,
        bias_tensor: ttnn.Tensor,
        out_channels: int,
        kernel_size: tuple,
        device: ttnn.Device,
    ) -> ttnn.Tensor:
        # Permute input from PyTorch BCHW (batch, channel, height, width)
        # to NHWC (batch, height, width, channel) which TTNN expects
        permuted_input = ttnn.permute(input_tensor, (0, 2, 3, 1))

        # Get shape after permutation
        B, H, W, C = permuted_input.shape

        # Reshape input to a flat image of shape (1, 1, B*H*W, C)
        # This flattens the spatial dimensions and prepares it for TTNN conv2d
        reshaped_input = ttnn.reshape(permuted_input, (1, 1, B * H * W, C))

        # Set up convolution configuration for TTNN conv2d
        conv_config = ttnn.Conv2dConfig(weights_dtype=weight_tensor.dtype)

        # Perform 2D convolution using TTNN
        out = ttnn.conv2d(
            input_tensor=reshaped_input,
            weight_tensor=weight_tensor,
            bias_tensor=bias_tensor,
            in_channels=C,
            out_channels=out_channels,
            device=device,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(1, 1),
            batch_size=1,
            input_height=1,
            input_width=B * H * W,
            conv_config=conv_config,
            groups=0,  # No grouped convolution
        )

        # Optionally convert back to torch tensor: out_torch = ttnn.to_torch(out)
        return out

    batch = 1
    in_channels = 3
    out_channels = 4
    height = width = 2  # Small dimensions to avoid device memory issues
    kernel_size = (3, 3)

    # Create random input tensor in BCHW format
    x = ttnn.rand((batch, in_channels, height, width), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Random weight tensor for convolution: (out_channels, in_channels, kH, kW)
    w = ttnn.rand(
        (out_channels, in_channels, *kernel_size), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )

    # Bias tensor, broadcastable to the output shape
    b = ttnn.zeros((1, 1, 1, out_channels), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run forward conv pass and print output shape
    out_torch = forward(x, w, b, out_channels, kernel_size, device)
    logger.info(f"✅ Success! Conv2D output shape: {out_torch.shape}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
