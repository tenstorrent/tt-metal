#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Conv2D Example Script
This script demonstrates how to use ttnn.conv2d to implement the same convolution
as DeiT patch embeddings, but with randomly generated input and weights.
"""

import torch
import ttnn
import numpy as np

def main():
    # Initialize device
    device = ttnn.CreateDevice(device_id=0, l1_small_size=32768)
    
    # DeiT patch embeddings parameters (matching deit_config.py)
    batch_size = 1
    num_channels = 3     # RGB input channels (config.num_channels)
    hidden_size = 768    # Embedding dimension (config.hidden_size)
    image_size = 224     # Input image size (config.image_size)
    patch_size = 16      # Patch size (config.patch_size)
    
    # Derived parameters
    in_channels = num_channels
    out_channels = hidden_size
    input_height = image_size
    input_width = image_size
    kernel_height = patch_size
    kernel_width = patch_size
    stride_h = patch_size    # Stride equal to patch size (non-overlapping patches)
    stride_w = patch_size
    pad_h = 0               # No padding
    pad_w = 0
    
    print("=== TTNN Conv2D Example (DeiT Patch Embeddings Implementation) ===")
    print(f"Implementing DeiT patch embeddings convolution with random data")
    print(f"Input shape: [{batch_size}, {input_height}, {input_width}, {in_channels}] (NHWC)")
    print(f"Weight shape: [{out_channels}, {in_channels}, {kernel_height}, {kernel_width}]")
    print(f"Kernel size: {kernel_height}x{kernel_width} (patch_size)")
    print(f"Stride: {stride_h}x{stride_w} (patch_size, non-overlapping)")
    print(f"Padding: {pad_h}x{pad_w} (no padding)")
    
    # Calculate expected output dimensions
    out_height = (input_height + 2 * pad_h - kernel_height) // stride_h + 1
    out_width = (input_width + 2 * pad_w - kernel_width) // stride_w + 1
    num_patches = out_height * out_width
    print(f"Expected output shape: [{batch_size}, {out_height}, {out_width}, {out_channels}] (NHWC)")
    print(f"Number of patches: {num_patches} ({out_height}x{out_width})")
    
    # Generate random input tensor (NCHW format first, then convert to NHWC)
    print("\n=== Generating Random Input ===")
    torch_input_nchw = torch.randn(batch_size, in_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))  # Convert to NHWC
    print(f"Generated input tensor with shape: {torch_input_nhwc.shape}")
    
    # Convert input to TTNN tensor
    ttnn_input_tensor = ttnn.from_torch(torch_input_nhwc, ttnn.bfloat16)
    
    # Generate random weight tensor
    print("\n=== Generating Random Weights ===")
    torch_weight_tensor = torch.randn(out_channels, in_channels, kernel_height, kernel_width, dtype=torch.bfloat16)
    print(f"Generated weight tensor with shape: {torch_weight_tensor.shape}")
    
    # Convert weight to TTNN tensor
    ttnn_weight_tensor = ttnn.from_torch(torch_weight_tensor, ttnn.bfloat16)
    
    # Generate random bias tensor
    print("\n=== Generating Random Bias ===")
    bias_shape = [1, 1, 1, out_channels]
    torch_bias_tensor = torch.randn(bias_shape, dtype=torch.bfloat16)
    print(f"Generated bias tensor with shape: {torch_bias_tensor.shape}")
    
    # Convert bias to TTNN tensor
    ttnn_bias_tensor = ttnn.from_torch(torch_bias_tensor, ttnn.bfloat16)
    
    # Configure convolution
    print("\n=== Configuring Convolution ===")
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16
    )
    
    # Perform convolution
    print("\n=== Performing Convolution ===")
    print("Running ttnn.conv2d...")
    
    try:
        [ttnn_output_tensor_on_device, [actual_out_height, actual_out_width]] = ttnn.conv2d(
            input_tensor=ttnn_input_tensor,
            weight_tensor=ttnn_weight_tensor,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
            bias_tensor=ttnn_bias_tensor,
            kernel_size=(kernel_height, kernel_width),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            return_output_dim=True,
        )
        
        print("✓ Convolution completed successfully!")
        print(f"Actual output dimensions: {actual_out_height} x {actual_out_width}")
        
        # Transfer output back to host
        ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor_on_device)
        torch_output_tensor = ttnn.to_torch(ttnn_output_tensor)
        
        print(f"Output tensor shape: {torch_output_tensor.shape}")
        print(f"Output tensor dtype: {torch_output_tensor.dtype}")
        
        # Print some statistics
        print("\n=== Output Statistics ===")
        print(f"Output mean: {torch_output_tensor.mean().item():.6f}")
        print(f"Output std: {torch_output_tensor.std().item():.6f}")
        print(f"Output min: {torch_output_tensor.min().item():.6f}")
        print(f"Output max: {torch_output_tensor.max().item():.6f}")
        
        # Verify output shape matches expectation
        expected_shape = (batch_size, out_height, out_width, out_channels)
        if torch_output_tensor.shape == expected_shape:
            print(f"✓ Output shape matches expectation: {expected_shape}")
        else:
            print(f"⚠ Output shape mismatch! Expected: {expected_shape}, Got: {torch_output_tensor.shape}")
            
    except Exception as e:
        print(f"✗ Convolution failed with error: {e}")
        raise
    
    finally:
        # Clean up device
        ttnn.CloseDevice(device)
        print("\n=== Device Closed ===")

if __name__ == "__main__":
    main()