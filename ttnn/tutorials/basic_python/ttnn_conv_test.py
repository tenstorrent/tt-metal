import torch
import ttnn


def forward(input_tensor, weight_tensor, bias_tensor, out_channels, kernel_size, device):
    # Convert input, weights, and bias to TTNN tensors in ROW_MAJOR_LAYOUT
    ttnn_input = ttnn.from_torch(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_weight = ttnn.from_torch(weight_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    ttnn_bias = ttnn.from_torch(bias_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

    # Data prep: permute and reshape input
    # BCHW -> BHWC
    permuted_input = ttnn.permute(ttnn_input, (0, 2, 3, 1))
    B, H, W, C = permuted_input.shape
    reshaped_input = ttnn.reshape(permuted_input, (1, 1, B * H * W, C))

    conv_config = ttnn.Conv2dConfig(dtype=ttnn.bfloat16, weights_dtype=ttnn.bfloat16)

    out = ttnn.conv2d(
        input_tensor=reshaped_input,
        weight_tensor=ttnn_weight,
        bias_tensor=ttnn_bias,
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
        groups=0,
    )

    # out_torch = ttnn.to_torch(out)
    return out


def test_conv2d():
    torch.manual_seed(0)
    device_params = {"l1_small_size": 8192}
    device = ttnn.CreateDevice(device_id=0, **device_params)

    try:
        batch = 1
        in_channels = 3
        out_channels = 4
        height = width = 2  # Keep small to avoid sharding/core error
        kernel_size = (3, 3)

        # Generate dummy input, weights, and bias
        x = torch.randn(batch, in_channels, height, width, dtype=torch.float32)  # BCHW
        w = torch.randn(out_channels, in_channels, *kernel_size, dtype=torch.float32)
        b = torch.zeros((1, 1, 1, out_channels), dtype=torch.float32)

        out_torch = forward(x, w, b, out_channels, kernel_size, device)
        print("✅ Success! Conv2D output shape:", out_torch.shape)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_conv2d()
