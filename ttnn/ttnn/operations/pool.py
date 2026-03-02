# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Tuple


def validate_maxpool2d_indices(
    input_tensor, torch_indices, ttnn_indices, kernel_size, stride, padding, dilation, dtype
):
    """
    Validate indices by checking if differences are due to valid tie-breaking.
    Note: input tensors should be in [N, H, W, C] format.
    Supports both uint16 and uint32 index tensors (indices should be converted to int64 before calling).
    Returns (indices_valid, tie_breaking_differences, actual_errors, value_differences, window_violations)
    """
    import torch
    import math

    batch_size, input_h, input_w, channels = input_tensor.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_tb, pad_lr = padding
    dilation_h, dilation_w = dilation

    # Check if indices are exactly equal first
    indices_match = torch.equal(torch_indices, ttnn_indices)
    if indices_match:
        return True, 0, 0, 0, 0

    # Find positions where indices don't match
    diff = torch.abs(torch_indices - ttnn_indices)
    mismatch_positions = torch.nonzero(diff, as_tuple=False)

    tie_breaking_differences = 0
    actual_errors = 0
    value_differences = 0
    window_violations = 0
    for pos in mismatch_positions:
        n, h, w, c = pos
        torch_idx = torch_indices[n, h, w, c]
        ttnn_idx = ttnn_indices[n, h, w, c]

        # Convert linear indices to spatial coordinates
        torch_h = torch_idx // input_w
        torch_w = torch_idx % input_w
        ttnn_h = ttnn_idx // input_w
        ttnn_w = ttnn_idx % input_w

        # Get input values at these positions
        if ttnn_h >= 0 and ttnn_w >= 0 and ttnn_h < input_h and ttnn_w < input_w:
            torch_input_val = input_tensor[n, torch_h, torch_w, c]
            ttnn_input_val = input_tensor[n, ttnn_h, ttnn_w, c]
        else:
            actual_errors += 1
            window_violations += 1
            continue

        # Check if this is a valid tie-breaking difference
        # Two conditions must be satisfied:
        # 1. The values must be the same
        # 2. Both indices must be within the same kernel window

        # Check if values are the same
        atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
        if dtype == ttnn.bfloat8_b:
            atol = 0.35
            values_same = math.isclose(torch_input_val, ttnn_input_val, abs_tol=atol, rel_tol=rtol)
        else:
            values_same = torch_input_val == ttnn_input_val

        # Check if both indices are within the same kernel window
        def is_in_dilated_kernel_window(
            input_h, input_w, kernel_top_left_h, kernel_top_left_w, kernel_h, kernel_w, dilation_h, dilation_w
        ):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    kernel_pos_h = kernel_top_left_h + kh * dilation_h
                    kernel_pos_w = kernel_top_left_w + kw * dilation_w
                    if kernel_pos_h == input_h and kernel_pos_w == input_w:
                        return True
            return False

        kernel_top_left_h = h * stride_h - pad_tb
        kernel_top_left_w = w * stride_w - pad_lr
        ttnn_in_window = is_in_dilated_kernel_window(
            ttnn_h, ttnn_w, kernel_top_left_h, kernel_top_left_w, kernel_h, kernel_w, dilation_h, dilation_w
        )

        if values_same and ttnn_in_window:
            tie_breaking_differences += 1
        elif not ttnn_in_window:
            actual_errors += 1
            window_violations += 1
        else:
            actual_errors += 1
            value_differences += 1

    # Indices are valid if there are no actual errors
    return (actual_errors == 0), tie_breaking_differences, actual_errors, value_differences, window_violations


def golden_maxpool2d(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding,  # Can be Tuple[int, int] or List[int, int, int, int]
    dilation: Tuple[int, int],
    ceil_mode: bool = False,
    return_indices: bool = False,
    **_,
):
    import torch

    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, -1).permute(
        0, 3, 1, 2
    )  # 1, 1, NHW, C -> N, C, H, W

    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        # Apply padding using torch.nn.functional.pad with 4D format
        # ttnn format: [pad_t, pad_b, pad_l, pad_r]
        # torch.nn.functional.pad expects: [pad_left, pad_right, pad_top, pad_bottom]
        pad_t, pad_b, pad_l, pad_r = padding
        input_nchw = torch.nn.functional.pad(
            input_nchw, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=float("-inf")
        )
        torch_padding = 0  # No padding in max_pool2d since we already padded
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        # Standard 2D padding format (pad_h, pad_w)
        torch_padding = padding
    else:
        # Assume it's already in the correct format
        torch_padding = padding

    output_tensor, indices = torch.nn.functional.max_pool2d(
        input_nchw,
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    N, C, H, W = output_tensor.shape
    output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)  # N, C, H, W -> 1, 1, NHW, C
    indices = indices.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)  # N, C, H, W -> 1, 1, NHW, C

    if return_indices:
        return output_tensor, indices

    return output_tensor


ttnn.attach_golden_function(ttnn.max_pool2d, golden_maxpool2d)


def golden_global_avg_pool2d(input_tensor: ttnn.Tensor):
    import torch

    output_size = (1, 1)
    return torch.nn.functional.global_avg_pool2d(input_tensor, output_size)


ttnn.attach_golden_function(ttnn.global_avg_pool2d, golden_global_avg_pool2d)


def golden_rotate(
    input_tensor: ttnn.Tensor,
    angle: float,
    center=None,
    fill: float = 0.0,
    expand: bool = False,
    interpolation_mode: str = "nearest",
    **_,
):
    """
    Golden function for rotate operation using torchvision.transforms.functional.rotate.

    Args:
        input_tensor: Input tensor in NHWC format
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Optional rotation center as (x, y) in pixel coordinates, where x is
                the horizontal/width coordinate and y is the vertical/height coordinate.
                If None, uses image center.
        fill: Fill value for areas outside the rotated tensor
        expand: Must be False (only same-size rotation supported)
        interpolation_mode: "nearest" or "bilinear"

    Returns:
        Rotated tensor in NHWC format
    """
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    # Convert NHWC to NCHW for torchvision
    torch_input_nchw = input_tensor.permute(0, 3, 1, 2).to(torch.float32)

    # Map interpolation mode
    if interpolation_mode == "nearest":
        torch_interp = InterpolationMode.NEAREST
    elif interpolation_mode == "bilinear":
        torch_interp = InterpolationMode.BILINEAR
    else:
        raise ValueError(f"Unsupported interpolation mode: {interpolation_mode}")

    # Apply rotation
    torch_output_nchw = TF.rotate(
        torch_input_nchw,
        angle=float(angle),
        interpolation=torch_interp,
        center=center,
        fill=fill,
    )

    # Convert back to NHWC and original dtype
    torch_output_nhwc = torch_output_nchw.permute(0, 2, 3, 1).to(input_tensor.dtype)

    return torch_output_nhwc


ttnn.attach_golden_function(ttnn.rotate, golden_rotate)


def golden_avg_pool2d(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding,  # Can be Tuple[int, int] or List[int, int, int, int]
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
    **_,
):
    """
    Golden function for avg_pool2d operation using torch.nn.functional.avg_pool2d.

    Args:
        input_tensor: Input tensor in (1, 1, N*H*W, C) format
        batch_size: Number of batches
        input_h: Input height
        input_w: Input width
        channels: Number of channels
        kernel_size: Pooling kernel size (kernel_h, kernel_w)
        stride: Pooling stride (stride_h, stride_w)
        padding: Can be 2D (pad_h, pad_w) or 4D [pad_t, pad_b, pad_l, pad_r]
        ceil_mode: Use ceiling for output shape calculation
        count_include_pad: Include padding in average calculation
        divisor_override: Override the divisor used in averaging

    Returns:
        Output tensor in (1, 1, N*out_H*out_W, C) format
    """
    import torch

    # Reshape from (1, 1, N*H*W, C) to (N, H, W, C) then to (N, C, H, W)
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    # Handle 4D padding (asymmetric) vs 2D padding (symmetric)
    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        # Apply padding manually using torch.nn.functional.pad with 4D format
        # ttnn format: [pad_t, pad_b, pad_l, pad_r]
        # torch.nn.functional.pad expects: [pad_left, pad_right, pad_top, pad_bottom]
        pad_t, pad_b, pad_l, pad_r = padding
        input_nchw = torch.nn.functional.pad(input_nchw, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0)
        torch_padding = 0  # No padding in avg_pool2d since we already padded
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        # Standard 2D padding format (pad_h, pad_w)
        torch_padding = padding
    else:
        # Assume it's already in the correct format
        torch_padding = padding

    output_tensor = torch.nn.functional.avg_pool2d(
        input_nchw,
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    N, C, H, W = output_tensor.shape
    # Convert from (N, C, H, W) to (1, 1, N*H*W, C)
    output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)

    return output_tensor


ttnn.attach_golden_function(ttnn.avg_pool2d, golden_avg_pool2d)


def golden_adaptive_avg_pool2d(
    input_tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    output_size: Tuple[int, int],
    **_,
):
    """
    Golden function for adaptive_avg_pool2d operation using torch.nn.functional.adaptive_avg_pool2d.

    Args:
        input_tensor: Input tensor in (1, 1, N*H*W, C) format
        batch_size: Number of batches
        input_h: Input height
        input_w: Input width
        channels: Number of channels
        output_size: Target output size (out_h, out_w)

    Returns:
        Output tensor in (1, 1, N*out_H*out_W, C) format
    """
    import torch

    # Reshape from (1, 1, N*H*W, C) to (N, H, W, C) then to (N, C, H, W)
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    output_tensor = torch.nn.functional.adaptive_avg_pool2d(input_nchw, output_size=output_size)

    N, C, H, W = output_tensor.shape
    # Convert from (N, C, H, W) to (1, 1, N*H*W, C)
    output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)

    return output_tensor


if hasattr(ttnn, "adaptive_avg_pool2d"):
    ttnn.attach_golden_function(ttnn.adaptive_avg_pool2d, golden_adaptive_avg_pool2d)


def golden_adaptive_max_pool2d(
    input_tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    output_size: Tuple[int, int],
    **_,
):
    """
    Golden function for adaptive_max_pool2d operation using torch.nn.functional.adaptive_max_pool2d.

    Args:
        input_tensor: Input tensor in (1, 1, N*H*W, C) format
        batch_size: Number of batches
        input_h: Input height
        input_w: Input width
        channels: Number of channels
        output_size: Target output size (out_h, out_w)

    Returns:
        Output tensor in (1, 1, N*out_H*out_W, C) format
    """
    import torch

    # Reshape from (1, 1, N*H*W, C) to (N, H, W, C) then to (N, C, H, W)
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    output_tensor, _ = torch.nn.functional.adaptive_max_pool2d(input_nchw, output_size=output_size, return_indices=True)

    N, C, H, W = output_tensor.shape
    # Convert from (N, C, H, W) to (1, 1, N*H*W, C)
    output_tensor = output_tensor.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)

    return output_tensor


if hasattr(ttnn, "adaptive_max_pool2d"):
    ttnn.attach_golden_function(ttnn.adaptive_max_pool2d, golden_adaptive_max_pool2d)


def golden_upsample(
    input_tensor: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    scale_factor: Tuple[int, int],
    mode: str = "nearest",
    align_corners: bool = None,
    **_,
):
    """
    Golden function for upsample operation using torch.nn.functional.interpolate.

    Args:
        input_tensor: Input tensor in (1, 1, N*H*W, C) format
        batch_size: Number of batches
        input_h: Input height
        input_w: Input width
        channels: Number of channels
        scale_factor: Upsampling scale factor (scale_h, scale_w)
        mode: Interpolation mode ("nearest" or "bilinear")
        align_corners: Whether to align corners (only for bilinear mode)

    Returns:
        Output tensor in (1, 1, N*out_H*out_W, C) format
    """
    import torch

    # Reshape from (1, 1, N*H*W, C) to (N, H, W, C) then to (N, C, H, W)
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    # Apply upsample
    output_nchw = torch.nn.functional.interpolate(
        input_nchw, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )

    # Convert back to (1, 1, N*H*W, C)
    N, C, H, W = output_nchw.shape
    output_tensor = output_nchw.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)

    return output_tensor


ttnn.attach_golden_function(ttnn.upsample, golden_upsample)


def golden_grid_sample(
    input_tensor: ttnn.Tensor,
    grid: ttnn.Tensor,
    batch_size: int,
    input_h: int,
    input_w: int,
    channels: int,
    output_h: int,
    output_w: int,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    **_,
):
    """
    Golden function for grid_sample operation using torch.nn.functional.grid_sample.

    Args:
        input_tensor: Input tensor in (1, 1, N*H*W, C) format
        grid: Grid tensor in (1, 1, N*out_H*out_W, 2) format
        batch_size: Number of batches
        input_h: Input height
        input_w: Input width
        channels: Number of channels
        output_h: Output height
        output_w: Output width
        mode: Interpolation mode ("bilinear" or "nearest")
        padding_mode: Padding mode ("zeros", "border", or "reflection")
        align_corners: Whether to align corners

    Returns:
        Output tensor in (1, 1, N*out_H*out_W, C) format
    """
    import torch

    # Reshape input from (1, 1, N*H*W, C) to (N, C, H, W)
    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)

    # Reshape grid from (1, 1, N*out_H*out_W, 2) to (N, out_H, out_W, 2)
    grid_nhwc = grid.reshape(batch_size, output_h, output_w, 2)

    # Apply grid_sample
    output_nchw = torch.nn.functional.grid_sample(
        input_nchw.float(), grid_nhwc.float(), mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    # Convert back to (1, 1, N*H*W, C)
    N, C, H, W = output_nchw.shape
    output_tensor = output_nchw.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C).to(input_tensor.dtype)

    return output_tensor


ttnn.attach_golden_function(ttnn.grid_sample, golden_grid_sample)


def prepare_grid_sample_grid(*args, **kwargs):
    """
    Precomputes grid sample data for optimized kernel execution.

    This function takes a normalized grid tensor and precomputes the pixel coordinates
    and bilinear interpolation weights needed for grid sampling.

    Args:
        grid (ttnn.Tensor): Grid tensor of shape (N, H_out, W_out, 2) with normalized coordinates in [-1, 1]
        input_shape (List[int]): Input tensor dimensions [N, H_in, W_in, C] in NHWC format

    Keyword Args:
        padding_mode (str): How to handle out-of-bounds coordinates. Currently only "zeros" is supported.
        output_dtype (ttnn.DataType, optional): Data type for the output tensor. Default: bfloat16

    Returns:
        ttnn.Tensor: Precomputed grid tensor of shape (N, H_out, W_out, 6) where:
                    - [:, :, :, 0]: North-west height coordinate (as integer stored in bfloat16)
                    - [:, :, :, 1]: North-west width coordinate (as integer stored in bfloat16)
                    - [:, :, :, 2]: Weight for north-west pixel
                    - [:, :, :, 3]: Weight for north-east pixel
                    - [:, :, :, 4]: Weight for south-west pixel
                    - [:, :, :, 5]: Weight for south-east pixel

    Example:
        >>> # Create a normalized grid
        >>> grid = ttnn.from_torch(torch.randn(1, 8, 8, 2), dtype=ttnn.float32)
        >>> input_shape = [1, 32, 32, 64]  # N, H, W, C
        >>> precomputed_grid = ttnn.prepare_grid_sample_grid(grid, input_shape)
        >>> print(precomputed_grid.shape)  # [1, 8, 8, 6]
    """
    return ttnn._ttnn.operations.pool.prepare_grid_sample_grid(*args, **kwargs)
