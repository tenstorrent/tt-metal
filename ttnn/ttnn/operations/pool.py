# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Tuple


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
    from tests.sweep_framework.sweep_utils.pool2d_common import prepare_torch_pool_input, pool_output_to_flat_nhwc

    input_nchw, torch_padding = prepare_torch_pool_input(
        input_tensor, batch_size, input_h, input_w, channels, padding, pad_fill_value=float("-inf")
    )

    output_tensor, indices = torch.nn.functional.max_pool2d(
        input_nchw,
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    output_tensor = pool_output_to_flat_nhwc(output_tensor)
    indices = pool_output_to_flat_nhwc(indices)

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
    from tests.sweep_framework.sweep_utils.pool2d_common import prepare_torch_pool_input, pool_output_to_flat_nhwc

    input_nchw, torch_padding = prepare_torch_pool_input(
        input_tensor, batch_size, input_h, input_w, channels, padding, pad_fill_value=0
    )

    output_tensor = torch.nn.functional.avg_pool2d(
        input_nchw,
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    return pool_output_to_flat_nhwc(output_tensor)


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
    from tests.sweep_framework.sweep_utils.pool2d_common import pool_output_to_flat_nhwc

    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)
    output_tensor = torch.nn.functional.adaptive_avg_pool2d(input_nchw, output_size=output_size)
    return pool_output_to_flat_nhwc(output_tensor)


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
    from tests.sweep_framework.sweep_utils.pool2d_common import pool_output_to_flat_nhwc

    input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)
    output_tensor, _ = torch.nn.functional.adaptive_max_pool2d(input_nchw, output_size=output_size, return_indices=True)
    return pool_output_to_flat_nhwc(output_tensor)


if hasattr(ttnn, "adaptive_max_pool2d"):
    ttnn.attach_golden_function(ttnn.adaptive_max_pool2d, golden_adaptive_max_pool2d)


def golden_upsample(
    input_tensor: ttnn.Tensor,
    scale_factor,
    mode: str = "nearest",
    batch_size: int = None,
    input_h: int = None,
    input_w: int = None,
    channels: int = None,
    align_corners: bool = None,
    **_,
):
    """
    Golden function for upsample operation using torch.nn.functional.interpolate.

    Supports two input formats:
    - (N, H, W, C): Used by attach_golden_function (ttnn.upsample passes this format)
    - (1, 1, N*H*W, C): Used by direct test calls with explicit batch_size/input_h/input_w/channels

    Args:
        input_tensor: Input tensor in (N, H, W, C) or (1, 1, N*H*W, C) format
        scale_factor: Upsampling scale factor - int, float, [int, int], or [float, float]
        mode: Interpolation mode ("nearest" or "bilinear")
        batch_size: Number of batches (required for (1, 1, N*H*W, C) format)
        input_h: Input height (required for (1, 1, N*H*W, C) format)
        input_w: Input width (required for (1, 1, N*H*W, C) format)
        channels: Number of channels (required for (1, 1, N*H*W, C) format)
        align_corners: Whether to align corners (only for bilinear mode)

    Returns:
        Output tensor in the same spatial format as input
    """
    import torch

    # Normalize scale_factor to a tuple
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)

    if batch_size is not None:
        # Direct call with (1, 1, N*H*W, C) format - reshape to (N, H, W, C) first
        input_nhwc = input_tensor.reshape(batch_size, input_h, input_w, channels)
        input_nchw = input_nhwc.permute(0, 3, 1, 2)
        output_nchw = torch.nn.functional.interpolate(
            input_nchw, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
        N, C, H, W = output_nchw.shape
        return output_nchw.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C)
    else:
        # attach_golden_function call with (N, H, W, C) format
        input_nchw = input_tensor.permute(0, 3, 1, 2)
        output_nchw = torch.nn.functional.interpolate(
            input_nchw, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
        return output_nchw.permute(0, 2, 3, 1)


ttnn.attach_golden_function(ttnn.upsample, golden_upsample)


def golden_grid_sample(
    input_tensor: ttnn.Tensor,
    grid: ttnn.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    batch_size: int = None,
    input_h: int = None,
    input_w: int = None,
    channels: int = None,
    output_h: int = None,
    output_w: int = None,
    **_,
):
    """
    Golden function for grid_sample operation using torch.nn.functional.grid_sample.

    Supports two input formats:
    - (N, H_in, W_in, C): Used by attach_golden_function (ttnn.grid_sample passes this format)
    - (1, 1, N*H*W, C): Used by direct test calls with explicit batch_size/input_h/input_w/channels

    Args:
        input_tensor: Input tensor in (N, H_in, W_in, C) or (1, 1, N*H*W, C) format
        grid: Grid tensor in (N, H_out, W_out, 2) or (1, 1, N*out_H*out_W, 2) format
        mode: Interpolation mode ("bilinear" or "nearest")
        padding_mode: Padding mode ("zeros", "border", or "reflection")
        align_corners: Whether to align corners
        batch_size: Number of batches (required for (1, 1, N*H*W, C) format)
        input_h: Input height (required for (1, 1, N*H*W, C) format)
        input_w: Input width (required for (1, 1, N*H*W, C) format)
        channels: Number of channels (required for (1, 1, N*H*W, C) format)
        output_h: Output height (required for (1, 1, N*out_H*out_W, 2) format)
        output_w: Output width (required for (1, 1, N*out_H*out_W, 2) format)

    Returns:
        Output tensor in the same spatial format as input
    """
    import torch

    if batch_size is not None:
        # Direct call with (1, 1, N*H*W, C) format
        input_nchw = input_tensor.reshape(batch_size, input_h, input_w, channels).permute(0, 3, 1, 2)
        grid_nhwc = grid.reshape(batch_size, output_h, output_w, 2)

        output_nchw = torch.nn.functional.grid_sample(
            input_nchw.float(), grid_nhwc.float(), mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )

        N, C, H, W = output_nchw.shape
        return output_nchw.permute(0, 2, 3, 1).reshape(1, 1, N * H * W, C).to(input_tensor.dtype)
    else:
        # attach_golden_function call with (N, H_in, W_in, C) format
        input_nchw = input_tensor.permute(0, 3, 1, 2)
        # grid is already (N, H_out, W_out, 2)

        output_nchw = torch.nn.functional.grid_sample(
            input_nchw.float(), grid.float(), mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )

        return output_nchw.permute(0, 2, 3, 1).to(input_tensor.dtype)


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
