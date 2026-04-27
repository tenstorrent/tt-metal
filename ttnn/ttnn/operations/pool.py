# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Tuple


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


def golden_grid_sample(
    input_tensor: ttnn.Tensor,
    grid: ttnn.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    batch_output_channels: bool = False,
    grid_batching_factor: int = None,
    **_,
):
    """
    Golden function for grid_sample operation using torch.nn.functional.grid_sample.
    """
    import torch
    from tests.sweep_framework.sweep_utils.pool2d_common import prepare_grid_batching_expected_output

    N, H_grid, W_grid, last_dim = grid.shape
    K_grid = last_dim // 2

    input_nchw = input_tensor.permute(0, 3, 1, 2)
    C = input_nchw.shape[1]

    total_W = W_grid * K_grid
    grid_unpacked = grid.reshape(N, H_grid, total_W, 2)

    output_nchw = torch.nn.functional.grid_sample(
        input_nchw.float(), grid_unpacked.float(), mode=mode, padding_mode=padding_mode, align_corners=align_corners
    )

    output_nhwc = output_nchw.permute(0, 2, 3, 1).to(input_tensor.dtype)

    effective_K = grid_batching_factor if grid_batching_factor is not None else K_grid

    expected_shape, output_nhwc = prepare_grid_batching_expected_output(
        output_nhwc, N, H_grid, total_W, C, effective_K, batch_output_channels
    )

    return output_nhwc.reshape(expected_shape)


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
