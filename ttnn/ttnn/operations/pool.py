# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Tuple


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


def golden_upsample(
    input_tensor: ttnn.Tensor,
    scale_factor,
    mode: str = "nearest",
    **_,
):
    """
    Golden function for upsample operation using torch.nn.functional.interpolate.

    Args:
        input_tensor: Input tensor in (N, H, W, C) format
        scale_factor: Upsampling scale factor - int, float, [int, int], or [float, float]
        mode: Interpolation mode ("nearest" or "bilinear")

    Returns:
        Output tensor in (N, H, W, C) format
    """
    import torch

    # Normalize scale_factor to a tuple
    if isinstance(scale_factor, (int, float)):
        scale_factor = (scale_factor, scale_factor)

    input_nchw = input_tensor.permute(0, 3, 1, 2)
    output_nchw = torch.nn.functional.interpolate(input_nchw, scale_factor=scale_factor, mode=mode)
    return output_nchw.permute(0, 2, 3, 1)


ttnn.attach_golden_function(ttnn.upsample, golden_upsample)


def _grid_sample_precomputed_reference(input_tensor, grid, mode, N, H_grid, total_W, C):
    """Sample a packed precomputed grid with zero padding."""
    import torch

    if grid.dtype != torch.bfloat16:
        raise ValueError(
            f"A precomputed grid must be BFLOAT16 (its corner indices are int16 bit patterns), got {grid.dtype}."
        )

    H_in, W_in = input_tensor.shape[1], input_tensor.shape[2]
    elements_per_point = 2 if mode == "nearest" else 6

    # Unpack batched points from the last dimension into the width dimension.
    packed = grid.reshape(N, H_grid, total_W, elements_per_point)

    # Elements 0 and 1 are h0/w0 stored as int16 bit patterns bit_cast into bfloat16.
    corner_indices = packed.view(torch.int16).to(torch.int64)
    h0, w0 = corner_indices[..., 0], corner_indices[..., 1]

    flat_input = input_tensor.reshape(N, H_in * W_in, C).float()

    def gather(h, w):
        """input[n, h, w, :], or zero where the point falls outside the image."""
        valid = (h >= 0) & (h < H_in) & (w >= 0) & (w < W_in)
        idx = (h.clamp(0, H_in - 1) * W_in + w.clamp(0, W_in - 1)).reshape(N, -1, 1).expand(-1, -1, C)
        gathered = torch.gather(flat_input, 1, idx).reshape(N, H_grid, total_W, C)
        return torch.where(valid.unsqueeze(-1), gathered, 0)

    if mode == "nearest":
        output_nhwc = gather(h0, w0)
    else:
        # Elements 2..5 are the four bilinear weights as real bfloat16, in nw/ne/sw/se order.
        weights = packed[..., 2:6].float()
        output_nhwc = (
            weights[..., 0:1] * gather(h0, w0)
            + weights[..., 1:2] * gather(h0, w0 + 1)
            + weights[..., 2:3] * gather(h0 + 1, w0)
            + weights[..., 3:4] * gather(h0 + 1, w0 + 1)
        )

    return output_nhwc.to(input_tensor.dtype)


def golden_grid_sample(
    input_tensor: ttnn.Tensor,
    grid: ttnn.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    use_precomputed_grid: bool = False,
    batch_output_channels: bool = False,
    grid_batching_factor: int = None,
    **_,
):
    """
    Golden function for grid_sample with normalized or precomputed grids.

    Args:
        input_tensor: Input tensor in (N, H_in, W_in, C) format
        grid: Grid tensor in (N, H_out, W_out, 2*K) format where K is the grid batching factor.
              When K=1, this is the standard (N, H_out, W_out, 2) grid.
              When K>1, K coordinate sets are packed into the last dimension.
        mode: Interpolation mode ("bilinear" or "nearest")
        padding_mode: Padding mode ("zeros", "border", or "reflection")
        align_corners: Whether to align corners
        use_precomputed_grid: Whether `grid` holds precomputed corner indices and interpolation
            weights (as produced by ttnn.prepare_grid_sample_grid) instead of normalized
            coordinates. Such a grid has 6*K elements per point for "bilinear" and 2*K for
            "nearest", must be BFLOAT16, and already has align_corners and padding_mode baked in,
            so those two arguments are ignored in this mode.
        batch_output_channels: Controls how grid batching factor K affects output dimensions.
            When False (default): extend W dimension - output shape (N, H_out, total_W // K, C).
            When True: batch output channels - output shape (N, H_out, total_W // K, C*K).
        grid_batching_factor: Explicit grid batching factor K. When provided, overrides the K
            derived from the grid's last dimension. This is needed when the grid is passed in
            un-batched form (last_dim=2, K=1) but the TTNN op uses a higher batching factor.

    Returns:
        Output tensor in (N, H_out, total_W // K, C) if batch_output_channels=False,
        or (N, H_out, total_W // K, C*K) if batch_output_channels=True,
        where total_W = W_out * K_grid (total sample points per row).
    """
    import torch
    from tests.sweep_framework.sweep_utils.pool2d_common import prepare_grid_batching_expected_output

    N, H_grid, W_grid, last_dim = grid.shape

    C = input_tensor.shape[3]

    if use_precomputed_grid:
        K_grid = last_dim // (2 if mode == "nearest" else 6)
        total_W = W_grid * K_grid
        output_nhwc = _grid_sample_precomputed_reference(input_tensor, grid, mode, N, H_grid, total_W, C)
    else:
        K_grid = last_dim // 2

        input_nchw = input_tensor.permute(0, 3, 1, 2)

        # Unpack K_grid coordinate sets from last dim into the W dimension:
        # (N, H_grid, W_grid, 2*K_grid) -> (N, H_grid, W_grid*K_grid, 2)
        total_W = W_grid * K_grid
        grid_unpacked = grid.reshape(N, H_grid, total_W, 2)

        output_nchw = torch.nn.functional.grid_sample(
            input_nchw.float(),
            grid_unpacked.float(),
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        # Convert to NHWC: (N, C, H_grid, total_W) -> (N, H_grid, total_W, C)
        output_nhwc = output_nchw.permute(0, 2, 3, 1).to(input_tensor.dtype)

    # Use explicit grid_batching_factor if provided, otherwise use K_grid from grid shape
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
