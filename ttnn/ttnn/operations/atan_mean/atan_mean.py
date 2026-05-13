# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
atan_mean — main entry point.

Computes ``torch.atan(x).mean(dim=-1)`` as a single fused TTNN kernel. The
function validates the input, allocates a rank-4 ``(N, C, H, 1)`` output tile-
laid out tensor, dispatches one ``ttnn.generic_op`` call, and then returns a
metadata-only ``ttnn.squeeze(out, dim=-1)`` view of rank 3.

See ``op_design.md`` next to this file for the full architectural rationale.
"""

import ttnn

from .atan_mean_program_descriptor import create_program_descriptor


def atan_mean(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    Fused ``atan`` then row-mean along ``dim=-1``.

    Args:
        input_tensor: ``float32`` / ``bfloat16`` / ``bfloat8_b`` TILE_LAYOUT
            tensor of rank exactly 4 with H and W tile-aligned (divisible by
            32), located on the device.

    Returns:
        Rank-3 ``(N, C, H)`` tensor (same dtype as ``input_tensor``) holding
        ``atan(x).mean(dim=-1)``.
    """
    _validate_input(input_tensor)

    device = input_tensor.device()
    output_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Allocate rank-4 ``(N, C, H, 1)`` TILE_LAYOUT — the matmul-mode REDUCE_ROW
    # path packs row results into column 0 of each output tile, so the trailing
    # dim of 1 (padded to a tile of width 32 in the physical layout) is the
    # canonical output footprint. ``ttnn.squeeze`` at the end is a metadata-only
    # view that does not dispatch a program.
    input_shape = list(input_tensor.shape)
    output_shape = [input_shape[0], input_shape[1], input_shape[2], 1]

    # NOTE: positional args only — keyword form is not supported by the binding.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor)

    # Output tensor MUST be last in the io_tensors list.
    out = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)

    # Squeeze the trailing length-1 dim → rank-3 ``(N, C, H)``. This is a
    # tensor-metadata operation, not a kernel dispatch.
    return ttnn.squeeze(out, dim=-1)


_SUPPORTED_DTYPES = (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Input validation. See op_design.md → 'Validation'."""
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError(
            "atan_mean: input tensor must be allocated on device " f"(got storage_type={input_tensor.storage_type()})."
        )

    if input_tensor.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"atan_mean: only {_SUPPORTED_DTYPES} are supported (got dtype={input_tensor.dtype}).")

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"atan_mean: input must be TILE_LAYOUT (got layout={input_tensor.layout}).")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"atan_mean: input rank must be exactly 4 in Phase 0 (got shape={shape}).")

    if shape[-2] % 32 != 0:
        raise ValueError(f"atan_mean: input H must be divisible by 32 (got H={shape[-2]} in shape={shape}).")

    if shape[-1] % 32 != 0:
        raise ValueError(f"atan_mean: input W must be divisible by 32 (got W={shape[-1]} in shape={shape}).")
