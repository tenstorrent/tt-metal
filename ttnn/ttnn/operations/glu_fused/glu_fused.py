# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
glu_fused — entry point.

Validates the input tensor, allocates the (half-width) output tensor on device,
builds the program descriptor, and dispatches a single ``ttnn.generic_op`` call.

The operation matches ``torch.nn.functional.glu(x, dim=-1)``:

    out[n, c, h, j] = x[n, c, h, j] * sigmoid(x[n, c, h, j + W/2])  for j in [0, W/2)

Supported configurations:
    - dtype: float32 (Phase 0) or bfloat16 (Refinement 2)
    - layout: TILE_LAYOUT, rank == 4
    - input on device, output inherits input memory config
    - W % 64 == 0  (each half tile-aligned)
    - H % 32 == 0  (logical H tile-aligned)

Compute config is dtype-aware:
    - fp32 input: HiFi4 + fp32_dest_acc_en + UnpackToDestFp32 on input CBs.
      Max precision; this is what the Phase 0 verifier locked in.
    - bf16 input: LoFi + fp32_dest_acc_en=False + default unpack mode.
      Matches the precision regime bf16 inputs already carry — running the
      fp32 settings on bf16 inputs is pure overhead with no precision gain
      (UnpackToDestFp32 just zero-extends; fp32 DEST halves auto-batching).
"""

import ttnn

from .glu_fused_program_descriptor import create_program_descriptor


def glu_fused(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    Gated Linear Unit along the last dim, fused into one TTNN kernel.

    Args:
        input_tensor: float32 or bfloat16 TILE_LAYOUT tensor of rank 4
            ``(N, C, H, W)`` on device, with ``W % 64 == 0`` and
            ``H % 32 == 0``.

    Returns:
        Tensor of shape ``(N, C, H, W/2)``, same dtype/layout/memory_config as
        the input, holding ``glu(input_tensor, dim=-1)``.
    """
    _validate_input(input_tensor)

    device = input_tensor.device()
    output_memory_config = input_tensor.memory_config()

    shape = list(input_tensor.shape)
    output_shape = shape[:-1] + [shape[-1] // 2]

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
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    """Phase-0 validation. See ``op_design.md`` → "Validation"."""
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError(
            "glu_fused: input tensor must be allocated on device " f"(got storage_type={input_tensor.storage_type()})."
        )

    if input_tensor.dtype not in (ttnn.float32, ttnn.bfloat16):
        raise ValueError(
            f"glu_fused: only float32 and bfloat16 are supported (got dtype={input_tensor.dtype})."
        )

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"glu_fused: input must be TILE_LAYOUT (got layout={input_tensor.layout}).")

    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError("glu_fused: input must be rank 4 (N, C, H, W) in Phase 0 " f"(got shape={shape}).")

    if shape[-1] % 64 != 0:
        raise ValueError(
            "glu_fused: input W must be divisible by 64 so each half is tile-aligned "
            f"(got shape={shape}; W={shape[-1]})."
        )

    if shape[-2] % 32 != 0:
        raise ValueError("glu_fused: input H must be divisible by 32 " f"(got shape={shape}; H={shape[-2]}).")
