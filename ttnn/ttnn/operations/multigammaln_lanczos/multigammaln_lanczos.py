# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
multigammaln_lanczos — main entry point.

Computes ``torch.special.multigammaln(x, p=4)`` as a single fused TTNN kernel.
Validates input, allocates the output tensor, builds the program descriptor, and
dispatches one ``ttnn.generic_op`` call.

See ``op_design.md`` next to this file for the full architectural rationale.
"""

import ttnn

from .multigammaln_lanczos_program_descriptor import create_program_descriptor


def multigammaln_lanczos(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    multigammaln(x, p=4) via the Lanczos 6-term polynomial.

    Args:
        input_tensor: float32 TILE_LAYOUT tensor of rank >= 2, with H and W
            tile-aligned (divisible by 32), located on the device. The safe
            numerical domain at fp32 is ``x ∈ [2.0, 10.0]``.

    Returns:
        Tensor with the same shape, dtype, layout, and memory config as the
        input, holding ``multigammaln(x, p=4)``.
    """
    _validate_input(input_tensor)

    device = input_tensor.device()
    output_memory_config = input_tensor.memory_config()

    output_shape = list(input_tensor.shape)

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
    """Phase-0 validation. See op_design.md → "Validation"."""
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError(
            "multigammaln_lanczos: input tensor must be allocated on device "
            f"(got storage_type={input_tensor.storage_type()})."
        )

    if input_tensor.dtype != ttnn.float32:
        raise ValueError(
            "multigammaln_lanczos: only float32 is supported in Phase 0 " f"(got dtype={input_tensor.dtype})."
        )

    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("multigammaln_lanczos: input must be TILE_LAYOUT " f"(got layout={input_tensor.layout}).")

    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError("multigammaln_lanczos: input must have at least 2 dimensions " f"(got shape={shape}).")

    if shape[-1] % 32 != 0 or shape[-2] % 32 != 0:
        raise ValueError(
            "multigammaln_lanczos: input H and W must be divisible by 32 "
            f"(got shape={shape}; H={shape[-2]}, W={shape[-1]})."
        )
