# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Softmax — main entry point.

Numerically-stable row-wise (dim=-1) or column-wise (dim=-2) softmax for
fp32 TILE-layout 4D tensors.

The entry point validates the Phase-0 envelope (rank=4, fp32, TILE, H/W
tile-aligned, dim ∈ {-1, -2}, compute_kernel_config is either None or the
Phase-0 explicit descriptor), allocates the output tensor, constructs the
ProgramDescriptor via `softmax_program_descriptor.create_program_descriptor`,
and launches it through `ttnn.generic_op`.
"""

import ttnn

from .softmax_program_descriptor import create_program_descriptor


# Phase-0 supported compute kernel config: HiFi4 + fp32_dest_acc_en=True.
# `math_approx_mode` is unspecified by the spec but Phase-0 default is False;
# we don't gate on it here (the test only constructs ComputeConfigDescriptor
# with the two named fields). The default we install when caller passes None.
def _default_compute_kernel_config() -> ttnn.ComputeConfigDescriptor:
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


def softmax(
    input_tensor: ttnn.Tensor,
    dim: int = -1,
    numeric_stable: bool = True,
    *,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Compute softmax along ``dim`` of a fp32 4D TILE-layout tensor.

    Args:
        input_tensor: Input tensor. Must be float32, TILE_LAYOUT, rank-4, on
            device, with H % 32 == 0 and W % 32 == 0.
        dim: Reduction axis. Phase-0 supports only ``-1`` and ``-2``.
        numeric_stable: If True (default), subtract the per-row/column max
            before exponentiating (standard numerically-stable softmax). If
            False, skip the max subtraction (faster, but overflows on large
            inputs).
        compute_kernel_config: Either ``None`` (entry point installs the
            Phase-0 default of math_fidelity=HiFi4, fp32_dest_acc_en=True),
            or an explicit ``ttnn.ComputeConfigDescriptor`` with those exact
            fields. Other configs are rejected.
        memory_config: Output memory config; default DRAM interleaved.

    Returns:
        Output tensor with the same shape, dtype, and layout as ``input_tensor``.
    """
    if compute_kernel_config is None:
        compute_kernel_config = _default_compute_kernel_config()

    _validate(input_tensor, dim, compute_kernel_config)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(input_tensor.shape)

    # allocate_tensor_on_device requires POSITIONAL args, not keyword args.
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        dim=dim,
        numeric_stable=numeric_stable,
        compute_kernel_config=compute_kernel_config,
    )

    # Output tensor MUST be last in the IO tensor list.
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate(
    input_tensor: ttnn.Tensor,
    dim: int,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> None:
    """Reject every input outside the Phase-0 envelope."""

    # ----- Rank -----
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"softmax: input must be rank-4 (N, C, H, W); got rank-{len(shape)} shape {shape}")

    # ----- dim -----
    if dim not in (-1, -2):
        raise ValueError(f"softmax: dim must be -1 or -2; got dim={dim}")

    # ----- dtype -----
    if input_tensor.dtype != ttnn.float32:
        raise ValueError(f"softmax: input dtype must be float32 (Phase-0); got {input_tensor.dtype}")

    # ----- layout -----
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"softmax: input layout must be TILE_LAYOUT (Phase-0); got {input_tensor.layout}")

    # ----- H/W tile alignment -----
    # Test parametrizes "bad" shapes via the un-padded logical shape (e.g.
    # H=17, W=50). ttnn.from_torch pads to a tile boundary internally, so
    # input_tensor.shape may still show 17 or 50 even though the storage is
    # tile-aligned. We check the *logical* H/W (input_tensor.shape) for the
    # alignment requirement.
    _, _, h, w = shape
    if h % 32 != 0:
        raise ValueError(f"softmax: H must be tile-aligned (H % 32 == 0); got H={h}")
    if w % 32 != 0:
        raise ValueError(f"softmax: W must be tile-aligned (W % 32 == 0); got W={w}")

    # ----- compute_kernel_config -----
    # Phase-0 accepts exactly: math_fidelity=HiFi4 AND fp32_dest_acc_en=True.
    if compute_kernel_config.math_fidelity != ttnn.MathFidelity.HiFi4:
        raise ValueError(
            f"softmax: compute_kernel_config.math_fidelity must be HiFi4 (Phase-0); "
            f"got {compute_kernel_config.math_fidelity}"
        )
    if not compute_kernel_config.fp32_dest_acc_en:
        raise ValueError("softmax: compute_kernel_config.fp32_dest_acc_en must be True (Phase-0)")
