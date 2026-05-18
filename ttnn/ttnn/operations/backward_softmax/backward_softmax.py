# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
backward_softmax — VJP of softmax.

    grad_input = output * (grad_output - sum(output * grad_output, dim))

Current constraints (post Refinement 4):
- Inputs: ``float32`` / ``bfloat16`` / ``bfloat8_b``, TILE_LAYOUT, rank-4,
  H/W tile-aligned (multiple of 32), identical shape and dtype between the
  two inputs.
- Reduce dimension: dim ∈ {-1, -2}.
- Output dtype matches input dtype.
- Compute config: dtype-aware default (HiFi4 + fp32_dest_acc for fp32;
  lower-fidelity defaults for bf16/bfp8). Refinement 4 adds caller overrides
  via ``compute_kernel_config`` (a :class:`ttnn.WormholeComputeKernelConfig`)
  and ``unpack_to_dest_mode`` (per-CB vector). When both are ``None`` the
  behaviour is bit-identical to R3.
"""

import ttnn
from .backward_softmax_program_descriptor import create_program_descriptor

_SUPPORTED_DTYPES = (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b)


def backward_softmax(
    grad_output: ttnn.Tensor,
    output: ttnn.Tensor,
    *,
    dim: int = -1,
    memory_config: ttnn.MemoryConfig = None,
    compute_kernel_config: ttnn.WormholeComputeKernelConfig = None,
    unpack_to_dest_mode: list = None,
) -> ttnn.Tensor:
    """
    Vector-Jacobian product (VJP) of softmax.

    Computes::

        grad_input = output * (grad_output - sum(output * grad_output, dim))

    Args:
        grad_output: Upstream gradient (dy). float32 / bfloat16 / bfloat8_b,
            TILE_LAYOUT, rank-4, H/W tile-aligned, on-device.
        output: Forward softmax output (y). Identical shape & dtype as
            grad_output.
        dim: Reduction dimension. Must be -1 (W) or -2 (H). Defaults to -1.
        memory_config: Output memory config (default: DRAM interleaved).
        compute_kernel_config: Optional override of the compute config
            (math_fidelity, math_approx_mode, fp32_dest_acc_en,
            dst_full_sync_en). When ``None``, a dtype-aware default is used
            (fp32 → HiFi4 + fp32_dest_acc; bf16 → HiFi2; bfp8_b → LoFi —
            see ``backward_softmax_program_descriptor.py``). ``packer_l1_acc``
            and ``throttle_level`` on the passed config are accepted for
            interface symmetry but **ignored** — the kernels do not consume
            them (no packer-side L1 accumulation; throttling is a host-side
            policy not applicable to this op).
        unpack_to_dest_mode: Optional per-CB vector of
            :class:`ttnn.UnpackToDestMode` (length 32, one entry per CB
            index). When ``None`` the default ``UnpackToDestMode.Default``
            applies to every CB. **CAUTION** — applying
            ``UnpackToDestMode.UnpackToDestFp32`` to ``CB_PROD`` (index 24)
            or ``CB_OUTPUT`` (index 1, with fp32 input) is known to produce
            ``inf`` outputs because the matmul-based REDUCE_ROW SUM path is
            incompatible with that unpack mode. Apply only to non-matmul
            input CBs (typically ``CB_GRAD_OUTPUT`` index 0 alone).

    Returns:
        grad_input: same shape and dtype as grad_output.
    """
    _validate(grad_output, output, dim)

    device = grad_output.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    output_shape = list(grad_output.shape)

    # CRITICAL: positional args, not keyword args.
    grad_input = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        grad_output.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(
        grad_output,
        output,
        grad_input,
        dim=dim,
        compute_kernel_config=compute_kernel_config,
        unpack_to_dest_mode=unpack_to_dest_mode,
    )

    # Output tensor MUST be last in the io list.
    return ttnn.generic_op([grad_output, output, grad_input], program_descriptor)


def _validate(grad_output: ttnn.Tensor, output: ttnn.Tensor, dim: int) -> None:
    # Dtype.
    if grad_output.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"backward_softmax: grad_output dtype must be one of {_SUPPORTED_DTYPES}, " f"got {grad_output.dtype}"
        )
    if output.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"backward_softmax: output dtype must be one of {_SUPPORTED_DTYPES}, " f"got {output.dtype}")
    if grad_output.dtype != output.dtype:
        raise ValueError(
            f"backward_softmax: grad_output dtype ({grad_output.dtype}) and "
            f"output dtype ({output.dtype}) must match"
        )

    # Layout.
    if grad_output.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"backward_softmax: grad_output must be TILE_LAYOUT, got {grad_output.layout}")
    if output.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"backward_softmax: output must be TILE_LAYOUT, got {output.layout}")

    # Rank.
    if len(grad_output.shape) != 4:
        raise ValueError(f"backward_softmax: grad_output rank must be 4, got {len(grad_output.shape)}")
    if len(output.shape) != 4:
        raise ValueError(f"backward_softmax: output rank must be 4, got {len(output.shape)}")

    # Shape match.
    if tuple(grad_output.shape) != tuple(output.shape):
        raise ValueError(
            f"backward_softmax: grad_output shape {tuple(grad_output.shape)} must "
            f"match output shape {tuple(output.shape)}"
        )

    # Tile alignment of H, W.
    H = grad_output.shape[-2]
    W = grad_output.shape[-1]
    if H % 32 != 0 or W % 32 != 0:
        raise ValueError(f"backward_softmax: H ({H}) and W ({W}) must each be a multiple of 32")

    # dim.
    if dim not in (-1, -2):
        raise ValueError(f"backward_softmax: dim must be -1 or -2, got {dim}")
