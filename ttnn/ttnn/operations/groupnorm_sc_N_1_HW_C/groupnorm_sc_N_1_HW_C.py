# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""groupnorm_sc_N_1_HW_C — GroupNorm over an ``(N, 1, HW, C)`` channel-last tensor.

Per-(image, group) statistics over ``HW x C/G``, centered two-pass variance and optional
per-channel affine. Groups may straddle 32-channel tile boundaries: the kernel never reduces
over channel lanes with a tile reduce — it forms per-channel column sums and aggregates them by
group with a 0/1 membership matmul built on-device from runtime args.

Supported: bf16 / fp32 / bf8b activations (bf8b TILE only) in TILE or ROW_MAJOR layout, DRAM or
L1 interleaved or L1 BLOCK_SHARDED (``in_place`` needs the shard); any HW and C (non-tile-aligned
extents are masked); gamma / beta as (1, 1, 1, C) bf16 / fp32 / bf8b tensors in either layout.
"""

from __future__ import annotations

import ttnn

from .groupnorm_sc_N_1_HW_C_program_descriptor import create_program_descriptor, default_compute_kernel_config

_OP = "groupnorm_sc_N_1_HW_C"

_DTYPES = (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b)
_LAYOUTS = (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)
_MEMORY_LAYOUTS = (ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.BLOCK_SHARDED)


def _validate_arguments(input_tensor, num_groups, gamma, beta, eps, in_place, compute_kernel_config):
    """Argument validation; every unsupported input raises ValueError."""
    if input_tensor.dtype not in _DTYPES:
        raise ValueError(f"{_OP}: unsupported dtype {input_tensor.dtype} (expected one of {_DTYPES})")
    if input_tensor.layout not in _LAYOUTS:
        raise ValueError(f"{_OP}: unsupported layout {input_tensor.layout}")
    if input_tensor.dtype == ttnn.bfloat8_b and input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{_OP}: bfloat8_b input requires TILE layout")
    memory_layout = input_tensor.memory_config().memory_layout
    if memory_layout not in _MEMORY_LAYOUTS:
        raise ValueError(f"{_OP}: unsupported memory layout {memory_layout} (INTERLEAVED or BLOCK_SHARDED)")
    if in_place and memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise ValueError(f"{_OP}: in_place requires a BLOCK_SHARDED input (the result overwrites the L1 shard)")
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"{_OP}: input must be 4D (N, 1, H*W, C), got rank {len(shape)}")
    if shape[1] != 1:
        raise ValueError(f"{_OP}: input dim[1] must be 1 for the (N, 1, H*W, C) layout, got {shape[1]}")
    C = shape[-1]
    if not isinstance(num_groups, int) or num_groups < 1:
        raise ValueError(f"{_OP}: num_groups must be a positive int, got {num_groups!r}")
    if C % num_groups != 0:
        raise ValueError(f"{_OP}: C={C} is not divisible by num_groups={num_groups}")
    for name, w in (("gamma", gamma), ("beta", beta)):
        if w is None:
            continue
        if w.dtype not in _DTYPES or w.layout not in _LAYOUTS:
            raise ValueError(f"{_OP}: {name} has unsupported dtype / layout {w.dtype} / {w.layout}")
        wshape = list(w.shape)
        if wshape != [1, 1, 1, C]:
            raise ValueError(f"{_OP}: {name} must have shape (1, 1, 1, {C}), got {tuple(wshape)}")
    if gamma is not None and beta is not None:
        if gamma.dtype != beta.dtype or gamma.layout != beta.layout:
            raise ValueError(f"{_OP}: gamma and beta must share dtype and layout")
    if eps <= 0:
        raise ValueError(f"{_OP}: eps must be > 0, got {eps}")
    # compute_kernel_config: the statistics path (column sums, membership / combine / expansion
    # matmuls, centered squares) accumulates in fp32 DEST into fp32 CBs — with a 16-bit DEST the
    # group mean / variance would round at 2^-8 (op_design.md "Precision") and the matmul_block
    # fidelity rule (#38306) would no longer hold, so fp32_dest_acc_en=False is refused rather than
    # silently degraded. packer_l1_acc does not compose with fp32 DEST (#28800).
    # math_fidelity / math_approx_mode / dst_full_sync_en pass through.
    for name in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en"):
        if not hasattr(compute_kernel_config, name):
            raise ValueError(
                f"{_OP}: compute_kernel_config must be a ttnn.WormholeComputeKernelConfig (missing {name})"
            )
    if not compute_kernel_config.fp32_dest_acc_en:
        raise ValueError(f"{_OP}: compute_kernel_config.fp32_dest_acc_en must be True (fp32 statistics path)")
    if compute_kernel_config.packer_l1_acc:
        raise ValueError(f"{_OP}: compute_kernel_config.packer_l1_acc is not supported (fp32 DEST accumulation)")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def groupnorm_sc_N_1_HW_C(
    input_tensor: ttnn.Tensor,
    num_groups: int,
    *,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    eps: float = 1e-5,
    in_place: bool = False,
    compute_kernel_config=None,
    activation: str = None,
) -> ttnn.Tensor:
    """GroupNorm over an (N, 1, H*W, C) tensor; one native device dispatch.

    `compute_kernel_config` (ttnn.WormholeComputeKernelConfig / BlackholeComputeKernelConfig, optional)
    drives math_fidelity / math_approx_mode / dst_full_sync_en; the default is HiFi4, fp32 DEST,
    exact SFPU. fp32_dest_acc_en must stay True (fp32 statistics).
    `activation="silu"` fuses SiLU into the apply pass (output = silu(groupnorm(x))).
    """
    if compute_kernel_config is None:
        compute_kernel_config = default_compute_kernel_config()
    _validate_arguments(input_tensor, num_groups, gamma, beta, eps, in_place, compute_kernel_config)
    if activation not in (None, "silu"):
        raise ValueError(f"{_OP}: activation must be None or 'silu', got {activation!r}")

    device = input_tensor.device()
    if in_place:
        # The result lands in the input's own L1 shard (pass 3 reads each input tile for the last
        # time exactly once, then packs the output over it); the returned tensor IS the input.
        output_tensor = input_tensor
    else:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(input_tensor.shape)),
            input_tensor.dtype,
            input_tensor.layout,
            device,
            input_tensor.memory_config(),
        )

    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)  # output MUST be last

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        num_groups,
        gamma=gamma,
        beta=beta,
        eps=eps,
        compute_kernel_config=compute_kernel_config,
        activation=activation,
    )
    return ttnn.generic_op(io_tensors, program_descriptor)
