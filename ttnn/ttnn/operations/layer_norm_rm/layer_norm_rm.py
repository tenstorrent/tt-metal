# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — Per-row (final-dim) LayerNorm on a ROW_MAJOR fp32 tensor.

Phase-0 implementation of the spec in op_design.md.

Pipeline (per tile-row work item, see op_design.md for the full reduce/sub/
reduce chain):

    reader   : DRAM RM sticks → cb_input_rm (tile-paged)
             + one-shot replicate-32× of optional gamma/beta sticks
             + one-shot scaler tile (1/W) for both reductions
    compute  : tilize → reduce(SUM, REDUCE_ROW) [mean]
             → sub<COL> [centered]
             → square [(x - mean)^2]
             → reduce(SUM, REDUCE_ROW) + (+eps, rsqrt) post-op [inv_std]
             → mul<COL> [normalized]
             → optional mul_in_place<ROW> by gamma
             → optional add_in_place<ROW> by beta
             → untilize → cb_output_tiles
    writer   : cb_output_tiles → DRAM RM sticks

Output shape, dtype (fp32), and layout (ROW_MAJOR) match input. Tilize and
untilize happen entirely in-kernel — the entry point does NOT cast to TILE.

Phase-0 envelope:
- input dtype = float32; input layout = ROW_MAJOR_LAYOUT
- rank ≥ 2; H % 32 == 0; W % 32 == 0; H ≥ 32; W ≥ 32; W ≤ 1024
- optional gamma / beta — same dtype/layout, total element count == W
- epsilon strictly > 0
"""

import ttnn

from .layer_norm_rm_program_descriptor import create_program_descriptor


_DEFAULT_EPSILON = 1e-5
_MAX_W_PHASE0 = 1024


def layer_norm_rm(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    *,
    epsilon: float = _DEFAULT_EPSILON,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Per-row LayerNorm over the final dimension.

    Args:
        input_tensor: fp32 ROW_MAJOR tensor (rank ≥ 2, tile-aligned H/W).
        gamma: optional fp32 ROW_MAJOR scale, total element count == input W.
        beta:  optional fp32 ROW_MAJOR shift, total element count == input W.
        epsilon: numerical stability term added before rsqrt; default 1e-5.
        memory_config: output memory config; default DRAM_MEMORY_CONFIG.

    Returns:
        Output tensor with the same shape, dtype, and layout as input.
    """
    _validate(input_tensor, gamma, beta, epsilon)

    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # Output has identical shape, dtype, and layout to the input.
    output_shape = list(input_tensor.shape)
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
        gamma=gamma,
        beta=beta,
        epsilon=epsilon,
    )

    # Output tensor MUST be last in the IO tensor list.
    io_tensors = [input_tensor]
    if gamma is not None:
        io_tensors.append(gamma)
    if beta is not None:
        io_tensors.append(beta)
    io_tensors.append(output_tensor)

    return ttnn.generic_op(io_tensors, program_descriptor)


def _validate(
    input_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor,
    beta: ttnn.Tensor,
    epsilon: float,
) -> None:
    # ---- Input tensor ----
    if input_tensor.dtype != ttnn.float32:
        raise NotImplementedError(f"layer_norm_rm: input dtype {input_tensor.dtype} not in SUPPORTED [float32]")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise NotImplementedError(
            f"layer_norm_rm: input layout {input_tensor.layout} not in SUPPORTED [ROW_MAJOR_LAYOUT]"
        )

    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise NotImplementedError(f"layer_norm_rm: input rank {len(shape)} not in SUPPORTED [rank >= 2]")

    H = shape[-2]
    W = shape[-1]
    if H < 32 or H % 32 != 0:
        raise NotImplementedError(f"layer_norm_rm: H={H} not in SUPPORTED [H >= 32 and H % 32 == 0]")
    if W < 32 or W % 32 != 0:
        raise NotImplementedError(f"layer_norm_rm: W={W} not in SUPPORTED [W >= 32 and W % 32 == 0]")
    if W > _MAX_W_PHASE0:
        raise NotImplementedError(f"layer_norm_rm: W={W} not in SUPPORTED [W <= {_MAX_W_PHASE0} for Phase 0]")

    # ---- Optional affine tensors ----
    for name, t in (("gamma", gamma), ("beta", beta)):
        if t is None:
            continue
        if t.dtype != ttnn.float32:
            raise NotImplementedError(f"layer_norm_rm: {name} dtype {t.dtype} not in SUPPORTED [float32]")
        if t.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise NotImplementedError(f"layer_norm_rm: {name} layout {t.layout} not in SUPPORTED [ROW_MAJOR_LAYOUT]")
        affine_numel = 1
        for d in list(t.shape):
            affine_numel *= d
        if affine_numel != W:
            raise NotImplementedError(f"layer_norm_rm: {name} numel {affine_numel} != input W {W}")

    # ---- Epsilon ----
    if not (epsilon > 0):
        raise NotImplementedError(f"layer_norm_rm: epsilon={epsilon} not in SUPPORTED [epsilon > 0]")
