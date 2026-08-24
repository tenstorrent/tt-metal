# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Toy variance op -- per-row population variance over the W dimension, on a Metal 2.0 ProgramSpec.

Two placements, two blocking schemes:

- INTERLEAVED input: one core streams the whole row. The reduction axis is chunked into blocks so W
  can be arbitrarily wide (e.g. 32 x 64000) without exceeding L1.
- WIDTH-SHARDED input: W is split across a row of cores, so the reduced axis itself is split and the
  combine has to cross cores. Supported on a deliberately narrow set of shapes (see
  `toy_variance_sharded_program_artifacts.validate`) -- everything else raises
  NotImplementedError rather than silently falling back.

Constraints common to both (intentionally narrow for a toy):
- Input is on-device, TILE_LAYOUT, bfloat16.
- Reduction is over the last dimension (W) only.
- NC = 1; output shape is the input shape with the last dim collapsed to 32, the variance value
  living in column 0.
"""

import ttnn

from . import toy_variance_sharded_program_artifacts as sharded
from .toy_variance_program_artifacts import create_program_artifacts


def toy_variance(
    input_tensor: ttnn.Tensor,
    *,
    std_dev: bool = False,
    block_size: int | None = None,
    memory_config: ttnn.MemoryConfig = None,
) -> ttnn.Tensor:
    """
    Per-row population variance (or standard deviation) over the W dimension.

    Args:
        input_tensor: TILE_LAYOUT bfloat16 tensor with tile-aligned H and W. Either DRAM/L1
            interleaved (single-core streaming path) or WIDTH_SHARDED in L1 (cross-core path).
        std_dev: If True, return std deviation = sqrt(variance). The sqrt is applied as the
            last-block post-op of the streaming reduce -- no extra pass over the data. Default
            False (returns variance).
        block_size: Optional override for the streaming block size (in tiles). Must divide the
            per-core Wt. Defaults to 8 (or the largest divisor <= 8). Interleaved path only.
        memory_config: Output memory config (default: DRAM interleaved).
    """
    device = input_tensor.device()
    output_memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # NOTE on implicit tile padding: padded W positions flow through sub<COL> + square_in_place
    # *before* the partial scaler zeros them in the reduce. The partial-scaler tile multiplies the
    # last W-tile of the last block by zero at padded positions, so any FINITE garbage there ends up
    # as (garbage - mean)^2 * 0 = 0 in the accumulator. Caller is responsible for ensuring padded
    # values are finite -- if you have inf/nan garbage, call
    # ttnn.fill_implicit_tile_padding(input, 0.0) first to avoid inf * 0 = nan propagating.

    input_shape = list(input_tensor.shape)
    output_shape = input_shape[:-1] + [32]

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config,
    )

    # Each factory owns every name it declares, so it hands back the tensor bindings alongside
    # the spec and run args. Nothing here needs to know what those names are.
    io_tensors = [input_tensor, output_tensor]

    if input_tensor.memory_config().is_sharded():
        sharded.validate(input_tensor)
        spec, run_args, tensor_indices = sharded.create_program_artifacts(input_tensor, output_tensor, std_dev=std_dev)
        return ttnn.generic_op(io_tensors, spec, run_args, tensor_indices)

    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, output_tensor, block_size=block_size, std_dev=std_dev
    )
    return ttnn.generic_op(io_tensors, spec, run_args, tensor_indices)
