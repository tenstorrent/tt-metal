# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""nlp_concat_heads_decode: the heads of one decode step, side by side again.

    output[0, 0, u, h * head_dim + d] = input[0, u, h, d]   for u < users, h < num_heads, d < head_dim

The input is height-sharded in L1, one shard `[32, head_dim]` per user; the output is width-sharded in L1, one
shard `[32, head_dim]` per head on the first `num_heads` compute cores (rule R4). One `ttnn.generic_op` dispatch
of one `ttnn.ProgramSpec` (see nlp_concat_heads_decode_program_artifacts.py): every output core pulls row
`head_idx` of every user's input tiles over the NoC straight into its own output shard. No compute.

Specification: eval/specs/nlp_concat_heads_decode/spec.py. Design: op_design.md beside this file.
"""

from __future__ import annotations

import ttnn

from eval import memory
from eval.specs.nlp_concat_heads_decode.spec import AXES
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .nlp_concat_heads_decode_program_artifacts import create_program_artifacts

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's AXES, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the specification's PHASE0, axis for axis.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "memory_config": ["L1"],
    "sharding": ["height"],
    "users": [1],
    "alignment": ["tile_aligned"],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS: none.
# ---------------------------------------------------------------------------

EXCLUSIONS = []

PADDED_USERS = 32  # the output's rows: the batch dimension padded to a tile (rule R3)
TILE_SHAPE = [32, 32]
FACE_SHAPE = [16, 16]


def _axes(input_tensor, num_heads):
    shape = tuple(int(d) for d in input_tensor.shape)
    axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout, **memory.memory_axes(input_tensor)}
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger((shape, num_heads), axes)
    return axes


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(input_tensor, *, num_heads, sub_core_grids=None):
    """Refuse a malformed call (ValueError), then anything outside SUPPORTED / EXCLUSIONS (NotImplementedError)."""
    # Malformed calls (REJECTIONS X1-X5, X7; rule R7).
    shape = tuple(int(d) for d in input_tensor.shape)
    if len(shape) != 4:
        raise ValueError(f"nlp_concat_heads_decode: the input must be of rank 4 [1, users, H, head_dim], got {shape}")
    if shape[0] != 1:
        raise ValueError(f"nlp_concat_heads_decode: the input's first dimension must be 1 (one token), got {shape}")
    if shape[2] > PADDED_USERS:
        raise ValueError(f"nlp_concat_heads_decode: at most {PADDED_USERS} head rows (one tile row), got {shape[2]}")
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or not 1 <= num_heads <= shape[2]:
        raise ValueError(f"nlp_concat_heads_decode: num_heads must be in [1, {shape[2]}], got {num_heads!r}")
    if sub_core_grids is not None:
        raise ValueError("nlp_concat_heads_decode: sub_core_grids must be None (rule R7)")

    # Support gate (rules R6, R9).
    axes = _axes(input_tensor, num_heads)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nlp_concat_heads_decode: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nlp_concat_heads_decode: unsupported combination (refinement candidate): {exc}")

    # One shard per user, [32, head_dim] (R6), and the 32x32 tile of 16x16 faces the kernel's offsets assume.
    shard_spec = input_tensor.memory_config().shard_spec
    if shard_spec is None or list(shard_spec.shape) != [PADDED_USERS, shape[3]]:
        got = None if shard_spec is None else list(shard_spec.shape)
        raise NotImplementedError(
            f"nlp_concat_heads_decode: the input shard must be [{PADDED_USERS}, {shape[3]}] (one per user), got {got}"
        )
    tile = input_tensor.tile
    if list(tile.tile_shape) != TILE_SHAPE or list(tile.face_shape) != FACE_SHAPE:
        raise NotImplementedError(
            f"nlp_concat_heads_decode: tile {list(tile.tile_shape)} / face {list(tile.face_shape)} not supported"
        )


def output_memory_config(device, num_heads, head_dim):
    """Rule R4: width-sharded in L1, shard [32, head_dim], row major, on the first num_heads cores row by row.

    Rule R5: raises RuntimeError("Target number of cores ...") on a device with fewer compute cores, before
    anything is allocated."""
    grid = device.compute_with_storage_grid_size()
    available = int(grid.x) * int(grid.y)
    if num_heads > available:
        raise RuntimeError(
            f"Target number of cores {num_heads} is greater than total number of available cores {available}"
        )
    cores = ttnn.num_cores_to_corerangeset(num_heads, grid, row_wise=True)
    shard = ttnn.ShardSpec(cores, [PADDED_USERS, head_dim], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def nlp_concat_heads_decode(
    input_tensor: ttnn.Tensor,
    *,
    num_heads: int,
    sub_core_grids: "ttnn.CoreRangeSet" = None,
) -> ttnn.Tensor:
    validate(input_tensor, num_heads=num_heads, sub_core_grids=sub_core_grids)
    device = input_tensor.device()
    head_dim = int(input_tensor.shape[3])
    memory_config = output_memory_config(device, num_heads, head_dim)
    out_spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, PADDED_USERS, num_heads * head_dim]),
        input_tensor.dtype,
        ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
    output = ttnn.allocate_tensor_on_device(out_spec, device)
    spec, run_args, tensor_indices = create_program_artifacts(input_tensor, output, num_heads=num_heads)
    return ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)
