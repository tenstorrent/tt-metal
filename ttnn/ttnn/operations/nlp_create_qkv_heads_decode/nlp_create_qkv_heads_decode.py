# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""nlp_create_qkv_heads_decode: cut one decode step's fused QKV row per user into the heads of q, k and v.

Specification: eval/specs/nlp_create_qkv_heads_decode/spec.py. Design: op_design.md beside this file.

Registry model (eval/op_template.py): INPUT_TAGGERS (the specification's AXES), SUPPORTED, EXCLUSIONS and
validate(); the entry point validates first, allocates the three height-sharded results (rule R5) and runs
ONE ttnn.generic_op dispatch of one ProgramSpec (nlp_create_qkv_heads_decode_program_artifacts.py).
"""

from __future__ import annotations

import ttnn

from eval import memory
from eval.specs.nlp_create_qkv_heads_decode.spec import AXES, canonical_memory
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .nlp_create_qkv_heads_decode_program_artifacts import FACE, TILE, create_program_artifacts

OP_NAME = "nlp_create_qkv_heads_decode"

# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS: the specification's axis functions, never redefined here.
# ---------------------------------------------------------------------------

INPUT_TAGGERS = AXES

# ---------------------------------------------------------------------------
# 2. SUPPORTED: the first phase equals the specification's PHASE0.
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype":         [ttnn.bfloat16],
    "layout":        [ttnn.TILE_LAYOUT],
    "memory_config": ["L1"],
    "sharding":      ["interleaved"],
    "output_memory": ["L1/height"],
    "users":         [1],
    "head_rows":     [32],
}

# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _head_counts(num_heads, num_kv_heads):
    """(num_heads, num_kv_heads) with the default filled in (R4); X1, X2, X3 refused."""
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    for name, value in (("num_heads", num_heads), ("num_kv_heads", kv_heads)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{OP_NAME}: {name} must be an int, got {value!r}")
        if value < 1:
            raise ValueError(f"{OP_NAME}: {name} must be at least 1, got {value}")
    if kv_heads > num_heads:
        raise ValueError(f"{OP_NAME}: num_kv_heads ({kv_heads}) must not exceed num_heads ({num_heads})")
    return num_heads, kv_heads


def validate(input_tensor, *, num_heads, num_kv_heads=None, overlap_qk_coregrid=True, memory_config=None):
    """Refuse a malformed call (REJECTIONS X1-X8) and a call outside SUPPORTED / inside EXCLUSIONS.

    `overlap_qk_coregrid` is accepted and never read (rule R7). Returns the derived geometry.
    """
    shape = tuple(int(d) for d in input_tensor.shape)
    # X6: [1, 1, users, width]
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
        raise ValueError(f"{OP_NAME}: the input must be [1, 1, users, width], got {list(shape)}")
    # X8 / R10: tile layout only, refused by name
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{OP_NAME}: the input must be in TILE_LAYOUT, got {input_tensor.layout}")
    # X1-X3, then X4, X5
    nh, nkv = _head_counts(num_heads, num_kv_heads)
    sections = nh + 2 * nkv
    width = shape[3]
    if width % sections != 0:
        raise ValueError(f"{OP_NAME}: width {width} is not a multiple of num_heads + 2 * num_kv_heads = {sections}")
    head_dim = width // sections
    if head_dim % TILE != 0:
        raise ValueError(f"{OP_NAME}: head_dim {head_dim} is not a multiple of {TILE}")
    # X7: canonical_memory raises TypeError for anything that is not a memory config
    output_memory = canonical_memory(memory_config, None)

    # The axes, exactly as the harness derives them (R9: placement through eval.memory.memory_axes).
    placed = memory.memory_axes(input_tensor)
    axes = {
        "dtype":         input_tensor.dtype,
        "layout":        input_tensor.layout,
        "memory_config": placed["memory_config"],
        "sharding":      placed["sharding"],
        "output_memory": output_memory,
    }
    inputs = (shape, num_heads, num_kv_heads)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"{OP_NAME}: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"{OP_NAME}: unsupported combination (refinement candidate): {exc}")

    # Mechanism cap: the face-row offsets assume the default 32x32 tile of 16x16 faces.
    tile = input_tensor.tile
    if list(tile.tile_shape) != [TILE, TILE] or list(tile.face_shape) != [FACE, FACE]:
        raise NotImplementedError(f"{OP_NAME}: only the default 32x32 tile of 16x16 faces is supported, got {tile}")

    return {"users": shape[2], "num_heads": nh, "num_kv_heads": nkv, "head_dim": head_dim}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _round_up_tiles(n):
    return -(-n // TILE) * TILE


def nlp_create_qkv_heads_decode(
    input_tensor: ttnn.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int = None,
    overlap_qk_coregrid: bool = True,
    memory_config: "ttnn.MemoryConfig" = None,
) -> list:
    geometry = validate(
        input_tensor,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        overlap_qk_coregrid=overlap_qk_coregrid,
        memory_config=memory_config,
    )
    users, nh, nkv, head_dim = (geometry[k] for k in ("users", "num_heads", "num_kv_heads", "head_dim"))
    device = input_tensor.device()

    # R5: each result height-sharded in L1, one shard [heads rounded up to 32, head_dim] per user,
    # on the first `users` compute cores, row by row; q, k and v on the same cores.
    cores = ttnn.num_cores_to_corerangeset(users, device.compute_with_storage_grid_size(), row_wise=True)

    def allocate(heads):
        config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [_round_up_tiles(heads), head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        spec = ttnn.TensorSpec([1, users, heads, head_dim], input_tensor.dtype, ttnn.TILE_LAYOUT, config)
        return ttnn.allocate_tensor_on_device(spec, device)

    q, k, v = allocate(nh), allocate(nkv), allocate(nkv)
    spec, run_args, tensor_indices = create_program_artifacts(
        input_tensor, q, k, v, cores=cores, num_heads=nh, num_kv_heads=nkv, head_dim=head_dim
    )
    ttnn.generic_op([input_tensor, q, k, v], spec, run_args, tensor_indices)
    return [q, k, v]
