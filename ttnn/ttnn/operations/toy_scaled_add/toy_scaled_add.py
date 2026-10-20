# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""toy_scaled_add: out = a + alpha * (b * gamma), gamma an optional row broadcast down the rows.

The operation is the native C++ ttnn.toy_scaled_add (ttnn/cpp/ttnn/operations/toy_scaled_add), which checks this
support contract and refuses inputs outside it with the same exceptions. This file keeps the contract tables
(INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / PROPERTIES), validate() for checking the contract without dispatching,
and the public entry point. The generic_op version stays in toy_scaled_add_generic.py, on the same kernels.

    a, b    tiled with 32 x 32 tiles, same padded shape, bfloat16 or float32; both interleaved, or both
            height-sharded on L1 with one shard spec (shard width = the full row, whole tiles high)
    gamma   optional, tiled, interleaved, one row: padded shape [..., 32, W]
    alpha   per-call scalar
    dtype / memory_config   output dtype / placement (default: a's); a sharded output takes a's shard spec
    compute_kernel_config   math fidelity, fp32 DEST accumulation, ...
    output_tensor           preallocated output; may be `a` itself (in place)

Inputs outside SUPPORTED raise UnsupportedAxisValue and cells in EXCLUSIONS raise ExcludedCell, both from
ttnn.operations._op_contract; inputs that do not fit together (mismatched shapes or shard specs) raise
ValueError.
"""

from __future__ import annotations

import math
import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .toy_scaled_add_program_descriptor import TILE


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_rank(inputs, axes):
    return len(inputs[0])


INPUT_TAGGERS = {
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "b_dtype": [ttnn.bfloat16, ttnn.float32],
    "gamma_dtype": [ttnn.bfloat16, ttnn.float32, "none"],
    "output_dtype": [ttnn.bfloat16, ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
    "tile": ["32x32"],
    "rank": [2, 3, 4],
    "memory_layout": [ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED],
    "buffer_type": [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = [
    # The sharded program backs its circular buffers with the shards, and circular buffers live in L1.
    {"memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED, "buffer_type": ttnn.BufferType.DRAM},
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _tile_name(tensor: ttnn.Tensor) -> str:
    h, w = tensor.tile.tile_shape
    return f"{h}x{w}"


def _operand_axis(operands, read, axis):
    """One axis over every tensor operand: the first value outside SUPPORTED, else the shared value."""
    values = [read(t) for t in operands]
    return next((v for v in values if v not in SUPPORTED[axis]), values[0])


def _output_memory_config(a, memory_config):
    if memory_config is None:
        return a.memory_config()
    if memory_config.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED and memory_config.shard_spec is None:
        if not a.memory_config().is_sharded():
            raise ValueError("toy_scaled_add: a height-sharded memory_config needs a shard spec when a is interleaved")
        return ttnn.MemoryConfig(memory_config.memory_layout, memory_config.buffer_type, a.memory_config().shard_spec)
    return memory_config


def _structure_checks(a, b, gamma, output_memory_config):
    """Inputs that cannot work together, whatever the support contract says."""
    if a.padded_shape != b.padded_shape:
        raise ValueError(f"toy_scaled_add: a {a.padded_shape} and b {b.padded_shape} must have the same padded shape")
    if gamma is not None:
        if gamma.padded_shape[-1] != a.padded_shape[-1] or math.prod(gamma.padded_shape) != TILE * a.padded_shape[-1]:
            raise ValueError(
                f"toy_scaled_add: gamma {gamma.padded_shape} must be one tile-row as wide as a {a.padded_shape}"
            )
        if gamma.memory_config().is_sharded():
            raise ValueError("toy_scaled_add: gamma must be interleaved")
    configs = (a.memory_config(), b.memory_config(), output_memory_config)
    if len({mc.memory_layout for mc in configs}) != 1:
        raise ValueError("toy_scaled_add: a, b and the output must be all interleaved or all height-sharded")
    if a.memory_config().is_sharded():
        shard_spec = a.memory_config().shard_spec
        if b.memory_config().shard_spec != shard_spec or output_memory_config.shard_spec != shard_spec:
            raise ValueError("toy_scaled_add: a, b and the output must share one shard spec")
        if shard_spec.shape[1] != a.padded_shape[-1] or shard_spec.shape[0] % TILE != 0:
            raise ValueError(
                f"toy_scaled_add: shard shape {shard_spec.shape} must span the full row and a whole number of tiles"
            )


def validate(a, b, *, gamma=None, dtype=None, memory_config=None, output_tensor=None):
    """Checks a call against the support contract without dispatching it. Returns the output dtype and
    memory config the call resolves to."""
    if output_tensor is not None:
        dtype = dtype or output_tensor.dtype
        memory_config = memory_config or output_tensor.memory_config()
    dtype = dtype or a.dtype
    output_memory_config = _output_memory_config(a, memory_config)

    operands = [t for t in (a, b, gamma) if t is not None]
    axes = {
        "dtype": a.dtype,
        "b_dtype": b.dtype,
        "gamma_dtype": gamma.dtype if gamma is not None else "none",
        "output_dtype": dtype,
        "layout": _operand_axis(operands, lambda t: t.layout, "layout"),
        "tile": _operand_axis(operands, _tile_name, "tile"),
        "memory_layout": output_memory_config.memory_layout,
        "buffer_type": output_memory_config.buffer_type,
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((list(a.shape),), axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"toy_scaled_add: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"toy_scaled_add: unsupported combination: {exc}")

    _structure_checks(a, b, gamma, output_memory_config)
    if output_tensor is not None and (
        output_tensor.shape != a.shape
        or output_tensor.dtype != dtype
        or output_tensor.layout != ttnn.TILE_LAYOUT
        or output_tensor.memory_config() != output_memory_config
    ):
        raise ValueError(f"toy_scaled_add: output_tensor {output_tensor.spec} does not match a {a.shape}, {dtype}")
    return dtype, output_memory_config


# ---------------------------------------------------------------------------
# Public entry point: the registered C++ operation
# ---------------------------------------------------------------------------

toy_scaled_add = ttnn.toy_scaled_add
