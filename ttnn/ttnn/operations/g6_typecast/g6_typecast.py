# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_typecast — convert every value of one tensor to `dtype` in one device program.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate(), entry point.
Design: op_design.md (regime R1 `tile_stream`: interleaved TILE in -> compute typecast -> interleaved
TILE out, contiguous tile ranges over the full compute grid; regime R2 `rm_pseudo_tile_stream`: interleaved
ROW_MAJOR rows gathered by bank runs, converted as dense 1024-value pseudo-tiles, scattered back).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_typecast_program_descriptor as _pd
from .g6_typecast_program_descriptor import create_program_descriptor, patch_buffer_addresses


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each projects the scenario dict `(scenario,)` onto one axis
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_output_dtype(inputs, axes):
    return inputs[0]["output_dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_buffer(inputs, axes):
    return inputs[0]["buffer"]


def tag_alignment(inputs, axes):
    shape = tuple(inputs[0]["input_shape"])
    w = shape[-1]
    h = shape[-2] if len(shape) >= 2 else 1
    if w % 32:
        return "w_non_aligned"
    if h % 32:
        return "h_non_aligned"
    return "tile_aligned"


def tag_rank(inputs, axes):
    return len(tuple(inputs[0]["input_shape"]))


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "output_dtype": tag_output_dtype,
    "layout": tag_layout,
    "buffer": tag_buffer,
    "alignment": tag_alignment,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED (Phase 0 + Refinement 1: ROW_MAJOR regime R2 + Refinement 2: fp32/int32/uint16 sources)
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32, ttnn.uint16],
    "output_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.int32, ttnn.uint8, ttnn.uint32],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "buffer": ["dram", "l1"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"],
    "rank": [1, 2, 3, 4],
}

# Conversion arms implemented (source -> destination). The per-axis SUPPORTED product admits pairs with no
# arm (e.g. uint32 -> uint8); validate() refuses those with ValueError rather than run a wrong arm.
# Both regimes (TILE / ROW_MAJOR) run every arm: the compute chain is layout-agnostic.
CONVERSION_ARMS = {
    (ttnn.bfloat16, ttnn.float32),  # packer only
    (ttnn.bfloat16, ttnn.bfloat8_b),  # packer only (bfp8_pack_precise)
    (ttnn.bfloat16, ttnn.int32),  # SFPU truncation toward zero
    (ttnn.bfloat16, ttnn.uint8),  # SFPU truncation, low byte
    (ttnn.uint32, ttnn.int32),  # all 32 bits kept (unpack-to-DEST, no SFPU)
    (ttnn.int32, ttnn.uint32),  # all 32 bits kept (unpack-to-DEST, no SFPU)
    (ttnn.float32, ttnn.bfloat16),  # SFPU RNE (ties-to-even), fp32 unpacked to DEST
    (ttnn.int32, ttnn.bfloat16),  # SFPU int32 -> fp32 -> bf16 (two roundings, as torch)
    (ttnn.uint16, ttnn.int32),  # SFPU zero-extend (unpack-to-DEST)
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — stays empty in every phase (every TARGET cell is a production case)
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer_tag(memory_config):
    return "l1" if memory_config.buffer_type == ttnn.BufferType.L1 else "dram"


def validate(input_tensor, dtype, *, sub_core_grids=None):
    if not isinstance(dtype, ttnn.DataType):
        raise ValueError(f"g6_typecast: dtype must be a ttnn.DataType, got {dtype!r}")
    if sub_core_grids is not None:
        raise ValueError("g6_typecast: sub_core_grids must be None")
    shape = tuple(input_tensor.shape)
    if len(shape) not in (1, 2, 3, 4):
        raise ValueError(f"g6_typecast: rank must be 1..4, got {len(shape)}")
    if input_tensor.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError("g6_typecast: input must be on device")
    mem = input_tensor.memory_config()
    if mem.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError("g6_typecast: input must be interleaved")

    scenario = {
        "input_shape": shape,
        "dtype": input_tensor.dtype,
        "output_dtype": dtype,
        "layout": input_tensor.layout,
        "buffer": _buffer_tag(mem),
    }
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    if axes["output_dtype"] == ttnn.bfloat8_b and axes["layout"] == ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError("g6_typecast: bfloat8_b has no ROW_MAJOR representation")

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_typecast: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"g6_typecast: unsupported combination (refinement candidate): {exc}")
    if (axes["dtype"], axes["output_dtype"]) not in CONVERSION_ARMS:
        raise ValueError(f"g6_typecast: no conversion arm for {axes['dtype']} -> {axes['output_dtype']}")
    return axes


# ---------------------------------------------------------------------------
# Plan reporting (BB-08) and program-descriptor cache (BB-07)
# ---------------------------------------------------------------------------

# The plan the last call selected: {"regime", "num_cores", "block_tiles", "effective_depth", ...}.
last_plan = None

# key -> (ProgramDescriptor, plan). Repeat calls only patch buffer addresses (RT args).
_PROGRAM_CACHE = {}


def _cache_key(input_tensor, dtype):
    mem = input_tensor.memory_config()
    device = input_tensor.device()
    return (
        device.id(),
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        dtype,
        input_tensor.layout,
        mem.memory_layout,
        mem.buffer_type,
        # Block knobs change CT args / CB sizes: part of the key so a turned knob never replays a stale program.
        _pd.BUFFER_DEPTH,
        _pd.BLOCK_TILES_CAP,
        _pd.STAGING_DEPTH,
        _pd.L1_OP_BUDGET_BYTES,
        _pd.L1_SLACK_BYTES,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def g6_typecast(input_tensor, dtype, *, sub_core_grids=None):
    """Convert `input_tensor` to `dtype` (bit-identical to torch.Tensor.to; bfloat8_b block-quantized).

    Output keeps the input's shape, layout and memory config. One device program per call.
    """
    global last_plan
    validate(input_tensor, dtype, sub_core_grids=sub_core_grids)

    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape,
        dtype,
        input_tensor.layout,
        device,
        input_tensor.memory_config(),
    )

    key = _cache_key(input_tensor, dtype)
    cached = _PROGRAM_CACHE.get(key)
    if cached is None:
        program_descriptor, plan = create_program_descriptor(input_tensor, output_tensor)
        _PROGRAM_CACHE[key] = (program_descriptor, plan)
    else:
        program_descriptor, plan = cached
        patch_buffer_addresses(program_descriptor, input_tensor, output_tensor)
    last_plan = plan

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
