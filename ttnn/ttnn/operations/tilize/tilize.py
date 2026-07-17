# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — convert a ROW_MAJOR tensor into TILE layout.

Pure layout conversion: element VALUES are unchanged, only the byte positions in
memory change (value-preserving cast tolerance when `dtype=` narrows the format).
The op tilizes natively via the `tilize_block` LLK — it never round-trips through
`ttnn.to_layout` / host.

Registry model (see eval/op_template.py): INPUT_TAGGERS / SUPPORTED / EXCLUSIONS
are declared inline and validate() gates any input outside the contract with a
support-refusal from ttnn.operations._op_contract. INVALID (int<->float casts)
lives in feature_spec.py and is not declared here.
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import create_program_descriptor

TILE = 32


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  — project scenario axes off the golden INPUTS scenario dict
#    (inputs[0] is the per-case scenario dict; see feature_spec.py).
# ---------------------------------------------------------------------------
def _buffer_name(buf) -> str:
    return "dram" if buf == ttnn.BufferType.DRAM else "l1"


def tag_use_multicore(inputs, axes) -> bool:
    return bool(inputs[0]["use_multicore"])


def tag_shard_api(inputs, axes) -> str:
    return inputs[0]["shard_api"]


def tag_out_scheme(inputs, axes):
    out = inputs[0]["out"]
    if out["kind"] == "interleaved":
        return "interleaved"
    scheme = out.get("scheme")
    return "nd" if scheme is None else scheme


def tag_buffer(inputs, axes) -> str:
    in_buf = _buffer_name(inputs[0]["in"]["buffer"])
    out_buf = _buffer_name(inputs[0]["out"]["buffer"])
    return f"{in_buf}_to_{out_buf}"


def tag_rank(inputs, axes) -> int:
    return len(inputs[0]["input_shape"])


INPUT_TAGGERS = {
    "use_multicore": tag_use_multicore,
    "shard_api": tag_shard_api,
    "out_scheme": tag_out_scheme,
    "buffer": tag_buffer,
    "rank": tag_rank,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED  — per-axis accepted values (grows per refinement toward TARGET)
# ---------------------------------------------------------------------------
SUPPORTED = {
    # Integer passthrough (uint32 + the uint16/int32 family it stands in for in
    # feature_spec.py): the tilize LLK reorders bytes with no arithmetic and no
    # cast (integers only pair with the same integer dtype; int<->float crosses
    # are pruned by INVALID in feature_spec.py). The integer path must NOT take
    # the fp32 branch (no Lossless / UnpackToDestFp32 / fp32_dest_acc_en) — those
    # are float precision levers only; the tilize helper falls back to the
    # standard (non-fast) tilize path for non-Float32/Float16_b formats.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.uint16, ttnn.int32],
    "output_dtype": [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.uint32, ttnn.uint16, ttnn.int32],
    "use_multicore": [False, True],
    "shard_api": ["none"],
    "out_scheme": ["interleaved"],
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_dram", "l1_to_l1"],
    "rank": [2, 3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS  — cells inside cartesian(SUPPORTED) refused for now
# ---------------------------------------------------------------------------
EXCLUSIONS = [
    # int<->float casts are pruned by INVALID (feature_spec.py); float->float
    # and float->bf8b casts are all handled by the pack-stage reconfigure.
]


PROPERTIES = {
    "multi_core": {"value": True, "source": "verified"},
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()  — runtime gate built from the real tensor + kwargs
# ---------------------------------------------------------------------------
def _scheme_of(mem_config):
    """interleaved -> 'interleaved'; sharded -> the scheme (or 'nd')."""
    if not mem_config.is_sharded():
        return "interleaved"
    layout = mem_config.memory_layout
    return "nd" if layout == ttnn.TensorMemoryLayout.INTERLEAVED else layout


def _shard_api_of(in_mc, out_mc):
    if not in_mc.is_sharded() and not out_mc.is_sharded():
        return "none"
    # Only interleaved is supported for now; anything sharded is refused via
    # the out_scheme / shard_api axes below.
    return "legacy_2d"


def validate(input_tensor, *, memory_config=None, dtype=None, use_multicore=True):
    # --- hard input validation (independent of the registry) ---
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("tilize: input tensor must be on device")
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(f"tilize: input must be ROW_MAJOR_LAYOUT, got {input_tensor.layout}")
    shape = list(input_tensor.shape)
    if len(shape) < 2:
        raise ValueError(f"tilize: input rank must be >= 2, got {len(shape)}")
    if shape[-1] % TILE != 0 or shape[-2] % TILE != 0:
        raise ValueError(f"tilize: last two dims must be divisible by {TILE} (op does NOT pad), got {shape[-2:]}")

    # --- registry axes from the real tensor + kwargs ---
    in_mc = input_tensor.memory_config()
    out_mc = memory_config if memory_config is not None else in_mc
    out_dtype = dtype if dtype is not None else input_tensor.dtype

    axes = {
        "dtype": input_tensor.dtype,
        "output_dtype": out_dtype,
        "use_multicore": bool(use_multicore),
        "shard_api": _shard_api_of(in_mc, out_mc),
        "out_scheme": _scheme_of(out_mc),
        "buffer": f"{_buffer_name(in_mc.buffer_type)}_to_{_buffer_name(out_mc.buffer_type)}",
        "rank": len(shape),
    }

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def tilize(
    input_tensor: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig | None = None,
    *,
    dtype: ttnn.DataType | None = None,
    use_multicore: bool = True,
) -> ttnn.Tensor:
    """Convert `input_tensor` (ROW_MAJOR) to TILE_LAYOUT.

    Args:
        input_tensor: ROW_MAJOR tensor on device, last two dims divisible by 32.
        memory_config: output memory config (default: input's).
        dtype: output dtype (default: input's). Value-preserving cast only.
        use_multicore: distribute tile-rows across the core grid (default True).
    """
    validate(input_tensor, memory_config=memory_config, dtype=dtype, use_multicore=use_multicore)

    device = input_tensor.device()
    out_mem_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = dtype if dtype is not None else input_tensor.dtype

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        out_dtype,
        ttnn.TILE_LAYOUT,
        device,
        out_mem_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, use_multicore)

    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
