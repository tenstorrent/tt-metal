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
    # Sharded I/O (Refinement 2): "legacy_2d" (HEIGHT/WIDTH/BLOCK ShardSpec) and
    # "nd" (NdShardSpec). The supported sharded path is same-spec, zero-copy:
    # RM-sharded L1 input -> TILE-sharded L1 output on the IDENTICAL shard spec,
    # each core tilizing its own resident block straight into the output shard
    # (no DRAM/NoC). Cross-spec / interleaved<->sharded crossovers are refused in
    # validate() (see _SHARDED_REFUSE); they are a follow-up (Refinement 2b).
    "shard_api": ["none", "legacy_2d", "nd"],
    "out_scheme": [
        "interleaved",
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "nd",
    ],
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_dram", "l1_to_l1"],
    "rank": [2, 3, 4],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS  — cells inside cartesian(SUPPORTED) the op refuses
# ---------------------------------------------------------------------------
# int<->float "casts" are value REINTERPRETATIONS, not the value-preserving
# layout+precision cast tilize implements — a different op entirely. They are
# declared INVALID in feature_spec.py (skipped test-side), but because `dtype`
# (input) and `output_dtype` (kwarg) are independent cartesian axes, every
# int<->float cross falls INSIDE cartesian(SUPPORTED) once both an integer and
# a float dtype are in the SUPPORTED lists. EXCLUSIONS is the registry's
# hole-punch for that: it removes those cells from the SUPPORTED rectangle so
# (a) validate() refuses them at runtime (ExcludedCell) instead of running the
# kernel on a garbage reinterpret, and (b) the completion gate's is_supported()
# does not count them as "responsible" cells that can never pass (they are
# INVALID-skipped, not passing). float->float and float->bf8b casts stay
# supported — the pack-stage reconfigure handles those value-preservingly.
_INT_DTYPES = (ttnn.uint32, ttnn.uint16, ttnn.int32)
_FLOAT_IN_DTYPES = (ttnn.bfloat16, ttnn.float32)  # bf8b is not an RM input
_FLOAT_OUT_DTYPES = (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b)

EXCLUSIONS = (
    [{"dtype": i, "output_dtype": f} for i in _INT_DTYPES for f in _FLOAT_OUT_DTYPES]
    + [{"dtype": f, "output_dtype": i} for f in _FLOAT_IN_DTYPES for i in _INT_DTYPES]
    # Sharding is inherently multi-core: each core owns and tilizes its own shard.
    # Single-core + sharded is structurally impossible for this op.
    + [
        {"use_multicore": False, "shard_api": "legacy_2d"},
        {"use_multicore": False, "shard_api": "nd"},
    ]
)


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
    return "nd" if layout == ttnn.TensorMemoryLayout.ND_SHARDED else layout


def _shard_api_of(in_mc, out_mc):
    if not in_mc.is_sharded() and not out_mc.is_sharded():
        return "none"
    # nd (NdShardSpec) carries memory_layout == INTERLEAVED; legacy 2D carries a
    # HEIGHT/WIDTH/BLOCK layout. Read from whichever side is sharded.
    ref = out_mc if out_mc.is_sharded() else in_mc
    return "nd" if ref.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED else "legacy_2d"


def _folded_shard_shape(mem_config):
    """(shard_h_folded, shard_w) — leading shard dims folded into height."""
    if mem_config.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
        shape = list(mem_config.nd_shard_spec.shard_shape)
    else:
        shape = list(mem_config.shard_spec.shape)
    shard_w = shape[-1]
    shard_h = 1
    for d in shape[:-1]:
        shard_h *= d
    return shard_h, shard_w


def _shard_spec_of(mc):
    return mc.nd_shard_spec if mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED else mc.shard_spec


def _shard_props(mc):
    """PHYSICAL shard placement, invariant to the nd<->legacy normalization ttnn
    applies (an nd spec expressible as legacy BLOCK/HEIGHT/WIDTH is stored under
    that legacy layout, so the memory_layout enum alone is not comparable).
    Two tensors with identical (buffer_type, folded shard shape, orientation,
    grid) occupy the same L1 regions on the same cores — the per-core zero-copy
    tilize then preserves identity."""
    spec = _shard_spec_of(mc)
    return (mc.buffer_type, _folded_shard_shape(mc), spec.orientation, spec.grid)


def _same_shard_spec(in_mc, out_mc):
    """True iff input and output describe the same physical shard placement, so
    each core tilizes its own resident block and global identity is preserved."""
    return _shard_props(in_mc) == _shard_props(out_mc)


def _num_shards(tensor_shape, mc):
    """Number of shards the tensor is divided into (product of per-dim ceil
    divisions). Rank-aligns the shard shape to the tensor from the right;
    legacy 2D specs cover the last-2-dims-folded view."""
    if mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
        shard = list(mc.nd_shard_spec.shard_shape)
    else:
        shard = list(mc.shard_spec.shape)
    ts = list(tensor_shape)
    if len(shard) < len(ts):
        # legacy 2D shard covers the folded [H_folded, W] view.
        h = 1
        for d in ts[:-1]:
            h *= d
        ts = [h, ts[-1]]
    k = min(len(shard), len(ts))
    shard = shard[-k:]
    ts = ts[-k:]
    n = 1
    for t, s in zip(ts, shard):
        n *= -(-t // s)  # ceil division
    return n


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

    # 3. Sharded sub-case gating (Refinement 2). The implemented sharded path is
    #    same-spec, zero-copy: both sides L1-sharded on the IDENTICAL shard spec.
    #    Cross-spec resharding and interleaved<->sharded crossovers are not yet
    #    wired (Refinement 2b) — refuse them cleanly (never hang).
    if in_mc.is_sharded() or out_mc.is_sharded():
        if not (in_mc.is_sharded() and out_mc.is_sharded()):
            raise UnsupportedAxisValue(
                "tilize: sharded path requires BOTH input and output sharded "
                "(interleaved<->sharded crossover not yet supported)"
            )
        if in_mc.buffer_type != ttnn.BufferType.L1 or out_mc.buffer_type != ttnn.BufferType.L1:
            raise UnsupportedAxisValue("tilize: sharded path requires L1 buffers")
        if not _same_shard_spec(in_mc, out_mc):
            raise UnsupportedAxisValue(
                "tilize: sharded path requires identical input/output shard spec "
                "(cross-spec resharding not yet supported)"
            )
        shard_h, shard_w = _folded_shard_shape(in_mc)
        if shard_h % TILE != 0 or shard_w % TILE != 0:
            raise UnsupportedAxisValue(f"tilize: sharded shard dims must be tile-aligned, got ({shard_h}, {shard_w})")
        # The zero-copy path tilizes exactly ONE resident shard per core. Configs
        # where a core owns multiple shards (num_shards > num_cores) or where cores
        # sit idle (num_shards < num_cores) are not yet wired (Refinement 2b) —
        # refuse cleanly rather than under-process the buffer (which corrupts output
        # and can hang the precompile real-alloc path).
        n_shards = _num_shards(shape, in_mc)
        n_cores = _shard_spec_of(in_mc).grid.num_cores()
        if n_shards != n_cores:
            raise UnsupportedAxisValue(
                f"tilize: sharded path requires one shard per core "
                f"(got {n_shards} shards over {n_cores} cores; multi-shard-per-core is Refinement 2b)"
            )


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
