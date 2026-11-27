# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""g6_to_memory_config: the input's tile pages, bit-identical, in the placement memory_config names.

One device program per call. Every page is read once from its place under the input's placement
and written once to its place under the target placement, through the L1 of the core that moves it
(op_design.md). No host-side tensor transformation of any kind.
"""

from __future__ import annotations

import hashlib

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from . import g6_to_memory_config_program_descriptor as descriptor_module

# The reports bindings live in ttnn._ttnn.reports; the op and its callers name them ttnn.reports.
if not hasattr(ttnn, "reports"):
    ttnn.reports = ttnn._ttnn.reports


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads only the scenario dict inputs[0]
# ---------------------------------------------------------------------------


def tag_dtype(inputs, axes):
    return inputs[0]["dtype"]


def tag_layout(inputs, axes):
    return inputs[0]["layout"]


def tag_rank(inputs, axes):
    return int(len(inputs[0]["shape"]))


def tag_alignment(inputs, axes):
    shape = inputs[0]["shape"]
    if shape[-2] % 32 == 0 and shape[-1] % 32 == 0:
        return "tile_aligned"
    return "h_non_aligned"


def tag_in_scheme(inputs, axes):
    return inputs[0]["in_memory"]["scheme"]


def tag_out_scheme(inputs, axes):
    return inputs[0]["out_memory"]["scheme"]


def tag_in_buffer(inputs, axes):
    return inputs[0]["in_memory"]["buffer"]


def tag_out_buffer(inputs, axes):
    return inputs[0]["out_memory"]["buffer"]


INPUT_TAGGERS = {
    "dtype": tag_dtype,
    "layout": tag_layout,
    "rank": tag_rank,
    "alignment": tag_alignment,
    "in_scheme": tag_in_scheme,
    "out_scheme": tag_out_scheme,
    "in_buffer": tag_in_buffer,
    "out_buffer": tag_out_buffer,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

_SCHEMES = ["interleaved", "height_sharded", "width_sharded", "block_sharded"]

SUPPORTED = {
    # Pages move whole, so a block format is just a smaller page: page_size comes from
    # buffer_aligned_page_size() and the bounce CB takes the input dtype.
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
    "layout": [ttnn.TILE_LAYOUT],
    "rank": [2, 3, 4],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "in_scheme": list(_SCHEMES),
    "out_scheme": list(_SCHEMES),
    "in_buffer": ["dram", "l1"],
    "out_buffer": ["dram", "l1"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_BUFFER_NAME = {ttnn.BufferType.DRAM: "dram", ttnn.BufferType.L1: "l1"}
_SCHEME_NAME = {
    ttnn.TensorMemoryLayout.INTERLEAVED: "interleaved",
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: "block_sharded",
}
_ORIENTATION_NAME = {ttnn.ShardOrientation.ROW_MAJOR: "row_major", ttnn.ShardOrientation.COL_MAJOR: "col_major"}


def _memory_entry(memory_config):
    entry = {
        "buffer": _BUFFER_NAME.get(memory_config.buffer_type, str(memory_config.buffer_type)),
        "scheme": _SCHEME_NAME.get(memory_config.memory_layout, str(memory_config.memory_layout)),
    }
    spec = memory_config.shard_spec if memory_config.is_sharded() else None
    if spec is not None:
        entry["grid"] = [[r.start.x, r.start.y, r.end.x, r.end.y] for r in spec.grid.ranges()]
        entry["shard_shape"] = list(spec.shape)
        entry["orientation"] = _ORIENTATION_NAME.get(spec.orientation, str(spec.orientation))
    return entry


def _scenario(input_tensor, memory_config):
    return {
        "shape": list(input_tensor.shape),
        "dtype": input_tensor.dtype,
        "layout": input_tensor.layout,
        "in_memory": _memory_entry(input_tensor.memory_config()),
        "out_memory": _memory_entry(memory_config),
    }


def _check_shard_spec(side, memory_config, rows_tiles, cols_tiles, tile_h, tile_w):
    """Structural checks of one sharded placement against the input's padded tile grid."""
    spec = memory_config.shard_spec
    if spec is None:
        raise ValueError(f"g6_to_memory_config: {side} is sharded but carries no shard spec")
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError(f"g6_to_memory_config: {side} shard orientation {spec.orientation} is not ROW_MAJOR")
    shard_h, shard_w = spec.shape
    if shard_h <= 0 or shard_w <= 0 or shard_h % tile_h or shard_w % tile_w:
        raise ValueError(f"g6_to_memory_config: {side} shard shape {list(spec.shape)} is not a multiple of the tile")
    shard_h_tiles, shard_w_tiles, n_rows, n_cols = descriptor_module.shard_grid(
        memory_config, rows_tiles, cols_tiles, tile_h, tile_w
    )
    layout = memory_config.memory_layout
    if layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED and shard_w_tiles != cols_tiles:
        raise ValueError(f"g6_to_memory_config: {side} height shard width {shard_w} != padded W {cols_tiles * tile_w}")
    if layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and shard_h_tiles != rows_tiles:
        raise ValueError(
            f"g6_to_memory_config: {side} width shard height {shard_h} != padded rows {rows_tiles * tile_h}"
        )
    if layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        ranges = spec.grid.ranges()
        if len(ranges) != 1:
            raise ValueError(f"g6_to_memory_config: {side} block shard grid must be one core range")
        width = ranges[0].end.x - ranges[0].start.x + 1
        height = ranges[0].end.y - ranges[0].start.y + 1
        if n_cols > width or n_rows > height:
            raise ValueError(
                f"g6_to_memory_config: {side} needs a {n_cols}x{n_rows} shard grid, its core grid is {width}x{height}"
            )
    elif n_rows * n_cols > spec.grid.num_cores():
        raise ValueError(
            f"g6_to_memory_config: {side} needs {n_rows * n_cols} shards, its grid holds {spec.grid.num_cores()}"
        )


def _memory_key(memory_config):
    shard = None
    if memory_config.is_sharded() and memory_config.shard_spec is not None:
        spec = memory_config.shard_spec
        shard = (
            tuple((r.start.x, r.start.y, r.end.x, r.end.y) for r in spec.grid.ranges()),
            tuple(spec.shape),
            spec.orientation,
        )
    return (memory_config.memory_layout, memory_config.buffer_type, shard)


def _cache_key(input_tensor, memory_config):
    """Everything that changes the program (BB-07): device, tile grid, dtype, both placements, knobs."""
    return (
        input_tensor.device().id(),
        tuple(input_tensor.padded_shape),
        tuple(input_tensor.tile.tile_shape),
        input_tensor.dtype,
        _memory_key(input_tensor.memory_config()),
        _memory_key(memory_config),
        descriptor_module.BOUNCE_DEPTH,
        descriptor_module.MOVERS_PER_OWNER,
        descriptor_module.BOUNCE_WRITE_RUN_PAGES,
        descriptor_module.BOUNCE_TARGET_ORDER,
        descriptor_module.BOUNCE_BANK_STAGGER,
        tuple(descriptor_module.ABLATE_DEFINES),
        descriptor_module.L1_OP_BUDGET_BYTES,
    )


_plan_cache = {}
_descriptor_cache = {}


def _plan_for(key, input_tensor, memory_config):
    plan = _plan_cache.get(key)
    if plan is None:
        plan = descriptor_module.build_plan(input_tensor, memory_config)
        _plan_cache[key] = plan
    return plan


def validate(input_tensor, memory_config):
    """Gate a call: support contract first, then structure. Returns (cache key, plan) of the call."""
    # (1) The axes are read from the device placement; a host tensor has none.
    if not isinstance(input_tensor, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("g6_to_memory_config: input_tensor must be a device tensor")
    if not isinstance(memory_config, ttnn.MemoryConfig):
        raise ValueError("g6_to_memory_config: memory_config must be a ttnn.MemoryConfig")

    # (2) Same scenario dict and taggers as the golden suite.
    scenario = _scenario(input_tensor, memory_config)
    axes = {}
    for axis, tagger in INPUT_TAGGERS.items():
        axes[axis] = tagger((scenario,), axes)

    # (3) SUPPORTED, per axis.
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"g6_to_memory_config: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # (4) EXCLUSIONS, per cell.
    for cell in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in cell.items()):
            raise ExcludedCell(f"g6_to_memory_config: excluded cell (refinement candidate): {cell}")

    # (5) Structure.
    rows_tiles, cols_tiles, tile_h, tile_w = descriptor_module.tile_grid(input_tensor)
    in_config = input_tensor.memory_config()
    if in_config.is_sharded():
        _check_shard_spec("input", in_config, rows_tiles, cols_tiles, tile_h, tile_w)
    if memory_config.is_sharded():
        _check_shard_spec("memory_config", memory_config, rows_tiles, cols_tiles, tile_h, tile_w)
    page_size = input_tensor.buffer_aligned_page_size()
    if page_size % ttnn.get_dram_alignment():
        raise ValueError(f"g6_to_memory_config: page size {page_size} is not a multiple of the DRAM alignment")

    key = _cache_key(input_tensor, memory_config)
    plan = _plan_for(key, input_tensor, memory_config)
    if plan["l1_op_bytes"] > descriptor_module.L1_OP_BUDGET_BYTES:
        raise ValueError(
            f"g6_to_memory_config: {plan['l1_op_bytes']} B of op L1 on the most loaded core exceeds the "
            f"{descriptor_module.L1_OP_BUDGET_BYTES} B budget"
        )
    return key, plan


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _program_hash(key):
    return int.from_bytes(hashlib.sha256(repr(("g6_to_memory_config", key)).encode()).digest()[:8], "little")


def _check_owner_cores(plan, input_tensor, output_tensor):
    """The derived owner order must be the buffer's: a pull into a non-local slot corrupts another tensor."""
    if plan["regime"] not in ("pull_shard", "push_shard"):
        return
    owned = output_tensor if plan["regime"] == "pull_shard" else input_tensor
    actual = [(c.x, c.y) for c in ttnn.get_optimal_worker_cores_for_sharded_tensor(owned)]
    if actual != plan["owner_cores"]:
        raise RuntimeError(
            f"g6_to_memory_config: derived shard owners {plan['owner_cores']} differ from the buffer's {actual}"
        )


def _check_dram_alignment(input_tensor, output_tensor):
    """Blackhole DRAM transfers need the L1 address congruent to the DRAM address mod the DRAM alignment."""
    alignment = ttnn.get_dram_alignment()
    in_l1 = input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    out_l1 = output_tensor.memory_config().buffer_type == ttnn.BufferType.L1
    if in_l1 != out_l1:
        l1_tensor = input_tensor if in_l1 else output_tensor
        if l1_tensor.buffer_address() % alignment:
            raise RuntimeError(
                f"g6_to_memory_config: L1 buffer address {l1_tensor.buffer_address()} is not {alignment} B aligned"
            )


def g6_to_memory_config(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Return input_tensor's values, shape, dtype and TILE layout, placed as memory_config names."""
    key, plan = validate(input_tensor, memory_config)

    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape, input_tensor.dtype, ttnn.TILE_LAYOUT, input_tensor.device(), memory_config
    )
    _check_dram_alignment(input_tensor, output_tensor)

    descriptor = _descriptor_cache.get(key)
    if descriptor is None:
        _check_owner_cores(plan, input_tensor, output_tensor)
        descriptor = descriptor_module.create_program_descriptor(plan, input_tensor, output_tensor, _program_hash(key))
        _descriptor_cache[key] = descriptor
    else:
        descriptor_module.patch_addresses(descriptor, input_tensor.buffer_address(), output_tensor.buffer_address())

    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def plan_of(input_tensor: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> dict:
    """The plan the call g6_to_memory_config(input_tensor, memory_config) selects (BB-08).

    Keys: regime, movers (owner cores in owned-shard order for shard regimes, every grid core for
    bank / bounce regimes), kernel_cores (sx, sy, ex, ey), roles {(x, y): role} (the BB-21 role dump),
    block sizes and the per-core op L1.
    """
    _, plan = validate(input_tensor, memory_config)
    return {k: v for k, v in plan.items() if k != "core_args"}
