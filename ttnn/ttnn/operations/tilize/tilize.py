# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — re-lay a Layout::ROW_MAJOR tensor into Layout::TILE in one generic_op dispatch.

Registry model (eval/op_template.py): INPUT_TAGGERS, SUPPORTED, EXCLUSIONS,
validate(), then the public entry point `tilize`.

The twelve INPUT_TAGGERS follow the `Expected INPUT_TAGGERS` block of
eval/golden_tests/tilize/feature_spec.py: each reads inputs[0] as a SCENARIO
DICT (keys: input_shape, shard_api, in, out, and optionally low_l1 / pad_mode /
output_padded_shape / pad_value / tile_height / in_tile_height). validate()
builds the same scenario dict from the live call (`_scenario_from_call`) and
runs the very same taggers over it, so the runtime gate and the golden
harness's xfail decisions share one set of rules.

Regimes: `row_split_interleaved`, `grid_2d_split`, `sharded_resident`, `sharded_accessor`,
`retile_l1_facewalk` (op_design.md -> Blocking Model -> Regimes).
"""

from __future__ import annotations

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import TILE_WIDTH, PadSpec, _tile_grid, create_program_descriptor


def _dominant():
    """Dimensionless ratio at which one tile-grid axis "dominates" the other.

    Imported (lazily, so importing the op never triggers the feature spec's arch
    probe) from the golden feature spec rather than restated (op_design.md). The
    fallback only exists for installs that ship ttnn without the eval tree.
    """
    global _DOMINANT
    if _DOMINANT is None:
        try:
            from eval.golden_tests.tilize.feature_spec import DOMINANT
        except Exception:  # noqa: BLE001 — eval tree absent outside the repo checkout
            DOMINANT = 16
        _DOMINANT = DOMINANT
    return _DOMINANT


_DOMINANT = None

LEGAL_TILE_HEIGHTS = (32, 16, 8, 4, 2, 1)


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS — each reads inputs[0], the whole scenario dict
# ---------------------------------------------------------------------------


def _is_sharded(spec):
    return spec["kind"] == "sharded"


def _is_nd(spec):
    return _is_sharded(spec) and spec.get("scheme") is None


def tag_low_l1(inputs, axes):
    return bool(inputs[0].get("low_l1", False))


def tag_shard_api(inputs, axes):
    s = inputs[0]
    if not _is_sharded(s["in"]) and not _is_sharded(s["out"]):
        return "none"
    if _is_nd(s["in"]) or _is_nd(s["out"]):
        return "nd"
    return "legacy_2d"


def tag_out_scheme(inputs, axes):
    out = inputs[0]["out"]
    if not _is_sharded(out):
        return "interleaved"
    if _is_nd(out):
        return "nd"
    return out["scheme"]


def _buffer_word(buffer_type):
    return "dram" if buffer_type == ttnn.BufferType.DRAM else "l1"


def tag_buffer(inputs, axes):
    s = inputs[0]
    return f"{_buffer_word(s['in']['buffer'])}_to_{_buffer_word(s['out']['buffer'])}"


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_orientation(inputs, axes):
    s = inputs[0]
    if _is_sharded(s["out"]):
        return s["out"]["orientation"]
    if _is_sharded(s["in"]):
        return s["in"]["orientation"]
    return "none"


def tag_pad_mode(inputs, axes):
    return inputs[0].get("pad_mode", "none")


def tag_pad_value(inputs, axes):
    s = inputs[0]
    if s.get("pad_mode", "none") == "none":
        return "none"
    v = s.get("pad_value", 0)
    if v is None or v == 0:
        return "zero"
    return "positive" if v > 0 else "negative"


def tag_tile_height(inputs, axes):
    return int(inputs[0].get("tile_height", 32))


def tag_in_tile_height(inputs, axes):
    v = inputs[0].get("in_tile_height", "none")
    return "none" if v is None else v


def tag_tile_grid(inputs, axes):
    shape = list(inputs[0]["input_shape"])
    if len(shape) < 2:
        return "single_tile"
    tile_height = axes.get("tile_height", inputs[0].get("tile_height", 32))
    rows, cols = _tile_grid(shape, int(tile_height))  # the same (R, C) the program descriptor splits
    dominant = _dominant()
    if rows == 1 and cols == 1:
        return "single_tile"
    if cols >= dominant * rows:
        return "short_wide"
    if rows >= dominant * cols:
        return "tall_narrow"
    if rows * cols < dominant * dominant:
        return "small"
    return "square_large"


def tag_alignment(inputs, axes):
    shape = list(inputs[0]["input_shape"])
    if len(shape) < 2:
        return "hw_non_aligned"
    tile_height = axes.get("tile_height", inputs[0].get("tile_height", 32))
    h_ok = shape[-2] % int(tile_height) == 0
    w_ok = shape[-1] % TILE_WIDTH == 0
    if h_ok and w_ok:
        return "tile_aligned"
    if h_ok:
        return "w_non_aligned"
    if w_ok:
        return "h_non_aligned"
    return "hw_non_aligned"


# Order matters: tile_grid and alignment read the resolved tile_height.
INPUT_TAGGERS = {
    "low_l1": tag_low_l1,
    "shard_api": tag_shard_api,
    "out_scheme": tag_out_scheme,
    "buffer": tag_buffer,
    "rank": tag_rank,
    "orientation": tag_orientation,
    "pad_mode": tag_pad_mode,
    "pad_value": tag_pad_value,
    "tile_height": tag_tile_height,
    "in_tile_height": tag_in_tile_height,
    "tile_grid": tag_tile_grid,
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------
#
# Placement (Refinement 1): every shard API / scheme / orientation and every
# DRAM/L1 buffer pairing, through the sharded_resident / sharded_accessor
# regimes of op_design.md (tilize_program_descriptor._core_assignment). Ranks
# 2-6 fold their leading dims into R.
#
# tile_grid: wherever no L1 shard fixes the core assignment, the host picks (g_r, g_c) by the
# grid_2d_split rule (tilize_program_descriptor.grid_2d_split): the busiest Tensix core's tile
# count is minimized over row groups x column groups, so short_wide (R is 1-2) spreads its
# tile-columns over the grid and square_large is balanced on both axes; g_c = 1 is the row split.
# Where an L1 shard fixes the core assignment (a legacy-sharded L1 output, or a legacy-sharded
# L1 input with an interleaved output) the shard grid IS the parallelism, whatever the shape.
#
# low_l1 (Refinement 5): CB_BUDGET_BYTES[True] = 64 KiB bounds the streamed CBs independently of
# the tensor (block_width re-derives from it); same data path, so the output is bit-identical.

_LEGACY_SCHEMES = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "output_dtype": [ttnn.bfloat16],
    "low_l1": [False, True],
    "shard_api": ["none", "legacy_2d", "nd"],
    "out_scheme": ["interleaved", *_LEGACY_SCHEMES, "nd"],
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"],
    # Ranks 0 / 1 only reach the op with a padding argument (rule 5): the pad synthesizes the
    # tile dims ([] -> [32, 32], [W] -> [32, round_up(W, 32)]).
    "rank": [0, 1, 2, 3, 4, 5, 6],
    "orientation": ["none", ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    # Padding (Refinement 4): the output is allocated at the padded shape and returned as a
    # zero-copy view at the input's logical shape; the stick reader walks the padded tile grid
    # through the per-image stick map and fills everything the input does not cover.
    "pad_mode": ["none", "auto", "explicit"],
    "pad_value": ["none", "zero", "positive", "negative"],
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned", "hw_non_aligned"],
    # Tile geometry (Refinement 2): every output tile height on every placement (both CBs carry
    # TileDescriptor(tile_h, 32); the output is allocated through a TensorSpec carrying the tile),
    # and every input tile height of a Layout::TILE input, re-tiled in the same dispatch by the
    # retile_l1_facewalk reader.
    "tile_height": list(LEGAL_TILE_HEIGHTS),
    "in_tile_height": ["none", *LEGAL_TILE_HEIGHTS],
    "tile_grid": ["single_tile", "small", "tall_narrow", "short_wide", "square_large"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------
#
# Padding x retile (a Layout::TILE input whose pad has something to fill): the pad fill lives in
# the stick reader; the retile face walk has no fill yet (it would have to clamp the staging
# reads and the walk to the input's tile-rows / tile-columns and fill the rest after the walk
# lands). An auto pad of a tile-aligned TILE input fills nothing and runs the plain retile path.
def _retile_pad_exclusions():
    cells = []
    for in_tile_h in LEGAL_TILE_HEIGHTS:
        cells.append({"pad_mode": "explicit", "in_tile_height": in_tile_h})
        for alignment in ("w_non_aligned", "h_non_aligned", "hw_non_aligned"):
            cells.append({"pad_mode": "auto", "in_tile_height": in_tile_h, "alignment": alignment})
    return cells


EXCLUSIONS = _retile_pad_exclusions()


# ---------------------------------------------------------------------------
# 3b. PROPERTIES
# ---------------------------------------------------------------------------

PROPERTIES = {
    # grid_2d_split over device.compute_with_storage_grid_size() (g_r * g_c Tensix cores; the row
    # split's min(R, N) when g_c = 1);
    # a resident L1 shard's grid when one fixes the core assignment (sharded_resident).
    "multi_core": {"value": True, "source": "declared"},
    # CB total = rows_per_quantum * block_width * per_col_tile_bytes <= CB_BUDGET_BYTES[low_l1] (l1_ledger.md).
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------

_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _created_with_nd_shard_spec(mem_config):
    """True iff the caller built this MemoryConfig from an NdShardSpec.

    `nd_shard_spec` alone cannot tell: once a legacy 2-D sharded tensor is
    allocated, its memory_config() carries a derived nd_shard_spec too (and an
    ND spec with a 2-D equivalent reports a legacy memory_layout + shard_spec).
    The C++ `created_with_nd_shard_spec` flag is only exposed through to_json.
    """
    return '"created_with_nd_shard_spec":true' in mem_config.to_json().replace(" ", "")


def _side_spec(mem_config):
    """Scenario-dict side spec ({kind, buffer, orientation, scheme}) from a live MemoryConfig.

    Same classification as eval/golden_tests/tilize/axes.py:_spec_of, except that
    legacy-vs-ND is read from `created_with_nd_shard_spec` (see above).
    """
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is not None and (mem_config.shard_spec is None or _created_with_nd_shard_spec(mem_config)):
        return {"kind": "sharded", "buffer": mem_config.buffer_type, "orientation": nd_spec.orientation, "scheme": None}
    if mem_config.memory_layout in _SHARDED_LAYOUTS:
        shard_spec = mem_config.shard_spec
        return {
            "kind": "sharded",
            "buffer": mem_config.buffer_type,
            "orientation": shard_spec.orientation if shard_spec is not None else "none",
            "scheme": mem_config.memory_layout,
        }
    return {"kind": "interleaved", "buffer": mem_config.buffer_type}


def _tile_height_of(tile):
    if tile is None:
        return None
    shape = getattr(tile, "tile_shape", None) or getattr(tile, "shape", None)
    return int(shape[0]) if shape is not None else None


def _input_tile_height(input_tensor):
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return None
    return _tile_height_of(getattr(input_tensor, "tile", None)) or 32


def _resolve_output_dtype(input_tensor, dtype):
    if dtype is not None:
        return dtype
    if input_tensor.dtype == ttnn.fp8_e4m3:
        return ttnn.float32
    return input_tensor.dtype


def _scenario_from_call(input_tensor, memory_config, *, low_l1, output_padded_shape, pad_value, tile):
    in_cfg = input_tensor.memory_config()
    out_cfg = memory_config if memory_config is not None else in_cfg
    scenario = {
        "input_shape": list(input_tensor.shape),
        "in": _side_spec(in_cfg),
        "out": _side_spec(out_cfg),
        "low_l1": bool(low_l1),
        "tile_height": _tile_height_of(tile) or _input_tile_height(input_tensor) or 32,
    }
    if output_padded_shape is not None:
        scenario["pad_mode"] = "explicit"
        scenario["output_padded_shape"] = list(output_padded_shape)
        scenario["pad_value"] = 0 if pad_value is None else pad_value
    elif pad_value is not None:
        scenario["pad_mode"] = "auto"
        scenario["pad_value"] = pad_value
    in_tile_h = _input_tile_height(input_tensor)
    if in_tile_h is not None:
        scenario["in_tile_height"] = in_tile_h
    return scenario


def validate(
    input_tensor,
    memory_config=None,
    *,
    dtype=None,
    low_l1=False,
    output_padded_shape=None,
    pad_value=None,
    tile=None,
):
    """Registry gate: SUPPORTED per-axis, then EXCLUSIONS. Raises NotImplementedError subclasses."""
    scenario = _scenario_from_call(
        input_tensor,
        memory_config,
        low_l1=low_l1,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
    )
    axes = {
        "dtype": input_tensor.dtype,
        "output_dtype": _resolve_output_dtype(input_tensor, dtype),
    }
    inputs = (scenario,)
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(inputs, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")
    return axes


# ---------------------------------------------------------------------------
# Malformed-call checks (op_design.md -> validation order, rules 1-6)
# ---------------------------------------------------------------------------


def _check_well_formed(input_tensor, *, output_padded_shape, pad_value, tile):
    # 1. on device
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("tilize: input tensor must be on device")
    # 2. layout
    if input_tensor.layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"tilize: input layout {input_tensor.layout} is neither ROW_MAJOR nor TILE")
    # 3. a TILE input needs a target geometry
    if input_tensor.layout == ttnn.TILE_LAYOUT and tile is None:
        raise ValueError("tilize: a TILE_LAYOUT input needs tile= (there is nothing to re-tile to)")
    # 4. tile geometry
    if tile is not None:
        th, tw = (int(d) for d in tile.tile_shape)
        if tw != TILE_WIDTH or th not in LEGAL_TILE_HEIGHTS:
            raise ValueError(
                f"tilize: tile {[th, tw]} must be [h, 32] with h a power-of-two fraction of 32 {LEGAL_TILE_HEIGHTS}"
            )
    tile_h = _tile_height_of(tile) or _input_tile_height(input_tensor) or 32
    shape = list(input_tensor.shape)
    has_pad = output_padded_shape is not None or pad_value is not None
    # 5. alignment without a padding argument
    if not has_pad:
        if len(shape) < 2 or shape[-2] % tile_h != 0 or shape[-1] % TILE_WIDTH != 0:
            raise ValueError(
                f"tilize: shape {shape} is not a multiple of ({tile_h}, {TILE_WIDTH}) and no padding argument was given"
            )
    # 6. explicit padded shape
    if output_padded_shape is not None:
        padded = [int(d) for d in output_padded_shape]
        expanded = [1] * max(0, len(padded) - len(shape)) + shape
        if len(padded) < len(shape) or any(p < s for p, s in zip(padded, expanded)):
            raise ValueError(f"tilize: output_padded_shape {padded} is smaller than the input shape {shape}")
        if len(padded) < 2 or padded[-2] % tile_h != 0 or padded[-1] % TILE_WIDTH != 0:
            raise ValueError(
                f"tilize: output_padded_shape {padded} last two dims must be multiples of ({tile_h}, {TILE_WIDTH})"
            )
        _check_leading_pad_expressible(padded, expanded)


def _check_leading_pad_expressible(padded, expanded):
    """Refuse leading-dim growth that TTNN's padded-shape model cannot express.

    A Layout::TILE tensor's logical view maps logical image k (flat over the LOGICAL leading
    dims) to physical image k (flat over the PADDED ones). That agrees with F.pad only while
    every input image keeps its flat index, i.e. each leading dim the input actually indexes
    (dim > 1) has the same stride on both sides. Growing the outermost indexed dim (or adding
    new outer dims) keeps them; growing an inner one (e.g. [2, 3, ...] -> [3, 4, ...]) would give
    a buffer whose padded readback is F.pad but whose logical readback is not the input.
    """
    lead_p, lead_x = padded[:-2], expanded[:-2]
    stride_p = stride_x = 1
    for d in reversed(range(len(lead_p))):
        if lead_x[d] > 1 and stride_p != stride_x:
            raise NotImplementedError(
                f"tilize: output_padded_shape {padded} grows a leading dim inside input dim {d} of {expanded}; "
                "a TTNN padded shape can only grow the outermost indexed leading dim and the last two dims"
            )
        stride_p *= lead_p[d]
        stride_x *= lead_x[d]


# ---------------------------------------------------------------------------
# Padding: the padded shape and the fill bits
# ---------------------------------------------------------------------------


def _resolve_padding(shape, *, output_padded_shape, pad_value, tile_h):
    """(padded_shape, input_shape_expanded) or None when no padding argument was given.

    auto: the last two dims rounded up to (tile_h, 32); ranks 0 / 1 are left-expanded to rank 2
    first, so the pad synthesizes the tile dims. explicit: `output_padded_shape`, the input
    left-expanded with 1s to its rank.
    """
    shape = [int(d) for d in shape]
    if output_padded_shape is not None:
        padded = [int(d) for d in output_padded_shape]
        return padded, [1] * (len(padded) - len(shape)) + shape
    if pad_value is None:
        return None
    expanded = [1] * max(0, 2 - len(shape)) + shape
    padded = expanded[:-2] + [-(-expanded[-2] // tile_h) * tile_h, -(-expanded[-1] // TILE_WIDTH) * TILE_WIDTH]
    return padded, expanded


def _fill_bits(pad_value, dtype):
    """The fill value as the input dtype's element bit pattern (low bits of a uint32)."""
    import torch

    v = 0 if pad_value is None else pad_value
    if dtype == ttnn.bfloat16:
        return int(torch.tensor([float(v)], dtype=torch.bfloat16).view(torch.int16).item()) & 0xFFFF
    if dtype == ttnn.float32:
        return int(torch.tensor([float(v)], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF
    if dtype in (ttnn.uint32, ttnn.int32):
        return int(v) & 0xFFFFFFFF
    if dtype == ttnn.uint16:
        return int(v) & 0xFFFF
    if dtype == ttnn.uint8:
        return int(v) & 0xFF
    raise NotImplementedError(f"tilize: no pad-fill encoding for input dtype {dtype}")


# ---------------------------------------------------------------------------
# Output allocation
# ---------------------------------------------------------------------------


def _resolve_output_memory_config(input_tensor, mem_config, out_shape, *, tile_h):
    """A legacy-sharded output MemoryConfig given WITHOUT a shard spec takes the input's.

    The input's shard grid and orientation are kept and the shard shape is re-derived over the
    output (padded) 2-D fold, rounded to whole output tiles: WIDTH keeps the input's shard width
    over all padded rows; HEIGHT and BLOCK keep the input's number of shards along each axis
    (so a shard of whole input images becomes a shard of whole padded images).
    """
    if (
        mem_config.memory_layout not in _SHARDED_LAYOUTS
        or mem_config.shard_spec is not None
        or getattr(mem_config, "nd_shard_spec", None) is not None
    ):
        return mem_config
    in_mc = input_tensor.memory_config()
    in_spec = in_mc.shard_spec
    if in_mc.memory_layout not in _SHARDED_LAYOUTS or in_spec is None:
        raise NotImplementedError(
            "tilize: a sharded output memory_config without a shard spec needs a legacy-sharded input to derive it from"
        )

    def _up(a, b):
        return -(-a // b) * b

    in_shape = [int(d) for d in input_tensor.shape] or [1]
    in_rows = 1
    for d in in_shape[:-1]:
        in_rows *= d
    in_width = in_shape[-1]
    out_rows = 1
    for d in out_shape[:-1]:
        out_rows *= int(d)
    out_width = int(out_shape[-1])
    in_sh, in_sw = (int(d) for d in in_spec.shape)
    n_h, n_w = -(-in_rows // in_sh), -(-in_width // in_sw)
    layout = mem_config.memory_layout
    if layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        shard = [out_rows, _up(-(-out_width // n_w), TILE_WIDTH)]
    elif layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        shard = [_up(-(-out_rows // n_h), tile_h), out_width]
    else:
        shard = [_up(-(-out_rows // n_h), tile_h), _up(-(-out_width // n_w), TILE_WIDTH)]
    return ttnn.MemoryConfig(layout, mem_config.buffer_type, ttnn.ShardSpec(in_spec.grid, shard, in_spec.orientation))


def _allocate_output(shape, dtype, device, mem_config, *, tile_h):
    """Device tensor in Layout::TILE with tile [tile_h, 32] on `mem_config`.

    allocate_tensor_on_device(shape, dtype, layout, device, mem_config) always
    lays out 32x32 tiles, so a tiny output tile goes through a TensorSpec that
    carries `tile` (the TensorSpec constructor takes the MemoryConfig's parts,
    one overload per placement: interleaved, legacy 2-D shard, ND shard).
    """
    if tile_h == 32:
        # CRITICAL: allocate_tensor_on_device takes positional args only.
        return ttnn.allocate_tensor_on_device(shape, dtype, ttnn.TILE_LAYOUT, device, mem_config)
    tile = ttnn.Tile([tile_h, TILE_WIDTH])
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is not None and (mem_config.shard_spec is None or _created_with_nd_shard_spec(mem_config)):
        spec = ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, nd_spec, mem_config.buffer_type, tile)
    elif mem_config.memory_layout in _SHARDED_LAYOUTS:
        spec = ttnn.TensorSpec(
            shape,
            dtype,
            ttnn.TILE_LAYOUT,
            mem_config.memory_layout,
            mem_config.shard_spec,
            mem_config.buffer_type,
            tile,
        )
    else:
        spec = ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, mem_config.buffer_type, tile)
    return ttnn.allocate_tensor_on_device(spec, device)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tilize(
    input_tensor: ttnn.Tensor,
    memory_config: ttnn.MemoryConfig | None = None,
    *,
    dtype: ttnn.DataType | None = None,
    low_l1: bool = False,
    output_padded_shape=None,
    pad_value=None,
    tile: ttnn.Tile | None = None,
) -> ttnn.Tensor:
    """Re-lay `input_tensor` (Layout::ROW_MAJOR) into Layout::TILE; values and logical shape unchanged."""
    _check_well_formed(input_tensor, output_padded_shape=output_padded_shape, pad_value=pad_value, tile=tile)
    validate(
        input_tensor,
        memory_config,
        dtype=dtype,
        low_l1=low_l1,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
    )

    device = input_tensor.device()
    out_mem_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = _resolve_output_dtype(input_tensor, dtype)
    tile_h = _tile_height_of(tile) or 32
    logical_shape = [int(d) for d in input_tensor.shape]
    padding = _resolve_padding(
        logical_shape, output_padded_shape=output_padded_shape, pad_value=pad_value, tile_h=tile_h
    )
    pad = None
    out_shape = logical_shape
    if padding is not None:
        out_shape, expanded = padding
        pad = PadSpec(out_shape, expanded, _fill_bits(pad_value, input_tensor.dtype))
    # The device program writes the whole padded tile grid, so the output is allocated at the
    # padded shape; only the returned tensor's metadata carries the logical shape.
    out_mem_config = _resolve_output_memory_config(input_tensor, out_mem_config, out_shape, tile_h=tile_h)
    output_tensor = _allocate_output(ttnn.Shape(out_shape), out_dtype, device, out_mem_config, tile_h=tile_h)

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        tile_h=tile_h,
        in_tile_h=_input_tile_height(input_tensor),
        low_l1=low_l1,
        pad=pad,
    )
    # Output tensor MUST be last in the list.
    output_tensor = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
    if out_shape == logical_shape:
        return output_tensor
    return _logical_view(output_tensor, logical_shape, out_shape)


def _logical_view(tensor, logical_shape, padded_shape):
    """The same device buffer with logical shape `logical_shape` and padded shape `padded_shape`.

    For a Layout::TILE tensor whose padded last two dims are tile multiples and whose padded
    last dim is unchanged, ttnn.reshape(tensor, logical, padded) resolves to the metadata-only
    view (tt::tt_metal::view): no device program. Checked, never assumed: a copy here would be
    a second dispatch.
    """
    view = ttnn.reshape(tensor, ttnn.Shape(logical_shape), ttnn.Shape(padded_shape))
    if (
        view.buffer_address() != tensor.buffer_address()
        or list(view.shape) != list(logical_shape)
        or list(view.padded_shape) != list(padded_shape)
    ):
        raise RuntimeError(f"tilize: the logical view {logical_shape} / {padded_shape} of the output is not zero-copy")
    return view
