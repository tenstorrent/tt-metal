# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ROW_MAJOR -> TILE layout re-lay, one native dispatch.

Registry-model op file: the four declarations (INPUT_TAGGERS, SUPPORTED,
EXCLUSIONS, validate) plus the public entry point.

INVALID is deliberately NOT declared here — it is a test-harness concept living
in `eval/golden_tests/tilize/feature_spec.py`.

The twelve INPUT_TAGGERS project off the golden SCENARIO DICT (`inputs[0]` is
the whole dict, not a shape tuple) as specified in feature_spec.py's
`Expected INPUT_TAGGERS` block; that is a deliberate deviation from
eval/op_template.py shared with the other pure-layout ops. Because scenario-dict
taggers cannot be replayed against a live call, validate() re-derives the same
axis values from the real tensors and kwargs — the same rules
`eval/golden_tests/tilize/axes.py:classify_call` implements.
"""

from __future__ import annotations

import math

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

from .tilize_program_descriptor import LEGAL_TILE_HEIGHTS, TILE_WIDTH, create_program_descriptor

# Dimensionless ratio at which one tile-grid axis "dominates" the other. NOT a
# core count: the classification of a shape must be identical on every arch,
# while HOW MANY pieces to cut is read from the device at runtime.
DOMINANT = 16


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS  — each reads inputs[0], the whole scenario dict
# ---------------------------------------------------------------------------

_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _side_is_sharded(side):
    return side.get("kind") == "sharded"


def _side_is_nd(side):
    """An nd (NdShardSpec) side is a sharded side whose `scheme` is None."""
    return _side_is_sharded(side) and side.get("scheme") is None


def _buffer_word(buffer_type):
    return "dram" if buffer_type == ttnn.BufferType.DRAM else "l1"


def _out_tile_height(scenario):
    """The OUTPUT tile height the scenario asks for (32 unless `tile=` moves it)."""
    return int(scenario.get("tile_height", 32))


def tag_low_l1(inputs, axes):
    return bool(inputs[0].get("low_l1", False))


def tag_shard_api(inputs, axes):
    scenario = inputs[0]
    side_in, side_out = scenario["in"], scenario["out"]
    if not _side_is_sharded(side_in) and not _side_is_sharded(side_out):
        return "none"
    if _side_is_nd(side_in) or _side_is_nd(side_out):
        return "nd"
    return "legacy_2d"


def tag_out_scheme(inputs, axes):
    side_out = inputs[0]["out"]
    if not _side_is_sharded(side_out):
        return "interleaved"
    return "nd" if _side_is_nd(side_out) else side_out["scheme"]


def tag_buffer(inputs, axes):
    scenario = inputs[0]
    return f"{_buffer_word(scenario['in']['buffer'])}_to_{_buffer_word(scenario['out']['buffer'])}"


def tag_rank(inputs, axes):
    return int(len(inputs[0]["input_shape"]))


def tag_orientation(inputs, axes):
    """ "none" when both sides are interleaved — there is no shard to orient.
    Otherwise the sharded side's orientation (output side wins when both are)."""
    scenario = inputs[0]
    side_in, side_out = scenario["in"], scenario["out"]
    if _side_is_sharded(side_out):
        return side_out["orientation"]
    if _side_is_sharded(side_in):
        return side_in["orientation"]
    return "none"


def tag_pad_mode(inputs, axes):
    return inputs[0].get("pad_mode", "none")


def tag_pad_value(inputs, axes):
    """The SIGN bucket of the fill, not the number."""
    scenario = inputs[0]
    if scenario.get("pad_mode", "none") == "none":
        return "none"
    value = scenario.get("pad_value")
    if value is None or value == 0:
        return "zero"
    return "positive" if value > 0 else "negative"


def tag_tile_height(inputs, axes):
    return _out_tile_height(inputs[0])


def tag_in_tile_height(inputs, axes):
    """ "none" IS row-major: a ROW_MAJOR input has no tile geometry of its own."""
    return inputs[0].get("in_tile_height", "none")


def tag_alignment(inputs, axes):
    """Four-way alignment of the last two dims. H is measured against the
    OUTPUT TILE HEIGHT (a tiny-tile call redefines "aligned" on that axis);
    W against the literal 32, a tile's width always being 32. Rank < 2 is
    hw_non_aligned — both tile dims are synthesized by the pad."""
    scenario = inputs[0]
    shape = scenario["input_shape"]
    if len(shape) < 2:
        return "hw_non_aligned"
    tile_h = _out_tile_height(scenario)
    h_ok = int(shape[-2]) % tile_h == 0
    w_ok = int(shape[-1]) % TILE_WIDTH == 0
    if h_ok and w_ok:
        return "tile_aligned"
    if h_ok:
        return "w_non_aligned"
    if w_ok:
        return "h_non_aligned"
    return "hw_non_aligned"


def tag_tile_grid(inputs, axes):
    """Which axis of the OUTPUT tile grid carries the parallelism.

    Holds NO grid size: DOMINANT is a dimensionless ratio between R and C, so
    the classification is identical on Wormhole, Blackhole and Quasar.
    """
    scenario = inputs[0]
    shape = scenario["input_shape"]
    if len(shape) < 2:
        return "single_tile"
    tile_h = _out_tile_height(scenario)
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)
    rows = leading * math.ceil(int(shape[-2]) / tile_h)
    cols = math.ceil(int(shape[-1]) / TILE_WIDTH)
    return _classify_tile_grid(rows, cols)


def _classify_tile_grid(rows, cols):
    if rows == 1 and cols == 1:
        return "single_tile"
    if cols >= DOMINANT * rows:
        return "short_wide"
    if rows >= DOMINANT * cols:
        return "tall_narrow"
    if rows * cols < DOMINANT * DOMINANT:
        return "small"
    return "square_large"


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
    "alignment": tag_alignment,
    "tile_grid": tag_tile_grid,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED — Phase 0
# ---------------------------------------------------------------------------
#
# `tile_grid` is declared COMPLETE, and that is a claim about the work split,
# not about correctness (every value of this axis is trivially correct under any
# scheme). The claim: the block grid is `num_row_groups x num_w_chunks` and BOTH
# indices go into `split_work_to_cores`, so the distribution reaches the full
# device grid on every geometry — including `short_wide`, where R == 1 and a
# row-only split would collapse onto one core. See op_design.md's worked table
# and tilize_program_descriptor.derive_plan.
#
# The five absent-argument sentinels (shard_api / orientation / pad_mode /
# pad_value / in_tile_height) carry "none", which is ALWAYS legal — four of the
# Phase 0 values below ARE sentinels.
SUPPORTED = {
    # --- the numerical cartesian (Refinement 5) -----------------------------
    # Both axes are DECLARATIONS ABOUT THE SAME ONE PATH. Nothing here forks the
    # kernels: the input CB's `data_format` is the input tensor's dtype, the
    # output CB's is the output tensor's, and the value-preserving cast happens
    # at PACK time inside the same `compute_kernel_lib::tilize` call that the
    # bfloat16 diagonal has always used. What the dtype pair selects is the
    # COMPUTE CONFIG the datapath needs to stay value-preserving — see
    # `is_lossless_fp32_relay` / `requires_fp32_dest_acc` /
    # `needs_srcb_alu_format_repair` in tilize_program_descriptor.py, which are
    # the single source of all three decisions.
    #
    # `bfloat8_b` / `bfloat4_b` are absent from the INPUT axis on purpose and
    # not for lack of trying: block float has no ROW_MAJOR form (16 values share
    # an exponent, so there is no stick to read), and the tilize helper asserts
    # `!is_block_float_format(unpack_src_format[input])` for exactly that
    # reason. They are legal OUTPUTS, where the packer does the compression.
    #
    # `fp8_e4m3` is INPUT-only for the mirror reason (no TILE form) and is
    # arch-gated to Blackhole; it is listed because the path is dtype-generic —
    # no kernel and no host derivation names a format — but see the changelog:
    # it could not be exercised on the Wormhole box this landed on, where the
    # golden suite skips all of its cells before `validate()` is reached.
    "dtype": [ttnn.bfloat16, ttnn.float32, ttnn.fp8_e4m3, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8],
    "output_dtype": [
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.uint32,
        ttnn.int32,
        ttnn.uint16,
        ttnn.uint8,
    ],
    # `low_l1` is not a regime — it is a value of the `block_width_tiles` knob
    # (`W_CAP = min(W_FIT, LOW_L1_WIDTH_CAP)`), so the kernels, the CBs and the
    # block operations are byte-identical at both settings. Promoted from
    # [False] by the verification pass on measured evidence: both settings are
    # bit-identical on `[1,1,32,8192]` (the `low_l1_forcing_width` geometry,
    # C=256) and neither OOMs, because `LOW_L1_WIDTH_CAP` is a host constant
    # independent of every tensor dimension. See l1_ledger.md's total, in which
    # no tensor dimension appears at either setting.
    "low_l1": [False, True],
    # A shard fixes the core assignment and the per-core extent, so the block
    # grid is READ off the shard spec instead of solved, and the sharded side's
    # CB is placed on the shard buffer (zero-copy) rather than re-read through a
    # TensorAccessor. tilize has no dependent axis — an output tile depends only
    # on its own column slice of its own tile_h sticks — so a shard IS a block
    # and there is no cross-core combine. Both APIs are the same partition seen
    # through two descriptors; a live tensor normalizes an ND spec down to the
    # equivalent legacy 2-D one whenever it has one, which is why `nd_in_legacy_out`
    # ends up fully native. See tilize_program_descriptor.shard_partition.
    "shard_api": ["none", "legacy_2d", "nd"],
    "out_scheme": [
        "interleaved",
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        "nd",
    ],
    # All four interleaved buffer transitions. Placement is a `TensorAccessor`
    # concern on both legs and the kernels never name a buffer type; promoted
    # from [dram_to_dram] by the verification pass after all three additional
    # transitions came back bit-exact.
    "buffer": ["dram_to_dram", "dram_to_l1", "l1_to_l1", "l1_to_dram"],
    # Rank is not a geometry branch at or above 2: `derive_plan` folds
    # `shape[:-2]` into R generically and the reader indexes sticks linearly, so
    # 2/3/5/6 exercise no new code path and all came back bit-exact. Promoted
    # from [4] by the verification pass. Ranks 0 and 1 joined in Refinement 2:
    # they have no tile dims of their own, so the pad SYNTHESIZES both (rank 0 ->
    # one [32,32] tile, rank 1 -> [32, W]) and they are reachable only with a
    # padding argument. Nothing special-cases them — a TILE TensorSpec's default
    # alignment is already rank 2, so the padded shape comes out [32,32] /
    # [32,W] on its own, and `derive_plan` left-pads the input's logical shape
    # to 2 so H=1 / W=1 fall out of the same pad arithmetic as any other tail.
    "rank": [0, 1, 2, 3, 4, 5, 6],
    # The orientation only re-linearizes shard -> core; the partition is read
    # through `corerange_to_cores(grid, n, row_wise=(orientation == ROW_MAJOR))`,
    # which is the same enumeration the buffer itself places shards with, so
    # COL_MAJOR needs no separate path.
    "orientation": ["none", ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
    # Padding is `grid2d_padded` in op_design.md, and it is ADDITIVE on the block
    # that already exists: the block grid, the core assignment, the CBs and the
    # compute call are byte-identical to the unpadded path, and only the reader's
    # `load_block` gains a fill. `auto` derives the target by rounding the last
    # two dims up to the tile; `explicit` takes `output_padded_shape` and may
    # exceed that round, in which case whole pad TILES are produced from the fill
    # alone. `pad_value`'s three sign buckets are one code path — the fill is
    # encoded host-side into the input dtype's bit pattern at the element width
    # (`pad_fill_word`), so a negative fill is a two's-complement bit_cast that
    # cannot truncate. See tilize_reader.cpp's PADDED INPUT block for the two
    # separate pad regions (the W tail and the fully padded row).
    "pad_mode": ["none", "auto", "explicit"],
    "pad_value": ["none", "zero", "positive", "negative"],
    # The W tail and the H tail are separate reader arithmetic, so they are
    # separate axis values: the W tail is an in-place fill of the end of a row
    # that has data, the H tail is a whole row sourced from `cb_pad_row`. The H
    # tail additionally breaks the contiguous-stick-run invariant
    # `read_sticks_for_tilize` is built on (source rows restart at each image
    # boundary), which is why the padded reader segments its block per image.
    "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned", "hw_non_aligned"],
    # Tiny tiles (Refinement 4). `tile_h` is a plan quantity everywhere already
    # -- `in_page_bytes = tile_h*32*elem`, `rows_per_image = ceil(H/tile_h)`, the
    # `TileDescriptor` on both CBs, and the reader's stick count -- so a sub-32
    # output tile turns the knob and changes no structure. The one behavioural
    # difference is that `can_use_fast_tilize` requires 32x32 output tiles
    # (`tilize_helpers.inl:77`), so a tiny tile takes the regular
    # `tilize_init`/`tilize_block` path, which is per-tile through DEST and
    # therefore has no width cap of its own. The list IS `LEGAL_TILE_HEIGHTS`
    # (the `_check_request` gate's own source), so the two cannot drift.
    "tile_height": list(LEGAL_TILE_HEIGHTS),
    # Retile (Refinement 4): a TILE input re-laid at another tile height. "none"
    # IS the ROW_MAJOR sentinel and stays legal. The added values put the reader
    # on a genuinely distinct block operation — `retile_block` walks FACES, not
    # sticks, because the source's pages are whole tiles — and that block
    # operation removes the compute stage entirely: a re-tile is a byte re-lay
    # between two tiled layouts, so `retile_copy_unit`'s runs go from the source
    # tile's faces straight into the destination tile's faces over the NoC. One
    # DRAM crossing each way, which is the minimum; the untilize-and-retilize
    # round trip op_design.md ranks `rejected` appears nowhere. The block grid,
    # the core assignment, the CBs and the writer are all unchanged.
    #
    # The value 32 is in the list because `tile=` must be HONORED on a TILE
    # input, so `in_tile_height=32 -> tile_height=32` (and every other
    # equal-height pair) is a legal identity re-lay, not a no-op to elide —
    # `retile_copy_unit` degenerates to "the whole tile" there and the walk
    # becomes a page copy.
    "in_tile_height": ["none"] + list(LEGAL_TILE_HEIGHTS),
    "tile_grid": ["single_tile", "small", "tall_narrow", "short_wide", "square_large"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS — cells inside cartesian(SUPPORTED) refused for now
# ---------------------------------------------------------------------------
#
# Both groups are RETILE crossings, and both are excluded for a stated
# structural reason rather than for lack of trying:
#
#   * retile x sharded — `retile_block` addresses the source by INTERLEAVED TILE
#     page index (`tile_row * C + tile_col`). A native (zero-copy) CB over a
#     resident TILE shard would need the shard's own page map on BOTH sides of
#     the face walk, and the alternative — reading a core's own shard back
#     through a TensorAccessor — is the interleaved path wearing a sharded hat,
#     which this op does not do anywhere else. Left excluded rather than
#     half-wired: retile is arch-gated to Blackhole (the LLK tiny-tile gate the
#     golden suite skips on), so a native sharded retile cannot be verified on
#     the box this landed on.
#   * retile x padding — the fill would have to be written into output FACES the
#     face walk never sources, which is a second, differently-shaped fill from
#     the row-major one `cb_pad_row` serves.
#
# Neither crossing is reached by any `tile_geometry_retile` golden case (all six
# are unpadded interleaved DRAM), so no cell moves from pass to xfail here.
#   * retile x dtype CAST (added by Refinement 5) — same root cause as the two
#     above, one level up: the re-tile removes the compute stage ENTIRELY (the
#     reader assembles the output tile's bytes out of the input tiles' faces
#     over the NoC), and a `dtype=` cast is a PACK-TIME conversion. With no
#     packer in the pipeline there is nothing to convert with, so the no-cast
#     diagonal is the whole of what a byte re-lay can express. `derive_plan`
#     already asserts this rather than emitting wrong bytes; the exclusion is
#     what turns that assert into the registry's own refusal.
#
# A THIRD crossing was expected here and is NOT excluded, because the mechanism
# turned out to be repairable rather than structural. Recorded because the
# repair is the least obvious line in the op:
#
#   * `uint8` was broken on Wormhole B0 (every output datum zero) by an LLK
#     defect. `_llk_math_hw_configure_`
#     (tt_llk_wormhole_b0/llk_lib/llk_math_common.h) writes srcA and srcB's ALU
#     format fields as ONE word under the union of their two 4-bit masks and
#     without `masked_data_format()`. `DataFormat::UInt8` is 30, so bit 4 of the
#     srcA value spills into srcB's low bit: srcA lands correct (14 = Int8),
#     srcB lands 15. UInt8 is the ONLY format in this op's matrix that both
#     spills and reaches SrcA/SrcB — UInt32 spills too, but
#     `_llk_unpack_tilize_init_` routes UInt32/Int32 straight to DEST, so its
#     corrupt srcB is never read, which is why every other integer width is
#     bit-exact. The UInt8 datacopy MOP is ELWADD (it READS the zero-filled
#     srcB), so the mistyped field zeroes every output datum.
#
#     `needs_srcb_alu_format_repair` + one `reconfig_data_format_srcb` call in
#     the compute kernel re-write that one field under a mask that drops the
#     spill bit, and uint8 comes back BIT-EXACT. So uint8 is supported, not
#     excluded. Blackhole does not have the defect at all (its
#     `_llk_math_hw_configure_` never programs the Src format fields), and the
#     repair is a no-op there, so the same code is correct on both.
#   * block-float OUTPUT x `tile_height == 16` (added by Refinement 5) — the one
#     cell of the tile_height x output_dtype cross that does not work, and it is
#     a WORMHOLE B0 LLK gap at one specific tile geometry rather than anything
#     this op chooses. `Tile` sets `partial_face = (tile_h < 32)` and
#     `face_shape = {min(tile_h,16), 16}`, so `tile_h == 16` is the ONLY height
#     that is `partial_face` while still having a FULL 16-row face. `llk_pack.h`
#     branches on `partial_face && IS_BFP_FORMAT(pack_dst_format)` into a MOP
#     written for sub-16-row faces ("addr_mod_0 will increment by 15") with
#     `PACKCNT = 1` instead of `num_faces`, so at a full-height face the second
#     face and its shared exponents are never packed. MEASURED: PCC collapses to
#     ~0.01 at `tile_height=16` for both bfp targets and is 0.99997 / 0.984 at
#     every other height (32, 8, 4, 2, 1), on four different shapes —
#     probes/probe_038.py. Nothing on the host side reaches that MOP: the
#     geometry IS the output tile the caller asked for.
#   * `uint16` / `uint8` INPUT x `pad_value=negative` (added by Refinement 5) —
#     an unsigned dtype has no negative domain, so "the pad region holds exactly
#     the fill value" has no satisfiable reading. The op CAN write the fill: it
#     encodes it as a two's-complement bit_cast at the element width
#     (`pad_fill_word`), which is the only thing an N-bit unsigned datum can
#     carry. What is unsatisfiable is the CONTRACT: a uint16 datum widened to a
#     signed comparison type is always >= 0, so `65536 - N` can never read back
#     as `-N`; and at 8 bits the expectation cannot even be BUILT (torch refuses
#     `F.pad(uint8_tensor, value=-3)` — "value cannot be converted to type
#     uint8_t without overflow").
#
#     `uint32` is deliberately NOT excluded, and the asymmetry is the point
#     rather than an oversight: at 32 bits the reference comparison reinterprets
#     at the SAME width, so the two's-complement bits the op writes are read back
#     as the negative value asked for and the cell is verifiably correct. The
#     refusal is scoped to where the op cannot be shown right, not to a whole
#     signedness family on principle.
#
#   * `rank == 0` x block-float OUTPUT (added by Refinement 5) — a rank-0 input
#     has ONE logical element, and a one-element tensor has no correlation to
#     measure. The reference comparison says so itself: `get_atol_rtol_pcc`
#     falls back from PCC to `torch.allclose(..., atol=1e-4)` at `numel() == 1`,
#     and block float's quantization step is ~1e-2 — two orders larger. The cell
#     is therefore unmeasurable rather than wrong (the value IS correct to
#     within the format: measured max_abs 0.0039 into bfp8, 0.0117 into bfp4),
#     and NO implementation can pass it. Rank 1 is unaffected: it carries a whole
#     row of elements, so the PCC is real and the same pair passes there.
_RETILE_HEIGHTS = list(LEGAL_TILE_HEIGHTS)
_NARROW_UNSIGNED = [ttnn.uint16, ttnn.uint8]
_BFP_OUTPUTS = [ttnn.bfloat8_b, ttnn.bfloat4_b]
_PARTIAL_FULL_FACE_HEIGHT = 16
_CAST_PAIRS = [(d, o) for d in SUPPORTED["dtype"] for o in SUPPORTED["output_dtype"] if d != o]
EXCLUSIONS = (
    [{"in_tile_height": h, "shard_api": api} for h in _RETILE_HEIGHTS for api in ("legacy_2d", "nd")]
    + [{"in_tile_height": h, "pad_mode": mode} for h in _RETILE_HEIGHTS for mode in ("auto", "explicit")]
    + [{"in_tile_height": h, "dtype": d, "output_dtype": o} for h in _RETILE_HEIGHTS for (d, o) in _CAST_PAIRS]
    + [{"tile_height": _PARTIAL_FULL_FACE_HEIGHT, "output_dtype": o} for o in _BFP_OUTPUTS]
    + [{"dtype": d, "pad_value": "negative"} for d in _NARROW_UNSIGNED]
    + [{"rank": 0, "output_dtype": o} for o in _BFP_OUTPUTS]
)


# ---------------------------------------------------------------------------
# 3b. PROPERTIES — non-axis capabilities
# ---------------------------------------------------------------------------
PROPERTIES = {
    # The block grid is 2-D and both indices are core-assignment indices; the
    # core count comes from device.compute_with_storage_grid_size() at runtime.
    "multi_core": {"value": True, "source": "declared"},
    # Per-core L1 = INPUT_DEPTH_ROWS*W*tb_in + OUTPUT_DEPTH_BATCHES*wrpb*W*tb_out
    # with W <= W_FIT, a constant of (dtype, tile_h, device). No tensor
    # dimension appears — see l1_ledger.md.
    "bounded_cb": {"value": True, "source": "declared"},
}


# ---------------------------------------------------------------------------
# Malformed-request checks (ValueError / RuntimeError) — NOT support refusals
# ---------------------------------------------------------------------------


def _tile_shape(tile):
    shape = getattr(tile, "tile_shape", None) or getattr(tile, "shape", None)
    return (int(shape[0]), int(shape[1]))


# The checks split into two groups by ONE question: is the request malformed
# whatever the op eventually supports, or only once the relevant axis value is
# supported at all?
#
#   * UNCONDITIONAL (`_check_request`) — wrong tensor placement, a layout the op
#     will never take, a tile geometry the hardware does not have, a TILE input
#     with nothing to re-tile to. These run AHEAD of the support gate, because
#     no future refinement makes them legal.
#   * SUPPORT-CONDITIONAL (`_check_pad_target`, `_check_alignment_request`) —
#     both are statements about the PADDING contract, and padding is an axis
#     (`pad_mode`) the registry gates. While `pad_mode`/`alignment` sit outside
#     SUPPORTED the honest refusal is the registry's `UnsupportedAxisValue`
#     ("this op does not do padding yet"), not a ValueError about the argument's
#     shape; the golden harness decorates those cells xfail(raises=
#     NotImplementedError) off exactly that reasoning. So they run AFTER the
#     per-axis loop, and re-arm as real ValueErrors the moment the padding
#     refinement lands. Note `UnsupportedAxisValue` is a `NotImplementedError`
#     and therefore also a `RuntimeError`, so a caller (or an acceptance test)
#     catching `(ValueError, RuntimeError)` still sees the refusal either way.


def _check_request(input_tensor, *, tile):
    """UNCONDITIONALLY malformed REQUESTS — illegal under every future
    SUPPORTED, so they run ahead of the per-axis support loop."""
    if not ttnn.is_tensor_storage_on_device(input_tensor):
        raise ValueError("tilize: input_tensor must be on device")

    if input_tensor.layout not in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
        raise ValueError(f"tilize: input layout must be ROW_MAJOR_LAYOUT or TILE_LAYOUT, got {input_tensor.layout}")

    if input_tensor.layout == ttnn.TILE_LAYOUT and tile is None:
        raise ValueError(
            "tilize: a TILE_LAYOUT input needs an explicit `tile=` to re-tile to — there is nothing to re-tile to otherwise"
        )

    if tile is not None:
        tile_h, tile_w = _tile_shape(tile)
        if tile_w != TILE_WIDTH:
            raise ValueError(f"tilize: `tile` width must be {TILE_WIDTH}, got {tile_w}")
        if tile_h not in LEGAL_TILE_HEIGHTS:
            raise ValueError(f"tilize: `tile` height must be a power-of-two fraction of 32, got {tile_h}")


def _check_pad_target(input_tensor, *, output_padded_shape, tile_h):
    """The explicit pad target must cover the input and be a whole number of
    tiles. Support-conditional: only reachable once `pad_mode="explicit"` is in
    SUPPORTED."""
    if output_padded_shape is None:
        return

    shape = [int(d) for d in list(input_tensor.shape)]
    target = [int(d) for d in list(output_padded_shape)]

    # A rank-0 or rank-1 input has no tile dims of its own — the pad SYNTHESIZES
    # them (rank 0 -> [H, W], rank 1 -> [H, W] with the input's W), which is why
    # feature_spec.TARGET["rank"] lists 0 and 1 as pad-only ranks and
    # tag_alignment reports `hw_non_aligned` there. A rank-2 target against such
    # an input is therefore well-formed, not a rank mismatch.
    if len(shape) < 2 and len(target) == 2:
        shape = [1] * (2 - len(shape)) + shape

    if len(target) != len(shape):
        raise ValueError(f"tilize: output_padded_shape rank {len(target)} != input rank {len(shape)}")
    for i, (t, s) in enumerate(zip(target, shape)):
        if t < s:
            raise ValueError(f"tilize: output_padded_shape[{i}]={t} is smaller than the input's {s}")

    # A TILE tensor's physical extent IS a whole number of tiles, so a target
    # that is not one names a shape the output cannot have. Malformed request,
    # not a support gap.
    if target[-2] % tile_h or target[-1] % TILE_WIDTH:
        raise ValueError(
            f"tilize: output_padded_shape's last two dims {target[-2:]} are not a whole number of "
            f"{tile_h}x{TILE_WIDTH} tiles"
        )


def _check_alignment_request(axes, *, pad_requested):
    """Padding is opt-in: with no padding argument, a non-tile-aligned input is
    refused rather than silently padded. A malformed REQUEST, not a support
    refusal — hence ValueError and not UnsupportedAxisValue. Support-conditional
    for the same reason as `_check_pad_target`: while `alignment` is
    tile-aligned-only the registry has already refused the cell."""
    if not pad_requested and axes["alignment"] != "tile_aligned":
        raise ValueError(
            "tilize: input's last two dims are not tile-aligned "
            f"({axes['alignment']}) and no padding argument was given; "
            "pass pad_value= or output_padded_shape= to opt into padding"
        )


# ---------------------------------------------------------------------------
# Runtime axis derivation (the live-call analogue of INPUT_TAGGERS)
# ---------------------------------------------------------------------------


# A freshly built nd MemoryConfig reports this layout; a LIVE tensor built from
# one reports the equivalent legacy layout instead (see `_spec_of`).
_ND_LAYOUT = getattr(ttnn.TensorMemoryLayout, "ND_SHARDED", None)


def _spec_of(mem_config):
    """(is_sharded, is_nd, out_scheme, orientation) for one side of the call.

    The nd-vs-legacy_2d discriminator is NOT "does it have an nd_shard_spec".
    Measured on device (`probes/probe_003.py` - `probe_005.py`): a live tensor's `memory_config()`
    populates BOTH `shard_spec` and `nd_shard_spec` whichever API allocated it,
    so "has an nd spec" tags every sharded call `nd` and the declared
    `tag_shard_api` (which reads the scenario's `scheme`) disagrees on every
    legacy_2d row. What does discriminate, for a freshly built config AND a
    live one: an nd config carries an nd spec and NO legacy 2-D spec, and
    reports `ND_SHARDED` rather than a HEIGHT/WIDTH/BLOCK layout.
    """
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    shard_spec = getattr(mem_config, "shard_spec", None)
    layout = mem_config.memory_layout
    if nd_spec is not None and (shard_spec is None or (_ND_LAYOUT is not None and layout == _ND_LAYOUT)):
        return True, True, "nd", nd_spec.orientation
    if layout in _SHARDED_LAYOUTS:
        return (
            True,
            False,
            layout,
            shard_spec.orientation if shard_spec is not None else "none",
        )
    return False, False, "interleaved", "none"


def _in_tile_height(input_tensor):
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return None
    tile = getattr(input_tensor, "tile", None)
    return _tile_shape(tile)[0] if tile is not None else 32


def _axes_from_call(input_tensor, memory_config, dtype, low_l1, output_padded_shape, pad_value, tile):
    in_cfg = input_tensor.memory_config()
    out_cfg = memory_config if memory_config is not None else in_cfg
    in_sharded, in_nd, _in_scheme, in_orientation = _spec_of(in_cfg)
    out_sharded, out_nd, out_scheme, out_orientation = _spec_of(out_cfg)

    if not in_sharded and not out_sharded:
        shard_api = "none"
    elif in_nd or out_nd:
        shard_api = "nd"
    else:
        shard_api = "legacy_2d"

    if output_padded_shape is not None:
        pad_mode = "explicit"
    elif pad_value is not None:
        pad_mode = "auto"
    else:
        pad_mode = "none"

    if pad_mode == "none":
        pad_bucket = "none"
    elif pad_value is None or pad_value == 0:
        pad_bucket = "zero"
    else:
        pad_bucket = "positive" if pad_value > 0 else "negative"

    # The OUTPUT tile height: `tile=` when given, else the op keeps the input's
    # geometry (its own tile for a TILE input, 32 for ROW_MAJOR).
    out_tile_h = (_tile_shape(tile)[0] if tile is not None else None) or _in_tile_height(input_tensor) or 32

    shape = list(input_tensor.shape)
    scenario = {
        "input_shape": shape,
        "in": {"kind": "interleaved" if not in_sharded else "sharded"},
        "out": {"kind": "interleaved" if not out_sharded else "sharded"},
        "tile_height": out_tile_h,
    }

    return {
        "dtype": input_tensor.dtype,
        "output_dtype": dtype if dtype is not None else input_tensor.dtype,
        "low_l1": bool(low_l1),
        "shard_api": shard_api,
        "out_scheme": out_scheme,
        "buffer": f"{_buffer_word(in_cfg.buffer_type)}_to_{_buffer_word(out_cfg.buffer_type)}",
        "rank": len(shape),
        "orientation": (out_orientation if out_sharded else in_orientation if in_sharded else "none"),
        "pad_mode": pad_mode,
        "pad_value": pad_bucket,
        # tag_alignment / tag_tile_grid reuse the tagger bodies verbatim so the
        # runtime rule and the declared rule cannot drift.
        "alignment": tag_alignment((scenario,), None),
        "tile_height": out_tile_h,
        "in_tile_height": _in_tile_height(input_tensor) or "none",
        "tile_grid": tag_tile_grid((scenario,), None),
    }


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def validate(
    input_tensor,
    memory_config=None,
    *,
    dtype=None,
    low_l1=False,
    output_padded_shape=None,
    pad_value=None,
    tile=None,
    compute_kernel_config=None,
):
    """Unconditionally-malformed requests (ValueError), then the registry
    support gate (UnsupportedAxisValue / ExcludedCell), then the
    support-conditional padding checks — see the comment above
    `_check_request` for why the padding pair sits on the far side of the
    gate."""
    _check_request(input_tensor, tile=tile)

    axes = _axes_from_call(input_tensor, memory_config, dtype, low_l1, output_padded_shape, pad_value, tile)

    # 1. SUPPORTED — per-axis
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"tilize: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")

    # 2. EXCLUSIONS — cell-level inside SUPPORTED
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"tilize: unsupported combination (refinement candidate): {exc}")

    # 3. Support-conditional malformed requests — the padding contract, which
    #    only becomes the caller's mistake once padding itself is supported.
    _check_pad_target(input_tensor, output_padded_shape=output_padded_shape, tile_h=axes["tile_height"])
    _check_alignment_request(axes, pad_requested=(axes["pad_mode"] != "none"))

    return axes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _auto_padded_shape(logical_shape, tile_h):
    """The target `pad_mode="auto"` implies: the last two dims rounded UP to the
    tile, everything else untouched.

    Rank 0 and 1 have no tile dims to round — the pad SYNTHESIZES them, which is
    the same rank promotion a TILE `TensorSpec`'s own (rank-2) default alignment
    performs, so `auto` at those ranks is exactly the default spec.
    """
    shape = [int(d) for d in list(logical_shape)]
    shape = [1] * (2 - len(shape)) + shape
    shape[-2] = math.ceil(shape[-2] / tile_h) * tile_h
    shape[-1] = math.ceil(shape[-1] / TILE_WIDTH) * TILE_WIDTH
    return shape


_SHARD_FAMILY_DERIVATION = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: "height_sharded",
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: "width_sharded",
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: "block_sharded",
}


def _output_tensor_spec(logical_shape, out_dtype, out_memory_config, out_tile, padded_shape=None, core_grid=None):
    """TensorSpec for the output at ANY placement.

    Three constructors, one per placement family: an ND config carries an
    `NdShardSpec` and NO legacy 2-D spec, so passing its (None) `shard_spec` to
    the sharded overload is a `bad optional access` — the ND overload is the one
    that takes it. Interleaved goes through the plain overload for the same
    reason in reverse.

    `padded_shape` is the fourth case and the only one that is NOT derivable from
    the logical shape: an explicit pad target may exceed the tile round
    (`[1,1,32,50] -> [1,1,32,128]`), and a spec's default alignment caps the
    padded shape AT that round. `TensorSpec.with_padded_shape` states it
    instead, for every placement at once (it takes the MemoryConfig whole), and
    degenerates to the matching default when the target IS the round — which is
    why it is used only where it has to be, keeping every already-covered call
    on the constructor it was verified with.
    """
    shape = ttnn.Shape(list(logical_shape))

    # A sharded `memory_config` may name only the FAMILY and leave the shard
    # shape to the op ("give me a width-sharded output, you pick the cut") —
    # `ttnn.MemoryConfig(WIDTH_SHARDED, L1)` with no ShardSpec. There is nothing
    # to read the block grid off then, and no shard spec to hand a TensorSpec
    # constructor, so the canonical derivation is applied: `TensorSpec`'s own
    # `height_sharded` / `width_sharded` / `block_sharded`, which cut the
    # PADDED 2-D view into one shard per core over the compute grid. Deriving it
    # off the padded spec is what makes the shard's height the padded height,
    # which is the shape the caller of a padded call is asking for.
    derivation = None
    if (
        out_memory_config.is_sharded()
        and out_memory_config.shard_spec is None
        and out_memory_config.nd_shard_spec is None
    ):
        derivation = _SHARD_FAMILY_DERIVATION.get(out_memory_config.memory_layout)
        if derivation is None:
            raise ValueError(
                f"tilize: memory_config names {out_memory_config.memory_layout} but carries no shard spec, "
                "and there is no canonical cut for that layout to derive one from"
            )
        if core_grid is None:
            raise ValueError("tilize: a shard-spec-less memory_config needs a core grid to derive the cut from")
        unsharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, out_memory_config.buffer_type)
        base = _output_tensor_spec(logical_shape, out_dtype, unsharded, out_tile, padded_shape)
        return getattr(base, derivation)(core_grid, ttnn.ShardOrientation.ROW_MAJOR)

    if padded_shape is not None:
        return ttnn.TensorSpec.with_padded_shape(
            shape, ttnn.Shape(list(padded_shape)), out_dtype, ttnn.TILE_LAYOUT, out_memory_config, out_tile
        )
    if not out_memory_config.is_sharded():
        return ttnn.TensorSpec(shape, out_dtype, ttnn.TILE_LAYOUT, out_memory_config.buffer_type, out_tile)
    if out_memory_config.shard_spec is None:
        return ttnn.TensorSpec(
            shape, out_dtype, ttnn.TILE_LAYOUT, out_memory_config.nd_shard_spec, out_memory_config.buffer_type, out_tile
        )
    return ttnn.TensorSpec(
        shape,
        out_dtype,
        ttnn.TILE_LAYOUT,
        out_memory_config.memory_layout,
        out_memory_config.shard_spec,
        out_memory_config.buffer_type,
        out_tile,
    )


def tilize(
    input_tensor: ttnn.Tensor,
    memory_config: "ttnn.MemoryConfig | None" = None,
    *,
    dtype: "ttnn.DataType | None" = None,
    low_l1: bool = False,
    output_padded_shape=None,
    pad_value=None,
    tile: "ttnn.Tile | None" = None,
    compute_kernel_config: "ttnn.DeviceComputeKernelConfig | None" = None,
) -> ttnn.Tensor:
    """Re-lay `input_tensor` from ROW_MAJOR into TILE layout on device.

    Values and logical positions are preserved; only the addresses move. The
    output carries the input's logical shape unchanged, the input's dtype unless
    `dtype=` is given, the input's placement unless `memory_config=` is given,
    and the tile geometry `tile=` names (32x32 otherwise).

    `compute_kernel_config` exposes `math_fidelity` / `math_approx_mode` /
    `dst_full_sync_en` / `fp32_dest_acc_en`. Passing nothing reproduces the
    op's own defaults exactly. The one field that is not a free choice is
    `fp32_dest_acc_en`: the op ORs it with what the dtype pair REQUIRES and
    never turns it off, because tilize is value-preserving and a 16-bit DEST at
    an fp32 or 8-bit-integer pair returns wrong bytes rather than a cheaper
    approximation. See `requires_fp32_dest_acc`.
    """
    validate(
        input_tensor,
        memory_config,
        dtype=dtype,
        low_l1=low_l1,
        output_padded_shape=output_padded_shape,
        pad_value=pad_value,
        tile=tile,
        compute_kernel_config=compute_kernel_config,
    )

    device = input_tensor.device()
    out_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()
    out_dtype = dtype if dtype is not None else input_tensor.dtype
    out_tile = tile if tile is not None else ttnn.Tile([32, TILE_WIDTH])
    out_tile_h = _tile_shape(out_tile)[0]

    # The output's LOGICAL shape is the input's — a padded call grows ONLY the
    # padded shape. Promoting the logical shape is the named bug, and it is what
    # `to_torch(out) == x` (the unpadded oracle) catches.
    #
    # The padded shape is stated explicitly ONLY when the request needs it: an
    # `output_padded_shape` that exceeds the tile round is unreachable from
    # (logical shape, tile) alone. `pad_mode="auto"` and an `explicit` target at
    # exactly the round are both the default spec's own answer, so they keep the
    # Phase 0 / Refinement 1 constructors verbatim.
    padded_shape = None
    if output_padded_shape is not None:
        target = [int(d) for d in list(output_padded_shape)]
        if target != _auto_padded_shape(input_tensor.shape, out_tile_h):
            padded_shape = target
    grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        _output_tensor_spec(input_tensor.shape, out_dtype, out_memory_config, out_tile, padded_shape, core_grid),
        device,
    )

    program_descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        low_l1=low_l1,
        pad_value=pad_value,
        compute_kernel_config=compute_kernel_config,
    )
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
