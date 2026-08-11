# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — ProgramDescriptor.

Implements the Blocking Model of `op_design.md`:

  * two axes — `row` (independent, flattened leading dims in 32-row tile-rows)
    and `hidden` (dependent, the reduced last dim in 32-column tiles);
  * the grid is partitioned into `num_row_groups` axis-aligned rectangles of
    `w_group_cols x w_group_rows` cores.  Each rectangle is one reduction group:
    its members own disjoint hidden slices of the SAME rows and combine their
    partial sums-of-squares over the NoC (gather-to-root + multicast-back);
  * `block_row_tiles` (R) is the block extent along `row`, chosen by the closed-
    form L1 residency solve below.  `core_w_tiles` (C) is the block extent along
    `hidden` — the block always spans a core's whole hidden slice, so one block
    is exactly one cross-core combine round.

EVERY block factor / buffer depth / core assignment below is a named parameter,
defined once, with every CB page count and loop bound derived from it.  Nothing
is sized off a whole-op dimension.

PLACEMENT (op_design.md lamp S1). An INTERLEAVED tensor lets `_select_regime`
*choose* the geometry above; a physically SHARDED one has it *supplied* by the
caller's shard spec, which is read off directly (`_ShardView`) rather than
re-chosen — HEIGHT cuts the independent `row` axis so `w_group_size == 1` and the
combine degenerates; WIDTH cuts the dependent `hidden` axis so the whole shard
grid is ONE reduction group; BLOCK cuts both, and one grid row of the shard
rectangle is a group.  The shard is then consumed IN PLACE: a TILE shard's block
CB is pinned zero-copy over the resident L1 buffer (`_sharded_cb`) and the
reader/writer drop their DRAM leg entirely.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"


# --------------------------------------------------------------------------
# Circular buffer indices (semantic names; the number is only a buffer slot)
# --------------------------------------------------------------------------
CB_INPUT_RM = 0
CB_INPUT_TILES = 1
CB_SCALER = 2
CB_WMASK = 3
CB_ZERO_TILE = 4
CB_STAT_SQ = 5
CB_STAT_PARTIAL = 7
CB_STAT_GATHER = 8
CB_STAT_SUM = 9
CB_RSTD_SEND = 10
CB_RSTD = 11
CB_GAMMA_RM = 12
CB_GAMMA_TILES = 13
CB_NORMED = 14
CB_OUTPUT_TILES = 16
CB_OUTPUT_RM = 17

# --------------------------------------------------------------------------
# Semaphores
# --------------------------------------------------------------------------
SEM_GATHER = 0  # members -> root: "my partial is in your gather buffer"
SEM_MCAST_READY = 1  # mcast data-ready flag (mcast_pipe)
SEM_MCAST_CONSUMED = 2  # mcast consumer-ready counter (mcast_pipe)

# --------------------------------------------------------------------------
# Kernel arg-layout contracts. Each mirrors a hardcoded offset in a kernel
# (McastArgs<CT, RT> / TensorAccessorArgs<CT>), which cannot import this module;
# the asserts below pin them so an added arg fails loudly at program build
# instead of silently shifting a peer's args.
# --------------------------------------------------------------------------
READER_ACCESSOR_CT_BASE = 9  # rms_norm_reader.cpp: TensorAccessorArgs<9>
WRITER_MCAST_CT_BASE = 7  # rms_norm_writer.cpp: MCAST_CT_BASE
WRITER_MCAST_RT_BASE = 16  # rms_norm_writer.cpp: MCAST_RT_BASE
_READER_NUM_ARGS = 17  # rms_norm_reader.cpp get_arg_val<uint32_t>(0..16)
_COMPUTE_NUM_ARGS = 6  # rms_norm_compute.cpp get_arg_val<uint32_t>(0..5)

# --------------------------------------------------------------------------
# Blocking / buffer-depth knobs — single source of truth.
# Each is a tunable parameter; every derived quantity below reads it.
# --------------------------------------------------------------------------
INPUT_CB_DEPTH = 2  # reader prefetches block b+1 while compute runs block b
OUTPUT_CB_DEPTH = 2  # writer drains tile-row r while compute produces r+1
RM_CB_DEPTH = 2  # overlaps stick reads/writes with tilize / untilize
L1_RESERVE_BYTES = 131072  # kernel binaries, stack, semaphores, allocator slack
MAX_GATHER_TILES = 64  # cap on block_row_tiles * w_group_size (cb_stat_gather)
# Perf lamp P2 — an upper bound on the cores per reduction group. Maximum
# occupancy is the default first step, but at tensor_row_tiles == 1 the selection
# pushes w_group_size to the whole grid, leaving 3-4 hidden tiles of real work per
# core against a full gather + multicast round. 0 = uncapped.
#
# MEASURED on blackhole_p150b (110-core grid), interleaved bf16 decode shapes,
# device kernel ns, uncapped -> capped at 32 (which admits G <= 22 on this grid):
#   (1,1,32,5120): 17676 -> 11088 ns (1.59x)   (1,1,32,7168): 18624 -> 12165 ns (1.53x)
#   (1,1,32,2304): 12056 ->  9677 ns (1.25x)   (1,1,32,1024):  8927 ->  8672 ns (1.03x)
# Prefill (tensor_row_tiles >> grid) is unaffected: it already selects G = 1..5.
MAX_W_GROUP_SIZE = 32
# Perf lamp P1 — the smallest number of blocks a core's row assignment is cut
# into. `input_cb_depth = 2` only buys read/compute overlap when there is a
# block b+1 for the reader to prefetch while compute runs block b; at
# num_blocks == 1 the DRAM read fully serializes against compute. Raising this
# trades combine rounds (one per block) for that overlap. 1 = coarsest block,
# no forced pipelining.
#
# MEASURED FLAT on blackhole_p150b (device kernel ns, 1 / 2 / 3 / 4 blocks):
#   (1,1,8192,1024) 104674 / 103894 / 106145 / 103801
#   (1,1,8192,2304) 219050 / 218364 / 222682 / 224441
#   (1,1,8192,5120) 425343 / 428822 / 425210 / 422477
#   (1,1,8192,7168) 605501 / 597240 / 595999 / 598579
#   (1,1,  32,7168)  12378 /  12423 /  12256 /  12208
# i.e. these shapes are not read-vs-compute serialization bound — the wall is
# aggregate DRAM bandwidth (the widest prefill case moves 33.5 MB in ~104 us =
# ~330 GB/s) plus the 3-vs-2 tile-row imbalance of 256 tile-rows over 110 cores.
# The knob is KEPT at its byte-identical default: it costs nothing there, and it
# is the lever to turn once a co-binding stage (compute, or the DRAM run length)
# is shortened. Follow-up: re-measure after any change that lowers the DRAM
# floor, and pair it with input_cb_depth 3-4 as op_design.md's lamp P1 suggests.
MIN_PIPELINE_BLOCKS = 1

# Buffer-depth ladder, tried in order by the residency solve. Depth buys overlap,
# so it is spent FIRST and given up LAST: step 0 is the knobs above (the only step
# any interleaved geometry ever needs, so the default path is byte-identical), and
# the later steps exist for a resident shard, which leaves far less CB budget —
# a HEIGHT-sharded core holds the tensor's WHOLE hidden slice (`C =
# tensor_w_tiles`, not a chosen split), and on the ROW_MAJOR legs that slice is
# staged twice more through cb_input_rm / cb_output_rm.
_DEPTH_LADDER = (
    (INPUT_CB_DEPTH, RM_CB_DEPTH),
    (INPUT_CB_DEPTH, 1),
    (1, 1),
)

# Hidden-axis chunking (op_design.md regime R3), the residency knob of LAST resort
# for a resident shard. `w_chunk_tiles` (WC) is the block extent along `hidden` of
# every CB that only STREAMS over that axis — cb_gamma_tiles, cb_normed,
# cb_output_*, cb_input_rm — so their footprint becomes O(WC) instead of O(C).
# The block itself stays whole-resident in cb_input_tiles, which is what keeps the
# input to ONE crossing: R3's cost in the traffic ranking was a second whole-tensor
# READ for the apply pass, and there is none here (the shard is already in L1 and
# the block is never re-fetched).
#
# It is tried only after the depth ladder above has failed at WC == C, so every
# geometry that fits today keeps its byte-identical single-chunk schedule, and the
# COARSEST chunk that fits is taken (block-size fidelity: a finer chunk repays the
# per-chunk phase-boundary reconfig and one extra stat column per chunk for no
# extra work).
#
# `w_chunk_tiles=None` everywhere below means "one chunk" (WC == C) — the value
# every interleaved geometry and every currently-fitting shard keeps.

TILE_HW = 32
FP32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048

# Block-float dtypes: 16 datums share one 8-bit exponent, so a tile is an
# exponent section plus a mantissa section and there is NO per-element byte size
# (`Tensor.element_size()` raises "datum for bfp2, bfp4, bfp8 is invalid"). The
# only consumers of an element size are the ROW_MAJOR stick legs — and a
# block-float tensor has no row-major form (ttnn refuses to build one), so 0 is
# the honest value there rather than a branch at every use site.
_BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)

# L1 per Tensix core, by arch. Queried by name because the Python device object
# does not expose l1_size_per_core().
_L1_SIZE_BY_ARCH = {
    "grayskull": 1048576,
    "wormhole_b0": 1499136,
    "blackhole": 1572864,
}
_L1_SIZE_DEFAULT = 1499136

# The three physical shard placements. A sharded tensor's spec SUPPLIES the block
# geometry that `_select_regime` otherwise chooses (op_design.md lamp S1), and its
# shard is consumed in place — see `_ShardView` / `_sharded_cb`.
_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _div_up(a, b):
    return (a + b - 1) // b


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _elem_bytes(tensor):
    """Bytes per element of `tensor`, or 0 for a block-float dtype.

    See `_BLOCK_FLOAT_DTYPES`: block-float has no element size, and the only
    users of this value are the ROW_MAJOR stick legs, which a block-float tensor
    can never take.
    """
    return 0 if tensor.dtype in _BLOCK_FLOAT_DTYPES else tensor.element_size()


def _dram_alignment():
    """DRAM address alignment in bytes, from the hal (64 on Blackhole).

    A NoC read whose DRAM source address is not aligned to this TRUNCATES the low
    address bits — silently returning a neighbouring slice, with no assert. See the
    gamma leg in the reader: a ROW_MAJOR shard's width granule is the L1 alignment,
    so a core's gamma slice can start at a DRAM offset this does not divide.
    """
    try:
        return int(ttnn._ttnn.device.get_dram_alignment())
    except Exception:  # pragma: no cover - defensive
        return 64


def _l1_cb_budget():
    try:
        arch = str(ttnn.get_arch_name()).lower()
    except Exception:  # pragma: no cover - defensive
        arch = ""
    return _L1_SIZE_BY_ARCH.get(arch, _L1_SIZE_DEFAULT) - L1_RESERVE_BYTES


# ==========================================================================
# Geometry + regime selection
# ==========================================================================


class _Geometry:
    """Alignment-aware tile geometry of the whole tensor. `floor` appears nowhere."""

    def __init__(self, input_tensor, gamma):
        shape = list(input_tensor.shape)
        self.shape = shape
        self.W = shape[-1]
        self.tensor_w_tiles = _div_up(self.W, TILE_HW)
        self.partial_w = self.W % TILE_HW

        self.in_memory_layout = input_tensor.memory_config().memory_layout
        self.is_rm_in = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        if self.is_rm_in:
            # Sticks are contiguous across the leading dims: no per-image pad.
            self.num_sticks = 1
            for d in shape[:-1]:
                self.num_sticks *= d
            self.tensor_row_tiles = _div_up(self.num_sticks, TILE_HW)
        else:
            # A TILE tensor pads EACH image's H to 32 independently.
            images = 1
            for d in shape[:-2]:
                images *= d
            self.num_sticks = 0  # unused on the tiled path
            self.tensor_row_tiles = images * _div_up(shape[-2], TILE_HW)

        self.in_dtype = input_tensor.dtype
        self.in_elem_bytes = _elem_bytes(input_tensor)
        self.in_tile_bytes = ttnn.tile_size(input_tensor.dtype)

        self.has_gamma = gamma is not None
        if self.has_gamma:
            self.gamma_dtype = gamma.dtype
            self.gamma_elem_bytes = _elem_bytes(gamma)
            self.gamma_tile_bytes = ttnn.tile_size(gamma.dtype)
            self.is_rm_gamma = gamma.layout == ttnn.ROW_MAJOR_LAYOUT
        else:
            self.gamma_dtype = None
            self.gamma_elem_bytes = 0
            self.gamma_tile_bytes = 0
            self.is_rm_gamma = False


def _stat_cols(C, WC, has_tail):
    """Stat columns per tile-row of `cb_stat_sq` — ONE per hidden chunk, plus the tail's.

    `reduce_stat_block` already folds a tile-row's columns together, so a chunked
    `sumsq_block` packs each chunk's partial `Σ x²` into its own column instead of
    accumulating through L1. Unchunked this is the Phase 0 expression `1 + has_tail`,
    which the kernel reproduces exactly (`bulk_cols = ceil(c_full / WC)`).
    """
    if WC >= C:
        return 1 + has_tail
    return _div_up(max(C - has_tail, 0), WC) + has_tail


def _cb_specs(
    geo,
    C,
    G,
    R,
    is_rm_out,
    has_tail,
    input_cb_depth=INPUT_CB_DEPTH,
    rm_cb_depth=RM_CB_DEPTH,
    pin_in=False,
    pin_out=False,
    w_chunk_tiles=None,
):
    """THE statement of this op's CB inventory — `(index, num_pages, page_size, format)`.

    Both the L1 residency solve (`_cb_bytes`) and the descriptor's CB list are
    built from this list, so a page-count change lands in exactly ONE place and
    the solve can never drift from what is actually allocated.

    Every `num_pages` here is affine in `R` (either constant or `k·R`), which is
    what lets `_cb_bytes` recover `fixed_bytes` / `per_row_bytes` exactly by
    differencing. Mirrors the CB table in `l1_ledger.md`.

    `input_cb_depth` is a parameter rather than a direct read of the module knob
    only so the descriptor can drop the prefetch buffer when the core's whole row
    assignment is a single block (there is then no block b+1 to prefetch). The
    solve always passes the knob itself, i.e. the conservative value.

    `pin_in` / `pin_out` mark the CBs PINNED zero-copy over a resident L1 shard
    (`_sharded_cb`). A pinned CB costs the CB arena nothing — its bytes are the
    tensor's own shard, which the budget subtracts once — so it is omitted from
    this inventory, which is what keeps the residency solve in agreement with what
    is actually allocated.
    """
    T_in = geo.in_tile_bytes
    # The output CBs carry T_in / the input dtype: create_program_descriptor
    # asserts the output dtype matches the input's, so they are the same format.
    in_dtype = geo.in_dtype
    # WC = the hidden extent of every CB that only STREAMS over that axis. WC == C
    # (one chunk) is the default and makes every expression below the Phase 0 one.
    WC = C if w_chunk_tiles is None else w_chunk_tiles
    # The block width the CHUNKED buffers actually span: chunks are a uniform
    # quantum, so a WC that does not divide C carries `NUM_CHUNKS*WC - C` pad
    # columns — the same pad the ragged hidden split already carries.
    C_pad = _div_up(C, WC) * WC
    nc_max = _stat_cols(C, WC, has_tail)  # cb_stat_sq columns per tile-row

    specs = [
        (CB_SCALER, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        (CB_ZERO_TILE, 1, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_SQ, R * nc_max, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_PARTIAL, R, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_GATHER, R * G, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_SUM, R, FP32_TILE_BYTES, ttnn.float32),
        (CB_RSTD_SEND, R, FP32_TILE_BYTES, ttnn.float32),
        (CB_RSTD, R, FP32_TILE_BYTES, ttnn.float32),
    ]
    if not pin_in:
        # The whole block stays resident here (both passes read it), so this CB is
        # sized on the block width, never on the chunk. On the ROW_MAJOR leg tilize
        # fills it chunk by chunk, hence C_pad.
        specs.append((CB_INPUT_TILES, input_cb_depth * R * (C_pad if geo.is_rm_in else C), T_in, in_dtype))
    if not pin_out:
        # RM output needs the whole CHUNK resident before untilize runs (both are
        # compute-side, so they cannot pipeline); the tiled path streams.
        specs.append((CB_OUTPUT_TILES, (R * WC) if is_rm_out else (OUTPUT_CB_DEPTH * WC), T_in, in_dtype))
    if has_tail:
        specs.append((CB_WMASK, 1, BF16_TILE_BYTES, ttnn.bfloat16))
    if geo.is_rm_in:
        specs.append((CB_INPUT_RM, rm_cb_depth * WC, T_in, in_dtype))
    if is_rm_out:
        specs.append((CB_OUTPUT_RM, rm_cb_depth * WC, T_in, in_dtype))
    if geo.has_gamma:
        specs.append((CB_GAMMA_TILES, WC, geo.gamma_tile_bytes, geo.gamma_dtype))
        specs.append((CB_NORMED, R * WC, T_in, in_dtype))
        if geo.is_rm_gamma and not _gamma_rm_aliases_input_rm(geo, WC >= C):
            specs.append((CB_GAMMA_RM, WC, geo.gamma_tile_bytes, geo.gamma_dtype))
    return specs


def _gamma_rm_aliases_input_rm(geo, unchunked=True):
    """Can the ROW_MAJOR gamma stick be staged in `cb_input_rm` instead of its own CB?

    Reuse pattern 3 (alias disjoint lifetimes, `l1-footprint-discipline.md`):
    `cb_gamma_rm` is filled ONCE before the block loop and dies the moment
    `load_gamma_slice` has tilized it, which is strictly before `cb_input_rm`'s first
    push. Both CBs have the SAME producer (the reader) and the SAME consumer
    (compute's tilize), so the single-producer/single-consumer invariant is
    preserved, and both hold `C` tile-sized pages. Saves `C` tiles — which is what
    lets several wide HEIGHT-sharded ROW_MAJOR geometries fit at all, since a
    HEIGHT shard pins `C = tensor_w_tiles`.

    Gated on identical page FORMAT: one CB index carries one data format, so a
    mixed-precision gamma (bf16 activations x fp32 weights) keeps its own buffer.
    Also gated on the UNCHUNKED schedule: once the hidden axis is chunked, gamma is
    re-fed per chunk from inside the block loop, so the two lifetimes interleave
    instead of being disjoint.
    """
    return (
        unchunked
        and geo.has_gamma
        and geo.is_rm_gamma
        and geo.is_rm_in
        and geo.gamma_dtype == geo.in_dtype
        and geo.gamma_tile_bytes == geo.in_tile_bytes
    )


def _cb_bytes(geo, C, G, is_rm_out, has_tail, depths=_DEPTH_LADDER[0], pin_in=False, pin_out=False, w_chunk_tiles=None):
    """`(fixed_bytes, per_row_bytes)` of the per-core CB footprint at `(C, G)`.

    DERIVED from `_cb_specs` — never restated — by differencing the inventory at
    R = 1 and R = 2. Every page count there is affine in R, so
    `footprint(R) = fixed_bytes + R · per_row_bytes` holds exactly.
    """

    def total(R):
        return sum(
            pages * page_size
            for _, pages, page_size, _ in _cb_specs(
                geo,
                C,
                G,
                R,
                is_rm_out,
                has_tail,
                input_cb_depth=depths[0],
                rm_cb_depth=depths[1],
                pin_in=pin_in,
                pin_out=pin_out,
                w_chunk_tiles=w_chunk_tiles,
            )
        )

    at_1, at_2 = total(1), total(2)
    per_row_bytes = at_2 - at_1
    return at_1 - per_row_bytes, per_row_bytes


def _max_block_row_tiles(
    geo,
    C,
    G,
    core_row_tiles,
    is_rm_out,
    has_tail,
    budget,
    depths=_DEPTH_LADDER[0],
    pin_in=False,
    pin_out=False,
    w_chunk_tiles=None,
):
    """Closed-form L1 residency solve (a single expression, not a search).

    Returns 0 when even R == 1 does not fit.
    """
    fixed_bytes, per_row_bytes = _cb_bytes(
        geo, C, G, is_rm_out, has_tail, depths=depths, pin_in=pin_in, pin_out=pin_out, w_chunk_tiles=w_chunk_tiles
    )
    if fixed_bytes + per_row_bytes > budget:
        return 0
    cap = min(core_row_tiles, max(1, MAX_GATHER_TILES // G))
    if MIN_PIPELINE_BLOCKS > 1:
        # Perf lamp P1: cut the assignment into >= MIN_PIPELINE_BLOCKS blocks so
        # the depth-`input_cb_depth` input CB has a block to prefetch.
        cap = min(cap, max(1, _div_up(core_row_tiles, MIN_PIPELINE_BLOCKS)))
    return max(1, min((budget - fixed_bytes) // per_row_bytes, cap))


def _select_candidates(geo, grid_x, grid_y, is_rm_out, budget, w_group_cap):
    """Every (gc, gr) split that clears the mechanism caps and fits L1.

    Score, in priority order:
      1. active cores        — fill the grid;
      2. -G                  — fewest combine partners (each round is a gather
         barrier + a root reduce + a multicast);
      3. R                   — coarsest block, i.e. fewest combine rounds.

    A `-max_tiles_per_core` term was measured as a second key (prefer the split
    whose BUSIEST core carries the least work, e.g. G=2 x 80 tiles over G=1 x 96
    tiles at (8192, 1024) — both fill all 110 cores). It is NOT used: measured on
    blackhole_p150b it won (1,1,8192,2304) 220420 -> 198307 ns but lost
    (1,1,8192,1024) 102445 -> 123259 ns, i.e. the extra combine round and the
    halved DRAM run length outweigh the balance gain more often than not.
    """
    has_tail = 1 if geo.partial_w != 0 else 0
    out = []
    for gc in _divisors(grid_x):
        for gr in _divisors(grid_y):
            G = gc * gr
            num_groups = (grid_x // gc) * (grid_y // gr)
            if G > geo.tensor_w_tiles:
                continue  # mechanism cap: a core owning zero hidden tiles hangs the gather
            if w_group_cap and G > w_group_cap:
                continue  # perf lamp P2
            C = _div_up(geo.tensor_w_tiles, G)
            active_groups = min(geo.tensor_row_tiles, num_groups)
            core_row_tiles = _div_up(geo.tensor_row_tiles, active_groups)
            R = _max_block_row_tiles(geo, C, G, core_row_tiles, is_rm_out, has_tail, budget)
            if R == 0:
                continue
            score = (active_groups * G, -G, R)
            out.append((score, (gc, gr, C, R)))
    return out


def _select_regime(geo, grid_x, grid_y, is_rm_out, budget):
    """Exact, deterministic regime-selection function (op_design.md).

    Returns (w_group_cols, w_group_rows, core_w_tiles_ceil, block_row_tiles).

    MAX_W_GROUP_SIZE is a PREFERENCE, not a mechanism cap: if no capped candidate
    fits L1 (a hidden dim so wide that residency needs a large group), the search
    is retried uncapped rather than failing.
    """
    candidates = _select_candidates(geo, grid_x, grid_y, is_rm_out, budget, MAX_W_GROUP_SIZE)
    if not candidates and MAX_W_GROUP_SIZE:
        candidates = _select_candidates(geo, grid_x, grid_y, is_rm_out, budget, 0)
    if not candidates:
        raise RuntimeError(
            "rms_norm: no work split fits L1 for shape "
            f"{tuple(geo.shape)} (regime R3, streaming two-pass, is not implemented). "
            "Reduce the hidden dimension or use a larger grid."
        )
    return max(candidates, key=lambda c: c[0])[1]


# ==========================================================================
# Physical shard placement (op_design.md lamp S1)
# ==========================================================================
#
# A sharded input does NOT re-run `_select_regime`: the shard spec already fixes
# every extent the selection function would have chosen, so we READ them off it.
#   HEIGHT ⇒ the shard cuts the INDEPENDENT `row` axis ⇒ w_group_size == 1 and the
#            reduction is local (the combine degenerates to a local copy).
#   WIDTH  ⇒ the shard cuts the DEPENDENT `hidden` axis ⇒ the WHOLE shard grid is
#            ONE reduction group; `core_w_tiles` is the shard width in tiles.
#   BLOCK  ⇒ both ⇒ one grid ROW of the shard rectangle is one reduction group.
#
# The shard itself is consumed IN PLACE: `_sharded_cb` pins cb_input_tiles /
# cb_output_tiles zero-copy over the resident L1 buffer and the reader/writer drop
# their DRAM leg entirely (no TensorAccessor for a core's own shard). On the
# ROW_MAJOR legs the shard's stick stride is the shard width, not the block's
# group-uniform tile-row stride that `tilize`/`untilize` require, so those legs
# re-stride the sticks LOCALLY (core-local L1→L1, still no DRAM crossing) — see
# `LocalShardAccessor` in the reader/writer kernels.


class _ShardView:
    """The caller's shard spec, read as this op's block geometry."""

    def __init__(self, tensor, geo, is_rm):
        mem = tensor.memory_config()
        self.memory_layout = mem.memory_layout
        spec = mem.shard_spec
        if spec is None:
            raise ValueError(
                "rms_norm: a sharded tensor must carry a 2D shard_spec "
                f"(got memory_layout={mem.memory_layout} with none)"
            )
        self.row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
        # Shard index -> core, in EXACTLY the order the buffer assigns shards
        # (tt_metal/impl/buffers/buffer.cpp:271 uses this same call).
        self.cores = [(int(c.x), int(c.y)) for c in ttnn.corerange_to_cores(spec.grid, None, self.row_wise)]
        box = spec.grid.bounding_box()
        self.x0, self.y0 = int(box.start.x), int(box.start.y)
        self.x1, self.y1 = int(box.end.x), int(box.end.y)
        self.nx = self.x1 - self.x0 + 1
        self.ny = self.y1 - self.y0 + 1
        self.shard_h = int(spec.shape[0])  # elements (ROW_MAJOR: sticks)
        self.shard_w = int(spec.shape[1])  # elements
        self.is_rm = is_rm

        # Page/bank facts straight off the buffer, so nothing here re-derives an
        # alignment ttnn already applied.
        probe = ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, tensor)
        self.bank_bytes = int(probe.total_size)
        self.page_bytes = int(probe.format_descriptors[0].page_size)

        if is_rm:
            self.shard_row_tiles = _div_up(self.shard_h, TILE_HW)
        else:
            if self.shard_h % TILE_HW or self.shard_w % TILE_HW:
                raise ValueError(
                    f"rms_norm: a TILE-layout shard must be tile-aligned, got {self.shard_h}x{self.shard_w}"
                )
            self.shard_row_tiles = self.shard_h // TILE_HW
        self.shard_w_tiles = _div_up(self.shard_w, TILE_HW)

    def blocks(self):
        """`(row_block, col_block)` per shard index — read off the scheme."""
        n = len(self.cores)
        if self.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            return [(i, 0) for i in range(n)]
        if self.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return [(0, i) for i in range(n)]
        if n != self.nx * self.ny:
            raise ValueError(
                f"rms_norm: BLOCK_SHARDED needs a rectangular shard grid, got {n} cores in a {self.nx}x{self.ny} box"
            )
        if self.row_wise:
            return [(i // self.nx, i % self.nx) for i in range(n)]
        return [(i % self.ny, i // self.ny) for i in range(n)]

    def matches(self, other):
        return (
            other is not None
            and self.memory_layout == other.memory_layout
            and self.cores == other.cores
            and self.shard_h == other.shard_h
            and self.shard_w == other.shard_w
        )


def _shard_core_infos(geo, sv):
    """Per-shard-core extents. Every field is a function of the spec, not a choice.

    A shard grid does not have to divide the tensor evenly (`auto_shard_config`
    ceil-splits and pads the last shard), so `row_count` / `w_elems` are CLAMPED to
    the tensor: a core whose shard is entirely padding lands at 0 and becomes an
    inactive core (see `_sharded_groups`).
    """
    infos = []
    for i, (row_block, col_block) in enumerate(sv.blocks()):
        if sv.is_rm:
            stick_base = row_block * sv.shard_h
            num_sticks = max(0, min(geo.num_sticks - stick_base, sv.shard_h))
            row_count = _div_up(num_sticks, TILE_HW)
        else:
            row_base = row_block * sv.shard_row_tiles
            num_sticks = 0
            row_count = max(0, min(geo.tensor_row_tiles - row_base, sv.shard_row_tiles))
        w_start_elems = col_block * sv.shard_w
        w_elems = max(0, min(geo.W - w_start_elems, sv.shard_w))
        infos.append(
            {
                "core": sv.cores[i],
                "row_block": row_block,
                "col_block": col_block,
                "row_count": row_count,
                "num_sticks": num_sticks,
                "w_start_elems": w_start_elems,
                "w_elems": w_elems,
                "core_w": _div_up(w_elems, TILE_HW),
            }
        )
    return infos


def _sharded_groups(sv, infos):
    """Reduction groups over the shard grid: `[{members, box_cores, row_*}]`.

    `box_cores` is the mcast RECTANGLE the group's `rstd` is broadcast over, and it
    is a superset of `members` whenever the shard grid is not a rectangle: a WIDTH
    shard of 16 slices on an 11-wide grid is a full row plus a 5-core row, whose
    bounding box holds 6 cores that own no shard. Those cores stay PROGRAM cores
    (identical L1 map, so the broadcast lands in their `cb_rstd` and not in some
    other tensor's L1) but carry no work and never ack — which is why the mcast is
    emitted with an explicit `num_active` instead of the dense fan-out.
    """
    ML = ttnn.TensorMemoryLayout
    live = [i for i in infos if i["row_count"] > 0 and i["w_elems"] > 0]
    if not live:
        raise RuntimeError("rms_norm: the shard spec assigns no valid data to any core")

    def group(members, box_cores):
        head = members[0]
        return {
            "members": members,
            "box_cores": box_cores,
            "row_count": head["row_count"],
            "num_sticks": head["num_sticks"],
            # The sharded legs address the shard's OWN L1 base, so both the global
            # tile-row index and the global stick index are unused (page 0 of a
            # core's slice is the shard's first page).
            "row_start": 0,
            "stick_start": 0,
        }

    if sv.memory_layout == ML.HEIGHT_SHARDED:
        # w_group_size == 1: each core is its own degenerate group, so the mcast
        # rectangle is that single core and no filler core exists.
        return [group([info], [info["core"]]) for info in live]

    if sv.memory_layout == ML.WIDTH_SHARDED:
        box = [(x, y) for y in range(sv.y0, sv.y1 + 1) for x in range(sv.x0, sv.x1 + 1)]
        return [group(live, box)]

    groups = []
    for row_block in range(sv.ny):
        members = [i for i in live if i["row_block"] == row_block]
        if not members:
            continue
        y = sv.y0 + row_block
        groups.append(group(members, [(x, y) for x in range(sv.x0, sv.x1 + 1)]))
    return groups


def _chunking_supported(groups, pin_in, pin_out):
    """May the hidden axis be chunked for this (sharded) geometry?

    Two structural preconditions, both cheap to state and both load-bearing:

    1. The per-core hidden geometry must be UNIFORM across the program. The chunk
       count sets `cb_stat_sq`'s column stride and every chunked CB's push/pop
       quantum, and a CB's capacity must be an exact multiple of that quantum
       (dataflow_api.h:216-221) on every core that shares the L1 map. A HEIGHT
       shard — the case this exists for — gives every core the tensor's whole
       hidden slice, so it is uniform by construction.
    2. Input and output must be BOTH pinned or BOTH staged. A pinned block CB is
       the shard's own row-major-C buffer, which the apply writes at a strided
       offset; a staged one is filled chunk-major by tilize. Mixing the two would
       need the apply to read one layout and write the other in the same pass.
    """
    if pin_in != pin_out:
        return False
    widths = {(m["core_w"], m["w_elems"] % TILE_HW) for g in groups for m in g["members"]}
    return len(widths) == 1


def _sharded_cb(index, tensor, core_ranges):
    """A CB pinned ZERO-COPY over `tensor`'s resident L1 shard.

    Page size, format and bank size all come from the shard spec. `core_ranges` is
    widened to the program's whole active core set so a mcast-box filler core has
    the same L1 map as its group's shard cores (`mcast_pipe.hpp:44-45`).
    """
    desc = ttnn.cb_descriptor_from_sharded_tensor(index, tensor)
    desc.core_ranges = core_ranges
    return desc


# ==========================================================================
# Program descriptor
# ==========================================================================


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=data_format,
                page_size=page_size,
            )
        ],
    )


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _core_box(cores):
    """The single CoreRange spanning `cores` — a reduction group's mcast rectangle.

    `cores` is a rectangle by construction (a `w_group_cols x w_group_rows` tile of
    the grid, a grid row of a BLOCK shard, or the bounding box of a WIDTH shard).
    """
    xs = [x for x, _ in cores]
    ys = [y for _, y in cores]
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))])


def create_program_descriptor(
    input_tensor,
    gamma,
    output_tensor,
    *,
    epsilon,
    compute_kernel_config,
):
    device = input_tensor.device()
    geo = _Geometry(input_tensor, gamma)

    # The whole work split — tile-row numbering, per-core row ranges and the
    # row-major stick range every core drains — is derived ONCE, from the INPUT
    # tensor's layout (a TILE tensor pads each image's H to 32 independently; a
    # ROW_MAJOR one does not, so the two give different tile-row counts for the
    # same shape). The writer replays that mapping, so it is only valid while the
    # output shares the input's layout. The public entry point always allocates
    # the output that way; assert it here so a future caller that passes a
    # differently-laid-out output gets a loud failure instead of a silently
    # unwritten buffer.
    if output_tensor.layout != input_tensor.layout:
        raise ValueError(
            f"rms_norm: output layout ({output_tensor.layout}) must match the input layout "
            f"({input_tensor.layout}) — the per-core row mapping is derived from the input."
        )
    if output_tensor.dtype != input_tensor.dtype:
        raise ValueError(
            f"rms_norm: output dtype ({output_tensor.dtype}) must match the input dtype " f"({input_tensor.dtype})."
        )

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)

    is_rm_out = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    out_elem_bytes = _elem_bytes(output_tensor)

    # ---- placement -------------------------------------------------------
    # INTERLEAVED: `_select_regime` CHOOSES the block geometry.
    # *_SHARDED:    the caller's shard spec SUPPLIES it (op_design.md lamp S1) and
    #               the shard is consumed in place, so the DRAM leg disappears.
    sv_in = _ShardView(input_tensor, geo, geo.is_rm_in) if geo.in_memory_layout in _SHARDED_LAYOUTS else None
    out_memory_layout = output_tensor.memory_config().memory_layout
    sv_out = _ShardView(output_tensor, geo, is_rm_out) if out_memory_layout in _SHARDED_LAYOUTS else None
    if sv_out is not None and not sv_out.matches(sv_in):
        raise ValueError(
            "rms_norm: a sharded output must carry the SAME shard spec as the input — "
            "the per-core block geometry is read off the input's spec."
        )
    # A TILE shard is the block itself, so its CB is PINNED zero-copy over the
    # resident buffer. A ROW_MAJOR shard cannot be: the block CB is the
    # tilize/untilize staging buffer, whose group-uniform tile-row stride is not the
    # shard's stick stride, so that leg keeps an arena CB and re-strides the sticks
    # core-locally (L1 -> L1, still no DRAM crossing).
    pin_in = sv_in is not None and not geo.is_rm_in
    pin_out = sv_out is not None and not is_rm_out
    # The resident shards occupy the same L1 address range in every bank, so they
    # come off the CB budget before the residency solve runs.
    budget = _l1_cb_budget()
    budget -= sv_in.bank_bytes if sv_in is not None else 0
    budget -= sv_out.bank_bytes if sv_out is not None else 0

    # ---- block factors + reduction groups --------------------------------
    # Both paths produce the SAME structure, so everything below (mcast wiring, CBs,
    # per-core args, kernels) is written once:
    #   groups[i] = {members: [{core, core_w, w_start_tiles, w_start_elems, w_elems}],
    #                box_cores: the mcast rectangle, row_start, row_count,
    #                stick_start, num_sticks}
    if sv_in is None:
        w_group_cols, w_group_rows, C, R = _select_regime(geo, grid_x, grid_y, is_rm_out, budget)
        G = w_group_cols * w_group_rows
        num_row_groups = (grid_x // w_group_cols) * (grid_y // w_group_rows)
        active_row_groups = min(geo.tensor_row_tiles, num_row_groups)

        # Hidden split inside a group: `rem` cores take ceil, the rest floor.
        w_floor = geo.tensor_w_tiles // G
        w_rem = geo.tensor_w_tiles % G

        # Row split across the ACTIVE row-groups (every core of a group gets the
        # same row range — the combine is a group-wide barrier).
        rows_per_group = geo.tensor_row_tiles // active_row_groups
        rows_extra = geo.tensor_row_tiles % active_row_groups

        has_tail_global = 1 if geo.partial_w != 0 else 0
        max_row_count = rows_per_group + (1 if rows_extra else 0)
        depths = _DEPTH_LADDER[0]
        # An interleaved geometry never chunks: `_select_regime` still has the
        # w_group_size knob, which shrinks C itself (and chunking a DRAM-fed input
        # would not be free — see the R3 note above).
        W_CHUNK = C

        groups = []
        for gi in range(active_row_groups):
            groups_across = grid_x // w_group_cols
            gx0 = (gi % groups_across) * w_group_cols
            gy0 = (gi // groups_across) * w_group_rows
            cores = [(gx0 + x, gy0 + y) for y in range(w_group_rows) for x in range(w_group_cols)]
            row_start = gi * rows_per_group + min(gi, rows_extra)
            row_count = rows_per_group + (1 if gi < rows_extra else 0)
            stick_start = row_start * TILE_HW
            members = []
            for slot in range(G):
                if slot < w_rem:
                    core_w, w_start = w_floor + 1, slot * (w_floor + 1)
                else:
                    core_w, w_start = w_floor, w_rem * (w_floor + 1) + (slot - w_rem) * w_floor
                members.append(
                    {
                        "core": cores[slot],
                        "core_w": core_w,
                        "w_start_tiles": w_start,
                        "w_start_elems": w_start * TILE_HW,
                        "w_elems": min(core_w * TILE_HW, geo.W - w_start * TILE_HW),
                    }
                )
            groups.append(
                {
                    "members": members,
                    "box_cores": cores,
                    "row_start": row_start,
                    "row_count": row_count,
                    "stick_start": stick_start,
                    "num_sticks": max(0, min(geo.num_sticks, (row_start + row_count) * TILE_HW) - stick_start),
                }
            )
    else:
        C = sv_in.shard_w_tiles
        groups = _sharded_groups(sv_in, _shard_core_infos(geo, sv_in))
        group_sizes = {len(g["members"]) for g in groups}
        if len(group_sizes) != 1:
            raise ValueError(
                "rms_norm: every reduction group of a shard spec must hold the same number of cores "
                f"(the mcast L1 map is group-uniform), got {sorted(group_sizes)}"
            )
        G = group_sizes.pop()
        for g in groups:
            for m in g["members"]:
                m["w_start_tiles"] = m["w_start_elems"] // TILE_HW
        # A ROW_MAJOR shard's width granule is the L1 alignment, not a tile, so a
        # WIDTH/BLOCK shard can start a core's hidden slice mid-tile. gamma read as
        # TILES cannot express that (a tile read has no sub-tile column offset); a
        # ROW_MAJOR gamma can, via its byte offset, and is what the golden suite
        # pairs with a ROW_MAJOR sharded input.
        if geo.has_gamma and not geo.is_rm_gamma:
            bad = next((m for g in groups for m in g["members"] if m["w_start_elems"] % TILE_HW), None)
            if bad is not None:
                raise ValueError(
                    "rms_norm: a TILE-layout gamma needs tile-aligned per-core hidden slices, but this "
                    f"shard spec starts one at element {bad['w_start_elems']}. Pass a ROW_MAJOR gamma."
                )
        has_tail_global = 1 if any(m["w_elems"] % TILE_HW for g in groups for m in g["members"]) else 0
        max_row_count = max(g["row_count"] for g in groups)
        # The shard spec pins G and C, so `R`, the BUFFER DEPTHS and the HIDDEN
        # CHUNK are the only residency knobs left. Walk the depth ladder first at
        # WC == C (so every geometry that fits today keeps its byte-identical
        # single-chunk schedule), and only then chunk the hidden axis.
        R, depths, W_CHUNK = 0, _DEPTH_LADDER[-1], C
        for candidate_depths in _DEPTH_LADDER:
            R = _max_block_row_tiles(
                geo,
                C,
                G,
                max_row_count,
                is_rm_out,
                has_tail_global,
                budget,
                depths=candidate_depths,
                pin_in=pin_in,
                pin_out=pin_out,
            )
            if R:
                depths = candidate_depths
                break
        if R == 0 and _chunking_supported(groups, pin_in, pin_out):
            # HIDDEN-AXIS CHUNKING (op_design.md R3). Take the COARSEST chunk that
            # fits, at full buffer depth; only if no chunk fits at that depth does
            # the depth ladder come back into play.
            for candidate_depths in _DEPTH_LADDER:
                for wc in range(C - 1, 0, -1):
                    R = _max_block_row_tiles(
                        geo,
                        C,
                        G,
                        max_row_count,
                        is_rm_out,
                        has_tail_global,
                        budget,
                        depths=candidate_depths,
                        pin_in=pin_in,
                        pin_out=pin_out,
                        w_chunk_tiles=wc,
                    )
                    if R:
                        depths, W_CHUNK = candidate_depths, wc
                        break
                if R:
                    break
        if R == 0:
            raise RuntimeError(
                f"rms_norm: the per-core CB working set for shard spec {sv_in.shard_h}x{sv_in.shard_w} on "
                f"{len(sv_in.cores)} cores does not fit L1 (C={C} hidden tiles, w_group_size={G}), even with the "
                f"hidden axis chunked to a single tile; the shard spec pins both, and the two resident shards "
                f"alone take {(sv_in.bank_bytes + (sv_out.bank_bytes if sv_out is not None else 0))} bytes of the "
                f"{_l1_cb_budget()}-byte budget."
            )

    all_cores = []
    for g in groups:
        # Root = the group's LAST member (matches the interleaved convention). It
        # always sits inside the mcast rectangle, so the broadcast loops back into
        # its own cb_rstd (INCLUDE_SRC) and cb_rstd keeps ONE producer everywhere.
        g["root"] = ttnn.CoreCoord(*g["members"][-1]["core"])
        g["rect"] = _core_box(g["box_cores"])
        all_cores.extend(g["box_cores"])

    all_core_ranges = _core_range_set(all_cores)

    # mcast wiring — one Mcast2D per reduction group, all adopting the same
    # semaphore ids so the CT block is uniform across groups. `num_active = G - 1`
    # is passed EXPLICITLY rather than left dense: when a shard grid is not a
    # rectangle its bounding box holds filler cores that receive the broadcast but
    # never ack, so the sender must wait for the member count, not the fan-out.
    mcast_cfg = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1,  # the writer kernel runs on NoC1
        sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
    )
    mcasts = [ttnn.Mcast2D(device, g["rect"], g["root"], mcast_cfg, G - 1) for g in groups]
    mcast_ct = list(mcasts[0].compile_time_args())
    for mc_other in mcasts[1:]:
        assert list(mc_other.compile_time_args()) == mcast_ct, (
            "rms_norm: reduction groups disagree on the mcast compile-time block; the writer kernel "
            "carries ONE CT block for the whole grid"
        )

    # ---- circular buffers ------------------------------------------------
    # `C` (= cb_w_tiles) is the UNIFORM per-core block width along `hidden`: the
    # ceil of the ragged split. Every CB is allocated on the FULL active core set
    # at that width, for two reasons:
    #   1. the L1 map must be identical on every group member — cb_rstd is a
    #      multicast destination and cb_stat_gather a gather destination, both
    #      addressed by a peer's LOCAL pointer;
    #   2. a CB's page capacity must be an exact multiple of its push/pop
    #      quantum (dataflow_api.h:216-221 — "no other wrap is legal"), and a
    #      ragged per-core quantum does not divide a uniform capacity.
    # A core whose VALID slice is narrower (core_w < C) still moves C pages per
    # tile-row; the trailing pad tiles are never read by the statistics phases
    # (which walk `core_w` columns at row stride C) and their apply-phase output
    # is never written to DRAM.
    #
    # A SHARDED input/output in TILE layout replaces the corresponding block CB with
    # a zero-copy CB pinned over the resident shard: `total_size` is then the shard's
    # own bank size (>= R*C pages), so the reader's per-block push walks straight
    # through the shard in tile-row-major order and never wraps.

    # `input_cb_depth` buys ONE thing: the reader prefetching block b+1 while
    # compute runs block b. When the busiest core's whole row assignment is a
    # single block there is no block b+1, so the second buffer is dead L1 —
    # allocate depth 1 there instead of reserving a prefetch slot that provably
    # cannot be used. The residency SOLVE above deliberately still used the full
    # knob (the conservative value), so this only ever lowers the footprint and
    # never changes the chosen (G, C, R) geometry.
    input_cb_depth = depths[0] if max_row_count > R else 1
    # Single source of truth for the hidden chunking: the CB page counts below, the
    # kernels' CT arg and the gamma-alias decision all read THIS value.
    alias_gamma_rm = _gamma_rm_aliases_input_rm(geo, W_CHUNK >= C)

    cbs = [
        _cb(index, all_core_ranges, num_pages, page_size, data_format)
        for index, num_pages, page_size, data_format in _cb_specs(
            geo,
            C,
            G,
            R,
            is_rm_out,
            has_tail_global,
            input_cb_depth=input_cb_depth,
            rm_cb_depth=depths[1],
            pin_in=pin_in,
            pin_out=pin_out,
            w_chunk_tiles=W_CHUNK,
        )
    ]
    if pin_in:
        cbs.append(_sharded_cb(CB_INPUT_TILES, input_tensor, all_core_ranges))
    if pin_out:
        cbs.append(_sharded_cb(CB_OUTPUT_TILES, output_tensor, all_core_ranges))

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=all_core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=all_core_ranges, initial_value=0),
    ]

    # ---- per-core runtime args ------------------------------------------
    # Collected per core first; the RuntimeArgs objects are built per kernel
    # core-range below so a descriptor only ever carries args for its own cores.
    reader_args = {}
    writer_args = {}
    compute_args = {}

    # ROW_MAJOR shard legs re-stride from/into the shard's own L1 at its stick
    # stride (the buffer's aligned page size). 0 on every other leg.
    shard_row_bytes_in = sv_in.page_bytes if (sv_in is not None and geo.is_rm_in) else 0
    shard_row_bytes_out = sv_out.page_bytes if (sv_out is not None and is_rm_out) else 0
    dram_align = _dram_alignment()

    for gi, g in enumerate(groups):
        mc = mcasts[gi]
        root_virtual = device.worker_core_from_logical_core(g["root"])
        row_start = g["row_start"]
        row_count = g["row_count"]
        num_blocks = _div_up(row_count, R)
        last_block_row_tiles = row_count - (num_blocks - 1) * R
        by_core = {m["core"]: (slot, m) for slot, m in enumerate(g["members"])}

        for cx, cy in g["box_cores"]:
            entry = by_core.get((cx, cy))
            if entry is None:
                # A mcast-box FILLER core: it holds no shard (the shard grid is not
                # a rectangle, so its bounding box is wider), hence no work. It stays
                # a program core purely so the group's L1 map is uniform and the rstd
                # broadcast lands in a reserved cb_rstd instead of unowned L1. It
                # never acks — which is why `num_active` above is the member count.
                # `num_blocks = 0` returns from all three kernels immediately.
                reader_args[(cx, cy)] = [0] * _READER_NUM_ARGS
                writer_args[(cx, cy)] = [0] * WRITER_MCAST_RT_BASE + list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))
                compute_args[(cx, cy)] = [0] * _COMPUTE_NUM_ARGS
                continue

            slot, m = entry
            core_w = m["core_w"]
            w_start = m["w_start_tiles"]
            w_elems = m["w_elems"]
            # Per-core, not per-tensor: a ROW_MAJOR WIDTH/BLOCK shard can make EVERY
            # core's hidden slice ragged, not just the one owning the tensor's last
            # tile. On every other path this reduces to exactly the old
            # `owns_last_w_tile && W % 32` condition.
            core_partial_w = w_elems % TILE_HW
            has_tail = 1 if core_partial_w else 0
            is_root = 1 if (cx, cy) == (int(g["root"].x), int(g["root"].y)) else 0

            # This core's hidden slice, in bytes, on the input/output row. A sharded
            # leg addresses the shard's OWN base, where the slice starts at byte 0.
            in_slice_bytes = w_elems * geo.in_elem_bytes
            out_slice_bytes = w_elems * out_elem_bytes
            gamma_slice_bytes = w_elems * geo.gamma_elem_bytes
            # A DRAM read truncates its source address to the DRAM alignment, so the
            # ROW_MAJOR gamma leg is handed an ALIGNED read offset plus the leading
            # bytes to drop. `lead` is 0 on every tile-aligned slice (all interleaved
            # geometries and all TILE shards), where the read is byte-identical.
            gamma_byte_offset = m["w_start_elems"] * geo.gamma_elem_bytes
            gamma_read_offset = (gamma_byte_offset // dram_align) * dram_align
            gamma_lead_bytes = gamma_byte_offset - gamma_read_offset
            in_byte_offset = 0 if sv_in is not None else m["w_start_elems"] * geo.in_elem_bytes
            out_byte_offset = 0 if sv_out is not None else m["w_start_elems"] * out_elem_bytes

            num_sticks = g["num_sticks"] if geo.is_rm_in else 0
            stick_start = g["stick_start"] if geo.is_rm_in else 0

            reader_own_args = [
                input_tensor.buffer_address(),
                gamma.buffer_address() if geo.has_gamma else 0,
                row_start,
                num_blocks,
                R,
                last_block_row_tiles,
                w_start,
                core_w,
                core_partial_w,
                num_sticks,
                stick_start,
                in_slice_bytes,
                in_byte_offset,
                gamma_slice_bytes,
                gamma_read_offset,
                gamma_lead_bytes,
                shard_row_bytes_in,
            ]
            assert len(reader_own_args) == _READER_NUM_ARGS, (
                f"rms_norm: reader runtime-arg layout drifted ({len(reader_own_args)} args); "
                f"update kernels/rms_norm_reader.cpp and _READER_NUM_ARGS here"
            )
            reader_args[(cx, cy)] = reader_own_args

            writer_own_args = [
                output_tensor.buffer_address(),
                row_start,
                num_blocks,
                R,
                last_block_row_tiles,
                w_start,
                core_w,
                slot,
                is_root,
                int(root_virtual.x),
                int(root_virtual.y),
                num_sticks,
                stick_start,
                out_slice_bytes,
                out_byte_offset,
                shard_row_bytes_out,
            ]
            # The writer kernel reads the mcast runtime args from a hardcoded
            # offset (`MCAST_RT_BASE`), so the count of the writer's own args is
            # a contract with the kernel. Pin it here: appending an arg above
            # without bumping the kernel constant would otherwise silently feed
            # McastArgs garbage.
            assert len(writer_own_args) == WRITER_MCAST_RT_BASE, (
                f"rms_norm: writer runtime-arg layout drifted ({len(writer_own_args)} own args); "
                f"update MCAST_RT_BASE in kernels/rms_norm_writer.cpp and WRITER_MCAST_RT_BASE here"
            )
            writer_args[(cx, cy)] = writer_own_args + list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))

            compute_own_args = [
                num_blocks,
                R,
                last_block_row_tiles,
                core_w,
                has_tail,
                is_root,
            ]
            assert len(compute_own_args) == _COMPUTE_NUM_ARGS
            compute_args[(cx, cy)] = compute_own_args

    # ---- kernels ---------------------------------------------------------
    # `cb_w_tiles` (= C) is a COMPILE-TIME template parameter of tilize/untilize
    # and the stride of every block CB, so it is uniform across the whole active
    # grid. The ragged hidden remainder rides as the per-core RUNTIME arg
    # `core_w` (the valid slice width, core_w <= C).
    inv_w_bits = _f32_bits(1.0 / float(geo.W))
    eps_bits = _f32_bits(epsilon)

    def _rt(per_core):
        rt = ttnn.RuntimeArgs()
        for (cx, cy), vals in per_core.items():
            rt[cx][cy] = vals
        return rt

    reader_ct = [
        C,
        geo.tensor_w_tiles,
        1 if geo.is_rm_in else 0,
        1 if geo.has_gamma else 0,
        1 if geo.is_rm_gamma else 0,
        has_tail_global,
        1 if sv_in is not None else 0,
        1 if alias_gamma_rm else 0,
        W_CHUNK,
    ]
    # Same contract as the writer's runtime args: the reader reads its
    # TensorAccessorArgs from a hardcoded CT offset.
    assert len(reader_ct) == READER_ACCESSOR_CT_BASE, (
        f"rms_norm: reader compile-time-arg layout drifted ({len(reader_ct)} own args); "
        f"update TensorAccessorArgs<...> in kernels/rms_norm_reader.cpp and READER_ACCESSOR_CT_BASE here"
    )
    # The accessor family is emitted for BOTH placements. A resident shard is read
    # from its own L1 and the kernel never calls this accessor on the sharded leg,
    # but the CT block must stay one shape (the kernel declares the family
    # unconditionally and its `if constexpr` legs still type-check).
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if geo.has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct = [
        C,
        geo.tensor_w_tiles,
        1 if is_rm_out else 0,
        G,
        SEM_GATHER,
        1 if sv_out is not None else 0,
        W_CHUNK,
    ]
    assert len(writer_ct) == WRITER_MCAST_CT_BASE, (
        f"rms_norm: writer compile-time-arg layout drifted ({len(writer_ct)} own args); "
        f"update MCAST_CT_BASE in kernels/rms_norm_writer.cpp and WRITER_MCAST_CT_BASE here"
    )
    writer_ct.extend(mcast_ct)
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [
        C,
        G,
        1 if geo.has_gamma else 0,
        1 if geo.is_rm_in else 0,
        1 if is_rm_out else 0,
        inv_w_bits,
        eps_bits,
        1 if geo.is_rm_gamma else 0,
        1 if alias_gamma_rm else 0,
        W_CHUNK,
    ]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
            core_ranges=all_core_ranges,
            compile_time_args=reader_ct,
            runtime_args=_rt(reader_args),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
            core_ranges=all_core_ranges,
            compile_time_args=writer_ct,
            runtime_args=_rt(writer_args),
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
            core_ranges=all_core_ranges,
            compile_time_args=compute_ct,
            runtime_args=_rt(compute_args),
            config=compute_kernel_config,
        ),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
