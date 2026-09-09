# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, work split).

Implements the ``grid2d_full_width`` / ``grid2d_width_chunked`` regimes of
``op_design.md``: the output tile grid ``R x C`` is blocked on BOTH axes and the
linearized block id ``row_group * num_w_chunks + w_chunk`` is what
``split_work_to_cores`` hands to the cores. ``num_w_chunks == 1`` is the
full-width parameterization; nothing else distinguishes the two regimes.

Every block knob below is a named host constant or a derived quantity with a
single source. CB page counts, loop trip counts and grid sizing are computed
FROM those knobs — never from a whole-op dimension and never as a literal.

``grid2d_sharded`` (Refinement 1) is the SAME schedule with the block grid read
off a shard spec instead of solved: ``num_row_groups`` / ``num_w_chunks`` /
``block_width_tiles`` come from the shard's own geometry, the shard's linear index
IS the block id the three kernels already derive ``(row_group, w_chunk)`` from, and
each core is handed exactly the block(s) whose bytes are resident in its own L1
(``start_block_id`` = the core's index in the shard's core order, ``block_stride`` =
the number of cores — which is round-robin for an ND spec and the identity for a
legacy 2-D one). The sharded side's CB is then **zero-copy over the shard buffer**
(``ttnn.cb_descriptor_from_sharded_tensor``) — a core's own shard is never re-read
through a ``TensorAccessor``. The accessor keeps the interleaved leg and the
genuinely non-local cross-spec leg.

Deviation from op_design.md (recorded here and in l1_ledger.md), one item:

  * ``block_width_tiles`` is constrained to be a DIVISOR of ``C`` instead of
    ``ceil(C / num_w_chunks_target)``, which removes ``block_width_tail_tiles``
    (every w-chunk is exactly ``block_width_tiles`` wide).
    Reason — a HARD mechanism cap the design's ragged tail violates: both CB
    endpoints refuse to wrap mid-transfer. ``llk_push_tiles``
    ``LLK_ASSERT(remaining >= num_words)`` (``llk_io_pack.h``) and
    ``cb_pop_front``'s ``ASSERT(fifo_rd_ptr <= fifo_limit)``
    (``dataflow_api.h:269``, commented "consumer always reads from contiguous
    memory, it cannot wrap") both require the CB's page count to be an exact
    multiple of every push/pop quantum used on that core. A core CAN be handed
    one full-width block and one tail block (``split_work_to_cores`` ranges are
    contiguous over a linearization that crosses w-chunk boundaries), so a
    per-core mix of two quanta is reachable and would need the capacity to be a
    multiple of ``lcm(block_width_tiles, tail)``. Making the extent a divisor of
    ``C`` costs at most a coarser chunk count and reproduces the design's whole
    worked table (``[1,1,32,16384]`` -> 8, ``[1,1,1024,1024]`` -> 16,
    ``[1,1,2048,2048]`` -> 64, ...) except ``[1,1,1,50304]`` (12 x 131 chunks
    rather than 25 x 63). The knob stays a live tunable at its coarsest correct
    value; only the tail vanishes.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names; the numeric slot is just the buffer index) ---
CB_INPUT_ROWS = 0  # reader -> compute: row-major sub-block, one page per tile
CB_OUTPUT_TILES = 1  # compute -> writer: whole output tile pages
# reader-local scratch: ONE block row pre-filled with the pad value, so a whole
# pad row costs one DM-engine transfer instead of a RISC store loop. Allocated
# only on the padded path (`plan.pad_active`).
CB_PAD_ROW = 2

# ---------------------------------------------------------------------------
# Named block knobs — the single source of each. None is a tensor dimension.
# ---------------------------------------------------------------------------
TILE_WIDTH = 32
# Output tile-page writes to keep in flight behind one write barrier. Sets
# `write_rows_per_barrier`, which is inert (== 1) once the block is >= this wide.
# MEASURED, not assumed: op_design.md's "Write batch depth" lamp asked whether 8
# sits past the knee, and on `[1,1,16384,32]` (C == 1, so this constant IS the
# whole knob) it does. Median device kernel ns over 3 fresh-cache runs at 64/64
# cores: wb=1 24430 (the one-write-per-barrier trap), wb=2 22519, wb=4 20723,
# wb=8 22568, wb=16 22403. 4 is the plateau — 1.18x over the trap and 1.09x over
# 8 — which is where the `double_buffer` catalog entry put it. Inert on every
# block already >= 4 wide (the perf-focus shape's bw=8 gives wrpb=1 either way),
# and it costs LESS L1 than 8. See
# tests/.../tilize/test_tilize_lever_write_batch.py for the harness.
WRITE_BATCH_MIN_TILES = 4
# can_use_fast_tilize requires block_width_tiles < 256 (tilize_helpers.inl:77).
FAST_TILIZE_WIDTH_CAP = 255
# low_l1=True cap on the column extent. A host constant, independent of EVERY
# tensor dimension — that independence is the low_l1 contract, structurally.
LOW_L1_WIDTH_CAP = 4
# cb_input_rows depth, in tile-rows of the block (reader/compute overlap).
# MEASURED and KEPT AT ITS DEFAULT, with the flat result recorded rather than
# hidden: op_design.md's "Overlap (input depth)" lamp asked for {2,3,4} on the
# two wide-block shapes, and the answer is that the knob does not move either of
# them. Device kernel ns, 2 fresh-cache runs each, all at 64/64 cores:
#   [1,1,32,32768]  (bw=16, 1 KiB reads)  d1 25358/25582  d2 25863/24653
#                                         d3 27056/25478  d4 26285/25509
#   [1,1,2048,2048] (bw=64, 4 KiB reads)  d1 91377/95444  d2 92394/91515
#                                         d3 92520/93470  d4 93191/93780
# Depth 1 being flat too is the informative part: it says the READS are the wall
# on these shapes, so the overlap slot is not what is missing. `[1,1,2048,2048]`
# moves 16 MiB in ~91.5 us = ~183 GB/s, ~64% of this part's peak on a
# simultaneous read+write stream — near the practical DRAM roofline, which no CB
# depth can move. So 2 is kept: it is the smallest value that overlaps at all,
# it is byte-identical in output, and it costs the least L1 of the values that do
# (d3 would cost 640 KB on square_large vs 512 KB). It stays a LIVE knob, not an
# inlined constant, because what would have to change for depth > 2 to pay is a
# shape where reads are not the wall — the fp32 refinement (which drops off the
# fast-tilize path, making compute the expensive stage) is the obvious candidate.
# Harness: tests/.../tilize/test_tilize_lever_input_depth.py.
INPUT_DEPTH_ROWS = 2
# cb_output_tiles depth, in WRITE BATCHES (one batch = write_rows_per_barrier
# tile-rows). Depth 2 buys the writer a full batch in flight while compute
# fills the next one, and keeps the capacity an exact multiple of the batch so
# a full batch never straddles the FIFO wrap.
OUTPUT_DEPTH_BATCHES = 2
# Pipeline waves a core should own, where ONE WAVE is one tile-row of the block —
# the reader's push quantum, the compute helper's per-call unit and the writer's
# minimum wait. A core overlaps its DRAM reads against its DRAM writes only if it
# owns MORE THAN ONE wave; with exactly one it reads, then computes, then writes,
# strictly in series, and with every core in lockstep the device alternates a
# read-only phase with a write-only phase instead of sustaining both.
#
# The wave count is fully determined by the column cut, because the tensor holds
# `R * num_w_chunks` tile-rows in total:
#     waves_per_core = R * num_w_chunks / num_cores
# so `w_chunks_for_occupancy` (fill the grid) and `w_chunks_for_waves` (fill the
# pipe) are the SAME expression, this constant being the factor between them —
# which is why there is one knob here and not two. At 1 it is byte-identical to
# the occupancy-only cut.
#
# This constant is the CAP on how deep the pipe is allowed to get; the value
# actually taken is the deepest one whose read transaction still clears
# `MIN_BLOCK_ROW_BYTES` below, because a wave is bought by HALVING the block
# width. Past 4 the pipe is full and the extra waves only shrink the read.
# MEASURED on the `attention:` LOOSE_CASE `[1,1,32,16384]` (R = 1, so a core
# owns exactly one wave at 1) and on `[1,1,2048,2048]` — see
# tests/.../tilize/test_tilize_lever_pipeline_waves.py and the ablation in
# changelog.md Refinement 3.
PIPELINE_WAVES_PER_CORE = 4
# The read-transaction floor that stops the trade above. A wave is bought by
# HALVING the block width, so it is only worth buying while the resulting read
# stays large enough that the NoC/DRAM per-transaction cost is still amortized.
# MEASURED across five geometries (device kernel ns, 64/64 cores throughout,
# medians where the numbers were close; harness
# tests/.../tilize/test_tilize_lever_pipeline_waves.py). Read size per wave
# setting, then the wall:
#   [1,1,32,16384]  512/256/128/64 B  -> 13759 / 13620 / 15204 / 26947
#   [1,1,32,32768] 1024/512/256/128 B -> 25267 / 24077 / 23757 / 28964
#   [1,1,1024,1024]1024/512/256/128 B -> 23322 / 23555 / 24436 / 23895
#   [1,1,2048,2048]4096/2048/1024/512 -> 92722 / 89365 / 86085 / 85380
#   [1,1,2048,64]   128/64 B          ->  4797 /  6146
# 512 B is the largest value that is never the wrong call: every geometry whose
# read stays at or above it either improves (1.08x on `[1,1,2048,2048]`, 1.05x
# on `[1,1,32,32768]`) or lands inside the +-3% run-to-run band, and every
# geometry that would have to go BELOW it to buy a wave loses (128 B on
# `[1,1,32,16384]`, 64 B on `[1,1,2048,64]`, 256 B on `[1,1,1024,1024]`).
# In BYTES rather than tiles so it means the same thing at every element size.
MIN_BLOCK_ROW_BYTES = 512
# Drop the tilize helper's per-call unpack+pack data-format reconfig
# (`ReconfigureRegisterDatatypeMode::NoReconfigure`). Correct in THIS kernel
# because `compute_kernel_hw_startup(cb_in, cb_out)` programs srcA/srcB and the
# pack format once and nothing else runs on the TRISCs, so every per-call
# reconfig rewrites the value already in the register — including on the casting
# diagonal, where the cast is carried by the two CBs' formats and not by the
# reconfig. Exposed as a knob rather than hardcoded because a refinement that
# adds a SECOND compute phase to this kernel would have to turn it back on.
COMPUTE_SKIP_FORMAT_RECONFIG = True
# Pay the tilize LLK init + uninit once per core instead of once per block
# (`InitUninitMode::InitOnly / Neither / UninitOnly`). Gated at emission time on
# a core actually owning more than one block — see the compute kernel's
# `amortize_init` comment for why the dead instantiations are not free.
COMPUTE_AMORTIZE_INIT = True

# Legal output tile heights (power-of-two fractions of 32).
LEGAL_TILE_HEIGHTS = (1, 2, 4, 8, 16, 32)
# A tile's FACE is 16 wide at every legal tile height, and `min(tile_h, 16)`
# tall (`tt_metal/impl/data_format/tile.cpp:TILE_FACE_HW_CHOICES`). A tile of
# height `h` is therefore `h / min(h,16)` face-ROWS of two faces each, laid out
# face-row-major, elements row-major inside a face. Everything the retile block
# operation does is derived from that one fact.
FACE_WIDTH = 16


def retile_copy_unit(in_tile_h: int, out_tile_h: int) -> tuple:
    """(rows, cols) of the largest run that is CONTIGUOUS IN BOTH tile layouts.

    This is the whole retile algorithm in one expression, and it is what makes
    the re-tile a pure NoC gather/scatter with no compute stage at all: the
    output tile's bytes are assembled directly out of the input tiles' faces.

    Let `a = min(h, 16)` be a layout's face height.

      * `a_in == a_out`: face `(fr, 0)` and face `(fr, 1)` are ADJACENT in both
        layouts, so a whole face-PAIR SLAB (`a` consecutive rows x all 32
        columns, stored left-face-then-right-face — NOT row-major) is one byte
        run with the same internal order on both sides. Consecutive slabs stay
        contiguous while they remain inside one input tile AND one output tile,
        so the run extends to `min(in_tile_h, out_tile_h)` rows.
      * `a_in != a_out`: the slabs interleave differently, and the largest
        common run is a single face FRAGMENT — `min(in_tile_h, out_tile_h)`
        rows x 16 columns. (When the face heights differ, that minimum is
        always < 16 and divides both face heights, both being powers of two.)

    Verified exhaustively against the layout formula for all 36 legal
    (in_tile_h, out_tile_h) pairs; see probes/probe_024.py.
    """
    a_in = min(int(in_tile_h), FACE_WIDTH)
    a_out = min(int(out_tile_h), FACE_WIDTH)
    rows = min(int(in_tile_h), int(out_tile_h))
    return rows, (TILE_WIDTH if a_in == a_out else FACE_WIDTH)


def _bfloat16_bits(value: float) -> int:
    """`value` as a bfloat16 bit pattern, rounded to nearest EVEN.

    bfloat16 is fp32's top 16 bits, so the encoding is a truncation plus the
    round the hardware and torch both apply — a plain truncation would put a
    different number in the pad region than the oracle's
    `torch.nn.functional.pad(x.bfloat16(), value=v)` for any fill that is not
    exactly representable. The `+1` carries into the exponent naturally, which
    is also what makes an overflowing fill saturate to inf rather than wrap.
    """
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    upper, lower = (bits >> 16) & 0xFFFF, bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and (upper & 1)):
        upper = (upper + 1) & 0xFFFF
    return upper


def pad_fill_word(dtype, value, elem_size: int) -> int:
    """The fill as an `elem_size`-wide bit pattern in the INPUT's format.

    The fill is written into `cb_input_rows`, whose data_format is the INPUT
    dtype, so the encoding is the input's — the output cast happens later, at
    pack time, exactly as it does for a real element.

    A NEGATIVE fill on an integer dtype is a two's-complement bit_cast at the
    element width and must not truncate: masking to `elem_size * 8` bits is the
    whole rule, and it is written width-generically here so the integer dtypes
    Refinement 5 adds need no second implementation.
    """
    if value is None:
        value = 0
    if dtype == ttnn.bfloat16:
        return _bfloat16_bits(value)
    if dtype == ttnn.float32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dtype in (ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8):
        return int(value) & ((1 << (elem_size * 8)) - 1)
    raise RuntimeError(f"tilize: no pad-fill encoding for dtype {dtype}")


def _largest_divisor_at_most(n: int, limit: int) -> int:
    """Coarsest divisor of `n` that is <= `limit`. Always exists (1 | n)."""
    limit = max(1, min(int(limit), int(n)))
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1


class TilizePlan:
    """The derived block plan: every knob, plus the per-core block assignment.

    Memoized (see `_PLAN_CACHE`) so a repeat call at the same shape / dtype /
    memory_config / tile / low_l1 / grid does NOT re-enter the W_FIT solve or
    `split_work_to_cores` — that derivation is exactly what the program-cache
    rule says must not be redone per call.
    """

    __slots__ = (
        "tile_h",
        "tensor_row_blocks",
        "tensor_col_tiles",
        "block_width_tiles",
        "num_w_chunks",
        "num_row_groups",
        "num_blocks_total",
        "write_rows_per_barrier",
        "input_depth_rows",
        "output_depth_batches",
        "in_page_bytes",
        "out_page_bytes",
        "block_row_bytes",
        "all_cores",
        # [(core, start_block_id, num_blocks, block_stride), ...]. `block_stride`
        # is 1 for the solved plan (contiguous ranges) and `num_cores` for the
        # shard-driven plan, which is exactly the ROUND_ROBIN_1D placement an ND
        # spec uses and degenerates to "one block per core" for a legacy 2-D one.
        "assignment",
        "fp32_dest_acc_en",
        # Zero-copy: the named side's CB is placed ON the tensor's own shard
        # buffer, so that side does no NoC access at all.
        "input_native",
        "output_native",
        # How the ROW_MAJOR source pages a row (1 = a page IS a row). > 1 puts
        # the reader on its strided branch.
        "input_pages_per_row",
        "in_page_width_bytes",
        # --- retile (Refinement 4): a TILE input re-laid at another tile
        # height. `in_tile_h` is 0 on the ROW_MAJOR path, where there is no
        # input tile geometry at all.
        "is_retile",
        "in_tile_h",
        "in_tile_rows_per_image",
        # --- padding (Refinement 2). All inert when `pad_active` is False, and
        # the reader's non-padded branches are then byte-identical to Phase 0. ---
        "pad_active",
        "pad_word",
        "elem_size",
        "in_num_images",
        "in_rows_per_image",
        "in_row_bytes",
        "rows_per_image_out",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    # --- derived L1 footprint, in pages and bytes --------------------------
    @property
    def input_cb_pages(self) -> int:
        # The retile path never reads through cb_input_rows (the reader
        # assembles output tiles straight into cb_output_tiles), so the CB
        # shrinks to the ONE page that keeps its JIT tile/format descriptor
        # well-formed for the reader's other, compile-time-discarded branches.
        if self.is_retile:
            return 1
        return self.input_depth_rows * self.block_width_tiles

    @property
    def output_cb_pages(self) -> int:
        return self.output_depth_batches * self.write_rows_per_barrier * self.block_width_tiles

    @property
    def pad_row_bytes(self) -> int:
        """cb_pad_row's single page: ONE block row of the fill (0 when unpadded)."""
        return self.block_row_bytes if self.pad_active else 0

    @property
    def l1_per_core_bytes(self) -> int:
        return (
            self.input_cb_pages * self.in_page_bytes + self.output_cb_pages * self.out_page_bytes + self.pad_row_bytes
        )


_PLAN_CACHE: dict = {}


# ---------------------------------------------------------------------------
# The shard partition — the block grid a sharded operand has ALREADY fixed
# ---------------------------------------------------------------------------


class ShardPartition:
    """A shard spec re-expressed as this op's block grid, in TILE units.

    tilize has no dependent axis: an output tile depends only on its own
    32-element column slice of its own `tile_h` sticks. So a shard IS a block —
    a HEIGHT shard is a `row_group`, a WIDTH shard a `w_chunk`, a BLOCK shard the
    2-D block the op already builds — and there is no cross-core combine, no
    mcast and no semaphore to design. What changes versus the solved plan is only
    WHERE the numbers come from.

    `cores[i]` is the core the i-th shard lives on, in the shard's own linear
    order (`shard_row * num_shard_cols + shard_col`, which is the order
    `buffer.cpp:core_to_host_pages` lays shards out in and therefore the order
    the L1 bytes are in). That linear index IS the `block_id` the three kernels
    already derive `(row_group, w_chunk)` from.
    """

    __slots__ = (
        "cores",
        "grid",
        "shard_rows_tiles",
        "shard_cols_tiles",
        "num_shard_rows",
        "num_shard_cols",
        "num_shards",
        "is_l1",
        "per_core_bytes",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def key(self) -> tuple:
        """Identity of the PARTITION (not of the spec that expressed it), so an
        ND spec and the legacy 2-D spec describing the same cut compare equal."""
        return (
            tuple((int(c.x), int(c.y)) for c in self.cores),
            self.shard_rows_tiles,
            self.shard_cols_tiles,
            self.num_shard_rows,
            self.num_shard_cols,
        )


def _folded_2d(padded_shape) -> tuple:
    """The (height, width) 2-D view a shard spec is written against: every
    leading dim folds into the height, exactly as R folds them into tile-rows."""
    height = 1
    for d in padded_shape[:-1]:
        height *= int(d)
    return height, int(padded_shape[-1])


def _shard_geometry(mem_config):
    """(shard_h, shard_w, grid, orientation) for either shard API, or None.

    A LIVE tensor normalizes an ND spec down to the equivalent legacy 2-D one
    whenever it has one (measured: `probes/probe_010.py`), so reading the live
    tensor is what makes `nd_same_spec` and `nd_in_legacy_out` compare equal
    without an API-shaped special case. The ND branch below is the residue: a
    spec that keeps a genuine ND shape.
    """
    shard_spec = getattr(mem_config, "shard_spec", None)
    if shard_spec is not None:
        shape = list(shard_spec.shape)
        return int(shape[-2]), int(shape[-1]), shard_spec.grid, shard_spec.orientation
    nd_spec = getattr(mem_config, "nd_shard_spec", None)
    if nd_spec is None:
        return None
    shape = [int(d) for d in list(nd_spec.shard_shape)]
    if len(shape) < 2 or any(d != 1 for d in shape[:-2]):
        # A shard that spans more than one leading-dim slice is not a contiguous
        # run of folded rows, so the 2-D block view does not hold.
        return None
    return shape[-2], shape[-1], nd_spec.grid, nd_spec.orientation


def shard_partition(tensor, tile_h: int):
    """`ShardPartition` for `tensor`, or None when its shards are not blocks.

    None means "this side cannot drive the block grid" — not sharded, a ragged
    partition (the last shard short, which would make `block_width_tiles` vary
    per core and it is a COMPILE-TIME template argument), or a shard that is not
    a whole number of tiles. The caller then falls back to the solved plan, which
    is correct for every placement because the accessor addresses a sharded
    tensor as readily as an interleaved one — just not natively.
    """
    if not tensor.is_sharded():
        return None
    mem_config = tensor.memory_config()
    geometry = _shard_geometry(mem_config)
    if geometry is None:
        return None
    shard_h, shard_w, grid, orientation = geometry

    height, width = _folded_2d(list(tensor.padded_shape))
    if shard_h <= 0 or shard_w <= 0:
        return None
    if shard_h % tile_h or shard_w % TILE_WIDTH:
        return None
    if height % shard_h or width % shard_w:
        return None  # ragged: block_width_tiles / block_row_extent would vary

    num_shard_rows = height // shard_h
    num_shard_cols = width // shard_w
    num_shards = num_shard_rows * num_shard_cols

    num_cores = grid.num_cores()
    if num_cores == 0 or num_shards < 1:
        return None
    # buffer.cpp: shards are placed on `corerange_to_cores(grid, n, row_major)`
    # in their linear order, row_wise iff the orientation is ROW_MAJOR.
    core_order = ttnn.corerange_to_cores(grid, num_cores, orientation == ttnn.ShardOrientation.ROW_MAJOR)
    cores = [core_order[i % num_cores] for i in range(num_shards)]

    total_bytes = int(tensor.buffer_num_pages()) * int(tensor.buffer_aligned_page_size())
    shards_per_core = math.ceil(num_shards / num_cores)
    return ShardPartition(
        cores=cores,
        grid=grid,
        shard_rows_tiles=shard_h // tile_h,
        shard_cols_tiles=shard_w // TILE_WIDTH,
        num_shard_rows=num_shard_rows,
        num_shard_cols=num_shard_cols,
        num_shards=num_shards,
        is_l1=(mem_config.buffer_type == ttnn.BufferType.L1),
        per_core_bytes=(total_bytes // num_shards) * shards_per_core,
    )


def _resident_l1_bytes(tensor, partition, num_cores: int) -> int:
    """Per-core L1 the tensor ITSELF holds, which the CB budget must not spend.

    `ttnn.get_max_worker_l1_unreserved_size()` reports the worker L1 arena, not
    what is free in it, and an L1-resident operand (every sharded one, by
    definition) is already sitting in that arena. l1_ledger.md finding, folded in
    here because a sharded refinement is where it starts to bite.
    """
    if tensor.memory_config().buffer_type != ttnn.BufferType.L1:
        return 0
    if partition is not None:
        return partition.per_core_bytes
    total = int(tensor.buffer_num_pages()) * int(tensor.buffer_aligned_page_size())
    return math.ceil(total / max(1, num_cores))


def _input_tile_height(input_tensor) -> int:
    """The input's own tile height, or 0 for a ROW_MAJOR input (no tile geometry).

    Non-zero IS the retile predicate: a TILE input is the only thing that puts
    the reader on its face-walking block operation.
    """
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        return 0
    tile = getattr(input_tensor, "tile", None)
    return int(tile.tile_shape[0]) if tile is not None else TILE_WIDTH


def plan_cache_key(input_tensor, output_tensor, *, low_l1: bool, grid, pad_value) -> tuple:
    """Hashable identity of everything the plan derivation reads."""
    return (
        tuple(input_tensor.shape),
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        tuple(output_tensor.padded_shape),
        output_tensor.dtype,
        str(output_tensor.memory_config()),
        output_tensor.buffer_page_size(),
        (output_tensor.tile.tile_shape[0], output_tensor.tile.tile_shape[1]),
        # The INPUT's tile height selects the reader's block operation on the
        # retile path, so it belongs in the plan's identity.
        input_tensor.buffer_page_size(),
        _input_tile_height(input_tensor),
        bool(low_l1),
        (grid.x, grid.y),
        None if pad_value is None else float(pad_value),
    )


def derive_plan(input_tensor, output_tensor, *, low_l1: bool, grid, pad_value=None) -> TilizePlan:
    """Rule 2, in its two ordered steps: (1) fill the grid, (2) take the
    coarsest block that fits L1. Nothing here is restated from a literal."""
    key = plan_cache_key(input_tensor, output_tensor, low_l1=low_l1, grid=grid, pad_value=pad_value)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    num_cores = int(grid.x) * int(grid.y)

    tile_h = int(output_tensor.tile.tile_shape[0])
    # `shape` is the INPUT's page grid (what the reader addresses); `out_shape`
    # is the OUTPUT's tile grid (what the block plan is cut on).
    shape = list(input_tensor.padded_shape)
    out_shape = list(output_tensor.padded_shape)

    # Geometry: R is per-image and folds the leading dims. NEVER
    # floor(num_images * H / tile_h) -- [8,1,249,2048] is the case that differs.
    #
    # Read off the OUTPUT's padded shape, not the input's. The tile grid IS the
    # output, and the two shapes are NOT interchangeable once a shard is in play:
    # a ROW_MAJOR tensor's padded shape rounds its last dim up to its PAGE width,
    # which a width-cutting shard sets (`[3,160,160]` sharded 64 wide reports
    # `[3,160,192]`), while a TILE tensor rounds to the tile. Taking C from the
    # input there yields 6 columns for a 5-column output — measured as the
    # `test_tilize_nd_sharded` value mismatch.
    num_images = 1
    for d in out_shape[:-2]:
        num_images *= int(d)
    rows_per_image = math.ceil(int(out_shape[-2]) / tile_h)
    tensor_row_blocks = num_images * rows_per_image  # R
    tensor_col_tiles = math.ceil(int(out_shape[-1]) / TILE_WIDTH)  # C

    # Page bytes. tb_in is one tile's worth of ROW-MAJOR bytes; tb_out is one
    # whole output tile page, read off the buffer so block-float / tiny-tile
    # page sizes are exact rather than computed.
    in_page_bytes = tile_h * TILE_WIDTH * input_tensor.element_size()
    out_page_bytes = output_tensor.buffer_page_size()

    # --- how the ROW_MAJOR source is PAGED --------------------------------
    # A stick-indexed read (`start_page + row`) is only a stick index when a page
    # IS a whole row — true for interleaved ROW_MAJOR and for a HEIGHT-sharded
    # one. A shard that cuts the width pages by SHARD width instead, so one row
    # spans `input_pages_per_row` pages and consecutive sticks are that far
    # apart. Native consumption sidesteps it entirely (no read at all); where
    # native is unavailable the reader takes its strided branch, for which the
    # block's row segment must sit inside a single source page.
    in_page_width_elems = int(input_tensor.buffer_page_size()) // input_tensor.element_size()
    input_pages_per_row = max(1, math.ceil(int(shape[-1]) / max(1, in_page_width_elems)))

    # --- retile: a TILE input, re-laid at another tile height --------------
    # The input's pages are whole TILES, not sticks, so `input_pages_per_row`
    # (a stick-paging quantity) is meaningless here and every stick-indexed
    # branch of the reader is off. What replaces them is one derived unit: the
    # largest byte run contiguous in both tile layouts (`retile_copy_unit`),
    # which the reader gathers straight into the OUTPUT tile — no row-major
    # intermediate, no compute stage, and never an untilize/tilize round trip
    # (op_design.md ranks that `rejected` at 2x the minimum DRAM traffic).
    in_tile_h = _input_tile_height(input_tensor)
    is_retile = in_tile_h > 0
    if is_retile:
        input_pages_per_row = 1

    # --- the pad region: what the output grid covers MINUS the input -------
    # Padding is `grid2d_padded` in op_design.md, and it is ADDITIVE on the
    # block that already exists: the block grid, the core assignment, the CBs
    # and the compute call are untouched; only `load_block` gains a fill.
    #
    # The pad extent is derived from the two shapes rather than from the
    # request, and the two agree by construction: `validate()` refuses a
    # non-tile-aligned input with no padding argument, so a pad REGION exists iff
    # the output's padded grid reaches past the input's LOGICAL extent. Reading
    # it off the geometry is what keeps `pad_active` False — and the Phase 0
    # reader branches byte-identical — on every already-aligned call, padded or
    # not (`explicit` at exactly the tile round asks for no fill and gets none).
    #
    # LOGICAL, not padded, on the input side: the pad boundary is where the
    # caller's data ends, and a ROW_MAJOR tensor's padded shape rounds its last
    # dim up to its PAGE width (a width-cutting shard sets that), which would
    # claim page padding as real data.
    elem_size = int(input_tensor.element_size())
    in_logical = [int(d) for d in list(input_tensor.shape)]
    in_logical = [1] * (2 - len(in_logical)) + in_logical  # rank 0/1: the pad SYNTHESIZES both tile dims
    in_num_images = 1
    for d in in_logical[:-2]:
        in_num_images *= d
    in_rows_per_image = in_logical[-2]
    in_row_bytes = in_logical[-1] * elem_size
    pad_active = (
        in_num_images < num_images
        or in_rows_per_image < rows_per_image * tile_h
        or in_row_bytes < tensor_col_tiles * TILE_WIDTH * elem_size
    )
    pad_word = pad_fill_word(input_tensor.dtype, pad_value, elem_size) if pad_active else 0

    # The input's tile-rows PER IMAGE. The output's tile grid and the input's do
    # NOT have to agree on the padded per-image height (`H=20` at in_tile_h=8 is
    # three input tile-rows, at out tile_h=4 five output tile-rows), so the
    # reader splits a folded output tile-row per IMAGE before turning it into an
    # input tile-row — the same segmentation the padded branch does, and the
    # reason no host guard on "the two padded heights must match" is needed.
    in_tile_rows_per_image = math.ceil(in_rows_per_image / in_tile_h) if is_retile else 0
    if is_retile:
        # A pure byte re-lay assembles the output tile's bytes from the input
        # tiles' faces; both preconditions below are what make "byte" the right
        # unit. They cannot be reached from SUPPORTED today (dtype and
        # output_dtype are both bfloat16, and retile x padding is EXCLUDED), and
        # they are asserted rather than assumed so the dtype refinement trips
        # here instead of producing wrong bytes.
        if input_tensor.dtype != output_tensor.dtype:
            raise RuntimeError(
                f"tilize: re-tiling {in_tile_h}->{tile_h} with a dtype cast "
                f"({input_tensor.dtype} -> {output_tensor.dtype}) is not implemented; the retile "
                "path is a pure byte re-lay and carries no pack-time conversion stage"
            )
        if (
            int(input_tensor.buffer_page_size()) != in_tile_h * TILE_WIDTH * elem_size
            or int(out_page_bytes) != tile_h * TILE_WIDTH * elem_size
        ):
            raise RuntimeError(
                "tilize: re-tiling needs densely packed tile pages on both sides "
                f"(in {input_tensor.buffer_page_size()} B vs {in_tile_h * TILE_WIDTH * elem_size}, "
                f"out {out_page_bytes} B vs {tile_h * TILE_WIDTH * elem_size}); a block-float or "
                "otherwise padded tile page carries per-face exponents the face walk does not move"
            )
        if pad_active:
            raise RuntimeError(
                "tilize: re-tiling a padded region is not implemented (EXCLUSIONS covers the cell); "
                "the fill would have to be written into output faces the face walk never sources"
            )

    # --- who fixes the block grid: a shard spec, or the L1 solve? ---------
    # tilize has no dependent axis, so a shard is already a block. When one side
    # is L1-sharded its OWN partition is used verbatim and its CB is placed on
    # its buffer (zero-copy); the other side falls back to the accessor. Both
    # sides go native only when the two partitions are the SAME cut, which is
    # what makes a same-spec call touch the NoC not at all.
    def _partition_tiles_the_grid(part):
        """A partition can only DRIVE the plan if its shards tile the output grid
        exactly. A shard whose last column is part page-padding (a ROW_MAJOR
        tensor cut 64 wide across a 160-wide row) covers more tile columns than
        the output has, so it is a placement the accessor still serves but the
        block grid cannot be read off."""
        return part is not None and (
            part.num_shard_rows * part.shard_rows_tiles == tensor_row_blocks
            and part.num_shard_cols * part.shard_cols_tiles == tensor_col_tiles
        )

    in_partition = shard_partition(input_tensor, tile_h)
    out_partition = shard_partition(output_tensor, tile_h)
    if not _partition_tiles_the_grid(in_partition):
        in_partition = None
    if not _partition_tiles_the_grid(out_partition):
        out_partition = None
    partition = None
    input_native = False
    output_native = False
    if is_retile:
        # Retile x sharded is EXCLUDED (see tilize.EXCLUSIONS): the face walk is
        # written against the interleaved TILE page index (`tile_row * C + col`)
        # and a zero-copy CB over a resident TILE shard would need the shard's
        # own page map on both sides. Forcing both partitions off keeps the
        # solved plan — which is correct for every placement — as the only
        # retile path, rather than half-wiring a native one no test can reach.
        in_partition = out_partition = None
    if pad_active:
        # A zero-copy input CB IS the resident shard, and the shard holds only
        # the caller's own bytes — there is nowhere in it to put the fill, and
        # nothing that could put it there without corrupting the input tensor.
        # So a padded call reads its input through the accessor and fills a
        # scratch CB. The OUTPUT side is unaffected: the packer writes whole
        # tiles, pad positions included, straight into the output shard.
        in_partition = None
    if in_partition is not None and in_partition.is_l1:
        partition, input_native = in_partition, True
        output_native = out_partition is not None and out_partition.is_l1 and out_partition.key == in_partition.key
    elif out_partition is not None and out_partition.is_l1:
        partition, output_native = out_partition, True

    if partition is not None and partition is out_partition and input_pages_per_row > 1:
        # The output shard would fix a block width the strided read cannot honour
        # (see below): a block row segment has to sit inside ONE source page, and
        # only the solved plan is free to choose a width that does. Correctness
        # over the output's zero-copy, in a mix (sub-row-paged non-native input +
        # L1-sharded output on a different cut) no test in the suite reaches.
        partition, output_native = None, False

    if not input_native and input_pages_per_row > 1 and in_page_width_elems % TILE_WIDTH:
        raise RuntimeError(
            f"tilize: input's ROW_MAJOR page is {in_page_width_elems} elements, which is neither a "
            f"whole {shape[-1]}-element row nor a whole number of {TILE_WIDTH}-element tiles; the "
            "block's row segment cannot be addressed within one source page"
        )

    # --- L1 budget. The arena minus what the OPERANDS already hold ---------
    # An L1-resident tensor sits in the same worker arena the CBs are cut from,
    # and every sharded operand is L1-resident by definition, so a budget that
    # ignores it over-promises exactly where a sharded call needs it most.
    budget = ttnn.get_max_worker_l1_unreserved_size()
    budget -= _resident_l1_bytes(input_tensor, in_partition, num_cores)
    budget -= _resident_l1_bytes(output_tensor, out_partition, num_cores)
    budget = max(budget, in_page_bytes + out_page_bytes)

    input_depth_rows = INPUT_DEPTH_ROWS
    output_depth_batches = OUTPUT_DEPTH_BATCHES

    if partition is not None:
        # --- grid2d_sharded: every number below is READ, not solved. -------
        block_width_tiles = partition.shard_cols_tiles
        num_w_chunks = partition.num_shard_cols
        num_row_groups = partition.num_shard_rows
        num_blocks_total = partition.num_shards
        write_rows_per_barrier = max(1, math.ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))

        # Only the NON-native side costs CB L1; the native side's CB is the
        # tensor. Shrink the depth knobs (never the block) if the pair overruns.
        def _scratch_bytes(depth_in, depth_out):
            total = 0
            if pad_active:
                total += block_width_tiles * (in_page_bytes // tile_h)  # cb_pad_row
            if not input_native:
                total += depth_in * block_width_tiles * in_page_bytes
            if not output_native:
                total += depth_out * write_rows_per_barrier * block_width_tiles * out_page_bytes
            return total

        while _scratch_bytes(input_depth_rows, output_depth_batches) > budget and (
            input_depth_rows > 1 or output_depth_batches > 1
        ):
            if input_depth_rows >= output_depth_batches and input_depth_rows > 1:
                input_depth_rows -= 1
            else:
                output_depth_batches -= 1

        # One block per shard, handed to the core the shard is resident on. The
        # shard's linear index IS the block id, and the ROUND_ROBIN_1D placement
        # both shard APIs use makes core i own shards {i, i+N, i+2N, ...} — a
        # start plus a stride, which is also the identity for a legacy 2-D spec
        # (N == num_shards, so each core owns exactly one).
        all_cores = partition.grid
        stride = partition.grid.num_cores()
        assignment = []
        for core_index, core in enumerate(ttnn.corerange_to_cores(all_cores, stride, True)):
            blocks_here = len(range(core_index, num_blocks_total, stride)) if core_index < num_blocks_total else 0
            assignment.append((core, core_index, blocks_here, stride))
        assert (
            sum(a[2] for a in assignment) == num_blocks_total
        ), f"tilize: shard assignment covered {sum(a[2] for a in assignment)} of {num_blocks_total} shards"
    else:
        # --- the column-extent L1 bound, in closed form -------------------
        # footprint = INPUT_DEPTH_ROWS*W*tb_in + OUTPUT_DEPTH_BATCHES*wrpb*W*tb_out
        # and wrpb*W <= WRITE_BATCH_MIN_TILES + W, so
        #   footprint <= W*(2*tb_in + 2*tb_out) + 2*WRITE_BATCH_MIN_TILES*tb_out
        # which inverts to W_FIT below. Budget is read from the device, never a
        # literal; no tensor dimension appears.
        # On the retile path cb_input_rows is a one-page stub (see
        # TilizePlan.input_cb_pages), so only the output CB scales with the
        # column extent and the solve gets the whole budget for it.
        denom = OUTPUT_DEPTH_BATCHES * out_page_bytes
        if not is_retile:
            denom += INPUT_DEPTH_ROWS * in_page_bytes
        if pad_active:
            # cb_pad_row is ONE block row: `block_width_tiles * TILE_WIDTH * elem`
            # bytes, i.e. `in_page_bytes / tile_h` per tile of block width. Small,
            # but it scales with the same knob, so it belongs in the same solve
            # rather than being spent behind the budget's back.
            denom += in_page_bytes // tile_h
        headroom = budget - OUTPUT_DEPTH_BATCHES * WRITE_BATCH_MIN_TILES * out_page_bytes
        w_fit = max(1, min(headroom // denom, FAST_TILIZE_WIDTH_CAP))
        w_cap = min(w_fit, LOW_L1_WIDTH_CAP) if low_l1 else w_fit

        # Step 1 (occupancy + pipeline waves) and step 2 (L1 fit) -> the smallest
        # column split that satisfies both. num_w_chunks is MINIMIZED: a column cut
        # multiplies the read transaction count, so it is spent only where it is
        # required. The tensor holds `R * num_w_chunks` tile-rows, so ONE
        # expression covers both demands on the column axis — fill the grid
        # (`waves == 1`) and fill each core's pipe (`waves > 1`, which is what
        # buys read/write overlap on a geometry whose R cannot supply a second
        # wave by itself).
        w_chunks_for_l1 = math.ceil(tensor_col_tiles / w_cap)
        # On a sub-row-paged source the width must ALSO divide the page width in
        # tiles, so `w_chunk * block_row_bytes` lands at a page boundary plus an
        # offset that leaves the whole segment inside that page. Expressed by
        # narrowing what the divisor is taken OF — gcd(C, page width) — so the
        # search itself stays the single `_largest_divisor_at_most` knob, and
        # degenerates to C exactly when a page is a whole row.
        page_width_tiles = in_page_width_elems // TILE_WIDTH if input_pages_per_row > 1 else tensor_col_tiles
        width_source = math.gcd(tensor_col_tiles, page_width_tiles)

        def _width_at(waves: int) -> int:
            """Coarsest DIVISOR of C that fits L1 and yields >= the target chunks.

            A divisor (rather than a ceil) is what removes the ragged column tail —
            see this module's docstring for the mechanism cap that forces it.
            """
            w_chunks_for_waves = min(tensor_col_tiles, math.ceil(num_cores * waves / tensor_row_blocks))
            target = max(w_chunks_for_l1, w_chunks_for_waves)
            return _largest_divisor_at_most(width_source, min(w_cap, tensor_col_tiles // target))

        # The wave count is bought FROM the column axis, so each doubling halves
        # the read transaction — the trade this knob makes. `MIN_BLOCK_ROW_BYTES`
        # is where the trade stops paying; take the deepest pipe whose read still
        # clears it, and fall back to the occupancy-only cut when none does.
        block_width_tiles = _width_at(1)
        for waves in range(PIPELINE_WAVES_PER_CORE, 1, -1):
            candidate = _width_at(waves)
            if candidate * TILE_WIDTH * elem_size >= MIN_BLOCK_ROW_BYTES:
                block_width_tiles = candidate
                break
        num_w_chunks = tensor_col_tiles // block_width_tiles  # exact, no tail

        num_row_groups = min(tensor_row_blocks, max(1, math.ceil(num_cores / num_w_chunks)))
        num_blocks_total = num_row_groups * num_w_chunks

        # Transactions-in-flight knob for the writer. Inert (1) once the block is
        # at least WRITE_BATCH_MIN_TILES wide; the whole knob on a C == 1 tensor.
        write_rows_per_barrier = max(1, math.ceil(WRITE_BATCH_MIN_TILES / block_width_tiles))

        # --- core assignment over the linearized block id -----------------
        # row_wise=True explicitly: a column line of cores measured 2.91x worse
        # than a row line on an interleaved DRAM->DRAM copy, and row_wise=False is
        # the default that hands you the column.
        (
            _num_cores_used,
            all_cores,
            core_group_1,
            core_group_2,
            blocks_per_core_g1,
            blocks_per_core_g2,
        ) = ttnn.split_work_to_cores(grid, num_blocks_total, row_wise=True)

        assignment = []
        start = 0
        for group, per_core in ((core_group_1, blocks_per_core_g1), (core_group_2, blocks_per_core_g2)):
            if per_core == 0:
                continue
            for core in ttnn.corerange_to_cores(group, None, True):
                assignment.append((core, start, per_core, 1))
                start += per_core
        assert start == num_blocks_total, f"tilize: block assignment covered {start} of {num_blocks_total} blocks"

    plan = TilizePlan(
        tile_h=tile_h,
        tensor_row_blocks=tensor_row_blocks,
        tensor_col_tiles=tensor_col_tiles,
        block_width_tiles=block_width_tiles,
        num_w_chunks=num_w_chunks,
        num_row_groups=num_row_groups,
        num_blocks_total=num_blocks_total,
        write_rows_per_barrier=write_rows_per_barrier,
        input_depth_rows=input_depth_rows,
        output_depth_batches=output_depth_batches,
        in_page_bytes=in_page_bytes,
        out_page_bytes=out_page_bytes,
        block_row_bytes=block_width_tiles * TILE_WIDTH * input_tensor.element_size(),
        all_cores=all_cores,
        assignment=assignment,
        fp32_dest_acc_en=(input_tensor.dtype == ttnn.float32),
        input_native=input_native,
        output_native=output_native,
        input_pages_per_row=input_pages_per_row,
        in_page_width_bytes=in_page_width_elems * input_tensor.element_size(),
        is_retile=is_retile,
        in_tile_h=in_tile_h,
        in_tile_rows_per_image=in_tile_rows_per_image,
        pad_active=pad_active,
        pad_word=pad_word,
        elem_size=elem_size,
        in_num_images=in_num_images,
        in_rows_per_image=in_rows_per_image,
        in_row_bytes=in_row_bytes,
        rows_per_image_out=rows_per_image,
    )
    _PLAN_CACHE[key] = plan
    return plan


def _cb_on_shard(cb_index, tensor, core_ranges, page_bytes: int, tile_desc):
    """A CB placed ON `tensor`'s resident shard — the native sharded path.

    `cb_descriptor_from_sharded_tensor` inherits the tensor's own page size
    (a stick for ROW_MAJOR, a tile for TILE). This op accounts in whole tiles on
    both sides, so the page size is re-stated as the tile-sized unit; the bytes
    are untouched and sit in L1 exactly as a tile-paged buffer, which is what
    keeps the whole leg zero-copy while the helpers still see clean tile pages.
    """
    cb = ttnn.cb_descriptor_from_sharded_tensor(cb_index, tensor, 0, 0, core_ranges)
    formats = cb.format_descriptors
    formats[0].page_size = page_bytes
    formats[0].tile = tile_desc
    cb.format_descriptors = formats
    assert (
        cb.total_size % page_bytes == 0
    ), f"tilize: shard bank size {cb.total_size} is not a whole number of {page_bytes} B tile pages"
    return cb


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    low_l1: bool = False,
    pad_value=None,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()  # never a hardcoded core count
    plan = derive_plan(input_tensor, output_tensor, low_l1=low_l1, grid=grid, pad_value=pad_value)

    tile_desc = ttnn.TileDescriptor(plan.tile_h, TILE_WIDTH)

    # ========== Circular buffers ==========
    # cb_input_rows: live set is ONE tile-row of the block (the tilize helper
    # waits/pops exactly block_width_tiles pages per iteration), so
    # block_row_extent does not enter the size. Capacity = depth * live set.
    #
    # ZERO-COPY when the input is consumed natively: the CB is PLACED ON the
    # shard buffer, so the resident bytes are the CB's contents and the reader
    # issues no NoC read at all. The page size is overridden to the tile-sized
    # accounting unit the tilize helper counts in — the shard's own paging is
    # one ROW_MAJOR stick, and `block_width_tiles` sticks-worth of contiguous
    # bytes IS one tile-row page group, because the shard's row width is exactly
    # the block's row width (`block_width_tiles == shard_cols_tiles`).
    if plan.input_native:
        cb_input_rows = _cb_on_shard(CB_INPUT_ROWS, input_tensor, plan.all_cores, plan.in_page_bytes, tile_desc)
    else:
        cb_input_rows = ttnn.CBDescriptor(
            total_size=plan.input_cb_pages * plan.in_page_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_ROWS,
                    data_format=input_tensor.dtype,
                    page_size=plan.in_page_bytes,
                    tile=tile_desc,
                )
            ],
        )

    # cb_output_tiles: live set is one write batch; capacity is
    # OUTPUT_DEPTH_BATCHES of them. Its data_format is where the
    # value-preserving `dtype=` cast happens, at pack time.
    #
    # ZERO-COPY when the output is produced natively: the packer writes the
    # output shard in place and there is no writer kernel at all. The tile order
    # matches because compute emits tile-rows left to right across the whole
    # shard width, which is the order a TILE shard stores its pages in.
    if plan.output_native:
        cb_output_tiles = _cb_on_shard(CB_OUTPUT_TILES, output_tensor, plan.all_cores, plan.out_page_bytes, tile_desc)
    else:
        cb_output_tiles = ttnn.CBDescriptor(
            total_size=plan.output_cb_pages * plan.out_page_bytes,
            core_ranges=plan.all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=plan.out_page_bytes,
                    tile=tile_desc,
                )
            ],
        )

    # cb_pad_row: reader-local scratch holding ONE block row pre-filled with the
    # fill value. Allocated only on the padded path. Its point is that a whole
    # pad ROW then costs one DM-engine transfer from L1 to L1 instead of a RISC
    # store loop over `block_row_bytes` — the reader seeds it once per kernel
    # (32 elements by hand, then doubling local reads) and reads from it for
    # every fully-padded row of every block. It has no producer/consumer pair:
    # nothing is ever pushed or popped, so its `total_size` is exactly its one
    # page and the reader only ever takes `get_write_ptr` of it.
    cbs = [cb_input_rows, cb_output_tiles]
    if plan.pad_active:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=plan.pad_row_bytes,
                core_ranges=plan.all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_PAD_ROW,
                        data_format=input_tensor.dtype,
                        page_size=plan.pad_row_bytes,
                    )
                ],
            )
        )

    # ========== Kernels ==========
    # The CT "plan" block is identical in all three kernels: each derives its
    # own view of a block from the same numbers, which is why there is no
    # cross-kernel handshake and no coordinator core.
    reader_ct_args = [
        CB_INPUT_ROWS,
        plan.block_width_tiles,
        plan.tile_h,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.block_row_bytes,
        int(plan.input_native),
        plan.input_pages_per_row,
        plan.in_page_width_bytes,
        # --- padding (inert, and the branch compiles out, when pad_active is 0)
        int(plan.pad_active),
        CB_PAD_ROW,
        plan.elem_size,
        plan.pad_word,
        plan.in_num_images,
        plan.in_rows_per_image,
        plan.in_row_bytes,
        plan.rows_per_image_out,
        # --- retile (inert, and the branch compiles out, when is_retile is 0)
        int(plan.is_retile),
        plan.in_tile_h,
        plan.in_tile_rows_per_image,
        CB_OUTPUT_TILES,
        plan.tensor_col_tiles,
        plan.out_page_bytes,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.tensor_col_tiles,
        plan.num_row_groups,
        plan.num_w_chunks,
        plan.write_rows_per_barrier,
        plan.out_page_bytes,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct_args = [
        CB_INPUT_ROWS,
        CB_OUTPUT_TILES,
        plan.block_width_tiles,
        plan.tensor_row_blocks,
        plan.num_row_groups,
        plan.num_w_chunks,
        int(COMPUTE_SKIP_FORMAT_RECONFIG),
        # The init/uninit amortization is only emitted where a core can actually
        # own more than one block; at one block per core its three extra
        # instantiations are dead code that still costs binary-dispatch time.
        int(COMPUTE_AMORTIZE_INIT and max(a[2] for a in plan.assignment) > 1),
    ]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for core, start_block_id, num_blocks, block_stride in plan.assignment:
        reader_rt_args[core.x][core.y] = [in_addr, start_block_id, num_blocks, block_stride]
        writer_rt_args[core.x][core.y] = [out_addr, start_block_id, num_blocks, block_stride]
        compute_rt_args[core.x][core.y] = [start_block_id, num_blocks, block_stride]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),  # reads on NoC0
    )
    # No writer at all on the native output path: the packer has already placed
    # every tile in the output shard, so there is nothing left to move. A writer
    # that re-wrote a core's own shard over the NoC would be the interleaved path
    # wearing a sharded hat.
    writer_kernel = (
        None
        if plan.output_native
        else ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
            core_ranges=plan.all_cores,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt_args,
            config=ttnn.WriterConfigDescriptor(),  # writes on NoC1
        )
    )
    # No compute kernel at all on the RETILE path. A re-tile is a pure byte
    # re-lay between two tiled layouts, so `retile_copy_unit`'s runs go from the
    # source tile's faces straight into the destination tile's faces over the
    # NoC — there is no row-major intermediate for a tilize LLK to consume and
    # nothing for the FPU to do. cb_output_tiles then has the READER as its
    # single producer and the writer as its single consumer, which is the same
    # one-producer/one-consumer contract as every other CB in this op.
    compute_kernel = (
        None
        if plan.is_retile
        else ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
            core_ranges=plan.all_cores,
            compile_time_args=compute_ct_args,
            runtime_args=compute_rt_args,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=plan.fp32_dest_acc_en),
        )
    )

    return ttnn.ProgramDescriptor(
        kernels=[k for k in (reader_kernel, writer_kernel, compute_kernel) if k is not None],
        semaphores=[],
        cbs=cbs,
    )
