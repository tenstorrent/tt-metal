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
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --- CB slots (semantic names; the numeric slot is just the buffer index) ---
CB_INPUT_ROWS = 0  # reader -> compute: row-major sub-block, one page per tile
CB_OUTPUT_TILES = 1  # compute -> writer: whole output tile pages

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

# Legal output tile heights (power-of-two fractions of 32).
LEGAL_TILE_HEIGHTS = (1, 2, 4, 8, 16, 32)


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
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    # --- derived L1 footprint, in pages and bytes --------------------------
    @property
    def input_cb_pages(self) -> int:
        return self.input_depth_rows * self.block_width_tiles

    @property
    def output_cb_pages(self) -> int:
        return self.output_depth_batches * self.write_rows_per_barrier * self.block_width_tiles

    @property
    def l1_per_core_bytes(self) -> int:
        return self.input_cb_pages * self.in_page_bytes + self.output_cb_pages * self.out_page_bytes


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


def plan_cache_key(input_tensor, output_tensor, *, low_l1: bool, grid) -> tuple:
    """Hashable identity of everything the plan derivation reads."""
    return (
        tuple(input_tensor.padded_shape),
        input_tensor.dtype,
        input_tensor.layout,
        str(input_tensor.memory_config()),
        tuple(output_tensor.padded_shape),
        output_tensor.dtype,
        str(output_tensor.memory_config()),
        output_tensor.buffer_page_size(),
        (output_tensor.tile.tile_shape[0], output_tensor.tile.tile_shape[1]),
        bool(low_l1),
        (grid.x, grid.y),
    )


def derive_plan(input_tensor, output_tensor, *, low_l1: bool, grid) -> TilizePlan:
    """Rule 2, in its two ordered steps: (1) fill the grid, (2) take the
    coarsest block that fits L1. Nothing here is restated from a literal."""
    key = plan_cache_key(input_tensor, output_tensor, low_l1=low_l1, grid=grid)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    num_cores = int(grid.x) * int(grid.y)

    tile_h = int(output_tensor.tile.tile_shape[0])
    shape = list(input_tensor.padded_shape)

    # Geometry: R is per-image and folds the leading dims. NEVER
    # floor(num_images * H / tile_h) -- [8,1,249,2048] is the case that differs.
    num_images = 1
    for d in shape[:-2]:
        num_images *= int(d)
    rows_per_image = math.ceil(int(shape[-2]) / tile_h)
    tensor_row_blocks = num_images * rows_per_image  # R
    tensor_col_tiles = math.ceil(int(shape[-1]) / TILE_WIDTH)  # C

    # Page bytes. tb_in is one tile's worth of ROW-MAJOR bytes; tb_out is one
    # whole output tile page, read off the buffer so block-float / tiny-tile
    # page sizes are exact rather than computed.
    in_page_bytes = tile_h * TILE_WIDTH * input_tensor.element_size()
    out_page_bytes = output_tensor.buffer_page_size()

    # --- who fixes the block grid: a shard spec, or the L1 solve? ---------
    # tilize has no dependent axis, so a shard is already a block. When one side
    # is L1-sharded its OWN partition is used verbatim and its CB is placed on
    # its buffer (zero-copy); the other side falls back to the accessor. Both
    # sides go native only when the two partitions are the SAME cut, which is
    # what makes a same-spec call touch the NoC not at all.
    in_partition = shard_partition(input_tensor, tile_h)
    out_partition = shard_partition(output_tensor, tile_h)
    partition = None
    input_native = False
    output_native = False
    if in_partition is not None and in_partition.is_l1:
        partition, input_native = in_partition, True
        output_native = out_partition is not None and out_partition.is_l1 and out_partition.key == in_partition.key
    elif out_partition is not None and out_partition.is_l1:
        partition, output_native = out_partition, True

    # The accessor's stick indexing (`start_page + row`) is only a stick index
    # when a page IS a whole row. That holds for interleaved ROW_MAJOR and for a
    # HEIGHT-sharded one; a WIDTH/BLOCK-sharded ROW_MAJOR input pages by SHARD
    # width, so reading it through the accessor would silently address the wrong
    # sticks. Native consumption is the answer, and this only fires when native
    # is unavailable (a ragged or DRAM-resident input shard).
    if (
        input_tensor.is_sharded()
        and not input_native
        and int(input_tensor.buffer_page_size()) != int(shape[-1]) * input_tensor.element_size()
    ):
        raise RuntimeError(
            "tilize: input is sharded by width and cannot be consumed natively "
            f"(page {input_tensor.buffer_page_size()} B is not a whole {shape[-1]}-element row); "
            "a width/block-sharded ROW_MAJOR input must be L1-resident with a "
            "tile-aligned, evenly-dividing shard"
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
        denom = INPUT_DEPTH_ROWS * in_page_bytes + OUTPUT_DEPTH_BATCHES * out_page_bytes
        headroom = budget - OUTPUT_DEPTH_BATCHES * WRITE_BATCH_MIN_TILES * out_page_bytes
        w_fit = max(1, min(headroom // denom, FAST_TILIZE_WIDTH_CAP))
        w_cap = min(w_fit, LOW_L1_WIDTH_CAP) if low_l1 else w_fit

        # Step 1 (occupancy) and step 2 (L1 fit) -> the smallest column split that
        # satisfies both. num_w_chunks is MINIMIZED: a column cut multiplies the
        # read transaction count, so it is spent only where it is required.
        w_chunks_for_l1 = math.ceil(tensor_col_tiles / w_cap)
        w_chunks_for_occupancy = min(tensor_col_tiles, math.ceil(num_cores / tensor_row_blocks))
        num_w_chunks_target = max(w_chunks_for_l1, w_chunks_for_occupancy)

        # Coarsest DIVISOR of C that both fits L1 and yields >= the target chunks.
        # A divisor (rather than a ceil) is what removes the ragged column tail —
        # see this module's docstring for the mechanism cap that forces it.
        width_limit = min(w_cap, tensor_col_tiles // num_w_chunks_target)
        block_width_tiles = _largest_divisor_at_most(tensor_col_tiles, width_limit)
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
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()  # never a hardcoded core count
    plan = derive_plan(input_tensor, output_tensor, low_l1=low_l1, grid=grid)

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
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=plan.all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=plan.fp32_dest_acc_en),
    )

    return ttnn.ProgramDescriptor(
        kernels=[k for k in (reader_kernel, writer_kernel, compute_kernel) if k is not None],
        semaphores=[],
        cbs=[cb_input_rows, cb_output_tiles],
    )
