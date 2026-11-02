# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""qkv_proj_heads_rope ProgramDescriptor: 2D token_rows x head_cols output-stationary grid; x K-blocks multicast
within each M-group (regime R3), W K-blocks multicast within each N-group (R2) where the layout allows it,
else W unicast per core (R1)."""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
HIDDEN = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN // NUM_HEADS
NUM_THIRDS = 3
TILES_PER_HEAD = HEAD_DIM // TILE
HIDDEN_TILES = HIDDEN // TILE
# The bias is not a K tile: compute seeds the Float32 partials with its row broadcast and K-block 0
# accumulates onto the seed (Perf 1: the 25th K tile was 1/25 of the FPU-bound matmul).
K_TILES = HIDDEN_TILES
TOTAL_HEADS = NUM_THIRDS * NUM_HEADS
TABLE_TILES_PER_ROW = 2 * TILES_PER_HEAD

# Blocking-model host constants (op_design.md, Blocking Model).
# Perf 1, 8x512 with the bias seed: k_block 3 / 4 / 5 (ragged) within ~1%; 4 divides K' and saves 76 KiB.
K_BLOCK = 4
ACT_DEPTH = 2
WEIGHT_DEPTH = 2
# R2 sender pipelining: one more cb_weight_block slot lets the N-group's sender issue the DRAM reads of
# round k+1 before it multicasts round k (read and multicast overlap instead of adding). Taken only when
# W is multicast, the footprint at the chosen block_m still fits CB_BUDGET_BYTES, and the modelled W stream
# is at least PREFETCH_MIN_W_SHARE of the matmul: measured on p100a, 2x512 / 4x256 (W stream ~0.95x the
# matmul) 41.7 -> 39.0 us, while compute-bound shapes lose 1-2.5 us (38x106 134.5 -> 137.1, ~0.45x).
WEIGHT_DEPTH_PREFETCH = 3
PREFETCH_MIN_W_SHARE = 0.75
OUT_DEPTH = 2
# Output rows alternate between the writer (NoC1) and the reader (NoC0); 1 = writer only. At 8x512 the
# store tail is NoC1-bandwidth bound once all 108 cores finish the K-loop together.
OUT_WRITE_NOCS = 2
CB_BUDGET_BYTES = 900 * 1024
HEADS_PER_CORE_CANDIDATES = tuple(h for h in range(1, NUM_HEADS + 1) if NUM_HEADS % h == 0)

BF16_TILE_BYTES = ttnn.tile_size(ttnn.bfloat16)
FP32_TILE_BYTES = ttnn.tile_size(ttnn.float32)

CB_ACTIVATION_BLOCK = 0
CB_WEIGHT_BLOCK = 1
CB_MATMUL_PARTIALS = 2
CB_ROTARY_TABLE = 3
CB_OUT_TILES = 4
CB_OUT_TILES_NOC0 = 5
CB_BIAS = 6

# Rectangles one multicast group may lower to (dataflow_kernel_lib::MAX_MCAST_RECTANGLES).
MAX_MCAST_RECTANGLES = 3
# Smallest N-group (= W DRAM crossings under R1) for which W is multicast (R2). Measured on p100a: at
# grid_m = 3 (3x36 grid) the per-round handshake over 3 one-core rectangles costs more than 3 W reads.
W_MCAST_MIN_GROUP = 4
# R2 senders per N-group, alternating rounds (one Mcast channel each: "wmc", "wmc1"); with two, one sender
# reads its next round while the other multicasts. Measured on p100a (Refinement 5): 2 is slower and noisy
# on every R2 shape (2x512 39 -> 41-49 us, 8x256 59 -> 69-83 us), so 1 (the second channel attached absent).
W_SENDERS = 1
W_CHANNEL_PREFIXES = ("wmc", "wmc1")

# Selection cost model (device ns per stream; the slowest overlapped stream binds). Fitted on p100a against
# forced head-group runs of 12 (B, S) shapes x 5-6 candidates; it ranks the measured best first on all of them.
NS_PER_TILE_PRODUCT_PHASE = 13  # matmul, per tile-product per fidelity phase (HiFi3: 39 ns)
NS_PER_CORE_W_TILE = 110  # one core's W stream (its own unicast reads, or the R2 sender's)
NS_PER_DRAM_W_TILE = 10.8  # W tiles crossing DRAM across the whole grid (R1: grid_m times per block)
NS_PER_X_TILE_RECEIVER = 7.1  # R3 x multicast, per tile per receiver: large M-groups serialize
FIDELITY_PHASES = {
    ttnn.MathFidelity.LoFi: 1,
    ttnn.MathFidelity.HiFi2: 2,
    ttnn.MathFidelity.HiFi3: 3,
    ttnn.MathFidelity.HiFi4: 4,
}
W_TILES = K_TILES * TILES_PER_HEAD * TOTAL_HEADS

assert K_TILES % K_BLOCK == 0, "k_block must divide K' (ragged last K-block not built)"
assert OUT_WRITE_NOCS in (1, 2)
assert 1 <= W_SENDERS <= len(W_CHANNEL_PREFIXES)


def _div_up(a, b):
    return -(-a // b)


@dataclass(frozen=True)
class Blocking:
    heads_per_core: int
    grid_m: int
    grid_n: int
    block_m: int
    k_block: int
    tensor_row_tiles: int  # Mt
    seq_tiles: int  # St
    sb_h: int
    sb_w: int
    weight_depth: int

    @property
    def n_tiles(self):
        return TILES_PER_HEAD * self.heads_per_core


def _table_rows(block_m, seq_tiles):
    """Rows resident in cb_rotary_table: one per distinct sequence position of a block (tile-row r of a
    block uses slot r % seq_tiles), so the reader fetches them all during the K-loop."""
    return min(block_m, seq_tiles)


def _cb_pages(block_m, n_tiles, k_block, seq_tiles, weight_depth=WEIGHT_DEPTH):
    """Page count per CB: the single source for the L1 footprint and the allocation (l1_ledger.md)."""
    pages = {
        CB_ACTIVATION_BLOCK: ACT_DEPTH * block_m * k_block,
        CB_WEIGHT_BLOCK: weight_depth * k_block * n_tiles,
        CB_MATMUL_PARTIALS: block_m * n_tiles,
        CB_ROTARY_TABLE: _table_rows(block_m, seq_tiles) * TABLE_TILES_PER_ROW,
        CB_OUT_TILES: OUT_DEPTH * n_tiles,
        CB_BIAS: n_tiles,
    }
    if OUT_WRITE_NOCS == 2:
        pages[CB_OUT_TILES_NOC0] = OUT_DEPTH * n_tiles
    return pages


CB_FORMATS = {
    CB_ACTIVATION_BLOCK: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_WEIGHT_BLOCK: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_MATMUL_PARTIALS: (FP32_TILE_BYTES, ttnn.float32),
    CB_ROTARY_TABLE: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_OUT_TILES: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_OUT_TILES_NOC0: (BF16_TILE_BYTES, ttnn.bfloat16),
    CB_BIAS: (BF16_TILE_BYTES, ttnn.bfloat16),
}


def _footprint(block_m, n_tiles, k_block, seq_tiles, weight_depth=WEIGHT_DEPTH):
    pages = _cb_pages(block_m, n_tiles, k_block, seq_tiles, weight_depth)
    return sum(count * CB_FORMATS[cb][0] for cb, count in pages.items())


def block_m_cap(n_tiles, seq_tiles, k_block=K_BLOCK):
    """Largest block_m whose CB footprint fits CB_BUDGET_BYTES (ledger formula, derived from _cb_pages).
    The footprint is linear in block_m up to seq_tiles (table rows) and linear past it."""
    fixed = _footprint(0, n_tiles, k_block, seq_tiles)
    per_row = _footprint(1, n_tiles, k_block, seq_tiles) - fixed
    cap = (CB_BUDGET_BYTES - fixed) // per_row
    if cap <= seq_tiles:
        return cap
    table_bytes = seq_tiles * TABLE_TILES_PER_ROW * CB_FORMATS[CB_ROTARY_TABLE][0]
    return (CB_BUDGET_BYTES - fixed - table_bytes) // (per_row - TABLE_TILES_PER_ROW * CB_FORMATS[CB_ROTARY_TABLE][0])


def _dest_tiles(compute_kernel_config):
    tiles = 4 if compute_kernel_config.fp32_dest_acc_en else 8
    return tiles * (2 if compute_kernel_config.dst_full_sync_en else 1)


def _subblock(block_m, n_tiles, dest_tiles):
    """Max sb_h*sb_w <= dest_tiles with sb_h | block_m and sb_w | n_tiles; ties -> wider sb_w."""
    best = (1, 1)
    for sb_w in range(1, n_tiles + 1):
        if n_tiles % sb_w:
            continue
        for sb_h in range(1, block_m + 1):
            if block_m % sb_h or sb_h * sb_w > dest_tiles:
                continue
            if sb_h * sb_w > best[0] * best[1] or (sb_h * sb_w == best[0] * best[1] and sb_w > best[1]):
                best = (sb_h, sb_w)
    return best


def _block_m(rows_max, rows_min, cap):
    """block_m <= cap: M-groups walking the same block count first (W rounds only line up across an
    N-group then, see _core_assignment), then fewest blocks, then fewest nominal rows (each block computes
    block_m rows, valid or not)."""

    def cost(block_m):
        blocks = _div_up(rows_max, block_m)
        return (blocks != _div_up(rows_min, block_m), blocks, blocks * block_m)

    return min(range(1, min(rows_max, cap) + 1), key=lambda b: (cost(b), -b))


def _stream_ns(compute_kernel_config, grid_m, grid_n, n_tiles, rows_max, num_blocks, w_multicast):
    """Device-time estimates of one core assignment's overlapped streams: the matmul, the per-core W stream,
    the grid-wide W DRAM traffic and the x multicast. Under R1 every M-group re-reads W, so a candidate that
    loses R2 pays grid_m crossings; a large M-group pays for every x receiver."""
    phases = FIDELITY_PHASES[compute_kernel_config.math_fidelity]
    return {
        "matmul": rows_max * n_tiles * K_TILES * phases * NS_PER_TILE_PRODUCT_PHASE,
        "w_core": num_blocks * n_tiles * K_TILES * NS_PER_CORE_W_TILE,
        "w_dram": W_TILES * (1 if w_multicast else grid_m * num_blocks) * NS_PER_DRAM_W_TILE,
        "x_mcast": rows_max * K_TILES * (grid_n - 1) * NS_PER_X_TILE_RECEIVER,
    }


def select_blocking(device, batch, seq, compute_kernel_config):
    """Selection function (op_design.md, Core assignment): least modelled device time (_modelled_ns, which
    sees whether the layout realizes R2), then fewest output tiles per core, then least per-core ingress."""
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    seq_tiles = _div_up(seq, TILE)
    mt = batch * seq_tiles

    candidates = []
    for heads_per_core in HEADS_PER_CORE_CANDIDATES:
        grid_n = TOTAL_HEADS // heads_per_core
        grid_m = min(num_cores // grid_n, mt)
        if grid_m == 0:
            continue
        n_tiles = TILES_PER_HEAD * heads_per_core
        cap = block_m_cap(n_tiles, seq_tiles)
        if cap < 1:
            continue
        rows_max = _div_up(mt, grid_m)
        block_m = _block_m(rows_max, mt // grid_m, cap)
        num_blocks = _div_up(rows_max, block_m)
        w_multicast = _w_multicast(grid, grid_m, grid_n, mt, block_m)
        streams = _stream_ns(compute_kernel_config, grid_m, grid_n, n_tiles, rows_max, num_blocks, w_multicast)
        tiles_max = rows_max * n_tiles
        ingress = K_TILES * (rows_max + num_blocks * n_tiles)
        key = (max(streams.values()), tiles_max, ingress, -heads_per_core)
        candidates.append((key, heads_per_core, grid_m, grid_n, block_m, w_multicast, streams))
    if not candidates:
        raise RuntimeError("qkv_proj_heads_rope: no core assignment fits the compute grid")

    _, heads_per_core, grid_m, grid_n, block_m, w_multicast, streams = min(candidates, key=lambda c: c[0])
    n_tiles = TILES_PER_HEAD * heads_per_core
    sb_h, sb_w = _subblock(block_m, n_tiles, _dest_tiles(compute_kernel_config))
    # block_m is sized at WEIGHT_DEPTH, so the deeper W ring never costs a smaller block.
    prefetch = (
        w_multicast
        and streams["w_core"] >= PREFETCH_MIN_W_SHARE * streams["matmul"]
        and _footprint(block_m, n_tiles, K_BLOCK, seq_tiles, WEIGHT_DEPTH_PREFETCH) <= CB_BUDGET_BYTES
    )
    weight_depth = WEIGHT_DEPTH_PREFETCH if prefetch else WEIGHT_DEPTH
    return Blocking(
        heads_per_core=heads_per_core,
        grid_m=grid_m,
        grid_n=grid_n,
        block_m=block_m,
        k_block=K_BLOCK,
        tensor_row_tiles=mt,
        seq_tiles=seq_tiles,
        sb_h=sb_h,
        sb_w=sb_w,
        weight_depth=weight_depth,
    )


def _cb(index, num_pages, cores):
    page_size, data_format = CB_FORMATS[index]
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _run_cover(cells):
    """Row runs merged down identical x-extents, as (x0, y0, x1, y1)."""
    runs = []
    for y in sorted({y for _, y in cells}):
        xs = sorted(x for x, cy in cells if cy == y)
        start = xs[0]
        for prev, x in zip(xs, xs[1:] + [None]):
            if x != prev + 1:
                runs.append([start, y, prev, y])
                start = x
    rects = []
    for run in runs:
        above = next((r for r in rects if r[0] == run[0] and r[2] == run[2] and r[3] == run[1] - 1), None)
        if above:
            above[3] = run[1]
        else:
            rects.append(run)
    return rects


def _rectangles(cells):
    """Cover a core set with few rectangles (the units Mcast lowers a receiver set to): the smaller of the
    row-first and column-first run covers."""
    by_rows = _run_cover(cells)
    by_cols = [[y0, x0, y1, x1] for x0, y0, x1, y1 in _run_cover([(y, x) for x, y in cells])]
    rects = min(by_rows, by_cols, key=len)
    return [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in rects]


def _row_major_layout(grid_x, grid_y, grid_n, grid_m):
    """Logical l = m*grid_n + n row-major: each M-group is a consecutive run (<= 3 rectangles)."""
    return {(m, n): divmod(m * grid_n + n, grid_x)[::-1] for m in range(grid_m) for n in range(grid_n)}


def _folded_layout(grid_x, grid_y, grid_n, grid_m, transpose):
    """n along the primary axis, m along the secondary; M-groups past the secondary extent fold into the
    leftover primary band, n running along the secondary there. Each group is one line plus folds."""
    primary, secondary = (grid_y, grid_x) if transpose else (grid_x, grid_y)
    per_band = secondary // grid_n
    if grid_n > primary or per_band == 0:
        return None
    layout = {}
    for m in range(grid_m):
        for n in range(grid_n):
            if m < secondary:
                p, q = n, m
            else:
                band, slot = divmod(m - secondary, per_band)
                p, q = grid_n + band, slot * grid_n + n
            if p >= primary:
                return None
            layout[(m, n)] = (q, p) if transpose else (p, q)
    return layout


def _choose_layout(grid_x, grid_y, grid_n, grid_m):
    """Physical placement of the grid_m x grid_n logical grid, and whether W is multicast (R2).

    Each multicast group must lower to <= MAX_MCAST_RECTANGLES rectangles. Prefer the row-major layout
    (the R3 placement) when its N-groups already qualify; otherwise fold the logical grid so both channels
    are lines. No qualifying layout, or N-groups below W_MCAST_MIN_GROUP: row-major with W unicast (R1).
    """
    row_major = _row_major_layout(grid_x, grid_y, grid_n, grid_m)
    if grid_m < W_MCAST_MIN_GROUP:
        return row_major, False
    for layout in (row_major, *(_folded_layout(grid_x, grid_y, grid_n, grid_m, t) for t in (False, True))):
        if layout is None:
            continue
        n_groups = [[layout[(m, n)] for m in range(grid_m)] for n in range(grid_n)]
        m_groups = [[layout[(m, n)] for n in range(grid_n)] for m in range(grid_m)]
        if all(len(_rectangles(g)) <= MAX_MCAST_RECTANGLES for g in n_groups + m_groups):
            return layout, True
    return row_major, False


def _row_ranges(mt, grid_m):
    """(row_start, rows) per M-group: Mt tile-rows split as evenly as possible."""
    base, extra = divmod(mt, grid_m)
    ranges = []
    start = 0
    for m_group in range(grid_m):
        rows = base + (1 if m_group < extra else 0)
        ranges.append((start, rows))
        start += rows
    return ranges


def _w_multicast(grid, grid_m, grid_n, mt, block_m):
    """R2 is on when the layout qualifies and every M-group walks the same block count (W rounds only line
    up across an N-group then)."""
    block_counts = {_div_up(rows, block_m) for _, rows in _row_ranges(mt, grid_m)}
    return len(block_counts) == 1 and _choose_layout(grid.x, grid.y, grid_n, grid_m)[1]


def _core_assignment(device, blocking):
    """(core, n_group, m_group, row_start, rows) per core, and whether W is multicast (R2)."""
    grid = device.compute_with_storage_grid_size()
    mt, grid_m, grid_n = blocking.tensor_row_tiles, blocking.grid_m, blocking.grid_n
    layout, _ = _choose_layout(grid.x, grid.y, grid_n, grid_m)
    w_multicast = _w_multicast(grid, grid_m, grid_n, mt, blocking.block_m)
    row_ranges = _row_ranges(mt, grid_m)
    assignment = []
    for m_group in range(grid_m):
        for n_group in range(grid_n):
            core = ttnn.CoreCoord(*layout[(m_group, n_group)])
            assignment.append((core, n_group, m_group, *row_ranges[m_group]))
    return assignment, w_multicast


def _group_kernels(descriptor, kernel_args, runtime_args, groups, prefixes, noc, first_semaphore_id, device):
    """One KernelDescriptor per group with one Mcast per channel prefix, so groups may be any <= 3-rectangle
    core sets (Mcast itself only partitions consecutive runs). Groups are disjoint, so they share semaphore
    IDs. Senders are staggered across groups and spread evenly across a group's channels. Returns the
    kernels and the next free semaphore ID."""
    kernels = []
    next_id = first_semaphore_id
    for index, group in enumerate(groups):
        group_cores = ttnn.CoreRangeSet(_rectangles([(c.x, c.y) for c in group]))
        rt = ttnn.RuntimeArgs()
        for core in group:
            rt[core.x][core.y] = runtime_args[(core.x, core.y)]
        kernel = ttnn.KernelDescriptor(core_ranges=group_cores, runtime_args=rt, **kernel_args)
        next_id = first_semaphore_id
        for channel, prefix in enumerate(prefixes):
            sender = (index + channel * len(group) // len(prefixes)) % len(group)
            mcast = ttnn.Mcast(
                device,
                ttnn.McastConfig(handshake=True, noc=noc),
                group_cores,
                len(group),
                ttnn.McastFixedSenderConfig(sender_index=sender),
                ttnn.McastCoreOrder.RowMajor,
            )
            mcast.attach(descriptor, prefix, [kernel], next_id)
            next_id = mcast.next_semaphore_id()
        kernels.append(kernel)
    return kernels, next_id


def create_program_descriptor(input_tensor, weight, bias, cos, sin, outputs, blocking, compute_kernel_config):
    device = input_tensor.device()
    n_tiles = blocking.n_tiles
    block_m = blocking.block_m
    k_block = blocking.k_block
    num_k_blocks = K_TILES // k_block
    weight_width_tiles = weight.padded_shape[-1] // TILE
    groups_per_third = NUM_HEADS // blocking.heads_per_core

    assignment, w_multicast = _core_assignment(device, blocking)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, *_ in assignment])
    cbs = [
        _cb(index, pages, cores)
        for index, pages in _cb_pages(block_m, n_tiles, k_block, blocking.seq_tiles, blocking.weight_depth).items()
    ]

    last_valid_rows = input_tensor.shape[-2] % TILE
    reader_ct_args = [
        CB_ACTIVATION_BLOCK,
        CB_ROTARY_TABLE,
        block_m,
        k_block,
        num_k_blocks,
        HIDDEN_TILES,
        blocking.seq_tiles,
        BF16_TILE_BYTES,
        TILES_PER_HEAD,
        OUT_WRITE_NOCS,
        CB_OUT_TILES_NOC0,
        n_tiles,
        NUM_HEADS,
        last_valid_rows,
        CB_BIAS,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(cos).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(sin).get_compile_time_args())
    for output in outputs:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(bias).get_compile_time_args())

    writer_ct_args = [
        CB_WEIGHT_BLOCK,
        CB_OUT_TILES,
        block_m,
        n_tiles,
        k_block,
        num_k_blocks,
        HIDDEN_TILES,
        weight_width_tiles,
        blocking.seq_tiles,
        NUM_HEADS,
        TILES_PER_HEAD,
        BF16_TILE_BYTES,
        last_valid_rows,
        OUT_WRITE_NOCS,
        blocking.weight_depth,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    for output in outputs:
        writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    compute_ct_args = [
        CB_ACTIVATION_BLOCK,
        CB_WEIGHT_BLOCK,
        CB_MATMUL_PARTIALS,
        CB_ROTARY_TABLE,
        CB_OUT_TILES,
        block_m,
        blocking.heads_per_core,
        TILES_PER_HEAD,
        k_block,
        num_k_blocks,
        blocking.sb_h,
        blocking.sb_w,
        OUT_WRITE_NOCS,
        CB_OUT_TILES_NOC0,
        blocking.seq_tiles,
        _dest_tiles(compute_kernel_config),
        CB_BIAS,
    ]

    reader_rt = {}
    writer_rt = {}
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    cos_addr, sin_addr = cos.buffer_address(), sin.buffer_address()
    w_addr, b_addr = weight.buffer_address(), bias.buffer_address()
    out_addrs = [o.buffer_address() for o in outputs]
    for core, n_group, m_group, row_start, core_rows in assignment:
        third = n_group // groups_per_third
        first_head = (n_group % groups_per_third) * blocking.heads_per_core
        first_col_tile = third * HIDDEN_TILES + TILES_PER_HEAD * first_head
        is_rotary = int(third < 2)
        reader_rt[(core.x, core.y)] = [
            x_addr,
            cos_addr,
            sin_addr,
            row_start,
            core_rows,
            is_rotary,
            *out_addrs,
            third,
            first_head,
            b_addr,
            first_col_tile,
        ]
        writer_rt[(core.x, core.y)] = [
            w_addr,
            *out_addrs,
            third,
            first_col_tile,
            first_head,
            row_start,
            core_rows,
            m_group,
            blocking.grid_m,
        ]
        compute_rt[core.x][core.y] = [core_rows, is_rotary]

    descriptor = ttnn.ProgramDescriptor(cbs=cbs)
    m_groups = [[core for core, _, m, *_ in assignment if m == m_group] for m_group in range(blocking.grid_m)]
    n_groups = [[core for core, n, *_ in assignment if n == n_group] for n_group in range(blocking.grid_n)]
    # R3: x does not vary along head_cols, so each M-group's grid_n cores share every x K-block: one core
    # reads it and multicasts it. Handshake on: the landing slot is a double-buffered CB slot a receiver
    # may still be consuming.
    readers, next_semaphore_id = _group_kernels(
        descriptor,
        dict(
            kernel_source=str(KERNEL_DIR / "qkv_proj_heads_rope_reader.cpp"),
            compile_time_args=reader_ct_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        reader_rt,
        m_groups,
        ["xmc"],
        ttnn.NOC.NOC_0,
        0,
        device,
    )
    # R2: W does not vary along token_rows, so each N-group's grid_m cores share every W K-block. Rides the
    # writer (NOC_1), so the two channels use different NoCs.
    writer_args = dict(
        kernel_source=str(KERNEL_DIR / "qkv_proj_heads_rope_writer.cpp"),
        compile_time_args=writer_ct_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    w_senders = min(W_SENDERS, blocking.grid_m)
    if w_multicast:
        writers, _ = _group_kernels(
            descriptor,
            writer_args,
            writer_rt,
            n_groups,
            W_CHANNEL_PREFIXES[:w_senders],
            ttnn.NOC.NOC_1,
            next_semaphore_id,
            device,
        )
    else:
        rt = ttnn.RuntimeArgs()
        for (x, y), args in writer_rt.items():
            rt[x][y] = args
        writers = [ttnn.KernelDescriptor(core_ranges=cores, runtime_args=rt, **writer_args)]
        w_senders = 0
    for writer in writers:
        for prefix in W_CHANNEL_PREFIXES[w_senders:]:
            ttnn.attach_absent_mcast(writer, prefix)
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "qkv_proj_heads_rope_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    # tail_v2 fork: QKV_TAIL=<name>[,<name>] -> TAIL_<NAME> define on every kernel; QKV_ABLATE likewise.
    names = [f"TAIL_{n.upper()}" for n in os.environ.get("QKV_TAIL", "").split(",") if n]
    names += [d for d in os.environ.get("QKV_ABLATE", "").split(",") if d]
    extra = [(d, "1") for d in names]
    if "TAIL_HALFSPLIT" in names:
        extra.append(("TAIL_WT", str(min(_dest_tiles(compute_kernel_config), n_tiles))))
    for kernel in [*readers, *writers, compute]:
        kernel.defines = list(kernel.defines) + extra
    descriptor.kernels = [*readers, *writers, compute]
    return descriptor
