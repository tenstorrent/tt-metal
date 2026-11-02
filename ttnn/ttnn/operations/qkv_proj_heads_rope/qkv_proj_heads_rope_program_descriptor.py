# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""qkv_proj_heads_rope ProgramDescriptor: 2D token_rows x head_cols output-stationary grid; x K-blocks multicast
within each M-group (regime R3), W K-blocks multicast within each N-group (R2) where the layout allows it,
else W unicast per core (R1)."""

from __future__ import annotations

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
# The bias rides as one extra K tile (in0: ones-column tile, in1: bias tile row).
K_TILES = HIDDEN_TILES + 1
TOTAL_HEADS = NUM_THIRDS * NUM_HEADS
TABLE_TILES_PER_ROW = 2 * TILES_PER_HEAD

# Blocking-model host constants (op_design.md, Blocking Model).
K_BLOCK = 5
ACT_DEPTH = 2
WEIGHT_DEPTH = 2
OUT_DEPTH = 2
# Output rows alternate between the writer (NoC1) and the reader (NoC0); 1 = writer only. At 8x512 the
# store tail is NoC1-bandwidth bound once all 108 cores finish the K-loop together.
OUT_WRITE_NOCS = 2
CB_BUDGET_BYTES = 900 * 1024
OCCUPANCY_SLACK = 0.05
HEADS_PER_CORE_CANDIDATES = tuple(h for h in range(1, NUM_HEADS + 1) if NUM_HEADS % h == 0)

BF16_TILE_BYTES = ttnn.tile_size(ttnn.bfloat16)
FP32_TILE_BYTES = ttnn.tile_size(ttnn.float32)

CB_ACTIVATION_BLOCK = 0
CB_WEIGHT_BLOCK = 1
CB_MATMUL_PARTIALS = 2
CB_ROTARY_TABLE = 3
CB_OUT_TILES = 4
CB_OUT_TILES_NOC0 = 5

# Rectangles one multicast group may lower to (dataflow_kernel_lib::MAX_MCAST_RECTANGLES).
MAX_MCAST_RECTANGLES = 3
# Smallest N-group (= W DRAM crossings under R1) for which W is multicast (R2). Measured on p100a: at
# grid_m = 3 (3x36 grid) the per-round handshake over 3 one-core rectangles costs more than 3 W reads.
W_MCAST_MIN_GROUP = 4

assert K_TILES % K_BLOCK == 0, "k_block must divide K' (ragged last K-block not built)"
assert OUT_WRITE_NOCS in (1, 2)


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

    @property
    def n_tiles(self):
        return TILES_PER_HEAD * self.heads_per_core


def _cb_pages(block_m, n_tiles, k_block):
    """Page count per CB: the single source for the L1 footprint and the allocation (l1_ledger.md).
    cb_rotary_table holds a whole block's rows so the reader can fetch them during the K-loop."""
    pages = {
        CB_ACTIVATION_BLOCK: ACT_DEPTH * block_m * k_block,
        CB_WEIGHT_BLOCK: WEIGHT_DEPTH * k_block * n_tiles,
        CB_MATMUL_PARTIALS: block_m * n_tiles,
        CB_ROTARY_TABLE: block_m * TABLE_TILES_PER_ROW,
        CB_OUT_TILES: OUT_DEPTH * n_tiles,
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
}


def _footprint(block_m, n_tiles, k_block):
    return sum(pages * CB_FORMATS[cb][0] for cb, pages in _cb_pages(block_m, n_tiles, k_block).items())


def block_m_cap(n_tiles, k_block=K_BLOCK):
    """Largest block_m whose CB footprint fits CB_BUDGET_BYTES (ledger formula, derived from _cb_pages)."""
    fixed = _footprint(0, n_tiles, k_block)
    per_row = _footprint(1, n_tiles, k_block) - fixed
    return (CB_BUDGET_BYTES - fixed) // per_row


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


def select_blocking(device, batch, seq, compute_kernel_config):
    """Selection function (op_design.md, Core assignment): occupancy first (5% slack), then min ingress."""
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
        cap = block_m_cap(n_tiles)
        if cap < 1:
            continue
        rows_max = _div_up(mt, grid_m)
        block_m = min(rows_max, cap)
        num_blocks = _div_up(rows_max, block_m)
        tiles_max = rows_max * n_tiles
        ingress = K_TILES * (rows_max + num_blocks * n_tiles)
        candidates.append((tiles_max, ingress, heads_per_core, grid_m, grid_n, block_m))
    if not candidates:
        raise RuntimeError("qkv_proj_heads_rope: no core assignment fits the compute grid")

    min_tiles = min(c[0] for c in candidates)
    kept = [c for c in candidates if c[0] <= (1 + OCCUPANCY_SLACK) * min_tiles]
    _, _, heads_per_core, grid_m, grid_n, block_m = min(kept, key=lambda c: (c[1], -c[2]))
    sb_h, sb_w = _subblock(block_m, TILES_PER_HEAD * heads_per_core, _dest_tiles(compute_kernel_config))
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


def _core_assignment(device, blocking):
    """(core, n_group, m_group, row_start, rows) per core, and whether W is multicast (R2)."""
    grid = device.compute_with_storage_grid_size()
    mt, grid_m, grid_n = blocking.tensor_row_tiles, blocking.grid_m, blocking.grid_n
    layout, w_multicast = _choose_layout(grid.x, grid.y, grid_n, grid_m)
    base, extra = divmod(mt, grid_m)
    row_ranges = []
    start = 0
    for m_group in range(grid_m):
        rows = base + (1 if m_group < extra else 0)
        row_ranges.append((start, rows))
        start += rows
    # W rounds line up across an N-group only if its M-groups walk the same number of blocks.
    block_counts = {_div_up(rows, blocking.block_m) for _, rows in row_ranges}
    w_multicast = w_multicast and len(block_counts) == 1
    assignment = []
    for m_group in range(grid_m):
        for n_group in range(grid_n):
            core = ttnn.CoreCoord(*layout[(m_group, n_group)])
            assignment.append((core, n_group, m_group, *row_ranges[m_group]))
    return assignment, w_multicast


def _group_kernels(descriptor, kernel_args, runtime_args, groups, prefix, noc, first_semaphore_id, device):
    """One KernelDescriptor + one Mcast per group, so groups may be any <= 3-rectangle core sets (Mcast
    itself only partitions consecutive runs). Groups are disjoint, so they share semaphore IDs. The sender
    is staggered across groups. Returns the kernels and the next free semaphore ID."""
    kernels = []
    next_id = first_semaphore_id
    for index, group in enumerate(groups):
        group_cores = ttnn.CoreRangeSet(_rectangles([(c.x, c.y) for c in group]))
        rt = ttnn.RuntimeArgs()
        for core in group:
            rt[core.x][core.y] = runtime_args[(core.x, core.y)]
        kernel = ttnn.KernelDescriptor(core_ranges=group_cores, runtime_args=rt, **kernel_args)
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(handshake=True, noc=noc),
            group_cores,
            len(group),
            ttnn.McastFixedSenderConfig(sender_index=index % len(group)),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, prefix, [kernel], first_semaphore_id)
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
    cbs = [_cb(index, pages, cores) for index, pages in _cb_pages(block_m, n_tiles, k_block).items()]

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
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(cos).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(sin).get_compile_time_args())
    for output in outputs:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

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
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(bias).get_compile_time_args())
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
        ]
        writer_rt[(core.x, core.y)] = [
            w_addr,
            b_addr,
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
        "xmc",
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
    if w_multicast:
        writers, _ = _group_kernels(
            descriptor, writer_args, writer_rt, n_groups, "wmc", ttnn.NOC.NOC_1, next_semaphore_id, device
        )
    else:
        rt = ttnn.RuntimeArgs()
        for (x, y), args in writer_rt.items():
            rt[x][y] = args
        writer = ttnn.KernelDescriptor(core_ranges=cores, runtime_args=rt, **writer_args)
        ttnn.attach_absent_mcast(writer, "wmc")
        writers = [writer]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "qkv_proj_heads_rope_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    descriptor.kernels = [*readers, *writers, compute]
    return descriptor
