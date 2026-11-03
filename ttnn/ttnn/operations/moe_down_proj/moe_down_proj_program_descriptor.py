# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_down_proj ProgramDescriptor: regime R1 (line-lockstep 2D, K-streamed, static multicasts).

Rows (expert-sorted tile rows, split on device from counts) over m-groups, output columns over n-lines. Per K round,
each m-group's rotating sender multicasts its rows' x block along the group, and each n-line's rotating sender
multicasts every in-wave expert's w chunk along the line. Every core accumulates its rows x n-slice in fp32 L1
partials (packer L1 accumulation), untilizes them per wave and stores bf16 row segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
NUM_EXPERTS = 8
HIDDEN = 768
FFN = 3072
TENSOR_K_TILES = FFN // TILE
TENSOR_N_TILES = HIDDEN // TILE

# Largest K block per round. Round 0 must land before compute starts, so a larger block lengthens the fill.
K_BLOCK_MAX = 8
# select_blocking takes the fewest waves among K blocks >= K_BLOCK_ONE_PASS_MIN while a wave's per-core rows stay
# <= ONE_PASS_MAX_ROWS, then the largest K block. Fewer waves skip re-reading the straddling expert's w; a smaller K
# block costs more per-round sync, and more rows per step lengthen the x chain. t3072 (8x384, 23 rows at 1 wave):
# 2 waves at k = 8 138.5-140.8 -> 1 wave at k = 6 133.5-134.8 us. t4096 (30 rows at 1 wave): L1 x needs k = 4
# (164 -> 177 us); DRAM x fits k = 6 (208 -> 223 us). Only with y in L1: one wave leaves the whole store as the tail,
# which a DRAM y store (about 4x the L1 one) cannot afford (t3072 DRAM y 159 -> 165 us).
K_BLOCK_ONE_PASS_MIN = 6
ONE_PASS_MAX_ROWS = 24
MAX_WAVES = 4
X_DEPTH = 2
W_DEPTH = 2
OUT_DEPTH = 2
# With num_waves >= 2, cb_output_rows holds a whole wave block, so compute untilizes wave p without waiting on the
# store, and the store drains over wave p + 1's rounds a quota per round instead of one burst that stalls compute and
# the w channel on the storing RISC. 8x512 layers 0-5 (L1 / L1, 2 waves): 172-174 -> 161-164 us.
STORE_PACED = True
MAX_TOKENS = 4096
MAX_ROWS = TILE * (-(-2 * MAX_TOKENS // TILE) + NUM_EXPERTS - 1)
# CBs are allocated upward from the L1 allocator base and must end below the lowest live L1 buffer, so the budget is
# read from the allocator at call time (other model tensors may be alive). Each CB start is aligned to this (the DRAM
# alignment, 64 B on Blackhole, 32 B on Wormhole).
CB_ADDR_ALIGN = 64

CB_X_ROUND = 0
CB_W_ROUND = 1
CB_PARTIALS = 2
CB_OUTPUT_ROWS = 3
CB_COUNTS_COMPUTE = 4
CB_COUNTS_WRITER = 5
# Sender-private staging: a rotating sender prefetches its next owned step here, then multicasts it into the slot.
CB_X_STAGE = 6
CB_W_STAGE = 7
# Writer -> reader: the w sender's (first owned step, period), for the reader's share of the w prefetch.
CB_W_OWNER = 8
W_OWNER_PAGE_BYTES = 16
# A staged prefetch's reads are issued over this many channel steps (a quota per step), so a sender never stalls
# the steps it receives in between on one long issue burst. 8x512 layer 0 / t = 1024: 1 -> 259 / 120 us,
# 4 -> 202 / 108, 9 -> 179 / 106, 16 -> 174 / 102, 32 -> 178 / 103, 128 -> 193 / 120.
PREFETCH_ISSUE_STEPS = 16
# Each owned w step's DRAM reads split by tile: w_reader_share / W_SHARE_DEN go on the reader RISC (NoC1) into the
# stage CB, the rest stay on the writer RISC (NoC0), which multicasts the stage. NoC1 DRAM reads are about half as
# fast here, but w reads sharing NoC0 with the w multicast cost t = 1024 about 22 us, so a share pays while the
# x channel (also NoC1) is light: the best share falls with the core's rows per step (x bytes / w bytes per step
# = rows / 16 at any K block). Best measured (eighths, us from share 0): 1 row 4 (44.1 -> 38.9), 2 rows 4-5
# (135 -> 76-81), 3 rows 3-4 (94.6 -> 79-82), 8 rows 3 (102.5 -> 84.8), 14-15 rows 1-2 (110 -> 103,
# 113 -> 106.5), 23 rows 0. With >= 2 waves every share measured worse (t3072 at 2 waves, t4096: x binds NoC1).
W_SHARE_DEN = 8
W_SHARE_MAX = 4
W_SHARE_ROWS_ZERO = 25
W_SHARE_ROWS_PER_STEP = 5

X_MAIN, X_STRIP, W_MAIN, W_STRIP = "xa", "xb", "wa", "wb"
# NoC of each channel's RISC (the reader RISC carries x, the writer RISC carries w and the y store). w's DRAM reads
# on NoC0 measured 8x512 layer 0 229 -> 176 us, t = 1024 125 -> 107 us against the swapped assignment.
READER_NOC = ttnn.NOC.NOC_1
WRITER_NOC = ttnn.NOC.NOC_0
# A DRAM y store runs on the reader RISC (NoC1, the DRAM-write NoC): 8x512 layer 0 / t = 1024 L1 x, DRAM y
# 254 -> 220 / 124 -> 113 us. An L1 y store stays on the writer RISC (moving it measured 172 -> 179 us at 8x512).


def _div_up(a, b):
    return -(-a // b)


def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


@dataclass
class Layout:
    name: str
    num_n_lines: int
    num_m_groups: int
    cores: object
    # (x, y) -> (n-line, m-group)
    roles: dict = field(default_factory=dict)
    # prefix -> (receivers, group size, sender config, receiver order)
    channels: dict = field(default_factory=dict)

    @property
    def core_n_tiles(self):
        return TENSOR_N_TILES // self.num_n_lines


def _rotating(group_size):
    # A one-core group has no other sender to rotate to.
    if group_size > 1:
        return ttnn.McastRotatingSenderConfig()
    return ttnn.McastFixedSenderConfig(sender_index=0)


def _rect_layout(grid):
    """n-line l = grid row y = l, m-group g = grid column x = g."""
    lines = max(d for d in range(1, grid.y + 1) if TENSOR_N_TILES % d == 0)
    layout = Layout("rect", lines, grid.x, _crs(0, 0, grid.x - 1, lines - 1))
    for x in range(grid.x):
        for y in range(lines):
            layout.roles[(x, y)] = (y, x)
    layout.channels[W_MAIN] = (layout.cores, grid.x, _rotating(grid.x), ttnn.McastCoreOrder.RowMajor)
    layout.channels[X_MAIN] = (layout.cores, lines, _rotating(lines), ttnn.McastCoreOrder.ColumnMajor)
    return layout


def _l_layout(grid):
    """n-lines 0..gx-1 = columns of the rect (gx x h, h = gy - 1); n-line gx = the strip y = gy - 1, x < h.
    m-group g = rect row g plus strip core (g, gy - 1). None where the shape does not exist."""
    h = grid.y - 1
    lines = grid.x + 1
    if h < 2 or h > grid.x or TENSOR_N_TILES % lines:
        return None
    rect = _crs(0, 0, grid.x - 1, h - 1)
    strip = _crs(0, grid.y - 1, h - 1, grid.y - 1)
    layout = Layout("L", lines, h, rect.merge(strip))
    for x in range(grid.x):
        for y in range(h):
            layout.roles[(x, y)] = (x, y)
    for x in range(h):
        layout.roles[(x, grid.y - 1)] = (grid.x, x)
    layout.channels[W_MAIN] = (rect, h, _rotating(h), ttnn.McastCoreOrder.ColumnMajor)
    layout.channels[W_STRIP] = (strip, h, _rotating(h), ttnn.McastCoreOrder.RowMajor)
    layout.channels[X_MAIN] = (rect, grid.x, _rotating(grid.x), ttnn.McastCoreOrder.RowMajor)
    # Group g's senders are rect row g, rotating in step with X_MAIN, so one core sends both copies of a round.
    layout.channels[X_STRIP] = (
        strip,
        1,
        ttnn.McastRotatingSenderConfig(rect, sender_order=ttnn.McastCoreOrder.RowMajor),
        ttnn.McastCoreOrder.RowMajor,
    )
    return layout


LAYOUTS = {"rect": _rect_layout, "L": _l_layout}


def select_layout(grid, tensor_row_tiles, force=None):
    """Lowest busiest-core output tile count wins; ties keep the rectangle. `force` pins one (tests only)."""
    if force is not None:
        layout = LAYOUTS[force](grid)
        if layout is None:
            raise RuntimeError(f"moe_down_proj: layout {force} does not exist on a {grid.x}x{grid.y} grid")
        return layout
    best = None
    for build in (_rect_layout, _l_layout):
        layout = build(grid)
        if layout is None:
            continue
        score = _div_up(tensor_row_tiles, layout.num_m_groups) * layout.core_n_tiles
        if best is None or score < best[0]:
            best = (score, layout)
    return best[1]


@dataclass(frozen=True)
class Blocking:
    tensor_row_tiles: int
    num_waves: int
    k_block_tiles: int
    core_row_tiles_max: int

    @property
    def num_k_rounds(self):
        return TENSOR_K_TILES // self.k_block_tiles


def _tile_bytes():
    return ttnn.tile_size(ttnn.bfloat8_b), ttnn.tile_size(ttnn.float32), ttnn.tile_size(ttnn.bfloat16)


def store_paced(num_waves):
    return STORE_PACED and num_waves > 1


def out_depth_blocks(core_row_tiles_max, num_waves):
    """Untilized tile-row blocks cb_output_rows holds."""
    return max(OUT_DEPTH, core_row_tiles_max) if store_paced(num_waves) else OUT_DEPTH


def store_blocks_per_round(core_row_tiles_max, num_waves, num_k_rounds, store_start_round):
    """Blocks of wave p one store call drains per round of wave p + 1, from round store_start_round on."""
    if not store_paced(num_waves) or num_k_rounds <= store_start_round:
        return core_row_tiles_max
    return _div_up(core_row_tiles_max, num_k_rounds - store_start_round)


def cb_plan(core_row_tiles_max, k_block_tiles, core_n_tiles, counts_page_bytes, num_waves):
    """(index, pages, page bytes, format) per CB. Single source for both the budget solve and the allocation."""
    in_tile, acc_tile, out_tile = _tile_bytes()
    x_round_tiles = core_row_tiles_max * k_block_tiles
    w_round_tiles = NUM_EXPERTS * core_n_tiles * k_block_tiles
    return [
        (CB_X_ROUND, X_DEPTH * x_round_tiles, in_tile, ttnn.bfloat8_b),
        (CB_W_ROUND, W_DEPTH * w_round_tiles, in_tile, ttnn.bfloat8_b),
        (CB_X_STAGE, x_round_tiles, in_tile, ttnn.bfloat8_b),
        (CB_W_STAGE, w_round_tiles, in_tile, ttnn.bfloat8_b),
        (CB_PARTIALS, core_row_tiles_max * core_n_tiles, acc_tile, ttnn.float32),
        (CB_OUTPUT_ROWS, out_depth_blocks(core_row_tiles_max, num_waves) * core_n_tiles, out_tile, ttnn.bfloat16),
        (CB_COUNTS_COMPUTE, 1, counts_page_bytes, ttnn.uint32),
        (CB_COUNTS_WRITER, 1, counts_page_bytes, ttnn.uint32),
        (CB_W_OWNER, 1, W_OWNER_PAGE_BYTES, ttnn.uint32),
    ]


def footprint_bytes(core_row_tiles_max, k_block_tiles, core_n_tiles, counts_page_bytes, num_waves):
    return sum(
        _div_up(pages * page_bytes, CB_ADDR_ALIGN) * CB_ADDR_ALIGN
        for _, pages, page_bytes, _ in cb_plan(
            core_row_tiles_max, k_block_tiles, core_n_tiles, counts_page_bytes, num_waves
        )
    )


def _l1_view(device):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    # Block addresses are relative to the allocator base, where the CB region starts.
    blocks = [(int(b["address"]), int(b["size"]), b["allocated"] == "yes") for b in view.block_table]
    return view.total_bytes_per_bank, view.num_banks, blocks


def cb_budget_bytes(device, pending_l1_bank_bytes=0):
    """Per-core bytes free for CBs: from the allocator base to the lowest live L1 buffer.

    pending_l1_bank_bytes: per-bank size of an L1 buffer not allocated yet (the output, at validate time). It is
    charged against the budget unless a free block above the lowest live buffer can hold it, which the allocator
    (top-down, smallest size class first) takes before the open space below."""
    total, _, blocks = _l1_view(device)
    lowest = min((addr for addr, _, used in blocks if used), default=total)
    if pending_l1_bank_bytes and not any(
        not used and addr >= lowest and size >= pending_l1_bank_bytes for addr, size, used in blocks
    ):
        lowest -= pending_l1_bank_bytes
    return max(0, lowest)


def output_l1_bank_bytes(device, rows, out_memory):
    """Per-bank bytes of the (rows, HIDDEN) bf16 ROW_MAJOR output if it goes to L1, else 0."""
    if out_memory.buffer_type != ttnn.BufferType.L1:
        return 0
    _, num_banks, _ = _l1_view(device)
    return _div_up(rows, num_banks) * HIDDEN * 2


def select_blocking(tensor_row_tiles, layout, counts_page_bytes, budget_bytes, out_in_dram=False):
    """Unless out_in_dram, the fewest waves (<= ONE_PASS_MAX_ROWS rows per core), then the largest
    k_block_tiles >= K_BLOCK_ONE_PASS_MIN, that fit budget_bytes; failing that the largest k_block_tiles (divisor of
    96, <= K_BLOCK_MAX), then the fewest waves."""
    k_blocks = [k for k in range(K_BLOCK_MAX, 0, -1) if TENSOR_K_TILES % k == 0]
    waves = range(1, MAX_WAVES + 1)
    # Unrestricted fewest waves first (one wave at k = 4) measured 8x512 layer 0 172 -> 177 us (L1 / L1), 221 -> 223 us
    # (DRAM x), hence the K block floor.
    order = (
        [] if out_in_dram else [(k, w, ONE_PASS_MAX_ROWS) for w in waves for k in k_blocks if k >= K_BLOCK_ONE_PASS_MIN]
    )
    order += [(k, w, None) for k in k_blocks for w in waves]
    for k_block_tiles, num_waves, max_rows in order:
        core_row_tiles_max = _div_up(_div_up(tensor_row_tiles, num_waves), layout.num_m_groups)
        if max_rows is not None and core_row_tiles_max > max_rows:
            continue
        if (
            footprint_bytes(core_row_tiles_max, k_block_tiles, layout.core_n_tiles, counts_page_bytes, num_waves)
            <= budget_bytes
        ):
            return Blocking(tensor_row_tiles, num_waves, k_block_tiles, core_row_tiles_max)
    raise UnsupportedAxisValue(
        f"moe_down_proj: no (num_waves <= {MAX_WAVES}, k_block_tiles) fits the {budget_bytes} B of L1 free for CBs "
        f"for {tensor_row_tiles} tile rows on the {layout.name} layout"
    )


def w_reader_share(blocking):
    """Eighths (W_SHARE_DEN) of each w step's DRAM reads the reader RISC issues; 0 keeps them all on the writer."""
    if blocking.num_waves > 1:
        return 0
    return max(0, min(W_SHARE_MAX, (W_SHARE_ROWS_ZERO - blocking.core_row_tiles_max) // W_SHARE_ROWS_PER_STEP))


def _cb(index, num_pages, page_bytes, dtype, cores):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def create_program_descriptor(input_tensor, weight, expert_counts, output, compute_kernel_config, layout_override=None):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    tensor_row_tiles = list(input_tensor.shape)[-2] // TILE
    counts_page_bytes = expert_counts.buffer_page_size()
    layout = select_layout(grid, tensor_row_tiles, layout_override)
    out_in_dram = output.memory_config().buffer_type == ttnn.BufferType.DRAM
    budget_bytes = cb_budget_bytes(device)
    blocking = select_blocking(tensor_row_tiles, layout, counts_page_bytes, budget_bytes, out_in_dram)
    core_n_tiles = layout.core_n_tiles
    in_tile = ttnn.tile_size(ttnn.bfloat8_b)
    cores = layout.cores

    store_on_reader = out_in_dram
    w_share = w_reader_share(blocking)
    # The store starts once the storing RISC has filled its channel's slots for wave p + 1 (X_DEPTH / W_DEPTH rounds).
    store_quota = store_blocks_per_round(
        blocking.core_row_tiles_max,
        blocking.num_waves,
        blocking.num_k_rounds,
        X_DEPTH if store_on_reader else W_DEPTH,
    )
    plan_ct = [tensor_row_tiles, blocking.num_waves, layout.num_m_groups]
    reader_ct = [
        CB_X_ROUND,
        CB_X_STAGE,
        CB_COUNTS_COMPUTE,
        CB_COUNTS_WRITER,
        *plan_ct,
        blocking.k_block_tiles,
        blocking.num_k_rounds,
        blocking.core_row_tiles_max,
        TENSOR_K_TILES,
        in_tile,
        counts_page_bytes,
        PREFETCH_ISSUE_STEPS,
        int(store_on_reader),
        CB_OUTPUT_ROWS,
        core_n_tiles,
        output.buffer_page_size(),
        X_DEPTH,
        store_quota,
        w_share,
        W_SHARE_DEN,
        CB_W_STAGE,
        CB_W_OWNER,
    ]
    for tensor in (input_tensor, expert_counts, output, weight):
        reader_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    writer_ct = [
        CB_OUTPUT_ROWS,
        CB_COUNTS_WRITER,
        CB_W_ROUND,
        *plan_ct,
        blocking.k_block_tiles,
        blocking.num_k_rounds,
        core_n_tiles,
        TENSOR_K_TILES,
        w_share,
        in_tile,
        output.buffer_page_size(),
        W_DEPTH,
        CB_W_STAGE,
        PREFETCH_ISSUE_STEPS,
        int(store_on_reader),
        store_quota,
        CB_W_OWNER,
        W_SHARE_DEN,
    ]
    for tensor in (output, weight):
        writer_ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    compute_ct = [
        CB_X_ROUND,
        CB_W_ROUND,
        CB_PARTIALS,
        CB_OUTPUT_ROWS,
        CB_COUNTS_COMPUTE,
        *plan_ct,
        blocking.k_block_tiles,
        blocking.num_k_rounds,
        blocking.core_row_tiles_max,
        core_n_tiles,
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for (x, y), (line, group) in layout.roles.items():
        reader_rt[x][y] = [
            input_tensor.buffer_address(),
            expert_counts.buffer_address(),
            group,
            output.buffer_address(),
            line,
            weight.buffer_address(),
        ]
        writer_rt[x][y] = [output.buffer_address(), weight.buffer_address(), group, line]
        compute_rt[x][y] = [group]

    cbs = [
        _cb(index, pages, page_bytes, dtype, cores)
        for index, pages, page_bytes, dtype in cb_plan(
            blocking.core_row_tiles_max, blocking.k_block_tiles, core_n_tiles, counts_page_bytes, blocking.num_waves
        )
    ]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "moe_down_proj_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=READER_NOC),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "moe_down_proj_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=WRITER_NOC),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "moe_down_proj_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    # x channel on the reader RISC, w channel on the writer RISC, each multicasting on its kernel's NoC.
    channel_kernels = {
        X_MAIN: (reader, READER_NOC),
        X_STRIP: (reader, READER_NOC),
        W_MAIN: (writer, WRITER_NOC),
        W_STRIP: (writer, WRITER_NOC),
    }
    semaphore_id = 0
    for prefix, (kernel, noc) in channel_kernels.items():
        if prefix not in layout.channels:
            ttnn.attach_absent_mcast(kernel, prefix)
            continue
        receivers, group_size, sender_config, order = layout.channels[prefix]
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=noc, handshake=True),
            receivers,
            group_size,
            sender_config,
            order,
        )
        mcast.attach(descriptor, prefix, [kernel], semaphore_id)
        semaphore_id = mcast.next_semaphore_id()
    descriptor.kernels = [reader, writer, compute]
    return descriptor
