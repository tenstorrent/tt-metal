# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_qkv_rotary_heads — ProgramDescriptor (regime `mcast2d_fused_rotary`, op_design.md).

Grid: logical cores (c, y), c in [0, num_columns), y in [0, NUM_LANES). Column c owns M rows
[c * core_m_tiles, min(M, (c + 1) * core_m_tiles)); row y owns lane y = N tiles [y * LANE_N_TILES, (y + 1) * LANE_N_TILES).

  RISCV_0 / NOC0 (reader): in0 stream. The column's top core (c, 0) gathers x from the norm's shards (x source
  table) and multicasts each (M block, K block) down the column; the other cores receive. q/k cores also load their
  M block's cos / sin table tiles.
  RISCV_1 / NOC1 (writer): in1 stream + head scatter. The row's in1 sender (W_SENDER: by default the right core
  (num_columns - 1, y)) reads each K block of the lane's weight pages from DRAM (M block 0 only) and multicasts it to
  the row; later M blocks re-push the resident ring.
  Every core writes its finished output tiles into their q / k / v pages.
  Compute: matmul_block (packer L1 accumulation, Out target); q/k cores rotate each head in fp32 DEST.

Every CB is declared on every core in the same order and size: the multicasts land at the sender's slot address.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import ttnn

from .nomic_qkv_rotary_heads import (
    BF16_TILE_BYTES,
    BIAS_K_TILE,
    CS_ON_RELOAD,
    FP32_TILE_BYTES,
    HEAD_TILES,
    HEADS,
    HIDDEN_TILES,
    K_BLOCK_TILES,
    K_TILES,
    LANE_N_TILES,
    NUM_K_BLOCKS,
    NUM_ROTARY_TENSORS,
    OUT_DEPTH_BLOCKS,
    OUT_DRAIN_TILES,
    QK_SUBBLOCK_W,
    OUT_LAYOUT_ROW_MAJOR,
    ROTARY_ON_PACK,
    ROTARY_TABLE_DEPTH,
    TABLE_LOAD_K_BLOCK,
    TABLE_TILES_PER_ROW,
    TENSOR_N_TILES,
    TILE,
    WEIGHT_PAGE_TILES,
    WEIGHT_PAGES_PER_K_ROW,
    WEIGHT_READ_AHEAD_CHUNKS,
    W_MCAST_CHUNK_K_ROWS,
    W_SENDER,
    W_SENDER_DIAGONAL,
    W_SENDER_RIGHT,
    W_SENDER_ROTATING,
    W_SENDER_STAGGER,
    WIDE_SUBBLOCK_W,
    X_BLOCK_DEPTH,
    Plan,
)

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names; declared in this order on every core).
CB_X_BLOCK = 0
CB_WEIGHTS = 1
CB_ROTARY_TABLE = 2
CB_PARTIALS = 3
CB_HEADS_OUT = 4


# Measurement-only overrides (perf ablation, /perf-measure): QKV_<NAME>=<int> replaces a knob or turns a stub on.
# Every default is the production value; the kernels compile the stubs out at 0.
def _env(name, default=0):
    return int(os.environ.get(f"QKV_{name}", default))


READER_NOC = ttnn.NOC.NOC_0
WRITER_NOC = ttnn.NOC.NOC_1


def _crs(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _cb(index, pages, page_size, dtype, cores):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _x_source_table(plan: Plan, x, device):
    """Per column: for each of its rows and each K part, the holder's packed NoC (x << 16 | y) and the L1 address of
    the row's first tile in that part. Derived from x's actual shard spec: an NdShardSpec (round-robin over the shard
    grid, ln_layout) or a legacy BLOCK_SHARDED ShardSpec (model_layout). Either way, the source of tile row m
    (= b * S_t + s_t, the padded height / 32) and K part j is one (holder core, L1 address)."""
    config = x.memory_config()
    padded = list(x.padded_shape)
    assert padded[-1] == HIDDEN_TILES * TILE
    page_bytes = x.buffer_aligned_page_size()
    base = x.buffer_address()  # a sharded buffer sits at the same L1 address on every holder
    noc_of = {}

    def noc_xy(core):
        key = (core.x, core.y)
        if key not in noc_of:
            noc = device.worker_core_from_logical_core(core)
            noc_of[key] = (noc.x << 16) | noc.y
        return noc_of[key]

    spec = config.nd_shard_spec
    if config.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED and config.shard_spec is not None:
        # Legacy 2D block sharding: shard (r, j) holds tile rows [r * bh, (r + 1) * bh) x K tiles
        # [j * kp, (j + 1) * kp), tiles row-major inside the shard. COL_MAJOR puts r on grid x and j on grid y;
        # ROW_MAJOR the other way round. bh, kp and the orientation come off the shard spec.
        shard = config.shard_spec
        shard_h, shard_w = (int(v) for v in shard.shape)
        assert shard_h % TILE == 0 and shard_w % TILE == 0, f"shard {shard.shape}"
        rows_per_shard = shard_h // TILE
        part_tiles = shard_w // TILE
        assert HIDDEN_TILES % part_tiles == 0
        num_parts = HIDDEN_TILES // part_tiles
        ranges = list(shard.grid.ranges())
        assert len(ranges) == 1, "BLOCK_SHARDED x must sit on one rectangular core range"
        origin = ranges[0].start
        grid_w = ranges[0].end.x - origin.x + 1
        grid_h = ranges[0].end.y - origin.y + 1
        col_major = shard.orientation == ttnn.ShardOrientation.COL_MAJOR
        row_blocks, part_blocks = (grid_w, grid_h) if col_major else (grid_h, grid_w)
        assert part_blocks == num_parts, f"{part_blocks} K shards on the grid, {num_parts} needed"
        assert row_blocks * rows_per_shard * TILE >= math.prod(padded[:-1]), "shard grid does not cover M"

        def entry(m, j):
            r, row_in_shard = divmod(m, rows_per_shard)
            assert r < row_blocks and j < num_parts
            gx, gy = (r, j) if col_major else (j, r)
            core = ttnn.CoreCoord(origin.x + gx, origin.y + gy)
            return [noc_xy(core), base + row_in_shard * part_tiles * page_bytes]

    else:
        assert spec is not None, "x must be ND-sharded or BLOCK_SHARDED"
        shard_shape = list(spec.shard_shape)
        assert len(shard_shape) == 4 and shard_shape[0] == 1 and shard_shape[1] == 1, f"shard {shard_shape}"
        assert shard_shape[-2] % TILE == 0 and shard_shape[-1] % TILE == 0
        assert spec.shard_distribution_strategy == ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
        rows_per_shard = shard_shape[-2] // TILE
        part_tiles = shard_shape[-1] // TILE
        assert HIDDEN_TILES % part_tiles == 0
        num_parts = HIDDEN_TILES // part_tiles
        cores = ttnn.corerange_to_cores(spec.grid, None, spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        num_cores = len(cores)
        shard_row_groups = math.ceil(plan.seq_tiles / rows_per_shard)
        num_shards = plan.batch * shard_row_groups * num_parts
        shard_bytes = rows_per_shard * part_tiles * page_bytes

        def entry(m, j):
            b, s_t = divmod(m, plan.seq_tiles)
            i = (b * shard_row_groups + s_t // rows_per_shard) * num_parts + j
            assert 0 <= i < num_shards
            addr = base + (i // num_cores) * shard_bytes + (s_t % rows_per_shard) * part_tiles * page_bytes
            return [noc_xy(cores[i % num_cores]), addr]

    tables = []
    for c in range(plan.num_columns):
        first, end = _column_rows(plan, c)
        words = []
        for m in range(first, end):
            for j in range(num_parts):
                words += entry(m, j)
        # Self-check: every (row, part) of the column has exactly one entry.
        assert len(words) == 2 * (end - first) * num_parts
        tables.append(words)
    return tables, part_tiles, num_parts, page_bytes


def _column_rows(plan: Plan, c):
    first = c * plan.core_m_tiles
    end = min(plan.m_tiles, first + plan.core_m_tiles)
    return first, end


def _column_m_blocks(plan: Plan, c):
    first, end = _column_rows(plan, c)
    return math.ceil((end - first) / plan.block_m_tiles)


def create_program_descriptor(plan: Plan, x, weights, tables, outputs, compute_config):
    device = x.device()
    query, key, value = outputs
    cols, lanes = plan.num_columns, plan.num_lanes
    rotary_lanes = (NUM_ROTARY_TENSORS * TENSOR_N_TILES) // LANE_N_TILES  # lanes 0..5 are q and k
    all_cores = _crs(0, 0, cols - 1, lanes - 1)
    rotary_cores = _crs(0, 0, cols - 1, rotary_lanes - 1)
    plain_cores = _crs(0, rotary_lanes, cols - 1, lanes - 1)
    block_m = plan.block_m_tiles
    out_drain_tiles = _env("OUT_DRAIN_TILES", OUT_DRAIN_TILES)
    assert LANE_N_TILES % out_drain_tiles == 0 and out_drain_tiles % HEAD_TILES == 0
    table_block_tiles = TABLE_TILES_PER_ROW * block_m  # C and S of every row of one M block

    # ---------------- circular buffers (ledger order; identical on every core) ----------------
    cbs = [
        _cb(CB_X_BLOCK, X_BLOCK_DEPTH * block_m * K_BLOCK_TILES, BF16_TILE_BYTES, ttnn.bfloat16, all_cores),
        _cb(CB_WEIGHTS, K_TILES * LANE_N_TILES, BF16_TILE_BYTES, ttnn.bfloat16, all_cores),
        _cb(CB_ROTARY_TABLE, ROTARY_TABLE_DEPTH * table_block_tiles, FP32_TILE_BYTES, ttnn.float32, all_cores),
        _cb(CB_PARTIALS, block_m * LANE_N_TILES, FP32_TILE_BYTES, ttnn.float32, all_cores),
        _cb(CB_HEADS_OUT, OUT_DEPTH_BLOCKS * block_m * LANE_N_TILES, BF16_TILE_BYTES, ttnn.bfloat16, all_cores),
    ]

    # ---------------- reader (RISCV_0, NOC0): x gather + in0 multicast, rotary tables ----------------
    source_tables, part_tiles, num_parts, x_page_bytes = _x_source_table(plan, x, device)
    assert x_page_bytes == BF16_TILE_BYTES
    table_accessor = ttnn.TensorAccessorArgs(tables).get_compile_time_args()

    def reader_named(load_tables):
        return [
            ("cb_x_block", CB_X_BLOCK),
            ("cb_rotary_table", CB_ROTARY_TABLE),
            ("block_m_tiles", block_m),
            ("table_block_tiles", table_block_tiles),
            ("k_block_tiles", K_BLOCK_TILES),
            ("num_k_blocks", NUM_K_BLOCKS),
            ("hidden_tiles", HIDDEN_TILES),
            ("bias_k_tile", BIAS_K_TILE),
            ("x_part_tiles", part_tiles),
            ("x_num_parts", num_parts),
            ("seq_tiles", plan.seq_tiles),
            ("load_tables", 1 if load_tables else 0),
            ("table_load_k_block", _env("TABLE_LOAD_K_BLOCK", TABLE_LOAD_K_BLOCK)),
            ("ablate_gather", _env("ABLATE_GATHER")),
        ]

    def reader_rt(core_range_rows):
        rt = ttnn.RuntimeArgs()
        for c in range(cols):
            first, end = _column_rows(plan, c)
            for y in core_range_rows:
                args = [tables.buffer_address(), first, end, _column_m_blocks(plan, c)]
                if y == 0:
                    args += source_tables[c]
                rt[c][y] = args
        return rt

    reader_kernels = []
    for load_tables, rows, cores in (
        (True, range(0, rotary_lanes), rotary_cores),
        (False, range(rotary_lanes, lanes), plain_cores),
    ):
        reader_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "nomic_qkv_rotary_heads_reader.cpp"),
                core_ranges=cores,
                compile_time_args=list(table_accessor),
                named_compile_time_args=reader_named(load_tables),
                runtime_args=reader_rt(rows),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=READER_NOC),
            )
        )

    # ---------------- writer (RISCV_1, NOC1): W stream + in1 multicast, head scatter ----------------
    writer_ct = list(ttnn.TensorAccessorArgs(weights).get_compile_time_args())
    for out in (query, key, value):
        writer_ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
    valid_last_rows = plan.seqlen - (plan.seq_tiles - 1) * TILE  # rows of the last position tile that are real
    w_sender = _env("W_SENDER", W_SENDER)
    assert w_sender in (W_SENDER_RIGHT, W_SENDER_DIAGONAL, W_SENDER_ROTATING)
    rotating = w_sender == W_SENDER_ROTATING
    chunk_k_rows = _env("W_MCAST_CHUNK_K_ROWS", W_MCAST_CHUNK_K_ROWS)
    assert K_BLOCK_TILES % chunk_k_rows == 0
    # The default read-ahead is the same number of K rows (one K block) whatever the chunk size.
    read_ahead = _env("WEIGHT_READ_AHEAD_CHUNKS", WEIGHT_READ_AHEAD_CHUNKS * W_MCAST_CHUNK_K_ROWS // chunk_k_rows)
    stagger = _env("W_SENDER_STAGGER", W_SENDER_STAGGER) if rotating else 0
    writer_named = [
        ("cb_weights", CB_WEIGHTS),
        ("cb_heads_out", CB_HEADS_OUT),
        ("block_m_tiles", block_m),
        ("k_block_tiles", K_BLOCK_TILES),
        ("num_k_blocks", NUM_K_BLOCKS),
        ("lane_n_tiles", LANE_N_TILES),
        ("out_drain_tiles", out_drain_tiles),
        ("weight_page_tiles", WEIGHT_PAGE_TILES),
        ("weight_pages_per_k_row", WEIGHT_PAGES_PER_K_ROW),
        ("head_tiles", HEAD_TILES),
        ("tensor_n_tiles", TENSOR_N_TILES),
        ("heads", HEADS),
        ("seq_tiles", plan.seq_tiles),
        ("valid_last_rows", valid_last_rows if valid_last_rows < TILE else 0),
        ("weight_read_ahead", read_ahead),
        ("w_mcast_chunk_k_rows", chunk_k_rows),
        ("weight_page_bytes", weights.buffer_aligned_page_size()),
        ("ablate_writes", _env("ABLATE_WRITES")),
        ("ablate_wread", _env("ABLATE_WREAD")),
        ("ablate_wmcast", _env("ABLATE_WMCAST")),
    ]
    writer_rt = ttnn.RuntimeArgs()
    for c in range(cols):
        first, end = _column_rows(plan, c)
        for y in range(lanes):
            writer_rt[c][y] = [
                weights.buffer_address(),
                query.buffer_address(),
                key.buffer_address(),
                value.buffer_address(),
                first,
                end,
                _column_m_blocks(plan, c),
                y * LANE_N_TILES,
                stagger * y,  # first in1 round of this row: the rotating sender of K block b is phase b + this
            ]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "nomic_qkv_rotary_heads_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        named_compile_time_args=writer_named,
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=WRITER_NOC),
    )

    # ---------------- compute ----------------
    compute_kernels = []
    for rotary, rows, cores in (
        (True, range(0, rotary_lanes), rotary_cores),
        (False, range(rotary_lanes, lanes), plain_cores),
    ):
        rt = ttnn.RuntimeArgs()
        for c in range(cols):
            for y in rows:
                rt[c][y] = [_column_m_blocks(plan, c)]
        config = ttnn.ComputeConfigDescriptor()
        config.math_fidelity = compute_config.math_fidelity
        config.fp32_dest_acc_en = True
        config.math_approx_mode = bool(getattr(compute_config, "math_approx_mode", False))
        config.dst_full_sync_en = False  # half-sync DEST is a mechanism cap (4 fp32 tiles a half)
        compute_kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "nomic_qkv_rotary_heads_compute.cpp"),
                core_ranges=cores,
                compile_time_args=[],
                named_compile_time_args=[
                    ("cb_x_block", CB_X_BLOCK),
                    ("cb_weights", CB_WEIGHTS),
                    ("cb_rotary_table", CB_ROTARY_TABLE),
                    ("cb_partials", CB_PARTIALS),
                    ("cb_heads_out", CB_HEADS_OUT),
                    ("block_m_tiles", block_m),
                    ("table_block_tiles", table_block_tiles),
                    ("head_tiles", HEAD_TILES),
                    ("k_block_tiles", K_BLOCK_TILES),
                    ("num_k_blocks", NUM_K_BLOCKS),
                    ("lane_n_tiles", LANE_N_TILES),
                    ("out_drain_tiles", out_drain_tiles),
                    (
                        "out_subblock_w",
                        _env("QK_SUBBLOCK_W", QK_SUBBLOCK_W) if rotary else WIDE_SUBBLOCK_W,
                    ),
                    (
                        "accum_subblock_w",
                        _env("QK_ACCUM_SUBBLOCK_W", WIDE_SUBBLOCK_W) if rotary else WIDE_SUBBLOCK_W,
                    ),
                    ("rotary", 1 if rotary else 0),
                    ("out_layout_row_major", _env("ROW_MAJOR", OUT_LAYOUT_ROW_MAJOR)),
                    ("rotary_on_pack", _env("ROTARY_ON_PACK", ROTARY_ON_PACK)),
                    ("cs_on_reload", _env("CS_ON_RELOAD", CS_ON_RELOAD)),
                    ("ablate_rotary", _env("ABLATE_ROTARY")),
                    ("ablate_compute_all", _env("ABLATE_COMPUTE_ALL")),
                ],
                defines=[("SKIP_COMPUTE", "1")] if _env("ABLATE_MATMUL") else [],
                runtime_args=rt,
                config=config,
            )
        )

    # ---------------- multicast channels ----------------
    # in0: groups = grid columns (receivers rows 1..lanes-1), sender = the column's top core, NOC0.
    x_mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=READER_NOC, handshake=True),
        _crs(0, 1, cols - 1, lanes - 1),
        lanes - 1,
        ttnn.McastExplicitFixedSenderConfig([ttnn.CoreCoord(c, 0) for c in range(cols)]),
        ttnn.McastCoreOrder.ColumnMajor,
    )
    # in1: groups = grid rows, NOC1 (W_SENDER). Right: the row's right core sends to columns 0..cols-2. Diagonal: row
    # y's sender is column y mod cols of a whole-row group. Rotating: every core of the row receives and, in turn,
    # sends K block b (phase b + the row's round offset, mod cols).
    if w_sender == W_SENDER_RIGHT:
        w_receivers, w_group = _crs(0, 0, cols - 2, lanes - 1), cols - 1
        w_senders = ttnn.McastExplicitFixedSenderConfig([ttnn.CoreCoord(cols - 1, y) for y in range(lanes)])
    else:
        w_receivers, w_group = all_cores, cols
        if rotating:
            w_senders = ttnn.McastRotatingSenderConfig()
        else:
            w_senders = ttnn.McastFixedSenderConfig(sender_index=0, placement=ttnn.McastSenderPlacement.Staggered)
    w_mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=WRITER_NOC, handshake=True),
        w_receivers,
        w_group,
        w_senders,
        ttnn.McastCoreOrder.RowMajor,
    )
    descriptor = ttnn.ProgramDescriptor(cbs=cbs)
    x_mcast.attach(descriptor, "x_mcast", reader_kernels, 0)
    w_mcast.attach(descriptor, "w_mcast", [writer_kernel], x_mcast.next_semaphore_id())
    descriptor.kernels = [*reader_kernels, writer_kernel, *compute_kernels]
    return descriptor
