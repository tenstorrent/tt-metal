# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_moe_dispatch: Xg[0, 0, r, :] = table[0, 0, gather_index[0, r], :], tilized, bit for bit.

One ttnn.generic_op program, regime R1 `local_tilize_two_noc` of op_design.md:
  * the output is ND-sharded in L1 (shard = 32 rows x w tiles, round robin over the 110-core grid), so a core tilizes
    exactly the shards it holds and no output byte crosses the NoC;
  * each core gathers its shards' row pieces on two reader RISCs (rows [0, rows_noc0) on RISCV_0 / NOC0, the rest on
    RISCV_1 / NOC1) with the NoC's stateful one-packet read;
  * gather_index is read from DRAM by one sender RISC a grid row, a round's span at a time, and multicast along the
    row to every core's ids scratch behind a count semaphore (R10 `ids_row_mcast`), so every reader's ids are local;
  * compute fast-tilizes each 32-row block straight into the output shard slot.

Registry model: INPUT_TAGGERS / SUPPORTED / EXCLUSIONS / validate() below; INVALID lives in
eval/golden_tests/nomic_moe_dispatch/feature_spec.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue

# ---------------------------------------------------------------------------
# Blocking knobs (single source of truth; every dependent quantity derives from these)
# ---------------------------------------------------------------------------

HIDDEN = 768
TILE = 32
BF16_BYTES = 2
TILE_BYTES = TILE * TILE * BF16_BYTES  # one bf16 tile
ROW_BYTES = HIDDEN * BF16_BYTES  # one table page (token row): 1536 B
HIDDEN_TILES = HIDDEN // TILE  # tensor_col_tiles = 24
GRID_X, GRID_Y = 11, 10
NUM_CORES = GRID_X * GRID_Y  # 110: the round robin of the output's ND spec
T_MIN, T_MAX = 1000, 4096
R_MIN, R_MAX = 2240, 8512
# tile_row axis: the shard height in tile rows (block_tile_rows). Fixed at 1 by the L1 budget (op_design.md).
SHARD_TILE_ROWS = 1
# column axis: candidate pieces_per_row k (w = 24 / k), the selection rule takes the smallest that fits the budget.
# k = 1 (whole 24-tile rows, 1-2 blocks a core) is not a candidate (perf lamp L4, measured by the verifier): k = 2 fits
# the budget wherever k = 1 does (ceil(2x) * 12 <= ceil(x) * 24), and is 1.36x / 2.0x faster at R = 3296 / 6368
# (6.59 -> 4.86 us, 14.97 -> 7.47 us): a core's last whole-row block leaves its 24-tile tilize unoverlapped.
PIECES_PER_ROW_CANDIDATES = (2, 3, 4, 6, 8, 12, 24)
# The model's L1 plan: a core's output share may exceed the interleaved share by at most this many tiles (8 KB).
SHARE_SLACK_TILES = 4
# row_in_block axis: rows [0, ROWS_NOC0) are read by the NOC0 reader, the rest by the NOC1 reader (perf lamp L1).
ROWS_NOC0 = 16
# Path-aware row split (Refinement 4, lamp L1): instead of the positional halves, each row of a block goes to the reader
# whose NoC carries its response over fewer torus hops, capped at ROWS_NOC0 (NOC0) / 32 - ROWS_NOC0 (NOC1) plus
# ROW_SPLIT_SLACK rows. Both readers walk all 32 ids and compute the same split. False = positional (byte-identical).
ROW_SPLIT_PATH = False
ROW_SPLIT_SLACK = 0
# Sender-column rebalance (Refinement 4, lamp L1): on grid column 0 the ids sender reader (IDS_SENDER_NOC) also fetches
# and multicasts its row's ids, and those cores finish last; it hands this many of its rows of every block to the
# core's other reader (column 0 gets its own reader kernels), per k; 0 = the positional split everywhere (one reader
# kernel pair for the grid, byte-identical). Measured per call: 2 is the best at every k (k = 3: R = 4320 5.60 -> 5.31,
# 8064 8.84 -> 8.21; k = 4: 7392 10.31 -> 9.59; k = 6: 2656 6.09 -> 5.76; 1 / 3 in between, 4 back to no gain); at k = 2
# it needs a stride-1 issue order (the halves must stay coprime with every row step, see ROW_STAGGER_BY_K).
SENDER_ROW_SHED_BY_K = {}
SENDER_ROW_SHED_DEFAULT = 2
# NoC reads in flight per reader RISC (perf lamp L2): at most READS_IN_FLIGHT, and at most READ_BYTES_IN_FLIGHT bytes
# of row pieces. Past a bytes threshold the NoC tips into a congestion collapse in a fraction of the calls (the call
# takes 1.5-2x; Refinement 3 found it at k = 2, 8 x 768 B). Refinement 4's cheaper per-row issue moves the cliff lower
# and makes it bite at k = 3 too, so the cap is re-measured per call (12-16 calls a point, with the issue orders below):
# k = 2 stable at 5 (6: 1/12 calls collapse at R = 8416; 7: most), k = 3 at 8, k = 4 at 10, k = 6 at 12 -- one ~4 KB
# rule: 4096 // piece_bytes = 5 / 8 / 10 / 16 -> 12.
READS_IN_FLIGHT = 12
READ_BYTES_IN_FLIGHT = 4096
# Blocks resident in the rows CB: None = the whole per-core assignment (max_blocks_per_core; perf lamp L3).
ROW_SLOTS = None
# gather_index fetch topology (Refinement 3, op_design.md R10 `ids_row_mcast`). Every core holds a full-size image of
# gather_index. A grid row's GRID_X cores hold consecutive shards in every round, so the row's round-j ids are one
# contiguous span: the row's x = 0 core (its NOC0 reader) reads it from the one gather_index DRAM bank and multicasts
# it along the row, published one block ahead of use and fetched IDS_LOOKAHEAD blocks before that (1 .. 6). 10 DRAM
# reads a round instead of one a reader (220); rows multicast on disjoint paths. Measured alternatives (changelog,
# Refinement 3): the whole index multicast to the grid (one or several senders) and a grid-multicast first round with
# per-reader fetches after it both lost at R = 8416 (14.4 / 15.0 / 9.8 us vs 9.8 Phase 0).
IDS_LOOKAHEAD = 1
# The NoC (and so the reader RISC) of the row senders. NOC1 measured no better (R = 2272 / 4320 / 8416: 3.87 / 7.98 /
# 9.12 us vs 3.74 / 6.92 / 9.08 on NOC0).
IDS_SENDER_NOC = 0
# Per-core issue stagger (Refinement 3): (stagger, strides) per k. Core c walks its half's rows from row
# (c * stagger) mod (rows of the half), in steps of 2 * (c mod strides) + 1, wrapping. With ids local the grid starts
# within ~0.3 us, and lockstep cores walking their sorted ids in the same order meet on the same banks at every step: at
# 8 reads in flight R = 8416 had 14-17 us outliers in 6/16 calls in order, 1/16 at (3, 4). At 6 in flight (k = 2),
# mean of 5 calls, R = 8416: (0, 1) 10.79 us, (3, 4) 9.11-9.24, (5, 2) 9.06-9.19; R = 6368: (3, 4) 7.48-7.52 (Phase 0
# 7.50), (5, 2) 7.59-7.64; R = 8224: (3, 4) 9.17-9.36, (5, 2) 9.07-9.15. k >= 3, (5, 2) is the best or within noise:
# R = 2272 3.74 (3, 4: 3.82), 4320 6.92 (7.09), 7392 13.87 (14.18), 8064 11.26 (11.53). (0, 1) = in order.
# Refinement 4 (lamp L6), with the cheaper issue path and the caps above, per call: at k = 3 / 8 in flight (5, 2) collapses
# in 40-50% of the calls at R = 4320 / 8064 (5.6 / 8.9 us -> 10.6-12.8 / 12.6-13.5) and (7, 1), (3, 4), (1, 8) in none:
# (7, 1) 5.60 / 8.85, (3, 4) 5.72 / 8.89, (1, 8) 5.62 / 8.91. (7, 1) is also the best or within noise at k = 4 / 6
# (R = 7392 10.32 vs 10.38, R = 2656 6.12 vs 6.16). At k = 2 (7, 1) and (1, 8) collapse even at 6 in flight; (3, 4)
# stays best of the multi-stride orders (R = 8416 at 5: (3, 4) 8.21, (5, 2) 8.33, (5, 4) 8.38), and with the sender
# shed (SENDER_ROW_SHED_*, 2 rows) the stride-1 (5, 1) beats it: R = 8416 7.99 (28 calls, none collapsed), 8224 8.15,
# 6368 6.72; (9, 1) 8.07 / 8.11; (3, 1) 8.34; (11, 1) and (13, 1) collapse in 1-2 of 14 calls.
ROW_STAGGER_BY_K = {2: (5, 1)}
ROW_STAGGER_DEFAULT = (7, 1)
# Start stagger: core c delays its first block by (37 c mod NUM_CORES) / NUM_CORES of this many cycles (0 = none).
# Measured no cure for the collapse (R = 8416, 700 / 1400 cycles: 3/16 and 7/16 outliers), so 0.
START_SPREAD_CYCLES = 0
# gather_index is read from DRAM at, and the ids scratch base is aligned to, this many bytes (BH DRAM read alignment).
IDS_ALIGN = 64
ID_BYTES = 4
CB_L1_BUDGET = 256 * 1024


def default_compute_kernel_config():
    config = ttnn.ComputeConfigDescriptor()
    config.math_fidelity = ttnn.MathFidelity.LoFi  # irrelevant: there is no math
    config.fp32_dest_acc_en = False
    config.dst_full_sync_en = False  # the fast tilize path needs half-sync DEST
    config.math_approx_mode = False
    return config


@dataclass(frozen=True)
class Plan:
    rows: int
    tile_rows: int
    pieces_per_row: int  # k
    block_width_tiles: int  # w
    piece_bytes: int
    num_shards: int
    max_blocks_per_core: int
    row_slots: int
    rows_noc0: int
    sender_rows_noc0: int  # rows_noc0 on grid column 0 (the ids sender cores)
    reads_in_flight: int
    ids_lookahead: int  # rounds a row sender fetches ahead of publishing them
    ids_sender_noc: int
    row_stagger: int
    row_strides: int
    start_spread_cycles: int
    row_split_path: bool
    row_split_slack: int

    @staticmethod
    def row_ranges(rows_noc0):
        return ((0, rows_noc0), (rows_noc0, TILE))


def _pieces_per_row(tile_rows: int) -> int:
    budget = math.ceil(HIDDEN_TILES * tile_rows / NUM_CORES) + SHARE_SLACK_TILES
    for k in PIECES_PER_ROW_CANDIDATES:
        if HIDDEN_TILES % k == 0 and math.ceil(tile_rows * k / NUM_CORES) * (HIDDEN_TILES // k) <= budget:
            return k
    raise UnsupportedAxisValue(f"nomic_moe_dispatch: no column split fits the L1 plan at {tile_rows} tile rows")


@lru_cache(maxsize=None)
def plan(rows: int) -> Plan:
    """The single source of the work split for R = rows (pure function of R; memoized)."""
    if rows % (TILE * SHARD_TILE_ROWS) != 0:
        raise UnsupportedAxisValue(f"nomic_moe_dispatch: R = {rows} is not a multiple of {TILE * SHARD_TILE_ROWS}")
    tile_rows = rows // TILE
    k = _pieces_per_row(tile_rows)
    w = HIDDEN_TILES // k
    num_shards = (tile_rows // SHARD_TILE_ROWS) * k
    max_blocks = math.ceil(num_shards / NUM_CORES)
    row_slots = max_blocks if ROW_SLOTS is None else max(1, min(ROW_SLOTS, max_blocks))
    assert 0 < ROWS_NOC0 < TILE
    row_stagger, row_strides = ROW_STAGGER_BY_K.get(k, ROW_STAGGER_DEFAULT)
    assert 0 <= ROW_SPLIT_SLACK <= min(ROWS_NOC0, TILE - ROWS_NOC0)
    assert 1 <= IDS_LOOKAHEAD <= 6 and row_stagger >= 0 and row_strides >= 1 and START_SPREAD_CYCLES >= 0
    shed = SENDER_ROW_SHED_BY_K.get(k, SENDER_ROW_SHED_DEFAULT)
    sender_rows_noc0 = ROWS_NOC0 - shed if IDS_SENDER_NOC == 0 else ROWS_NOC0 + shed
    assert shed >= 0 and 0 < sender_rows_noc0 < TILE
    assert not (shed and ROW_SPLIT_PATH), "the path-aware split takes its share from ROWS_NOC0 everywhere"
    for split in (ROWS_NOC0, sender_rows_noc0):
        for half_rows in (split, TILE - split):  # every per-core row step must walk the whole half
            assert all(2 * i + 1 <= half_rows and math.gcd(2 * i + 1, half_rows) == 1 for i in range(row_strides))
    return Plan(
        rows=rows,
        tile_rows=tile_rows,
        pieces_per_row=k,
        block_width_tiles=w,
        piece_bytes=w * TILE * BF16_BYTES,
        num_shards=num_shards,
        max_blocks_per_core=max_blocks,
        row_slots=row_slots,
        rows_noc0=ROWS_NOC0,
        sender_rows_noc0=sender_rows_noc0,
        reads_in_flight=max(1, min(READS_IN_FLIGHT, READ_BYTES_IN_FLIGHT // (w * TILE * BF16_BYTES))),
        ids_lookahead=IDS_LOOKAHEAD,
        ids_sender_noc=IDS_SENDER_NOC,
        row_stagger=row_stagger,
        row_strides=row_strides,
        start_spread_cycles=START_SPREAD_CYCLES,
        row_split_path=bool(ROW_SPLIT_PATH),
        row_split_slack=ROW_SPLIT_SLACK,
    )


def _ids_cb_bytes(p: Plan) -> int:
    # A full-size gather_index image (shared by both readers; a row multicast lands at the same address on every core;
    # a core receives only its row's spans), one IDS_ALIGN cell a round for the row sender's published count (its
    # multicast source), plus one page of slack to align the scratch base to IDS_ALIGN.
    return p.rows * ID_BYTES + (p.max_blocks_per_core + 1) * IDS_ALIGN


def _rows_cb_bytes(p: Plan) -> int:
    return p.row_slots * p.block_width_tiles * TILE_BYTES


def circular_buffer_bytes(rows: int) -> int:
    """L1 a core the op's CBs take beyond the io tensors (the output-backed CB is the output itself)."""
    p = plan(rows)
    return _rows_cb_bytes(p) + _ids_cb_bytes(p)


def _grid():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])


@lru_cache(maxsize=None)
def _l1_output_memory_config(rows: int) -> ttnn.MemoryConfig:
    p = plan(rows)
    spec = ttnn.NdShardSpec(
        ttnn.Shape([SHARD_TILE_ROWS * TILE, p.block_width_tiles * TILE]),
        _grid(),
        ttnn.ShardOrientation.ROW_MAJOR,
        ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    # Canonical form: the config a tensor allocated from this spec reports.
    tensor_spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, rows, HIDDEN]), ttnn.bfloat16, ttnn.TILE_LAYOUT, spec, ttnn.BufferType.L1
    )
    return tensor_spec.memory_config


def output_memory_config(rows, device=None, buffer_type=ttnn.BufferType.L1) -> ttnn.MemoryConfig:
    """The layout Xg is written in for `buffer_type`. DRAM is DRAM interleaved (refused by validate())."""
    if buffer_type == ttnn.BufferType.DRAM:
        return ttnn.DRAM_MEMORY_CONFIG
    return _l1_output_memory_config(int(rows))


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_window(inputs, axes):
    """T % 32 == 0 ("aligned") or not ("ragged"); inputs[0] is the table's shape."""
    return "aligned" if inputs[0][-2] % TILE == 0 else "ragged"


INPUT_TAGGERS = {
    "window": tag_window,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    # dram = R4 `table_dram` (deferred, op_design.md Regimes): the model's in-range tables are L1.
    "table_buffer": ["l1"],
    # dram = R3 `output_dram` (deferred): a scheme change; the incumbent serves DRAM Xg.
    "output_buffer": ["l1"],
    "window": ["aligned", "ragged"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _buffer(config) -> str:
    return "l1" if config.buffer_type == ttnn.BufferType.L1 else "dram"


def validate(table, gather_index, *, memory_config=None, compute_kernel_config=None):
    axes = {
        "dtype": table.dtype,
        "table_buffer": _buffer(table.memory_config()),
        "output_buffer": _buffer(memory_config or ttnn.L1_MEMORY_CONFIG),
    }
    shapes = (tuple(table.shape), tuple(gather_index.shape))
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger(shapes, axes)

    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nomic_moe_dispatch: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nomic_moe_dispatch: unsupported combination (refinement candidate): {exc}")

    # Operand contract (op_design.md Validation): every refusal is a SupportRefusal.
    t_shape = tuple(table.shape)
    if len(t_shape) != 4 or t_shape[0] != 1 or t_shape[1] != 1 or t_shape[3] != HIDDEN:
        raise UnsupportedAxisValue(f"nomic_moe_dispatch: table shape {t_shape} is not (1, 1, T, {HIDDEN})")
    if not (T_MIN <= t_shape[2] <= T_MAX):
        raise UnsupportedAxisValue(f"nomic_moe_dispatch: T = {t_shape[2]} outside [{T_MIN}, {T_MAX}]")
    if (
        table.layout != ttnn.ROW_MAJOR_LAYOUT
        or table.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise UnsupportedAxisValue("nomic_moe_dispatch: table must be ROW_MAJOR interleaved")
    i_shape = tuple(gather_index.shape)
    if len(i_shape) != 2 or i_shape[0] != 1:
        raise UnsupportedAxisValue(f"nomic_moe_dispatch: gather_index shape {i_shape} is not (1, R)")
    rows = i_shape[1]
    if rows % TILE != 0 or not (R_MIN <= rows <= R_MAX):
        raise UnsupportedAxisValue(f"nomic_moe_dispatch: R = {rows} must be a multiple of 32 in [{R_MIN}, {R_MAX}]")
    if (
        gather_index.dtype != ttnn.uint32
        or gather_index.layout != ttnn.ROW_MAJOR_LAYOUT
        or gather_index.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    ):
        raise UnsupportedAxisValue("nomic_moe_dispatch: gather_index must be uint32 ROW_MAJOR interleaved")
    grid = table.device().compute_with_storage_grid_size()
    if (grid.x, grid.y) != (GRID_X, GRID_Y):
        raise UnsupportedAxisValue(f"nomic_moe_dispatch: the plan is built for an {GRID_X}x{GRID_Y} grid, not {grid}")
    if circular_buffer_bytes(rows) > CB_L1_BUDGET:
        raise UnsupportedAxisValue("nomic_moe_dispatch: the plan's circular buffers exceed 256 KiB a core")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def nomic_moe_dispatch(table, gather_index, *, memory_config=None, compute_kernel_config=None):
    validate(table, gather_index, memory_config=memory_config, compute_kernel_config=compute_kernel_config)
    from .nomic_moe_dispatch_program_descriptor import create_program_descriptor

    device = table.device()
    rows = int(gather_index.shape[1])
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, HIDDEN]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        output_memory_config(rows, device, ttnn.BufferType.L1),
    )
    descriptor = create_program_descriptor(plan(rows), table, gather_index, output, compute_kernel_config)
    return ttnn.generic_op([table, gather_index, output], descriptor)
