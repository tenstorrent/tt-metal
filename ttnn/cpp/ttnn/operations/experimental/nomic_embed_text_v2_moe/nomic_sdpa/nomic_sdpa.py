# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""nomic_sdpa: the bidirectional 12-head attention of nomic-embed-text-v2-moe in one generic_op dispatch.

    out[b, 0, i, 64 h + d] = sum_j softmax_j(scale * q[b, h, i, :] . k[b, h, j, :]) * v[b, h, j, d]

Regime (op_design.md § Blocking Model): row-split + pair multicast (Refinement 2). The flattened query tile-rows
g = (b*12 + h)*S_t + st are split in contiguous balanced ranges over the first `num_cores` cores (row-major on the
11x10 grid); a core holds the whole key range of each (b, h) pair it touches (a *segment*) and runs the exact
two-pass softmax per block of its rows: scores = Q K^T, m = rowmax, P = exp(scale (scores - m)) in place,
[O | l] = P [V | 1], out = O / l. The cores sharing a pair at a fresh K/V slot read its K/V once (the group's first
core whose first segment it is) and receive it by NoC multicast (kv_multicast_topology in the program descriptor).

Every block knob (q_block_rows, key_block_tiles, the buffer depths, the core-assignment table) is defined once in
`plan()` and handed to the kernels as CT / common-RT args; nothing downstream restates it.
"""

from __future__ import annotations

import math
from functools import cache
import struct
from dataclasses import dataclass

import ttnn

from ttnn.operations._op_contract import ExcludedCell, SupportRefusal, UnsupportedAxisValue

HEADS = 12
HEAD_DIM = 64
TILE = 32
MAX_SEQLEN = 512  # residency predicate: a pair's K, V and one score row-block fit one core
MAX_TOKENS = 4096  # B * S_t * 32 (the domain: 12 * B * S_t <= 1536 query rows)
GRID = (11, 10)  # Blackhole p300c compute grid
BF16_TILE_BYTES = 2 * TILE * TILE

# ---------------------------------------------------------------------------
# Block knobs (op_design.md § Blocking Model). Single source of truth: plan() derives every extent from these.
# ---------------------------------------------------------------------------
Q_BLOCK_ROWS_CAP = 14  # ceil(1536 / 110): the domain's largest per-core assignment
HEAD_DIM_TILES = HEAD_DIM // TILE  # 2: contracted in Q K^T, output columns of P V
# cb_query depth, in nominal blocks, per q/k/v placement (like the K/V ring: plan() takes the cap, the entry point falls
# back to 1 when it does not fit). 2 (Refinement 4): the reader loads the next block's Q (and so reaches its K / V)
# while compute is still on the current block. Measured L1, 176 KB reserved, device us, median of 7 (with
# Q_LEAD_GROUPS 2): 4x512 50.4 -> 49.7, 8x512 103.6 -> 100.5 (noise band ~4 us), 2x512 27.8 -> 27.8. DRAM keeps 1:
# 16x256 DRAM measured 75.1 -> 84.6 at depth 2 (the reader running ahead on DRAM Q delays the next segment's K / V).
Q_DEPTH_L1 = 2
Q_DEPTH_DRAM = 1
# Q row GROUPS a block reads ahead of its segment's K / V; the rest of the block's Q follows V (Refinement 4). The
# row-group schedule consumes Q a row at a time and needs V from its second step on, so groups 0 and 1 are all the Q
# compute can use before V: the startup burst every core issues at t = 0 becomes lead Q + K + V instead of the whole
# Q block + K + V (8x512: compute's first K landed at 31.5 -> 18 us, its first V at 46 -> 35 us). 0 = the whole block
# first (the Phase 0 order). The block-sequential schedule waits for the whole block before Q K^T, so it keeps the
# whole block first (measured 2x512 bf16 DEST exact exp, L1: 53.4 -> 57.2 us with a 2-row lead).
Q_LEAD_GROUPS = 2
# cb_key_t / cb_value_aug depth, in pairs, per q/k/v placement: a cap; plan() takes min(cap, the most segments any
# core has), and the entry point falls back to 1 when the deeper ring does not fit below live L1. Every segment with
# index < kv_depth lands in a slot no earlier segment used, which is what lets a core receive that pair's K/V by
# multicast with no handshake (Refinement 2): at 1 only cores whose FIRST segment is the pair share it; at 2 the core
# starting before the pair joins too (it pre-sends the pair at t = 0, so the t = 0 read burst grows).
# Measured (Blackhole, device us, median of 3, 4x512 / 8x512): L1 depth 1 55.5 / 121.7, depth 2 59.2 / 120.5;
# DRAM depth 1 64.6 / 130.3, depth 2 63.2 / 118.8. Refinement 4 (the row-group schedule halved the CBs, so depth 2
# fits the 176 KB-reserved perf case): L1 8x512 103.6 -> 109.1, 4x512 50.4 -> 52.3 at depth 2 (median of 7, with
# Q_LEAD_GROUPS 2 / Q_DEPTH_L1 2): the first-segment K / V of almost every core then comes through a store-and-forward
# multicast hop on the startup's critical path (first V at 55 vs 46 us). Kept at 1, a live knob.
KV_DEPTH_L1 = 1
KV_DEPTH_DRAM = 2
# Refinement 2 lever: read each pair's K/V once per sharing group and multicast it (False = every core reads its own
# segments, the Phase 0 dataflow; kept as the A/B knob).
KV_MCAST = True
# Path gate: multicast only pairs of at least this many key tiles (K + V >= 64 KiB). Below it the sender's read plus the
# store-and-forward hop on the receivers' critical path costs more than the reads it saves (measured 1x128, 48 cores at
# one row each: 4.7 -> 5.3 us with the multicast).
KV_MCAST_MIN_SEQ_TILES = 8
# Refinement 4 (perf lamp L9, catalog bank_stagger): rotate each core's K / V key-row read start by core % N. False =
# every core walks from key row 0. Measured flat to slightly slower (median of 5, Q_LEAD_GROUPS 2): 4x512 L1 49.9 ->
# 50.1, DRAM 59.0 -> 59.4, 8x512 DRAM 110.7 -> 111.2 us. Consecutive pairs already start ~32 pages apart and each pair
# group has one reader, so there is no same-bank start to break up. Parked, a live knob.
KV_BANK_STAGGER = False
OUT_DEPTH = 1  # cb_out depth, in nominal blocks
# probs_block exp form under math_approx_mode (perf lamp L3): 0 = kernel_lib Exp<Fast> (input clamp),
# 1 = the incumbent's unclamped fast exp + pack ReLU.
EXP_FORM = 1
# Refinement 3 (op_design.md perf lamp L2): overlap the SFPU exp with the FPU matmuls. A block runs as ROW GROUPS of
# ROW_GROUP_ROWS query tile-rows (a schedule inside the block: Q and the output stay whole-block CBs). The exp moves to
# the PACK thread and each DEST window of group g's sub + exp is paired with a Q K^T window of group g+1, so the exp of
# one group runs while the FPU computes the next group's scores. Scores and P then live in separate group-sized rings.
# Applies to the approximate exp form (EXP_FORM 1 under math_approx_mode; its scale folds into the exp init); the exact
# exp keeps the block-sequential schedule. False = the Phase 0 block-sequential schedule for every config (A/B knob).
EXP_OVERLAP = True
ROW_GROUP_ROWS = 1  # rows a group (scores / probs ring extent); clamped to q_block_rows
# (Refinement 3's PV_WINDOW_COLS knob is gone since Perf 1: normalize in DEST needs O and l in one DEST window, so a
# P V window is always the whole [O | l] row. One column a window had measured 115.0 vs 113.3 us at 8x512 anyway.)
# Path gate: the row-group schedule only for blocks of at least this many rows. A 1-2 row block has nothing to overlap
# (prologue + epilogue are the whole block) and pays the window overheads: 1x512 (q_block_rows 2) measured 17.9 ->
# 19.4 us with it, 2x512 (4 rows) 31.8 -> 30.7, 32x128 (14 rows) 38.2 -> 34.1 (device us, L1, median of 3).
EXP_OVERLAP_MIN_ROWS = 3
# Refinement 4 (perf lamp L7): a pair-aligned core_row_start table (one segment a core) where it does not raise the
# largest per-core range (2x512, 1x512); elsewhere the balanced split. False = always the balanced split. Measured
# (median of 5-9): 2x512 L1 29.2 -> 27.8, DRAM 33.5 -> 32.2; 1x512 L1 19.8 -> 18.6, DRAM 20.7 -> 21.0 (noise band).
PAIR_ALIGNED_ROWS = True
# Perf 1 (perf lamp L5, perf_experiments/normalize_in_dest): O / l runs in DEST on the pack thread after the P V window
# (all-ones ones tile; kernels/nomic_sdpa_compute.cpp "Normalize in DEST"), replacing Refinement 4's L8 column-0 recip
# chain and the cb_pv_out round trip. Measured compute floor (reads + writes stubbed, L1, 176 KB reserved, us): 8x512
# 75.2 -> 72.0, 4x512 40.0 -> 38.4, 32x128 27.0 -> 21.7, 16x256 42.8 -> 38.8; real rel-RMS 0.0255 -> 0.0216 at 8x512.
# Perf 1 (perf_experiments/dual_noc_reads): split reads — the writer RISC (BRISC, NOC1) reads the Q and V streams,
# the reader (NCRISC, NOC0) keeps K and the multicast sends, so the read flood uses both NoCs' links. Measured (device
# us, median of 5, L1 176 KB reserved): 8x512 ~108.5 -> 92.6 (DM floor ~69 -> ~44), 8x384 73.0 -> 59.2, 16x256 57.9 ->
# 50.9, 32x128 32.8 -> 29.7, 2x512 28.9 -> 27.4; DRAM S >= 288 -4..-8%. Carve-out (measured regression, the only
# one): a 1-deep cb_query (DRAM) with S_t <= SPLIT_READS_LEGACY_MAX_SEQ_TILES keeps every read on the reader — DRAM
# reads over NOC1 are ~2x slower per page there and block b's Q load waits behind block b-1's output drain
# (DRAM 32x128 57.4 -> 61.8, 16x256 76.9 -> 81.4, 1x128 6.6 -> 7.6 us split).
SPLIT_READS = True
SPLIT_READS_LEGACY_MAX_SEQ_TILES = 8
# Perf 2 (perf_experiments/startup_k_first): split startup — under split reads, segment 0's K and V (the t = 0 burst
# of every core) are each read half on NOC0 by the reader and half on NOC1 by the writer (kernels: reader file header),
# and the writer prepares the reduce scaler under its first reads. Measured whole op (us, median of 5, L1 176 KB
# reserved; before -> after): 8x512 L1 63.7 -> 56.2, 4x512 L1 38.5 -> 30.7, 8x384 L1 42.0 -> 35.6, 16x256 L1 36.8 ->
# 32.2, 8x512 DRAM 85.7 -> 83.7. Carve-out (measured regression): the K split stays off — V still split — when the
# t = 0 K burst on NOC0 (tiles of segment-0 K the cores read themselves) is <= SPLIT_K_LEGACY_MAX_BURST_TILES. There
# NOC0 is not congested, and the writer's K part + the same-core handshake only delay its own V: K split vs V split
# alone 2x512 DRAM 28.2 vs 25.4, 1x512 DRAM 20.0 vs 18.5, 1x512 L1 16.9 vs 16.3, 2x512 L1 21.0 vs 20.7, 1x128 L1 4.6
# vs 4.2 (bursts 384-768 tiles); 32x128 L1 (880) flat 24.7 vs 24.9; >= 1536 the K split wins (4x512 L1 30.7 vs 35.1).
SPLIT_K = True
SPLIT_K_LEGACY_MAX_BURST_TILES = 768
# Perf 2 (perf_experiments/startup_first_qk): throttled startup — when the t = 0 K burst (Plan.k_burst_tiles) exceeds
# READ_CHUNK_MIN_BURST_TILES, the segment-0 page reads of every core (K, V, the first block's Q lead rows, on both RISCs)
# retire in chunks of READ_CHUNK_TILES (a read barrier after every chunk), and the reader takes READ_CHUNK_K_SPLIT_16THS
# sixteenths of segment 0's key rows (the writer reads the Q lead rows ahead of its K part). Unthrottled, ~200 RISCs with
# 16-32 reads each in flight collapse the NoC: 8x512 L1 whole op 55.1 .. 67.1 us run to run (the first K landed at 5-11
# us, the first V at 9-22 us); throttled 53.1 .. 57.2 (K at ~4.4 us, V at ~8 us). Measured (us, median of 9-15,
# interleaved with the op before, L1 176 KB reserved): 8x512 60.9 -> 53.6, 8x448 53.4 -> 49.2, 7x512 51.5 -> 51.0,
# 6x512 44.1 -> 43.9, 8x384 36.4 -> 34.9, 10x384 44.4 -> 43.7, 16x256 33.1 -> 32.1, 12x256 26.9 -> 26.1, 5x512 flat
# 37.9; DRAM 8x512 83.9 -> 83.4, 16x256 69.6 -> 68.7, 8x384 / 6x512 flat. Chunks of 4 / 8 / 12 measured 54.9 / 54.7 /
# 55.2 at 8x512 (6: 54.2); the reader's K share 9 / 10 / 11 / 12 / 13 sixteenths 53.8 / 53.6 / 53.3 / 53.6 / 54.7.
# Carve-out (measured regression): a burst of <= 1536 tiles keeps every read unthrottled — the NoC is not saturated
# there and each chunk only adds a round trip: throttled, 4x512 L1 31.0 -> 31.5 (32.2 without the K re-split), 2x512
# 21.2 -> 22.3, 1x512 16.8 -> 18.5. Second carve-out (measured regression, Perf 2 coordinator guard sweep): the
# block-sequential schedule (bf16 DEST + exact exp) with a 1-deep cb_query (DRAM) stays unthrottled — throttled vs not
# (us, median of 5-9): 8x512 DRAM 172.3 vs 169.7, 8x384 DRAM 109.8 vs 107.3, 16x256 DRAM 114.4 vs 113.3, while the same
# schedule in L1 gains (8x512 145.3 vs 155.8, 8x384 91.2 vs 104.0, 16x256 73.6 vs 78.8) and the row-group schedule on
# DRAM gains or is flat (8x512 HiFi2 82.3 vs 83.1, HiFi3 94.1 vs 94.4). Mechanism not isolated.
READ_CHUNK_TILES = 6
READ_CHUNK_MIN_BURST_TILES = 1536
READ_CHUNK_K_SPLIT_16THS = 11

# Caller compute configs that cannot meet the golden bar are refused (prompt § Precision). bf16 DEST with the
# approximate exp misses it at S = 512 (verifier, measured 2026-11-05: real rel-RMS 0.0269-0.0273 > 0.0265 at
# 2x512 / 4x512 / 8x512, flat 0.0194-0.0196 > 0.0175, flat still failing at HiFi3 / HiFi4) and passes at S <= 384
# (8x384 real 0.0255, 16x256 0.0242, 32x128 0.0220). bf16 DEST with the exact exp passes at every measured shape.
BF16_DEST_APPROX_EXP_MAX_SEQLEN = 384


# ---------------------------------------------------------------------------
# 1. INPUT_TAGGERS
# ---------------------------------------------------------------------------


def tag_alignment(inputs, axes):
    """S (inputs[0][-2]) a multiple of 32 or not."""
    return "tile_aligned" if inputs[0][-2] % TILE == 0 else "ragged"


INPUT_TAGGERS = {
    "alignment": tag_alignment,
}


# ---------------------------------------------------------------------------
# 2. SUPPORTED
# ---------------------------------------------------------------------------

SUPPORTED = {
    "dtype": [ttnn.bfloat16],
    "buffer": ["l1", "dram"],
    "mask_mode": ["none"],
    "alignment": ["tile_aligned"],
}


# ---------------------------------------------------------------------------
# 3. EXCLUSIONS
# ---------------------------------------------------------------------------

EXCLUSIONS = []


PROPERTIES = {
    "multi_core": {"value": True, "source": "declared"},
    "bounded_cb": {"value": True, "source": "declared"},
    "math_fidelity": {"value": ["HiFi2", "HiFi3", "HiFi4"], "source": "declared"},
}


def default_compute_kernel_config():
    """HiFi2, fp32 DEST, approximate exp. fp32 DEST is the precision floor (op_design.md § Key Risks remedy 2,
    measured): with a bf16 DEST the P V accumulation over 512 keys plus the approximate exp miss the golden bar at
    S = 512 (real rel-RMS 0.0273 > 0.0265, flat 0.0194 > 0.0175), and HiFi3 alone does not close flat. Every CB stays
    bf16: only the DEST accumulation needs the width."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=True,
        dst_full_sync_en=False,
    )


# ---------------------------------------------------------------------------
# Plan: the single source of every block extent, CB size and the core-assignment table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    batch: int
    seqlen: int
    seq_tiles: int  # S_t
    tensor_query_rows: int  # 12 * B * S_t
    num_cores: int
    core_row_start: tuple  # num_cores + 1 entries: core c owns rows [start[c], start[c+1])
    q_block_rows: int  # Q: nominal rows of a block (every CB page count derives from it)
    key_block_tiles: int  # N: = S_t (whole key range resident)
    head_dim_tiles: int  # HD
    pv_out_tiles: int  # PV = HD + 1 (the ones / row-sum column)
    q_depth: int
    q_lead_rows: int  # Q rows a block reads before its segment's K / V (<= q_block_rows; Q_LEAD_GROUPS groups)
    kv_depth: int
    out_depth: int
    kv_mcast: bool
    exp_overlap: bool  # Refinement 3 row-group schedule with the pack-thread exp
    row_group_rows: int  # G: rows a group (exp_overlap only)

    @property
    def split_reads(self):
        """Q and V read by the writer RISC on NOC1 (Perf 1), except the measured carve-out (see SPLIT_READS)."""
        legacy = self.q_depth == 1 and self.seq_tiles <= SPLIT_READS_LEGACY_MAX_SEQ_TILES
        return bool(SPLIT_READS) and not legacy

    @property
    def k_burst_tiles(self):
        """K tiles the cores read themselves for their first segment at t = 0 (multicast receivers read none)."""
        readers = (
            len({start // self.seq_tiles for start in self.core_row_start[:-1]}) if self.kv_mcast else self.num_cores
        )
        return readers * self.key_block_tiles * self.head_dim_tiles

    @property
    def read_chunk(self):
        """Segment-0 page reads a RISC issues between read barriers (Perf 2, see READ_CHUNK_TILES); 0 = unthrottled."""
        legacy = self.k_burst_tiles <= READ_CHUNK_MIN_BURST_TILES or (not self.exp_overlap and self.q_depth == 1)
        return 0 if legacy else READ_CHUNK_TILES

    @property
    def k_split_rows(self):
        """Key rows of segment 0 the reader reads (the writer reads the rest); key_block_tiles = no K split (Perf 2,
        see SPLIT_K: the measured carve-out is a small t = 0 K burst). Under the throttled startup the reader takes
        READ_CHUNK_K_SPLIT_16THS sixteenths: the writer reads the Q lead rows before its part (Perf 2)."""
        n = self.key_block_tiles
        legacy = self.k_burst_tiles <= SPLIT_K_LEGACY_MAX_BURST_TILES
        if not (bool(SPLIT_K) and self.split_reads and n >= 2 and not legacy):
            return n
        if self.read_chunk:
            return min(max((n * READ_CHUNK_K_SPLIT_16THS + 8) // 16, 1), n - 1)
        return n // 2

    def core_pairs(self, core):
        """The pairs of core's segments, in schedule order (segment k = the k-th entry)."""
        begin, end = self.core_row_start[core], self.core_row_start[core + 1]
        return list(range(begin // self.seq_tiles, (end - 1) // self.seq_tiles + 1))

    @property
    def core_list(self):
        return [ttnn.CoreCoord(c % GRID[0], c // GRID[0]) for c in range(self.num_cores)]

    @property
    def cores(self):
        full_rows, last = divmod(self.num_cores, GRID[0])
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID[0] - 1, full_rows - 1)))
        if last:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(last - 1, full_rows)))
        return ttnn.CoreRangeSet(ranges)


def exp_overlap_for(compute) -> bool:
    """Whether a compute config runs the Refinement 3 row-group schedule (pack-thread exp overlapping the matmuls)."""
    return bool(EXP_OVERLAP) and bool(compute.math_approx_mode) and EXP_FORM == 1


def _balanced_starts(rows: int, cores: int) -> list:
    base, rem = divmod(rows, cores)
    starts = [0]
    for c in range(cores):
        starts.append(starts[-1] + base + (1 if c < rem else 0))
    return starts


def _core_row_starts(pairs: int, seq_tiles: int) -> list:
    """The core_row_start table: contiguous query-tile-row ranges over the first cores of the grid. Default: balanced
    over min(grid, rows) cores. PAIR_ALIGNED_ROWS (perf lamp L7): every core inside one pair (one K/V segment, one
    sharing group), each pair split balanced over its share of the grid — taken only when its largest range is no
    larger than the balanced split's, so it never raises the per-core compute."""
    rows = pairs * seq_tiles
    balanced = _balanced_starts(rows, min(GRID[0] * GRID[1], rows))
    if not PAIR_ALIGNED_ROWS or pairs > GRID[0] * GRID[1]:
        return balanced
    base, extra = divmod(GRID[0] * GRID[1], pairs)
    aligned = [0]
    for pair in range(pairs):
        pair_cores = min(seq_tiles, base + (1 if pair < extra else 0))
        aligned += [pair * seq_tiles + r for r in _balanced_starts(seq_tiles, pair_cores)[1:]]
    return aligned if _widest(aligned) <= _widest(balanced) else balanced


def _widest(starts) -> int:
    return max(b - a for a, b in zip(starts, starts[1:]))


@cache  # a pure function of its arguments returning a frozen Plan: the per-call host cost was ~80 us a call
def plan(
    batch: int, seqlen: int, kv_depth: int | None = None, exp_overlap: bool | None = None, q_depth: int | None = None
) -> Plan:
    """kv_depth / q_depth: the K/V ring-depth cap and the cb_query depth (default: the deepest any placement uses).
    exp_overlap: the row-group schedule (default: what the default compute config runs)."""
    if exp_overlap is None:
        exp_overlap = exp_overlap_for(default_compute_kernel_config())
    kv_depth_cap = max(KV_DEPTH_L1, KV_DEPTH_DRAM) if kv_depth is None else kv_depth
    seq_tiles = math.ceil(seqlen / TILE)
    rows = HEADS * batch * seq_tiles
    starts = _core_row_starts(batch * HEADS, seq_tiles)
    num_cores = len(starts) - 1
    q_block_rows = min(_widest(starts), Q_BLOCK_ROWS_CAP)
    # A core's segments = the pairs its row range touches; deeper K/V rings than that buy nothing.
    pv_out_tiles = HEAD_DIM_TILES + 1  # [O | l]: the ones / row-sum column
    row_group_rows = max(1, min(ROW_GROUP_ROWS, q_block_rows))
    exp_overlap_on = bool(exp_overlap) and q_block_rows >= EXP_OVERLAP_MIN_ROWS
    max_segments = max((starts[c + 1] - 1) // seq_tiles - starts[c] // seq_tiles + 1 for c in range(num_cores))
    return Plan(
        batch=batch,
        seqlen=seqlen,
        seq_tiles=seq_tiles,
        tensor_query_rows=rows,
        num_cores=num_cores,
        core_row_start=tuple(starts),
        q_block_rows=q_block_rows,
        key_block_tiles=seq_tiles,
        head_dim_tiles=HEAD_DIM_TILES,
        pv_out_tiles=pv_out_tiles,
        q_depth=max(Q_DEPTH_L1, Q_DEPTH_DRAM) if q_depth is None else q_depth,
        # the block-sequential schedule waits for the whole Q block before Q K^T: its lead is the whole block
        q_lead_rows=(
            min(Q_LEAD_GROUPS * row_group_rows, q_block_rows) if Q_LEAD_GROUPS and exp_overlap_on else q_block_rows
        ),
        kv_depth=min(kv_depth_cap, max_segments),
        out_depth=OUT_DEPTH,
        kv_mcast=bool(KV_MCAST) and seq_tiles >= KV_MCAST_MIN_SEQ_TILES,
        exp_overlap=exp_overlap_on,
        row_group_rows=row_group_rows,
    )


def _plan_cb_bytes(p: Plan) -> int:
    from .nomic_sdpa_program_descriptor import circular_buffer_specs  # noqa: PLC0415

    return sum(pages * page_bytes for _, pages, page_bytes in circular_buffer_specs(p))


def circular_buffer_bytes(batch: int, seqlen: int, kv_depth: int | None = None, compute_kernel_config=None) -> int:
    """L1 a core the op's circular buffers take beyond the io tensors (the single CB sizing source). kv_depth: the K/V
    ring-depth cap (default: the deepest one any placement uses); compute_kernel_config: the call's (default: the
    op's default), which selects the schedule and so the scores / probs ring sizes."""
    compute = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    return _plan_cb_bytes(plan(batch, seqlen, kv_depth, exp_overlap_for(compute)))


# ---------------------------------------------------------------------------
# 4. validate()
# ---------------------------------------------------------------------------


def _refuse(message):
    raise SupportRefusal(f"nomic_sdpa: {message}")


def _buffer_axis(tensor):
    return "l1" if tensor.memory_config().buffer_type == ttnn.BufferType.L1 else "dram"


def _is_interleaved(config):
    return config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _check_malformed(query, key, value, key_lengths):
    """ValueError for malformed calls; runs before any support check so the error is stable across refinements."""
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if len(tensor.shape) != 4:
            raise ValueError(f"nomic_sdpa: {name} must be rank 4 (B, heads, S, head_dim), got {tuple(tensor.shape)}")
    shape = tuple(query.shape)
    if tuple(key.shape) != shape or tuple(value.shape) != shape:
        raise ValueError(
            f"nomic_sdpa: query / key / value shapes differ: {shape}, {tuple(key.shape)}, {tuple(value.shape)}"
        )
    if key_lengths is not None:
        batch, seqlen = shape[0], shape[2]
        if len(key_lengths) != batch:
            raise ValueError(f"nomic_sdpa: len(key_lengths) = {len(key_lengths)} != B = {batch}")
        for length in key_lengths:
            if not (1 <= int(length) <= seqlen):
                raise ValueError(f"nomic_sdpa: key length {length} outside 1 .. {seqlen}")


def _compute_key(config):
    return (config.math_fidelity, config.fp32_dest_acc_en, config.math_approx_mode, config.dst_full_sync_en)


def validate(
    query,
    key,
    value,
    *,
    scale=1.0,
    key_lengths=None,
    memory_config=None,
    compute_kernel_config=None,
):
    """The runtime gate (op_design.md § Support Contract and Validation, steps 1-7; the live-L1 step 8 runs in the
    entry point around the output allocation). Returns the call's Plan."""
    # 1. malformed calls
    _check_malformed(query, key, value, key_lengths)

    # 2. the registry cell
    axes = {
        "dtype": query.dtype,
        "buffer": _buffer_axis(query),
        "mask_mode": "none" if key_lengths is None else "key_padding",
    }
    for axis_name, tagger in INPUT_TAGGERS.items():
        axes[axis_name] = tagger((tuple(query.shape),), axes)
    for axis, allowed in SUPPORTED.items():
        if axes[axis] not in allowed:
            raise UnsupportedAxisValue(f"nomic_sdpa: {axis}={axes[axis]!r} not in SUPPORTED {allowed}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"nomic_sdpa: unsupported combination (refinement candidate): {exc}")

    # 3. layouts and placement
    buffer_type = query.memory_config().buffer_type
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.layout != ttnn.TILE_LAYOUT:
            _refuse(f"{name} must be TILE layout")
        if tensor.dtype != query.dtype:
            _refuse(f"{name} dtype {tensor.dtype} differs from query's {query.dtype}")
        if not _is_interleaved(tensor.memory_config()):
            _refuse(f"{name} must be interleaved")
        if tensor.memory_config().buffer_type != buffer_type:
            _refuse("query, key and value must share one buffer type")
    out_config = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    if not _is_interleaved(out_config):
        _refuse("memory_config must be interleaved")

    # 4. domain
    batch, heads, seqlen, head_dim = (int(d) for d in query.shape)
    if heads != HEADS or head_dim != HEAD_DIM:
        _refuse(f"needs {HEADS} heads of {HEAD_DIM}, got {heads} x {head_dim}")
    if seqlen > MAX_SEQLEN:
        _refuse(f"S = {seqlen} > {MAX_SEQLEN}: a (b, h) pair's K and V no longer fit one core (use stock SDPA)")
    if batch * math.ceil(seqlen / TILE) * TILE > MAX_TOKENS:
        _refuse(f"B * S_t * 32 = {batch * math.ceil(seqlen / TILE) * TILE} > {MAX_TOKENS}")

    # 5. grid
    grid = query.device().compute_with_storage_grid_size()
    if (grid.x, grid.y) != GRID:
        _refuse(f"the compute grid {grid.x}x{grid.y} is not {GRID[0]}x{GRID[1]}")

    # 6. scale
    if not (isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0):
        _refuse(f"scale must be finite and > 0, got {scale!r}")

    # 7. compute config
    compute = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    if compute.math_fidelity == ttnn.MathFidelity.LoFi:
        _refuse("LoFi is refused (precision)")
    if not compute.fp32_dest_acc_en and compute.math_approx_mode and seqlen > BF16_DEST_APPROX_EXP_MAX_SEQLEN:
        _refuse(
            f"a bfloat16 DEST with the approximate exp misses the precision bar at S = {seqlen} > "
            f"{BF16_DEST_APPROX_EXP_MAX_SEQLEN} (use fp32_dest_acc_en=True or math_approx_mode=False)"
        )

    l1 = axes["buffer"] == "l1"
    return plan(
        batch,
        seqlen,
        KV_DEPTH_L1 if l1 else KV_DEPTH_DRAM,
        exp_overlap_for(compute),
        Q_DEPTH_L1 if l1 else Q_DEPTH_DRAM,
    )


# ---------------------------------------------------------------------------
# Host call path: per-key cached descriptor, rebound every call
# ---------------------------------------------------------------------------


@dataclass
class _CallState:
    plan: Plan
    compute: object
    descriptor: object = None


_CALL_CACHE = {}


def _l1_free_below_live(device):
    """Bytes a core's CBs (allocated up from the allocator base) can take below the lowest live L1 allocation."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    live = [int(block["address"]) for block in view.block_table if block["allocated"] == "yes"]
    lowest = min(live, default=view.total_bytes_per_bank)
    return lowest - ttnn.get_allocator_base_address(device, ttnn.BufferType.L1)


def _scale_bits(scale):
    return struct.unpack("<I", struct.pack("<f", float(scale)))[0]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def nomic_sdpa(
    query,
    key,
    value,
    *,
    scale=1.0,
    key_lengths=None,
    memory_config=None,
    compute_kernel_config=None,
):
    p = validate(
        query,
        key,
        value,
        scale=scale,
        key_lengths=key_lengths,
        memory_config=memory_config,
        compute_kernel_config=compute_kernel_config,
    )
    from .nomic_sdpa_program_descriptor import bind_call, create_program_descriptor  # noqa: PLC0415

    device = query.device()
    out_config = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    compute = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()

    # 8. live L1, before allocating the output and again after it. A deeper K/V ring that does not fit falls back to
    # kv_depth = 1 (the multicast then shares each pair among the cores whose first segment it is), then a deeper
    # cb_query to q_depth = 1.
    candidates = [p]
    for kv_depth, q_depth in ((1, p.q_depth), (1, 1)):
        if (kv_depth, q_depth) != (candidates[-1].kv_depth, candidates[-1].q_depth):
            candidates.append(plan(p.batch, p.seqlen, kv_depth, p.exp_overlap, q_depth))
    free = _l1_free_below_live(device)
    fitting = [c for c in candidates if _plan_cb_bytes(c) <= free]
    if not fitting:
        _refuse(
            f"circular buffers ({_plan_cb_bytes(candidates[-1])} B a core) do not fit below the live L1 allocations"
        )
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([p.batch, 1, p.seqlen, HEADS * HEAD_DIM]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        out_config,
    )
    free = _l1_free_below_live(device)
    fitting = [c for c in fitting if _plan_cb_bytes(c) <= free]
    if not fitting:
        ttnn.deallocate(output)
        _refuse(
            f"circular buffers ({_plan_cb_bytes(candidates[-1])} B a core) do not fit below the live L1 allocations"
        )
    p = fitting[0]

    cache_key = (
        p.batch,
        p.seqlen,
        p.kv_depth,
        p.q_depth,
        query.memory_config().buffer_type,
        repr(out_config),
        "none" if key_lengths is None else "key_padding",
        _scale_bits(scale),
        _compute_key(compute),
        device.id(),
    )
    state = _CALL_CACHE.get(cache_key)
    if state is None:
        state = _CallState(plan=p, compute=compute)
        _CALL_CACHE[cache_key] = state

    if state.descriptor is None:
        state.descriptor = create_program_descriptor(state.plan, query, key, value, output, state.compute, scale)
    bind_call(state.descriptor, query, key, value, output)
    return ttnn.generic_op([query, key, value, output], state.descriptor)
