# hop_aware_noc: the op's tilize_program_descriptor.py with graduate_writer_hop.patch applied (verification copy;
# the harness variant "grad" points its KERNEL_DIR at kernels_dedg_hop/). Not imported by the op.
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor for the `row_split_interleaved`, `sharded_resident`,
`sharded_accessor` and `retile_l1_facewalk` regimes (op_design.md -> Regimes).

Retile (a Layout::TILE input, `in_tile_h`): only `load_block` changes. The reader
stages whole input tiles (or reads a resident input shard in place) and face-walks
them into cb_input_sticks; compute and writer are the stick path's. The work unit
along tile_row is `row_align` tile-rows so an input tile-row never straddles cores.

Core assignment (`_core_assignment`, one source for all three regimes):
  * an L1-sharded OUTPUT whose shard is `resident_ok` fixes it (the output shard
    grid wins when both sides are sharded); else
  * an L1-sharded INPUT whose shard is `resident_ok`; else
  * `split_work_to_cores(grid, R, row_wise=True)` over the runtime grid.
Each side is then RESIDENT (its CB is backed on its own shard via
`ttnn.cb_descriptor_from_sharded_tensor`, zero NoC bytes) iff its per-core shard
rectangle equals the core's assigned rectangle; every other side is STREAMED
through `TensorAccessor` (interleaved DRAM/L1, DRAM-sharded, cross-spec remote
shards, ND specs without a 2-D equivalent).

Blocking knobs, each defined ONCE here and passed to the kernels as a single CT
or RT arg (every dependent quantity derives from it):

  tile_row  axis  ->  block_height = core_row_tiles   (RT, from split_work_to_cores)
                      rows_per_quantum = streaming window along tile_row (CT; from QUANTUM_MIN_TILES)
  tile_col  axis  ->  block_width  = balanced_width(core_col_tiles_max, block_width_cap)   (CT)
                      num_col_groups = grid_2d_split(R, C, N) (1 = the row split)
  depth knobs     ->  DEPTH_IN, DEPTH_OUT   (CB total_size and per_col_tile_bytes only)
  in-flight       ->  read_ahead / write_ahead = CB quanta covering READ_WINDOW_MIN_TILES /
                      WRITE_WINDOW_MIN_TILES; depth_in / depth_out = max(DEPTH_*, window)
  L1 budget       ->  CB_BUDGET_BYTES[low_l1]

Per Tensix core the kernels process the output-tile rectangle
[row_start, row_start + core_row_tiles) x [col_start, col_start + core_col_tiles),
cut into ceil(core_col_tiles / block_width) column blocks; the walk over them is
streamed rows_per_quantum tile-rows (rows_per_quantum * block_width tiles) per CB
quantum through two CBs of depth_in / depth_out quanta (2 at the default knobs).
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
# Per-stage device zones (MaybeDeviceZoneScope, ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp)
# are compiled in only when the kernels get KERNEL_PERF_ZONES: set TT_METAL_KERNEL_PERF_ZONES=1 for a
# perf investigation. Off by default because the markers sit on the critical path (+14 % device-kernel
# ns on [1,1,128,64], Perf 1), so a plain profiled run measures the production kernel.
PERF_ZONES_ENV = "TT_METAL_KERNEL_PERF_ZONES"


def _kernel_defines():
    return [("KERNEL_PERF_ZONES", "1")] if os.environ.get(PERF_ZONES_ENV, "0") == "1" else []


TILE_WIDTH = 32  # elements per tile row (a tile's width is always 32)
FULL_TILE_HEIGHT = 32  # rows of a full (non-tiny) tile; tiny tiles are power-of-two fractions of it

# CB slots (semantic names; the index is just a slot).
CB_INPUT_STICKS = 0  # reader -> compute: tile_h stick segments per tile-row, block_width tile-sized pages
CB_OUTPUT_TILES = 1  # compute -> writer: block_width TILE pages per tile-row
CB_INPUT_STICKS_ODD = 2  # split reader only: BRISC -> compute, the odd tile-rows (one producer per CB)
CB_RETILE_STAGING = 3  # retile only: reader-private staging of whole input tiles (or the resident input shard)
CB_PAD_SOURCE = 4  # padded only: reader-private region of fill values, the source of NoC loopback fills
# bank_coalesced only: reader-private staging ring of bank-major stick runs. Aliases the retile
# staging slot: the two regimes are disjoint (a CB index, and the reader's CT arg, serve both).
CB_COALESCE_STAGING = CB_RETILE_STAGING

# Buffer-depth knobs, counted in CB quanta (one quantum = rows_per_quantum tile-rows
# of block_width pages each).
DEPTH_IN = 2
DEPTH_OUT = 2

# CB quanta of stick reads a producer keeps in flight before it waits on the oldest
# one's (transaction-id) barrier (op_design.md perf lamp "Read in-flight depth").
# 1 = barrier every quantum before issuing the next. Must be <= DEPTH_IN.
# Measured flat on WH 64 cores (2 vs 1: 26.5 vs 26.9 us on [1,1,16384,64], 20.2 vs
# 19.8 us on [1,1,16384,32]): the pipeline is DRAM-traffic-bound there, not
# barrier-bound. Parked at the trivial value; the knob stays live.
READ_AHEAD = 1

# retile_l1_facewalk: units of whole input tiles the staging ring holds (the reader keeps
# RETILE_STAGE_DEPTH - 1 units of page-sized tile reads in flight ahead of the face walk).
# 1 = read, barrier, walk, serially.
RETILE_STAGE_DEPTH = 2

# retile_l1_facewalk: who moves the face rows (16 elements each) from the staged input
# tiles into cb_input_sticks. True = NoC loopback reads on the core (the RISC-V only issues
# commands); False = RISC-V word copies.
RETILE_FACEWALK_NOC = True

# Minimum CB quantum in tiles along the tile_row streaming window. The reader's read
# barrier + CB push and the writer's flush + CB pop happen once per quantum of
# rows_per_quantum tile-rows (compute still tilizes one block_width tile-row per helper
# block). rows_per_quantum = ceil(QUANTUM_MIN_TILES / block_width), then capped by the
# busiest core's walk length // DEPTH_IN (so the double buffer still has quanta to
# overlap: one quantum per core serializes read -> tilize -> write) and by what the CB
# budget leaves after block_width is chosen. It never shrinks block_width, and it only
# engages where one tile-row is narrower than QUANTUM_MIN_TILES tiles. 1 = one
# handshake per tile-row everywhere.
# Measured on WH B0, 64 Tensix cores, bf16 (median device-kernel ns, 10 reps), by the
# tile-rows per quantum it yields:
#   [1,1,16384,64] (block_width 2)  1: 26651  2: 26468  4: 25590  8 (uncapped): 28724
#   [1,1,16384,32] (block_width 1)  1: 19155  2: 19656  4: 18212  8 (uncapped): 20531
# Wider tile-rows (block_width >= 8, e.g. [1,1,8192,256]) measured flat-to-worse when
# coarsened, hence the floor is in tiles, not in tile-rows.
QUANTUM_MIN_TILES = 8

# Per-core CB budget in bytes; the only thing low_l1 changes (l1_ledger.md -> Footprint).
CB_BUDGET_BYTES = {False: 524288, True: 65536}

# Fast-tilize eligibility cap on block_width (tilize_helpers.inl: block_width_tiles < 256).
FAST_TILIZE_MAX_BLOCK_WIDTH = 255

# grid_2d_split (op_design.md -> Regimes): the tile_col axis is split into num_col_groups column
# groups across Tensix cores when that lowers the busiest core's tile count (grid_2d_split()).
# Transaction-size floor (op_design.md perf lamp "Transaction size vs occupancy"): a column group
# holds at least MIN_GROUP_COL_TILES tile-columns (rounded up to col_align_tiles), so a stick
# segment read is at least MIN_GROUP_COL_TILES * 32 * elem_bytes bytes. 1 = maximum participation.
# Measured on WH B0 (64 Tensix cores), device-kernel ns, floor 1 / 2 / 4 / 8 / 16 tile-columns:
#   [1,1,32,2048]  3828 (64 cores) / 3969 (32) / 4068 (16) / 4368 (8) / 5395 (4); row split 14273 (1)
#   [1,1,32,8192]  7227 (64) / 7376 (64) / 7376 (64) / 8134 (32) / 8197 (16);    row split 30847 (1)
# Maximum participation wins: the floor is parked at 1, a live knob.
MIN_GROUP_COL_TILES = 1
# Minimum walk positions per Tensix core (tile-rows x column blocks) before the column axis is cut
# into extra blocks: with one position the depth-2 CBs have nothing to overlap (read -> tilize ->
# write run back to back). Only engages while every block keeps stick segments of at least
# PIPELINE_MIN_SEGMENT_BYTES (narrow segments cost more in transactions than overlap buys).
# 1 = off. Measured on WH, 64 Tensix cores, device-kernel ns (median of 3 where given):
#   [1,1,2048,2048] (1 tile-row x 64 tiles per core)  1: 92293   2 (2 x 2 KiB-segment blocks): 87033
#   [1,1,2048,1024] (1 x 32, 1 KiB segments at 2)     1: 45171   2: 45412 (flat)
#   ungated, short_wide: [1,1,32,8192] 7505 -> 10006 (2) / 11523 (4); [1,1,64,4096] 8128 -> 9398 / 11946;
#   [1,1,32,4096] 4985 -> 6763 / 6657 -- hence the segment floor.
PIPELINE_MIN_POSITIONS = 2
PIPELINE_MIN_SEGMENT_BYTES = 2048
# Fixed cost of one tile-row on a Tensix core, in output tiles: the rule's makespan is
# rows * (cols + ROW_COST_TILES), not rows * cols. A tile-row always costs tile_h stick-segment
# reads plus a CB handshake whatever its width, so a column split that multiplies a core's
# tile-rows pays that again for every extra tile-row. 0 = the pinned tile-count rule. Fitted
# from the row split's per-core timings (t ~ rows * (a + b * cols) + c on WH: a / b ~ 1.5).
# Measured, WH, device-kernel ns (0 vs 1.5 -> the split the rule picks):
#   [4,3,256,96] (R 96, C 3): 0 -> 3 column groups, 11183 ns; 1.5 -> row split, 8860 ns
ROW_COST_TILES = 1.5

# Split-reader knob (op_design.md perf lamp "Reader issue-rate"). When a core's
# stick-segment reads are small, one RISC-V is issue-bound; BRISC (the writer,
# which issues only valid_width tile writes per tile-row) then also produces the
# odd tile-rows through its own CB. Engaged when the stick segment is at most
# this many bytes and the core has at least two tile-rows to alternate.
# 0 disables the split reader entirely.
# Measured on WH 64 cores: correct but slower (19.2 -> 20.2 us on [1,1,16384,32],
# 26.9 -> 31.9 us on [1,1,16384,64]): BRISC then carries the odd reads AND the
# writes, and the writes alone are DRAM-throughput-bound (~130 GB/s aggregate).
# Parked disabled; what still has to shorten first is the write path.
SPLIT_READER_MAX_SEGMENT_BYTES = 0

# ---- Refinement 3 levers (block-quantum / depth / NoC co-tune). Measured on WH B0, 64 Tensix
# cores, [1,1,16384,64] bf16 DRAM interleaved (the perf-focus shape), median device-kernel ns,
# baseline 25.3-25.6 us. The shape is bound by aggregate DRAM throughput for its access mix:
# half the Tensix cores (32) is only 17 % slower; reads-only 16.1 us, writes-only 19.2 us and
# the no-transfer floor 8.1 us nearly add up (ablations). Every lever below is correct
# (bit-exact) and parked at its trivial default, which leaves the compiled kernels on the
# pre-Refinement-3 path; each stays a live tunable.

# In-flight windows, in full 32-row tile equivalents: read_ahead (reader) and write_ahead
# (writer) become the CB quanta that cover the window, and DEPTH_IN / DEPTH_OUT grow to hold
# them. Only a small quantum (a narrow tile-row) opens a window wider than one quantum.
# 0 = one quantum (READ_AHEAD, write_ahead 1). Measured with QUANTUM_MIN_TILES = 2 and both
# windows at 8: [1,1,16384,64] 25.4 -> 24.7..25.0 us in four runs, but 25.8 vs 25.5 us in a
# fifth (noise band); [1,1,16384,32] 18.0 -> 19.7-20.0, [1,1,32768,64] 51.7 -> 54.4-54.7,
# [1,1,4096,64] 7.7 -> 7.9-8.0 us; wide rows flat; only [4,3,256,96] (1-2 tile-rows per core)
# wins clearly, 9.0 -> 8.1 us. Not adopted: no shape rule separates that win from the losses.
READ_WINDOW_MIN_TILES = 0
WRITE_WINDOW_MIN_TILES = 0
MAX_WINDOW_QUANTA = 8  # <= 15: one NoC transaction id per CB slot on each side
# Outstanding reads one NoC transaction id can count (noc_parameters.h); caps a slot's reads.
NOC_MAX_TRANSACTION_ID_COUNT = 255

# Eager publish: the reader pushes every slot whose reads have landed before it issues the next
# tile-row (non-blocking trid poll), so a deep read_ahead never holds finished rows back. Without
# it, read_ahead == the core's slot count serializes the core (all reads, then tilize, then
# writes: 27.4 us at 1-row quanta, depth 8). With it that case is 25.1 us; at the default
# schedule it is flat (25.5 vs 25.3 us).
EAGER_PUBLISH = False

# Split one stream across both NoCs: every READ_NOC_SPLIT-th stick read of a tile-row (every
# WRITE_NOC_SPLIT-th tile write) leaves on the other NoC; the data-movement kernels then run in
# DM_DYNAMIC_NOC (dynamic mode alone, nothing moved: 26.4 vs 25.9 us). 0 = off. Reads on NoC1
# and writes on NoC0 run against the DRAM geometry here: reads 1-in-2 68.8 -> 36.0 us once the
# kernel NoCs were right, 1-in-8 33.4 us, 1-in-32 ~ +18 %; writes 1-in-2 45.4 us, 1-in-8 26.2
# (flat). Parked off.
READ_NOC_SPLIT = 0
WRITE_NOC_SPLIT = 0

# Hop-aware write NoC (Perf 2, perf_experiments/hop_aware_noc). The writer (BRISC, NoC1) sends DRAM
# bank b's tile writes on NoC0 instead when NoC0's Tensix core -> bank path is at least this many
# hops shorter (28 % of (core, bank) pairs on WH n150), still in DM_DEDICATED_NOC. Only where NoC0
# carries no DRAM read traffic (input not in DRAM): with DRAM reads on NoC0 it measured +12..30 %.
# Same-session medians of 3: [1,1,2048,512] HEIGHT_SHARDED L1 -> DRAM 16671 -> 14376 ns, L1
# interleaved [1,1,16384,64] -> DRAM 22120 -> 20179 ns. 0 = off.
HOP_WRITE_MIN_SAVING = 6
# ...and only on outputs of at least this many full-tile equivalents: below it the writes are
# latency-bound, not NoC1-congested, and NoC0's slower DRAM write path costs (8 Tensix cores x
# 16 tiles = 128 tiles: +5..10 %; 256 tiles on 2 or 16 Tensix cores: -6 / -4 %).
HOP_WRITE_MIN_TILES = 256
HOP_SEM = 1  # semaphore id of the writer's "NoC0 writes ACKed" flag (CO_READ_SEM is 0)

# ---- Padding (Refinement 4). The reader fills everything of the padded tile grid the input does
# not cover (W tail, H tail, whole pad sticks / tile-rows / images) before it publishes a slot.
# Fills shorter than PAD_NOC_MIN_BYTES are RISC-V stores (fill_l1_range); longer ones are NoC
# loopback copies, in chunks of up to PAD_SOURCE_BYTES, from a reader-private source region
# (cb_pad_source) filled once per kernel, so the RISC-V only issues commands for whole pad
# sticks and tiles. Both are live knobs; PAD_NOC_MIN_BYTES must be >= 2 * L1 alignment.
PAD_SOURCE_BYTES = 1024
PAD_NOC_MIN_BYTES = 128
# W-tail persistence: with one column block per core, the W-tail band of a CB row is filled
# on the walk's first pass through the CB ring only (nothing overwrites it afterwards).
# False = band-fill every tile-row.
PAD_W_TAIL_PERSIST = True
MAX_PAD_LEAD_DIMS = 8  # PadMap::MAX_LEAD_DIMS (tilize_stick_reads.hpp)


class PadSpec:
    """What the reader fills: the padded output shape, the input left-expanded to its rank, the fill bits."""

    def __init__(self, padded_shape, input_shape_expanded, fill_bits):
        assert len(padded_shape) == len(input_shape_expanded) >= 2
        self.padded_shape = [int(d) for d in padded_shape]
        self.input_shape = [int(d) for d in input_shape_expanded]
        self.fill_bits = int(fill_bits)

    @property
    def needs_fill(self):
        return self.padded_shape != self.input_shape

    def reader_rt_args(self, tile_h, in_elem_bytes):
        """PadMap RT args (tilize_stick_reads.hpp): rows per padded image, input H, input stick
        data bytes, then the leading dims to decode (innermost first, up to the outermost one the
        pad grows; the outer, equal dims fold into the quotient)."""
        P, X = self.padded_shape, self.input_shape
        lead_p, lead_x = P[:-2], X[:-2]
        differ = [k for k in range(len(lead_p)) if lead_p[k] != lead_x[k]]
        decode = list(range(differ[0], len(lead_p)))[::-1] if differ else []
        assert len(decode) <= MAX_PAD_LEAD_DIMS
        args = [P[-2] // tile_h, X[-2], X[-1] * in_elem_bytes, len(decode)]
        for k in decode:
            args += [lead_p[k], lead_x[k]]
        return args


# Reader address generation on a DRAM-interleaved input: 1 = bank-stride reuse (stick p + NB
# is stick p's bank, one aligned page further, so only NB accessor calls per core), 2 = the
# same plus bank-major issue order within a tile-row. 0 = one accessor call per stick. The
# reader issues ~42 cycles per 128-byte read uncontended, ~23 of them address math, yet
# cheaper issue made reads-only SLOWER (16.1 -> 18.9 us for 1, 19.5 us for 2: faster request
# issue congests the DRAM banks) and the full op flat (25.2 / 25.8 vs 25.4 us). Parked off; it
# is the addressing a bank-coalesced read (Refinement 6) builds on.
BANK_STRIDE = 0

# ---- Refinement 6: bank-coalesced stick reads (bank_coalesced load_block, tilize_reader.cpp).
# On a DRAM-interleaved input, stick page p lives in bank p % NB at offset (p // NB) * page, so
# the sticks of one run of consecutive tile-rows that share a bank are contiguous there: one NoC
# read per bank fetches them into a reader-private staging ring (CB_COALESCE_STAGING), and NoC
# loopback reads then move each stick to its tilize position in cb_input_sticks. Engaged only
# when every Tensix core reads WHOLE sticks in one column block (block_width == C) of at most
# BANK_COALESCE_MAX_STICK_BYTES, with the StickProducer-only levers at their defaults.
# Measured on WH B0, 64 Tensix cores, bf16, device-kernel ns (median of 3), off -> on:
#   [1,1,16384,64] 25967 -> 23303   [1,1,16384,32] 18170 -> 13347   [1,1,32768,64] 51477 -> 46344
#   [1,1,4096,64] / [4,3,256,96] / [1,1,16384,128] flat; [1,1,8192,256] flat; [1,1,2048,1024] +7.6 %
# Ablations on [1,1,16384,64]: the no-transfer floor fell 8.1 -> 2.3 us (NB accessor calls per
# run instead of one per stick) and reads-only 16.1 -> 12.0 us, flat in the per-bank read size
# (3..21 sticks); the loopback scatter costs ~22 cycles per stick and is ~1.3 us of the wall.
# BANK_COALESCE_STAGE_DEPTH: staging ring depth in CB quanta (units in flight = depth - 1);
# 0 = off (StickProducer, one read per stick). 3 measured flat-to-worse vs 2.
BANK_COALESCE_STAGE_DEPTH = 2
# The bank_coalesced path's CB quantum in tile-rows (full 32-row equivalents; replaces the
# QUANTUM_MIN_TILES floor there). Each quantum is a serial land -> loopback scatter -> push step,
# so a finer quantum than StickProducer's pipelines better; the best unit measured the same
# 2 tile-rows (64 sticks) at every narrow width ([1,1,16384,64]: 1 / 2 / 4 rows 24449 / 23303 /
# 24304; [1,1,16384,32]: 2 / 4 / 8 rows 13347 / 14459 / 14287; [1,1,32768,64]: 1 / 2 / 4 rows
# 49436 / 46344 / 46774).
BANK_COALESCE_QUANTUM_ROWS = 2
# Path gate: only sticks of at most this many bytes coalesce. Larger sticks are already large
# transactions and the scatter only adds work ([1,1,2048,1024], 2 KiB sticks: +7.6 %).
BANK_COALESCE_MAX_STICK_BYTES = 256
# Scatter by NoC loopback WRITES (the write command buffer) instead of reads. Measured slower
# (scatter-only 8.8 vs 8.1 us; [1,1,16384,32] 14259 vs 13215): parked off, a live knob.
BANK_COALESCE_SCATTER_WRITE = False

# ---- Refinement 8: co-read (tiny work). When every Tensix core's walk is ONE position (one
# tile-row of one column block: [1,1,128,64] on 8 cores, [1,1,32,2048] on 64), nothing overlaps
# across tile-rows and the op is a latency chain: read issue -> read landing -> tilize -> write.
# Its longest link is the stick-read issue on one RISC-V (~45 cycles per NoC read: five command
# registers + the ready poll; 32 reads ~1.46 us of a ~3.0 us op), while the writer RISC-V sits
# idle in cb_wait_front. Co-read hands the last CO_READ_SHARE of every tile-row's stick reads to
# the writer RISC-V (BRISC, NoC1), which reads them straight into cb_input_sticks' slot and raises
# a local semaphore NCRISC waits on before its push (NCRISC stays the CB's only producer).
# 0 = off (NCRISC reads every stick). Only the plain stick walk (not padded / coalesced / split
# reader / bank-stride / NoC-split) takes it.
# Measured on WH B0, bf16, device-kernel ns (median of 2-3), off -> on at 0.5:
#   DRAM  [1,1,128,64] (8 cores, 64 B segments) 2988 -> 2300   [1,1,32,2048] (64, 64 B) 3818 -> 3463
#         [1,1,2048,32] (64, 64 B; was bank_coalesced) 4318 -> 3405   [1,1,2048,64] (128 B) 5949 -> 5344
#   share 0.375 / 0.4375 / 0.5 / 0.5625 / 0.625 on [1,1,128,64]: 2591 / 2402 / 2300 / 2521 / 2447
CO_READ_SHARE = 0.5
CO_READ_SEM = 0  # semaphore id of the writer's "co-read landed" flag (the op's only semaphore)
# Where it engages, by the input's BufferType: (min, max) stick-segment bytes, None = unbounded.
# The writer's share travels on NoC1, which reads DRAM poorly: past ~128 B a DRAM read is no
# longer issue-bound and the NoC1 half loses (192 B +4.6 %, 256 B +3..6 %, 512 B..2 KiB +26..31 %,
# medians of 3). An L1 source reads well on both NoCs: interleaved L1 wins at every width
# ([1,1,2048,W], 64 cores: 128 B -14 %, 512 B -24 %, 1 KiB -17 %, 2 KiB -28 %). A sharded L1 input
# that streams (i.e. is not consumed resident) reads the same L1 banks through the accessor.
CO_READ_SEGMENT_BYTES = {ttnn.BufferType.DRAM: (0, 128), ttnn.BufferType.L1: (0, None)}


# ---- Numeric formats (Refinement 7). cb_input_sticks carries the input dtype and cb_output_tiles
# the output dtype; the value-preserving cast happens at pack. A 32-bit page (Float32 / Int32 /
# UInt32) on either side needs fp32 DEST (a 16-bit DEST would round it); a 32-bit INPUT also
# needs the lossless tilize path: its CB is tagged UnpackToDestFp32 (unpacked straight into
# DEST, bypassing SrcA's tf32) and the helper runs Fp32Mode::Lossless. Everything else (the
# bf16 -> bf16 perf path included) keeps 16-bit DEST + fast tilize.
WIDE_DTYPES = (ttnn.float32, ttnn.int32, ttnn.uint32)
# DTYPES whose pages need fp32 DEST: the 32-bit ones, plus UInt8 (measured on WH: through a 16-bit
# DEST every uint8 datum packs as 0; through fp32 DEST it is bit-exact). UInt16 is exact on 16-bit DEST.
FP32_DEST_DTYPES = WIDE_DTYPES + (ttnn.uint8,)
# DTYPES whose pages must NOT meet fp32 DEST: UInt16 (measured on WH: fp32_dest_acc_en=True scrambles
# every datum, PCC ~0). A caller's fp32_dest_acc_en request is ignored for them.
FP32_DEST_FORBIDDEN_DTYPES = (ttnn.uint16,)
# Block-float outputs: bfp8_pack_precise (the packer converts straight from the DEST format instead
# of through a Bfp8 pack-source format). Measured on WH, golden suite: fp32 -> bfloat8_b on randn
# becomes bit-identical to the host's quantization (default: max |diff| 0.0625), but a rank-0
# datum that bfp8 represents exactly comes back off by 0.0078 and a bfloat4_b pad cell drops to
# PCC 0.9799 (< 0.98). bfloat4_b is truncated by the packer either way. Parked off.
BFP_PACK_PRECISE = False
BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)
# unpack_to_dest_mode is indexed by CB id; sized for the largest CB count of any arch.
MAX_CIRCULAR_BUFFERS = 64


class NumericConfig:
    """The single source of the compute-side numeric configuration for one (in, out) dtype pair.

    `compute_kernel_config` (optional, ttnn.ComputeKernelConfig): tilize does no arithmetic, so
    math_fidelity / math_approx_mode are passed through without effect; fp32_dest_acc_en can
    only be turned ON by the caller (the page formats force it where a 32-bit / uint8 page exists)
    and is ignored where a page format forbids it (uint16);
    dst_full_sync_en disables fast tilize (correct, slower).
    """

    def __init__(self, in_dtype, out_dtype, compute_kernel_config=None):
        ckc = compute_kernel_config
        self.lossless = in_dtype in WIDE_DTYPES
        requested = bool(ckc is not None and ckc.fp32_dest_acc_en)
        forbidden = in_dtype in FP32_DEST_FORBIDDEN_DTYPES or out_dtype in FP32_DEST_FORBIDDEN_DTYPES
        self.fp32_dest_acc_en = (
            in_dtype in FP32_DEST_DTYPES or out_dtype in FP32_DEST_DTYPES or (requested and not forbidden)
        )
        self.dst_full_sync_en = bool(ckc is not None and ckc.dst_full_sync_en)
        self.bfp_pack_precise = BFP_PACK_PRECISE and out_dtype in BLOCK_FLOAT_DTYPES
        self.math_fidelity = ckc.math_fidelity if ckc is not None else None
        self.math_approx_mode = bool(ckc.math_approx_mode) if ckc is not None else None

    def compute_config(self, input_cbs):
        kwargs = dict(fp32_dest_acc_en=self.fp32_dest_acc_en, dst_full_sync_en=self.dst_full_sync_en)
        if self.math_fidelity is not None:
            kwargs["math_fidelity"] = self.math_fidelity
            kwargs["math_approx_mode"] = self.math_approx_mode
        config = ttnn.ComputeConfigDescriptor(**kwargs)
        config.bfp8_pack_precise = self.bfp_pack_precise
        if self.lossless:
            modes = [ttnn.UnpackToDestMode.Default] * MAX_CIRCULAR_BUFFERS
            for cb in input_cbs:
                modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
            config.unpack_to_dest_mode = modes
        return config


def _div_up(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _div_up(a, b) * b


def _tile_grid(shape, tile_h):
    """(R, C) of the output tile grid; leading dims fold into R (per-image ceil)."""
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)
    R = leading * _div_up(int(shape[-2]), tile_h)
    C = _div_up(int(shape[-1]), TILE_WIDTH)
    return R, C


def _col_align_tiles(input_tensor, in_elem_bytes):
    """Tile-columns per NoC-alignment unit for stick-segment offsets (source and L1 destination)."""
    src_align = (
        ttnn.get_dram_alignment()
        if input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        else ttnn.get_l1_alignment()
    )
    align_bytes = max(src_align, ttnn.get_l1_alignment())
    return max(1, _div_up(align_bytes, TILE_WIDTH * in_elem_bytes))


def balanced_width(core_col_tiles_max, *, per_col_tile_bytes, low_l1, col_align_tiles, min_col_blocks=1):
    """op_design.md `balanced_width`: coarsest column block that fits the CB budget, balanced over blocks.

    `min_col_blocks` (> 1 only for PIPELINE_MIN_POSITIONS) cuts the core's columns into at least
    that many blocks, never below col_align_tiles tile-columns per block.
    """
    cap = min(FAST_TILIZE_MAX_BLOCK_WIDTH, CB_BUDGET_BYTES[low_l1] // per_col_tile_bytes)
    cap = max(col_align_tiles, (cap // col_align_tiles) * col_align_tiles)
    num_col_blocks = max(
        _div_up(core_col_tiles_max, cap), min(min_col_blocks, _div_up(core_col_tiles_max, col_align_tiles))
    )
    width = _round_up(_div_up(core_col_tiles_max, num_col_blocks), col_align_tiles)
    return min(width, cap)


def grid_2d_split(R, C, num_cores, *, col_align_tiles, row_align=1, min_group_col_tiles=None, row_cost_tiles=None):
    """op_design.md `grid_2d_split` assignment rule -> (g_r, g_c).

    Rows are counted in units of `row_align` tile-rows (retile: an input tile-row never straddles
    two Tensix cores), columns in units of `col_align_tiles` tile-columns (NoC alignment of every
    column start). Minimizes the busiest core's tiles; tie-break 1: wider column groups (larger
    stick segments); tie-break 2: fewer Tensix cores. g_c == 1 is the row split. The busiest core's
    cost counts `row_cost_tiles` (ROW_COST_TILES) extra tiles per tile-row it owns.
    """
    if min_group_col_tiles is None:
        min_group_col_tiles = MIN_GROUP_COL_TILES
    if row_cost_tiles is None:
        row_cost_tiles = ROW_COST_TILES
    row_units = R // row_align
    col_units = _div_up(C, col_align_tiles)
    max_g_c = max(1, col_units // _div_up(min_group_col_tiles, col_align_tiles))
    best = None
    for g_r in range(1, min(row_units, num_cores) + 1):
        rows_busiest = _div_up(row_units, g_r) * row_align
        for g_c in range(1, min(col_units, max_g_c, num_cores // g_r) + 1):
            units_busiest = _div_up(col_units, g_c)
            key = (rows_busiest * (units_busiest * col_align_tiles + row_cost_tiles), -units_busiest, g_r * g_c)
            if best is None or key < best[0]:
                best = (key, g_r, g_c)
    return best[1], best[2]


def _balanced(total, groups):
    """[(start, count)] of `total` units over `groups` groups, the first `total % groups` one larger."""
    base, rem = divmod(total, groups)
    out, start = [], 0
    for g in range(groups):
        n = base + (1 if g < rem else 0)
        out.append((start, n))
        start += n
    return out


def _column_groups(C, num_col_groups, col_align_tiles):
    """[(col_start, core_col_tiles)] balanced over `num_col_groups`, in col_align_tiles units."""
    col_units = _div_up(C, col_align_tiles)
    groups = []
    start_unit = 0
    base, rem = divmod(col_units, num_col_groups)
    for g in range(num_col_groups):
        units = base + (1 if g < rem else 0)
        col_start = start_unit * col_align_tiles
        col_end = min(C, (start_unit + units) * col_align_tiles)
        groups.append((col_start, col_end - col_start))
        start_unit += units
    return groups


_SHARDED_LAYOUTS = (
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
)


def _shard_rects(tensor, rows_total, width):
    """Per-core valid shard rectangles of an L1-sharded tensor over its 2-D fold.

    Returns ((shard_h, shard_w), [(core, r0, nr, c0, nc), ...]) in elements
    (rows = folded sticks, cols = elements of a stick), cores with no data
    omitted; or None when the shard has no one-rectangle-per-core 2-D
    description (interleaved, DRAM, an ND spec without a 2-D equivalent, or
    more shards than cores). Shard -> core follows the shard orientation:
    HEIGHT / WIDTH enumerate the grid row-wise (ROW_MAJOR) or column-wise
    (COL_MAJOR); BLOCK maps shard (i, j) to core (x0 + j, y0 + i) for ROW_MAJOR
    and (x0 + i, y0 + j) for COL_MAJOR.
    """
    mc = tensor.memory_config()
    spec = mc.shard_spec
    if mc.buffer_type != ttnn.BufferType.L1 or mc.memory_layout not in _SHARDED_LAYOUTS or spec is None:
        return None
    shard_h, shard_w = (int(d) for d in spec.shape)
    row_major = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    n_h, n_w = _div_up(rows_total, shard_h), _div_up(width, shard_w)
    if mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        start = spec.grid.bounding_box().start
        placed = [
            (ttnn.CoreCoord(start.x + j, start.y + i) if row_major else ttnn.CoreCoord(start.x + i, start.y + j), i, j)
            for i in range(n_h)
            for j in range(n_w)
        ]
    else:
        cores = ttnn.corerange_to_cores(spec.grid, None, row_major)
        shards = [(k, 0) for k in range(n_h)] if mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else None
        if shards is None:
            shards = [(0, k) for k in range(n_w)]
        if len(shards) > len(cores) or n_h * n_w != len(shards):
            return None
        placed = [(cores[k], i, j) for k, (i, j) in enumerate(shards)]
    rects = []
    for core, i, j in placed:
        nr = min(shard_h, rows_total - i * shard_h)
        nc = min(shard_w, width - j * shard_w)
        if nr > 0 and nc > 0:
            rects.append((core, i * shard_h, nr, j * shard_w, nc))
    return (shard_h, shard_w), rects


def _resident_ok(shard_rects, tile_h):
    """op_design.md `resident_ok`: a one-rectangle-per-core 2-D shard, tile-aligned on both axes."""
    if shard_rects is None:
        return False
    (shard_h, shard_w), rects = shard_rects
    return shard_h % tile_h == 0 and shard_w % TILE_WIDTH == 0 and len(rects) > 0


def _rects_key(shard_rects):
    """Hashable identity of a shard placement: nominal shard shape + every core's valid rectangle."""
    shard_shape, rects = shard_rects
    return shard_shape, [((core.x, core.y), r0, nr, c0, nc) for core, r0, nr, c0, nc in rects]


def _core_assignment(input_tensor, output_tensor, *, rows_total, width, tile_h, max_block_width, input_may_reside=True):
    """(assignment, input_resident, output_resident, shard_block_width).

    `assignment` is [(core, row_start, core_row_tiles, col_start, core_col_tiles)]
    over the output tile grid, or None for the interleaved row split. A shard
    wider than `max_block_width` tiles cannot be one resident block (the
    streamed partner CB would not fit the budget), so it is streamed instead.
    `input_may_reside=False` keeps the input side streamed (a Layout::TILE input
    whose H padding makes its physical rows differ from the logical fold).
    """

    def usable(tensor):
        if tensor is input_tensor and not input_may_reside:
            return None
        rects = _shard_rects(tensor, rows_total, width)
        if not _resident_ok(rects, tile_h) or rects[0][1] // TILE_WIDTH > max_block_width:
            return None
        return rects

    out_rects, in_rects = usable(output_tensor), usable(input_tensor)
    owner = out_rects if out_rects is not None else in_rects
    if owner is None:
        return None, False, False, None
    (_, shard_w), rects = owner
    assignment = [
        (core, r0 // tile_h, _div_up(nr, tile_h), c0 // TILE_WIDTH, _div_up(nc, TILE_WIDTH))
        for core, r0, nr, c0, nc in rects
    ]
    output_resident = out_rects is not None
    input_resident = in_rects is not None and (not output_resident or _rects_key(in_rects) == _rects_key(out_rects))
    return assignment, input_resident, output_resident, shard_w // TILE_WIDTH


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    tile_h: int = 32,
    in_tile_h: int | None = None,
    low_l1: bool = False,
    pad: PadSpec | None = None,
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    """`output_tensor` is allocated at the PADDED shape (the input's shape when nothing is padded);
    the output tile grid, and so the whole schedule, is that shape's. `pad` describes the fill."""
    device = input_tensor.device()

    # ---------------- geometry ----------------
    shape = list(input_tensor.shape)
    out_shape = list(output_tensor.shape)  # the padded shape
    padded = pad is not None and pad.needs_fill
    R, C = _tile_grid(out_shape, tile_h)
    rows_total = R * tile_h  # folded output sticks (the padded H is a multiple of tile_h per image)
    width = int(out_shape[-1])  # elements of an output stick
    in_width = int(shape[-1]) if len(shape) >= 1 else 1  # elements of an input stick (rank 0: one)

    in_elem_bytes = input_tensor.element_size()
    in_tile_bytes = tile_h * TILE_WIDTH * in_elem_bytes  # tile-sized page holding tile_h stick segments
    out_tile_bytes = output_tensor.buffer_page_size()  # one TILE page of the output dtype
    stick_page_bytes = (
        input_tensor.buffer_aligned_page_size()
    )  # input page stride (a stick, a shard-width stick, or a tile)
    # A WIDTH / BLOCK / ND-sharded Layout::ROW_MAJOR input cuts every stick into pages of the
    # shard width; a stick-segment read then splits at page boundaries (reader CT args).
    in_page_bytes = input_tensor.buffer_page_size()
    pages_per_stick = _div_up(in_width * in_elem_bytes, in_page_bytes)

    # retile_l1_facewalk (Layout::TILE input): the reader stages whole input tiles (one
    # [in_tile_h, 32] page each) and face-walks them into cb_input_sticks. A unit is row_align
    # output tile-rows fed by unit_in_rows input tile-rows of one image; row_align > 1 only
    # when every image's H is a whole number of input tile-rows (else each output tile-row
    # reads its input tile-row whole: correct, in_tile_h / tile_h read amplification).
    retile = in_tile_h is not None
    assert not (retile and padded), "retile x padding is refused by EXCLUSIONS"
    if retile:
        in_page_bytes = stick_page_bytes  # one input tile, the unit of every retile NoC read
        pages_per_stick = 1
        H = int(shape[-2])
        in_rows_per_image = _div_up(H, in_tile_h)
        out_rows_per_image = H // tile_h
        input_whole_tiles = H % in_tile_h == 0
        row_align = in_tile_h // tile_h if in_tile_h > tile_h and input_whole_tiles else 1
        unit_in_rows = max(1, tile_h // in_tile_h)
    else:
        in_rows_per_image = out_rows_per_image = 0  # unused by the stick reader
        input_whole_tiles = True
        row_align = unit_in_rows = 1

    # ---------------- block knobs ----------------
    col_align_tiles = _col_align_tiles(input_tensor, in_elem_bytes)

    def _per_col_tile_bytes(
        num_input_cbs, *, input_resident=False, output_resident=False, depth_in=DEPTH_IN, depth_out=DEPTH_OUT
    ):
        """Bytes of STREAMED CBs per tile-column of one tile-row (resident CBs are the tensor's own L1)."""
        if retile:
            # cb_input_sticks always streams (the face walk fills it); a resident input backs
            # only cb_retile_staging, which otherwise holds RETILE_STAGE_DEPTH units.
            streamed_in = depth_in * in_tile_bytes
            if not input_resident:
                streamed_in += RETILE_STAGE_DEPTH * unit_in_rows * in_page_bytes
        else:
            streamed_in = 0 if input_resident else num_input_cbs * depth_in * in_tile_bytes
        streamed_out = 0 if output_resident else depth_out * out_tile_bytes
        return streamed_in + streamed_out

    # ---------------- core assignment (op_design.md -> Regimes, selection function) ----------------
    # A resident shard is ONE block (block_width = shard width in tiles), so its streamed partner
    # CB must hold depth * shard-width tiles within the budget.
    max_shard_block_width = min(FAST_TILIZE_MAX_BLOCK_WIDTH, CB_BUDGET_BYTES[low_l1] // _per_col_tile_bytes(1))
    assignment, input_resident, output_resident, shard_block_width = _core_assignment(
        input_tensor,
        output_tensor,
        rows_total=rows_total,
        width=width,
        tile_h=tile_h,
        max_block_width=max_shard_block_width,
        # A padded input's stick layout is not the tilize layout of the padded grid: stream it.
        input_may_reside=input_whole_tiles and not padded,
    )
    any_resident = input_resident or output_resident

    if assignment is None:
        # No shard fixes the core assignment (interleaved sides, and every streamed-only sharded
        # case: DRAM-sharded sides, ND specs without a 2-D equivalent). The Tensix core count is
        # read from the device. The split's row unit is row_align tile-rows (1 except retile), so
        # an input tile-row never straddles two Tensix cores.
        grid = device.compute_with_storage_grid_size()
        assert R % row_align == 0
        num_row_groups, num_col_groups = grid_2d_split(
            R, C, grid.x * grid.y, col_align_tiles=col_align_tiles, row_align=row_align
        )
        assignment = []
        if num_col_groups == 1:
            # row_split_interleaved: every core owns columns [0, C).
            _n, all_cores, core_group_1, core_group_2, units_g1, units_g2 = ttnn.split_work_to_cores(
                grid, R // row_align, row_wise=True
            )
            row_start = 0
            for group, units in ((core_group_1, units_g1), (core_group_2, units_g2)):
                for core in ttnn.corerange_to_cores(group, None, True):
                    assignment.append((core, row_start, units * row_align, 0, C))
                    row_start += units * row_align
            assert row_start == R, f"row split covered {row_start} of {R} tile-rows"
        else:
            # grid_2d_split: g_r row groups x g_c column groups on the first g_r * g_c Tensix
            # cores (row-wise). Core k owns row group k // g_c and column group k % g_c, so the
            # column groups of one row group sit side by side along a grid row.
            num_cores = num_row_groups * num_col_groups
            all_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
            cores = ttnn.corerange_to_cores(all_cores, None, True)
            row_groups = _balanced(R // row_align, num_row_groups)
            col_groups = _column_groups(C, num_col_groups, col_align_tiles)
            for k, core in enumerate(cores):
                row_unit_start, row_units = row_groups[k // num_col_groups]
                col_start, core_col_tiles = col_groups[k % num_col_groups]
                assignment.append((core, row_unit_start * row_align, row_units * row_align, col_start, core_col_tiles))
            assert sum(rows * cols for _, _, rows, _, cols in assignment) == R * C
    else:
        # sharded_resident: the owning shard grid (only the Tensix cores holding data).
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core, *_ in assignment])
        if any(start % row_align or rows % row_align for _, start, rows, _, _ in assignment):
            row_align = 1  # an output shard cuts inside input tile-rows: each core reads them whole

    core_row_tiles_max = max(rows for _, _, rows, _, _ in assignment)
    core_col_tiles_max = max(cols for _, _, _, _, cols in assignment)

    # PIPELINE_MIN_POSITIONS: a core whose walk is shorter than that (short_wide: one tile-row)
    # cuts its columns into more blocks so read, tilize and write overlap through the CBs.
    min_col_blocks = 1
    if PIPELINE_MIN_POSITIONS > 1 and not retile:
        segment_floor_tiles = _round_up(
            max(1, _div_up(PIPELINE_MIN_SEGMENT_BYTES, TILE_WIDTH * in_elem_bytes)), col_align_tiles
        )
        min_col_blocks = max(
            1,
            min(_div_up(PIPELINE_MIN_POSITIONS, core_row_tiles_max), core_col_tiles_max // segment_floor_tiles),
        )

    def _block_width_for(num_input_cbs):
        if any_resident:
            return shard_block_width  # the whole resident shard width is one block
        return balanced_width(
            core_col_tiles_max,
            per_col_tile_bytes=_per_col_tile_bytes(num_input_cbs),
            low_l1=low_l1,
            col_align_tiles=col_align_tiles,
            min_col_blocks=min_col_blocks,
        )

    # The split reader decision needs the segment width, which needs block_width, which
    # needs the CB count: resolve it for the split configuration first (the tighter
    # budget), and keep the split only if its segments are small enough. It only
    # applies when both sides stream.
    block_width_split = _block_width_for(2)
    split_segment_bytes = block_width_split * TILE_WIDTH * in_elem_bytes
    num_col_blocks_max = _div_up(core_col_tiles_max, block_width_split)
    split_reader = (
        not any_resident
        and not retile
        and not padded
        and split_segment_bytes <= SPLIT_READER_MAX_SEGMENT_BYTES
        and core_row_tiles_max * num_col_blocks_max >= 2
    )
    num_input_cbs = 2 if split_reader else 1
    block_width = block_width_split if split_reader else _block_width_for(1)
    assert core_col_tiles_max <= block_width or not any_resident

    # Streaming window along tile_row (the streamed side only): fill what the budget leaves
    # after block_width, but keep at least DEPTH_IN quanta per core so the double buffer still
    # overlaps (one quantum per core serializes read -> tilize -> write). The split reader
    # alternates CBs per tile-row, so it keeps one tile-row per quantum.
    per_row_bytes = block_width * _per_col_tile_bytes(
        num_input_cbs, input_resident=input_resident, output_resident=output_resident
    )
    # bank_coalesced (BANK_COALESCE_STAGE_DEPTH): every core reads whole sticks of a DRAM-interleaved
    # input in one column block, and the StickProducer-only levers are at their defaults. Its
    # staging ring holds BANK_COALESCE_STAGE_DEPTH quanta of page-strided sticks.
    in_mc = input_tensor.memory_config()
    max_positions = core_row_tiles_max * _div_up(core_col_tiles_max, block_width)  # busiest core's walk length
    # Co-read (CO_READ_SHARE): every core's walk is one position, streamed by the plain stick reader,
    # with a stick segment inside its input BufferType's CO_READ_SEGMENT_BYTES window.
    co_read = min(tile_h - 1, int(tile_h * CO_READ_SHARE))
    co_read_min, co_read_max = CO_READ_SEGMENT_BYTES.get(in_mc.buffer_type, (0, -1))
    segment_bytes = min(block_width, core_col_tiles_max) * TILE_WIDTH * in_elem_bytes
    if not (
        co_read > 0
        and co_read_min <= segment_bytes
        and (co_read_max is None or segment_bytes <= co_read_max)
        and max_positions == 1
        and not (input_resident or retile or split_reader or padded)
        and READ_NOC_SPLIT == 0
        and BANK_STRIDE == 0
    ):
        co_read = 0
    coalesce_row_bytes = BANK_COALESCE_STAGE_DEPTH * tile_h * stick_page_bytes  # staging per tile-row
    coalesce = (
        BANK_COALESCE_STAGE_DEPTH > 0
        and co_read == 0  # a one-position walk has nothing to overlap the coalesced scatter with
        and not (input_resident or retile or split_reader or padded)
        and pages_per_stick == 1
        and in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and in_mc.buffer_type == ttnn.BufferType.DRAM
        and block_width == C
        and block_width * TILE_WIDTH * in_elem_bytes <= BANK_COALESCE_MAX_STICK_BYTES
        and all(col_start == 0 and cols == C for _, _, _, col_start, cols in assignment)
        and READ_AHEAD == 1
        and READ_WINDOW_MIN_TILES == 0
        and READ_NOC_SPLIT == 0
        and BANK_STRIDE == 0
        and not EAGER_PUBLISH
        and per_row_bytes + coalesce_row_bytes <= CB_BUDGET_BYTES[low_l1]
    )
    if coalesce:
        per_row_bytes += coalesce_row_bytes
    # QUANTUM_MIN_TILES counts full 32-row tiles: a tiny tile carries tile_h / 32 of one
    # tile's bytes and per-quantum costs are per handshake, so the floor scales with 32 / tile_h
    # (at tile_h = 1 one "tile" is a single stick segment).
    quantum_floor = BANK_COALESCE_QUANTUM_ROWS * block_width if coalesce else QUANTUM_MIN_TILES
    quantum_min_tiles = quantum_floor * (FULL_TILE_HEIGHT // tile_h)
    # The stick reader (StickProducer) streams cb_input_sticks with NoC reads; a resident input only
    # publishes pages, retile face-walks, and the split reader has its own schedule.
    stick_reader = not (input_resident or retile or split_reader)
    assert stick_reader or not padded, "the pad fill lives in the stick reader"
    # One NoC transaction id counts at most NOC_MAX_TRANSACTION_ID_COUNT outstanding reads, and a
    # stick slot's reads share one id: cap the slot's read count (tile_h per tile-row, times the
    # pages a stick segment can straddle on a paged input).
    reads_per_row = tile_h * (
        1 if pages_per_stick == 1 else _div_up(block_width * TILE_WIDTH * in_elem_bytes, in_page_bytes) + 1
    )
    max_rows_per_trid = NOC_MAX_TRANSACTION_ID_COUNT // reads_per_row if stick_reader else max_positions
    if split_reader:
        rows_per_quantum = 1
    else:
        rows_per_quantum = max(
            1,
            min(
                _div_up(quantum_min_tiles, block_width),
                max_positions // DEPTH_IN,
                CB_BUDGET_BYTES[low_l1] // max(1, per_row_bytes),
                max_rows_per_trid,
            ),
        )
    quantum_tiles = rows_per_quantum * block_width  # pages per CB push / pop on a streamed CB

    # In-flight windows (op_design.md perf lamp "Read in-flight depth"), in CB quanta: enough quanta
    # to keep READ_WINDOW_MIN_TILES / WRITE_WINDOW_MIN_TILES full-tile equivalents in flight, never
    # more than the walk has, and each CB deep enough to hold its window. A window of one quantum
    # (every wide tile-row, every resident side) leaves the depth and handshake schedule as before.
    def _window_quanta(min_tiles):
        full_tiles_per_quantum = max(1, quantum_tiles // (FULL_TILE_HEIGHT // tile_h))
        return max(1, min(MAX_WINDOW_QUANTA, max_positions, _div_up(min_tiles, full_tiles_per_quantum)))

    read_ahead = max(READ_AHEAD, _window_quanta(READ_WINDOW_MIN_TILES)) if stick_reader else READ_AHEAD
    write_ahead = _window_quanta(WRITE_WINDOW_MIN_TILES) if not (output_resident or split_reader) else 1
    depth_in = max(DEPTH_IN, read_ahead)
    depth_out = max(DEPTH_OUT, write_ahead)
    # The windows only deepen CBs whose quantum is small; keep the streamed total inside the budget.
    while quantum_tiles * _per_col_tile_bytes(
        num_input_cbs,
        input_resident=input_resident,
        output_resident=output_resident,
        depth_in=depth_in,
        depth_out=depth_out,
    ) + (rows_per_quantum * coalesce_row_bytes if coalesce else 0) > CB_BUDGET_BYTES[low_l1] and (
        read_ahead > READ_AHEAD or write_ahead > 1
    ):
        if read_ahead >= write_ahead and read_ahead > READ_AHEAD:
            read_ahead -= 1
        else:
            write_ahead -= 1
        depth_in, depth_out = max(DEPTH_IN, read_ahead), max(DEPTH_OUT, write_ahead)

    # Parked NoC levers (see their knobs). A NoC split moves the side's kernel to DM_DYNAMIC_NOC;
    # the write-ahead window's per-quantum trids live on the writer's own NoC only, so a write
    # split keeps one write quantum in flight. Bank-stride addressing needs an interleaved DRAM
    # input with one page per stick, read on the reader's own NoC.
    read_noc_split = READ_NOC_SPLIT if stick_reader and pages_per_stick == 1 and not padded else 0
    write_noc_split = WRITE_NOC_SPLIT if not (output_resident or split_reader) else 0
    if write_noc_split != 0:
        write_ahead = 1
    dynamic_noc = read_noc_split != 0 or write_noc_split != 0
    out_mc = output_tensor.memory_config()
    hop_write_t = (
        HOP_WRITE_MIN_SAVING
        if ttnn.device.is_wormhole_b0(device)  # the kernel's NoC-grid / DRAM-coordinate model is Wormhole's
        and in_mc.buffer_type != ttnn.BufferType.DRAM
        and out_mc.buffer_type == ttnn.BufferType.DRAM
        and out_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and not (output_resident or split_reader or dynamic_noc or BANK_COALESCE_SCATTER_WRITE)
        and write_ahead == 1
        and R * C * tile_h // FULL_TILE_HEIGHT >= HOP_WRITE_MIN_TILES
        else 0
    )
    bank_stride = (
        BANK_STRIDE
        if stick_reader
        and not padded
        and read_noc_split == 0
        and pages_per_stick == 1
        and in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and in_mc.buffer_type == ttnn.BufferType.DRAM
        else 0
    )

    # ---------------- circular buffers ----------------
    tile_desc = ttnn.TileDescriptor(tile_h, TILE_WIDTH)
    input_format = ttnn.CBFormatDescriptor(
        buffer_index=CB_INPUT_STICKS,
        data_format=input_tensor.dtype,
        page_size=in_tile_bytes,
        tile=tile_desc,
    )
    if input_resident and not retile:
        # Zero-copy: cb_input_sticks IS the input shard. A Layout::ROW_MAJOR shard row has
        # stride shard_w * elem_bytes = block_width * tile_col_bytes, exactly the tilize input
        # layout, so tile_h sticks of the shard are block_width tile-sized pages.
        shard_h = input_tensor.memory_config().shard_spec.shape[0]
        cb_input_sticks = ttnn.cb_descriptor_from_sharded_tensor(
            CB_INPUT_STICKS,
            input_tensor,
            total_size=(shard_h // tile_h) * block_width * in_tile_bytes,
            core_ranges=all_cores,
        )
        cb_input_sticks.format_descriptors = [input_format]
    else:
        cb_input_sticks = ttnn.CBDescriptor(
            total_size=depth_in * quantum_tiles * in_tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[input_format],
        )
    cbs = [cb_input_sticks]
    if retile:
        staging_format = ttnn.CBFormatDescriptor(
            buffer_index=CB_RETILE_STAGING,
            data_format=input_tensor.dtype,
            page_size=in_page_bytes,
            tile=ttnn.TileDescriptor(in_tile_h, TILE_WIDTH),
        )
        if input_resident:
            # Zero-copy: the staging CB IS the input shard; the face walk reads its tiles in place.
            cb_retile_staging = ttnn.cb_descriptor_from_sharded_tensor(
                CB_RETILE_STAGING, input_tensor, core_ranges=all_cores
            )
            cb_retile_staging.format_descriptors = [staging_format]
        else:
            cb_retile_staging = ttnn.CBDescriptor(
                total_size=RETILE_STAGE_DEPTH * unit_in_rows * block_width * in_page_bytes,
                core_ranges=all_cores,
                format_descriptors=[staging_format],
            )
    if coalesce:
        assert read_ahead == 1 and not retile
        cb_coalesce_staging = ttnn.CBDescriptor(
            total_size=BANK_COALESCE_STAGE_DEPTH * rows_per_quantum * tile_h * stick_page_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_COALESCE_STAGING, data_format=input_tensor.dtype, page_size=stick_page_bytes
                )
            ],
        )
    if padded:
        cb_pad_source = ttnn.CBDescriptor(
            total_size=PAD_SOURCE_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_PAD_SOURCE, data_format=input_tensor.dtype, page_size=PAD_SOURCE_BYTES
                )
            ],
        )
    if split_reader:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=depth_in * quantum_tiles * in_tile_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_INPUT_STICKS_ODD,
                        data_format=input_tensor.dtype,
                        page_size=in_tile_bytes,
                        tile=tile_desc,
                    )
                ],
            )
        )
    assert len(cbs) == num_input_cbs
    if retile:
        cbs.append(cb_retile_staging)  # reader-private: not a compute input
    if padded:
        cbs.append(cb_pad_source)  # reader-private: not a compute input
    if coalesce:
        cbs.append(cb_coalesce_staging)  # reader-private: not a compute input
    if output_resident:
        # Zero-copy: compute packs straight into the output shard (TILE pages, shard order).
        cb_output_tiles = ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_TILES, output_tensor, core_ranges=all_cores)
    else:
        cb_output_tiles = ttnn.CBDescriptor(
            total_size=depth_out * quantum_tiles * out_tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=out_tile_bytes,
                    tile=tile_desc,
                )
            ],
        )

    # ---------------- kernel args ----------------
    # CT args: config only (dtype pair, tile, block_width, residency, accessor args) ->
    # program-cache friendly; buffer addresses ride on RT args and on the resident CBs.
    tile_col_bytes = TILE_WIDTH * in_elem_bytes
    assert 1 <= read_ahead <= depth_in and 1 <= write_ahead <= depth_out <= MAX_WINDOW_QUANTA
    reader_ct_args = [
        CB_INPUT_STICKS,
        block_width,
        tile_h,
        tile_col_bytes,
        stick_page_bytes,
        int(split_reader),
        depth_in,
        read_ahead,
        rows_per_quantum,
        int(input_resident),
        in_page_bytes,
        pages_per_stick,
        in_tile_h if retile else 0,
        row_align,
        CB_RETILE_STAGING,
        RETILE_STAGE_DEPTH,
        int(RETILE_FACEWALK_NOC),
        read_noc_split,
        int(EAGER_PUBLISH and stick_reader and not padded),
        bank_stride,
        int(padded),
        CB_PAD_SOURCE,
        PAD_SOURCE_BYTES,
        PAD_NOC_MIN_BYTES,
        in_elem_bytes,
        int(PAD_W_TAIL_PERSIST),
        BANK_COALESCE_STAGE_DEPTH if coalesce else 0,
        int(BANK_COALESCE_SCATTER_WRITE),
        co_read,
        CO_READ_SEM,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend([hop_write_t, HOP_SEM])
    writer_ct_args = [
        CB_OUTPUT_TILES,
        block_width,
        out_tile_bytes,
        int(split_reader),
        CB_INPUT_STICKS_ODD,
        tile_h,
        tile_col_bytes,
        stick_page_bytes,
        depth_in,
        read_ahead,
        rows_per_quantum,
        int(output_resident),
        in_page_bytes,
        pages_per_stick,
        write_noc_split,
        depth_out,
        write_ahead,
        CB_INPUT_STICKS,
        co_read,
        CO_READ_SEM,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    writer_ct_args.extend([hop_write_t, HOP_SEM])
    numeric = NumericConfig(input_tensor.dtype, output_tensor.dtype, compute_kernel_config)
    compute_ct_args = [
        CB_INPUT_STICKS,
        CB_OUTPUT_TILES,
        block_width,
        int(split_reader),
        CB_INPUT_STICKS_ODD,
        int(numeric.lossless),
    ]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    pad_map_args = pad.reader_rt_args(tile_h, in_elem_bytes) if padded else []

    for core_idx, (core, row_start, core_row_tiles, col_start, core_col_tiles) in enumerate(assignment):
        # Per-core traversal rotation (single source for reader AND writer): spreads the
        # cores' concurrent stick reads / tile writes over the DRAM banks. A resident side
        # fixes the tile-row order to shard order, so only the in-tile-row stick order rotates.
        # Retile rotates in whole units (row_align tile-rows) so a unit never wraps.
        row_rotation = 0 if any_resident else core_idx * row_align
        stick_rotation = core_idx
        reader_rt_args[core.x][core.y] = [
            in_addr,
            row_start,
            core_row_tiles,
            col_start,
            core_col_tiles,
            row_rotation,
            stick_rotation,
            C,
            out_rows_per_image,
            in_rows_per_image,
        ]
        if padded:
            reader_rt_args[core.x][core.y].extend([pad.fill_bits, *pad_map_args])
        writer_rt_args[core.x][core.y] = [
            out_addr,
            row_start,
            core_row_tiles,
            col_start,
            core_col_tiles,
            C,
            row_rotation,
            in_addr,
            stick_rotation,
        ]
        compute_rt_args[core.x][core.y] = [core_row_tiles, core_col_tiles]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        defines=_kernel_defines(),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=(
            ttnn.DataMovementConfigDescriptor(
                ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0, ttnn.NOC_MODE.DM_DYNAMIC_NOC
            )
            if dynamic_noc
            else ttnn.ReaderConfigDescriptor()
        ),  # NCRISC / NoC0
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        defines=_kernel_defines(),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=(
            ttnn.DataMovementConfigDescriptor(
                ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1, ttnn.NOC_MODE.DM_DYNAMIC_NOC
            )
            if dynamic_noc
            else ttnn.WriterConfigDescriptor()
        ),  # BRISC / NoC1
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        defines=_kernel_defines(),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        # 16-bit DEST for 16-bit pages (the fast-tilize path); fp32 DEST + UnpackToDestFp32 on the
        # compute input CBs wherever a 32-bit page exists (NumericConfig). Half-sync DEST is a
        # fast-tilize requirement.
        config=numeric.compute_config([CB_INPUT_STICKS] + ([CB_INPUT_STICKS_ODD] if split_reader else [])),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=(
            ([ttnn.SemaphoreDescriptor(id=CO_READ_SEM, core_ranges=all_cores, initial_value=0)] if co_read else [])
            + ([ttnn.SemaphoreDescriptor(id=HOP_SEM, core_ranges=all_cores, initial_value=0)] if hop_write_t else [])
        ),
        cbs=cbs + [cb_output_tiles],
    )
