# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""genify_matmul program descriptor — regime R1 `interleaved_out_split` (op_design.md, Blocking Model).

Host side of the blocking scheme:
  * choose_split()    — the core-assignment knob: logical (grid_b, grid_m, grid_n) factorization.
                        Every candidate is blocked (choose_blocking) and ranked by its estimated time
                        (split_cost: max of the padded K-walk, the DRAM stream and the operand
                        injectors; Refinement 11), then aggregate read traffic, W, cores.
  * choose_blocking() — the extent knobs (block_batches, block_m_tiles, block_n_tiles, block_k_tiles /
                        num_k_blocks / last_block_k_tiles), solved coarsest-first against the L1 budget
                        (steps S1..S4), plus the derived output subblock (sb_h, sb_w).
Every CB page count, kernel loop trip count and grid size below derives from those knobs; each knob is
defined once and handed to the kernels as a CT / RT arg.

Regime R1a `interleaved_2d_mcast` (Refinement 3) sub-selects from R1 on a 2-D rectangle split
(`uses_operand_mcast`): the same region / blocking / CBs / compute kernel, but A crosses DRAM once per
grid line instead of once per core -- the line's first core reads each A K-block and multicasts it
along the line of cores that share its rows, and likewise for B down the line sharing its columns.

Regime R4 `dram_sharded` (Refinement 4, `create_dram_sharded_program_descriptor`): B is DRAM width-
sharded one column shard per bank, A is L1 width-sharded over K. Each bank's columns are split over
`cores_per_bank` workers (Refinement 6; the bank's optimal worker plus its nearest free cores), each
streaming its column slice of the bank shard, and every A shard crosses the NoC once: its owning core
multicasts it, in its K turn (rotating senders), to the workers -- from the writer RISC (BRISC / NoC1,
DS_ACT_ON_WRITER) so the rounds never stall the B stream. With one worker per bank the output is packed
in place into its L1 width shard; split slices stream into the bank's shard through the output accessor.

Regime R3 `block_sharded` (Refinement 5, `create_block_sharded_program_descriptor`): A is L1 block-
sharded (rows over grid rows, K over grid columns), B interleaved, the output L1 block-sharded on A's
grid. Core (column c, row r) owns its shard rows x output shard c and walks K one A shard per K-block:
its own shard zero-copy (cb_act_shard), its row peers' shards over a rotating-sender row multicast
(BS_ROW_MCAST; unicast peer reads otherwise), B's column slice from DRAM. The output is packed in place
into the core's own output shard.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# ---- CB indices (op_design.md, Circular Buffers) ----
CB_ACT_TILES = 0  # A K-blocks, reader -> compute
CB_WEIGHT_TILES = 1  # B K-blocks, reader -> compute
CB_BIAS_ROW = 2  # bias N-block row (bias / bias_relu), reader -> compute, one per block
CB_GATE_ROW = 3  # gate N-block row (addcmul), reader -> compute, one per block
CB_RESIDUAL_TILES = 4  # residual row groups (addcmul), reader -> compute
CB_ACT_SHARD = 5  # R3: A's own L1 shard (zero-copy), reader (push only) -> compute (own K-block)
CB_OUTPUT_TILES = 16  # output row groups, compute -> writer
CB_OUTPUT_SHARD = 17  # R4: the output's own L1 shard (zero-copy), compute -> writer (wait/pop only)
CB_PARTIALS = 24  # K-accumulator (packer L1 acc), compute -> compute; num_k_blocks > 1 or epilogue != none

# ---- Epilogue kinds (compile-time arg shared by reader and compute) ----
EPILOGUES = {"none": 0, "bias": 1, "bias_relu": 2, "swiglu": 3, "addcmul": 4}

# ---- Buffer-depth knobs ----
DEPTH_ACT = 2  # A K-blocks in flight
DEPTH_WEIGHT = 2  # B K-blocks in flight
DEPTH_OUT = 2  # output row groups in flight
DEPTH_RESIDUAL = 2  # residual row groups in flight (addcmul)
# Epilogue chain DEST lanes per acquire (eltwise_chain block_size; the chain clamps it to DEST).
EPILOGUE_BLOCK_TILES = 4
# swiglu silu precision knob: the fp32-precision sigmoid (exp polynomial + Newton reciprocal per row)
# made the epilogue SFPU-bound. With a non-fp32 output the bf16-precision sigmoid is enough (the output
# rounding dominates); a float32 output always keeps the fp32 sigmoid.
SWIGLU_FAST_SILU = True

# ---- Blocking knobs ----
BLOCK_K_FLOOR = 8  # smallest K-block the solve accepts (clamped to tensor_k_tiles)
# Precision knob, 16-bit DEST only: the longest K chain accumulated inside DEST before the packer
# folds it into cb_partials (packer L1 accumulation). A bf16 DEST running sum swamps small addends,
# so the error grows with the in-DEST chain: K = 16384 as one 171-tile K-block measured rel-RMS 0.185
# (stock ttnn.matmul 0.027); capped at 8 tiles 0.019 (32 tiles: 0.046). fp32 DEST is uncapped.
DEST16_BLOCK_K_MAX = 8
# A per-core walk that would be ONE K-block (S1, whole region resident) is split into this many K-blocks
# when K allows (>= BLOCK_K_FLOOR each): otherwise compute waits for the whole read and the second slot of
# the depth-2 operand CBs is dead L1. 2 measured 0.79x on 384x1536x1024, 0.92x on 32x1024x2048, neutral
# elsewhere. 1 = unsplit.
SINGLE_WALK_K_BLOCKS = 2
# R1a K-walk granularity (Refinement 11): an R1a walk of fewer K-blocks than this is re-split into this many
# (each >= BLOCK_K_FLOOR and still above the R1a gate MCAST_MIN_KBLOCK_BYTES). The coarsest K-block that fits
# L1 left mid-M walks at 2-3 K-blocks, and every R1a K-block is a serial chain (line injector read ->
# barrier -> A line multicast -> B line multicast) that nothing overlaps before compute's first block: text
# encoder o_proj 512x4096x2560 (3 K-blocks of 43) 96.1 -> 85.1 us at 8, Qwen prefill M = 512 rows -4 to
# -12%. 1 = the coarsest K-block (no re-split). Multi-K-block walks only (a one-K-block walk accumulates K
# in DEST; splitting it would change precision, not speed), and only where the Perf 2 head ramp does not
# already shorten the fill (the raw walk: 384x1536x1024 bias measured +5% with both).
MIN_K_BLOCKS = 8
# R1a injector stall-shadow (Refinement 11). Each R1a K-block is a serial chain on the line injector: read its
# line's K-block (barrier), then multicast it; the zones of text-encoder down 512x16384x5120 show the readers
# never waiting on CB space (DM-bound, 545 us against a ~425 us K-walk). 2: the injector issues K-block kb+1's
# reads (own transaction id) before it multicasts kb, so read and multicast overlap; needs operand rings of
# R1A_PREFETCH_DEPTH K-blocks (compute holds kb-1 while kb goes out and kb+1 lands; the solve shrinks block_k
# to fit). Plain R1a helper walks only (the raw walk's head ramp schedules its own deliveries; residency walks
# are one K-block). 1 = read, barrier, multicast (no overlap). Measured: FLUX proj_out 2048x3072x128 (A line
# injectors ~4x the K-walk) 66.6 -> 58.8 us; where the K-walk is about as long as the injector stream it
# loses (text-encoder o_proj 85.4 -> 89.3 us, Qwen M = 512 rows +4..8%, text-encoder down flat), so it is
# gated to injector-bound walks: split_cost's busiest-core read term >= R1A_PREFETCH_MIN_RATIO x the K-walk.
R1A_INJECTOR_AHEAD = 2
R1A_PREFETCH_DEPTH = 3
R1A_PREFETCH_MIN_RATIO = 2.0
# Epilogue-none K-walk (Refinement 11): True runs an R1 / R1a / R1b epilogue-none multi-K-block walk with 1 x sb_w
# subblocks on the raw K-walk (kw::matmul_accumulate_interm, Perf 1: no per-tile DEST-offset write) into
# cb_partials, then an identity copy to the output, instead of matmul_block<Out>'s last-K-block partials reload.
# False: the helper walk. Measured: FLUX single QKV / out 1176 -> 1121 / 1492 -> 1428 us, Qwen prefill qkvg
# M = 4096 390 -> 374 us; on walks whose data movement binds, the copy tail (serialized after the walk, where the
# helper's reload overlapped its last K-block) loses 2-4% (text-encoder qkv / o_proj, Qwen M = 512 in_proj), so
# it is gated to compute-bound splits: split_cost's K-walk >= NONE_RAW_WALK_MIN_RATIO x every other term.
NONE_RAW_WALK = True
NONE_RAW_WALK_MIN_RATIO = 1.25
# Operand residency knob (Refinement 8). A per-core walk that is ONE block with ONE K-block that K cannot
# split (S1, K < SINGLE_WALK_K_BLOCKS * BLOCK_K_FLOOR: short-K shapes, whose output is many times their
# inputs, so the output write stream binds) reads its whole A and B before compute starts, and the
# output stream only starts after that. Split the region into this many blocks along ONE axis instead:
# the smaller per-core operand (bytes) is read once and held fronted across the blocks (matmul_block
# WaitAndRetainOnLastBlock on it, the caller pops it after its last block), the other one streams one
# block at a time, so compute and the first writes start after the resident operand + one block.
# A resident -> N-blocks (`act_resident`), B resident -> M-blocks (`weight_resident`). PERF FOCUS 3
# (32 x 512x128x1024, bf8b): 4 measured best (8-column B blocks: 2 / 4 / 8 blocks 74.9 / 65.0 /
# 68.5 us on the 32x2x1 split). 1 = off (the whole region as one block).
RESIDENT_SPLIT_BLOCKS = 4
# Output write batching knob (Refinement 8, R1): row groups the writer drains per wait / flush / pop
# (the largest divisor of the block's row groups <= this; cb_output_tiles holds DEPTH_OUT batches).
# Measured flat to +1.6% on PERF FOCUS 3 at 4 / 8 (the writer never stalled on its per-row-group flush:
# the DRAM write stream itself binds), so it is parked at 1 = one row group (the Refinement 7 shape).
OUT_WRITE_BATCH_ROW_GROUPS = 1
# Tail write split knob (Perf, tail_writes_both_nocs; R1 / R1a / R1b): the core's LAST block's output rows
# are written TAIL_NC_PCT % by the reader RISC on NoC0 (idle once its K-walk is done), the rest by the
# writer on NoC1. NoC0-only writes measured 1.11x slower than NoC1-only on PERF FOCUS 1's 11x10 grid (the
# DRAM-column links concentrate them), so NoC0 takes the smaller share: 36 % (5 of 14 tiles) measured best
# (PERF FOCUS 1, epilogue stubbed: write tail 60 -> 43 us, ~414 GB/s; 25 / 30 / 43 / 50 % all worse).
# It is the R1 descriptor's only write path (R1 / R1a / R1b; R4 in-place output has nothing to write).
TAIL_NC_PCT = 36
# Subblock tie-break knob (same DEST area): True prefers the square-most subblock (fewest operand
# unpacks per K step). Measured on PERF FOCUS 1 (LoFi, block 6x28): 2x2 477.8 us vs 1x4 463.7 us, so
# the wide subblock stays the default.
SUBBLOCK_MIN_UNPACK = False
Q_N_MAX = 4  # subblock quantum on core_n_tiles (a 4-tile subblock always exists)
# Core-assignment knob: a factorization whose per-core output tiles W are within (1 + GRID_W_SLACK)
# of the best W competes on aggregate DRAM read traffic (R summed over the used cores). R1 has no
# operand reuse across cores, so it is DRAM-bound on every measured shape (HiFi2 ~= HiFi4); ranking
# W strictly first picked e.g. a 1x108 N-only split for M=9472, K=5120, N=3456 (W 296 vs 324) that
# made every core read all of A (10.5 GB instead of 1.3 GB, 4.5x slower).
GRID_W_SLACK = 0.25
# Split-ranking cost model (Refinement 11). The W-slack + traffic ranking above never let a split trade
# per-core work for traffic or injector load: FLUX proj_out 2048x3072x128 kept a 64x1 R1 split that re-read
# all of B on every core (161 us; 11x4 R1a 64 us), text-encoder qkv 512x5120x3072 an R1b split capped by
# its single A injector (233 us; 8x8 R1a 136 us), and the FLUX single blocks 9x9 / 9x10 splits over
# 11x10 / 11x9 ones that do 12-18% less work per core. True: rank every candidate by split_cost (cycles):
#   max(padded K-walk, (operand reads + output writes) / DRAM rate, busiest reading core / one core's rate)
# with the K-walk = the core's padded block work (nb_m*block_m x nb_n*block_n, every core walks it) x K x
# the tile-K cost (FPU cycles per fidelity phase, or the subblock's operand unpacks when they bind).
# False: the legacy ranking (W within GRID_W_SLACK, then traffic).
SPLIT_COST_MODEL = True
FPU_CYCLES_PER_TILE_K = 17.5  # per fidelity phase (LoFi 1, HiFi2 2, HiFi3 3, HiFi4 4); measured 16.6-18.0 LoFi
UNPACK_CYCLES_PER_TILE = 12  # one operand tile unpack per K step: a 1 x w subblock costs (1 + w) / w of it
DRAM_BYTES_PER_CYCLE = 296  # ~400 GB/s interleaved tile reads / writes at 1.35 GHz
# One core's operand stream: an R1 core reading its own A + B (proj_out 13x1: 1.77 MB per core in 78 us,
# ~23 GB/s), an R1a line injector reading its line's K-block then multicasting it (serial; proj_out 11x4:
# 1.18 MB per A line in 63.9 us, 8x4: 1.57 MB in 81 us, ~19 GB/s). R1b's injector has its own price
# (ACT_STREAM_INJECTOR_COST, inside read_traffic).
CORE_READ_BYTES_PER_CYCLE = 15
# One core's output stream (the writer's tile writes + per-row-group flush): PERF FOCUS 3 (17.8 MB of 1088 B
# tiles) wrote 557 KB per core in 69.8 us on a 32-core split (~8 GB/s) against 66.6 us on 96 cores.
CORE_WRITE_BYTES_PER_CYCLE = 6
L1_RESERVE_BYTES = 0  # L1 held back for foreign tensors
# Split-ranking cost of an operand byte read from L1 (interleaved L1 tensor) relative to one read from
# DRAM (Refinement 8). DRAM bandwidth is what R1 runs out of -- including on write-bound shapes, where
# DRAM operand reads compete with the output stream -- while L1-sourced reads only load the NoC.
# PERF FOCUS 3 (B in L1): ranked at 1.0 the split re-read A (DRAM) 3x to save L1 B reads (32x1x3,
# 6.6 MB of DRAM reads); discounted it reads A once (32x3x1, B re-read from L1). 1.0 = bytes are bytes.
L1_READ_COST = 0.25
# Regime knob: sub-select R1a `interleaved_2d_mcast` (row / column operand multicast) whenever the split
# is a 2-D physical rectangle. False keeps every core reading its own operands from DRAM (plain R1).
OPERAND_MCAST = True
# R1a gate: the larger operand K-block an injector multicasts per round must reach this many bytes.
# Each round serializes injector read -> handshake -> multicast, so tiny K-blocks are latency-bound and
# a core reading its own operands (R1) wins: 64x16384x256 16-bit DEST (64 K-blocks of 16 KB / 16 KB,
# 16 cores) measured 119.8 us under R1a vs 107.7 us under R1. Every R1a winner of the guard set
# multicasts >= 51 KB on at least one operand.
MCAST_MIN_KBLOCK_BYTES = 32 * 1024
# R1a injector placement knob. False: every line injects from its first core, so core (0, 0) reads and
# multicasts BOTH operands each K-block (the serial critical path of the data movement). True: the
# injectors are staggered along the lines (in-line index = offset + line index, wrapped) with offsets
# chosen so no core injects both operands, and the DRAM readers spread over the grid.
STAGGER_INJECTORS = True
# Regime knob (Refinement 9): sub-select R1b `interleaved_stream_mcast_a` on a 1-D N split (grid_m == 1,
# every core owns all rows, so A is reuse-shared by every core): every core streams its own B columns
# from DRAM while ONE injector reads each A K-block and broadcasts it to all cores, on the writer RISC
# (BRISC / NoC1) so the A handshake never stalls the B stream. Plain R1 re-reads A on every core
# (FLUX timestep linear_2: 96 x 384 KB). False keeps plain R1 on 1-D N splits.
ACT_STREAM_MCAST = True
# R1b gate: the A K-block the injector broadcasts per round must reach this many bytes (each round is
# one handshake with every core of the split; below it a core reading its own A is cheaper).
ACT_STREAM_MIN_KBLOCK_BYTES = 32 * 1024
# R1b operand CB depth (K-blocks in flight, cb_act_tiles and cb_weight_tiles) of a K-walk (S2 / S3; the
# solve shrinks block_k to fit). Every core waits for the A round of each K-block, so the slowest core's
# B read of a K-block gates the next broadcast; deeper rings give that coupling slack and let the B stream
# run ahead of compute. PERF FOCUS 4: depth 2 / 3 / 4 = 683 / 666 / 633 us (B-stream floor 578 us).
ACT_STREAM_DEPTH = 4
# R1b injector stall-shadow: A K-blocks the injector keeps in flight (read trids), so the DRAM read of
# K-block kb+1 overlaps the multicast of kb (clamped to the act ring depth). 1 = read, barrier, send.
ACT_STREAM_AHEAD = 2
# R1b injector cost in the split ranking: one core reads every A byte and multicasts it (A read + multicast
# measured ~23 GB/s on the writer RISC), while the interleaved B stream of the whole split runs at ~411
# GB/s, so an A byte through the injector is worth this many streamed bytes. R1b ranks at
# max(A + B, A * cost): text-encoder down 512x16384x5120 (A = 19% of B) measured 732 us under R1b against
# 552 us under R1a and is refused; FLUX context_embedder 256x15360x3072 (8%) keeps R1b (350 us vs 2226).
ACT_STREAM_INJECTOR_COST = 18
# R1b injector: index of the broadcasting core in the split's row-wise core order.
ACT_STREAM_INJECTOR = 0
# R1b NoC: the writer RISC's (BRISC, WriterConfigDescriptor).
ACT_STREAM_NOC = ttnn.NOC.NOC_1
# N quantum on a 1-D N split (Refinement 9). Q_N_MAX makes a 4-wide subblock exist when a core owns one
# tile row; when it owns all rows (grid_m == 1) a subblock of height h (the largest power of two <= 4
# dividing core_m) needs only q_n = Q_N_MAX / h columns. FLUX FF1 256x6144x18432 swiglu: core_n 8 -> 6
# (72 -> 96 cores), subblock 2x2. False: Q_N_MAX regardless of core_m.
Q_N_ROW_AWARE = True
# R4 `dram_sharded` (Refinement 4) uses DEPTH_ACT / DEPTH_WEIGHT / BLOCK_K caps above; its own knobs:
# the NoC the bank workers read B on (the reader runs on NCRISC = NOC_0, the preferred DRAM-read NoC),
# which also selects get_optimal_dram_bank_to_logical_worker_assignment's worker per bank.
DS_READ_NOC = ttnn.NOC.NOC_0
# R4 compute spread (Refinement 6, the `cores_per_bank` split). With one core per bank a decode-shaped
# walk (M = 1) is compute-bound: PERF FOCUS 2 measured 38.7 us of matmul alone against a 28-29 us B
# stream. Each bank's column shard is therefore split over C cores (the bank's optimal worker plus its
# nearest free cores), each streaming ITS column sub-range of every K row and computing those columns.
# DS_CORE_N_TILES_MIN: the narrowest per-core column range the split may cut to (one fp32-DEST subblock
# wide); the split takes the smallest divisor of the bank width >= it whose core count fits the grid.
# DS_CORES_PER_BANK: None = that automatic choice; an int forces C (it must divide the bank width, in
# whole swiglu pairs); 1 = one core per bank, the Refinement 4 topology (output packed in place).
DS_CORE_N_TILES_MIN = 4
DS_CORES_PER_BANK = None
# R4 B stream depth (Refinement 6, `trid_double_issue`). DS_B_AHEAD: B K-blocks kept in flight, each
# retired by its own transaction-id barrier so the bank pipe does not drain once per K-block (1 = the
# Refinement 4 shape: one K-block in flight, issued before its A round). DS_DEPTH_WEIGHT: cb_weight_tiles
# depth in K-blocks (>= DS_B_AHEAD + 1 so compute holds one block while the ring is full); the solve
# falls back to DEPTH_WEIGHT (and the ring to its depth) when that does not fit L1.
DS_B_AHEAD = 1
DS_DEPTH_WEIGHT = 2
# R4 A broadcast placement (Refinement 6, `split_reader`). True: the A rounds run on the writer RISC
# (BRISC / NoC1), idle until the block's output, so they no longer sit between the B stream's issues and
# barriers on NCRISC / NoC0. False: the Refinement 4 shape, rounds interleaved with the B stream.
DS_ACT_ON_WRITER = True
# The writer RISC's NoC (BRISC, WriterConfigDescriptor): the A broadcast's NoC when DS_ACT_ON_WRITER.
DS_ACT_WRITER_NOC = ttnn.NOC.NOC_1
# R3 `block_sharded` (Refinement 5) peer K-slice exchange. True (R3a): a rotating-sender row multicast,
# every A shard crosses the NoC once per grid row. False (R3 stepping stone): every core unicast-reads
# its row peers' shards through A's sharded TensorAccessor (each shard is read cols - 1 times).
BS_ROW_MCAST = True
# R2a / R3a weight multicast (Refinement 10). B does not vary along the shard rows, so every core of a
# B-sharing group (R3: a grid column, which owns one output column slice; R2: all of A's shard cores) needs
# the same B K-blocks. True: one fixed injector per group reads each K-block from DRAM once and multicasts
# it to the group (B crosses DRAM once instead of once per grid row / per core). False: the R2 / R3 stepping
# stone, every core reads its own B. Single (non-batched) B and groups of > 1 core only; an R2 core set the
# multicast cannot cover (> 3 rectangles) keeps the per-core reads.
SHARDED_B_MCAST = True
# The injector's in-group index; staggered by group so R3's column injectors sit on different grid rows.
SHARDED_B_INJECTOR = 0
# A B round's multicast goes out in K-row chunks of >= this many bytes (at most SHARDED_B_MAX_CHUNKS,
# each read under its own transaction id), so chunk c is on the wire while chunk c + 1 is still landing.
# Parked at one chunk: each chunk is a full handshake + write-ack round (~1.2 us on PDL's 86-core HS group),
# measured 6.9 / 10.3 / 15.1 us for 1 / 4 / 8 chunks on res3 reduce (8192x512x128).
SHARDED_B_CHUNK_BYTES = 16 * 1024
SHARDED_B_MAX_CHUNKS = 1
# One B round carries g K-blocks (g | num_k_blocks, the smallest with >= this many bytes, cb_weight_tiles
# deepened to 2g K-blocks when L1 allows): a round costs a handshake on the reader's serial chain beside the
# A round, so thin K-blocks (R3's 1-2 output tiles per grid column) amortize it over several K-blocks.
SHARDED_B_ROUND_BYTES = 32 * 1024
# R3 only: the B rounds share the reader's serial chain with the A row exchange, so the B multicast pays only
# when it removes enough DRAM traffic -- the (rows - 1) re-reads of B it saves. Measured on PDL (device ns,
# multicast vs per-core reads): 0.97-2.1 MB saved runs 0-9% slower (res4 reduce / expand, ASPP projection),
# 3.9 MB and up 3-61% faster (res4 shortcut, ASPP 1x1, res5). R2 has no A exchange: always on.
BS_B_MCAST_MIN_SAVED_BYTES = 3 * 1024 * 1024
# own_kb sentinel for every regime without a resident own K-slice (R1 / R1a / R4).
NO_OWN_KB = 0xFFFFFFFF


def _cdiv(a, b):
    return -(-a // b)


def _round_up(a, b):
    return _cdiv(a, b) * b


def _pow2_floor(x):
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def _divisors(x):
    return [d for d in range(1, x + 1) if x % d == 0]


@dataclass(frozen=True)
class Geometry:
    tensor_batches: int
    tensor_m_tiles: int
    tensor_k_tiles: int
    tensor_n_tiles: int
    out_n_tiles: int
    weight_batch_stride: int
    a_tile_bytes: int
    b_tile_bytes: int
    o_tile_bytes: int
    p_tile_bytes: int
    p_dtype: object
    dest_limit: int
    block_k_cap: int  # longest in-DEST K chain (DEST16_BLOCK_K_MAX with 16-bit DEST, else all of K)
    epilogue: str = "none"
    row_tile_bytes: int = 0  # bias (bias / bias_relu) or gate (addcmul) tile bytes, else 0
    r_tile_bytes: int = 0  # residual tile bytes (addcmul), else 0
    a_in_l1: bool = False  # operand A / B is an L1 tensor (split ranking: L1_READ_COST)
    b_in_l1: bool = False
    fidelity_phases: int = 4  # math fidelity passes per tile-K (split_cost); LoFi 1 .. HiFi4 4

    @property
    def swiglu(self):
        return self.epilogue == "swiglu"

    @property
    def q_n_min(self):
        # swiglu: whole (gate, up) tile pairs per core and per block (N % 64 == 0 makes Nt even).
        return 2 if self.swiglu else 1

    def out_block_n(self, block_n_tiles):
        return block_n_tiles // 2 if self.swiglu else block_n_tiles


@dataclass(frozen=True)
class GridChoice:
    grid_b: int  # used logical grid along each axis
    grid_m: int
    grid_n: int
    core_batches: int
    core_m_tiles: int
    core_n_tiles: int
    q_n: int
    placement: str  # "rect" (x=n, y=m), "rect_t" (x=m, y=n) or "flat" (row-wise list)


@dataclass(frozen=True)
class Blocking:
    block_batches: int
    block_m_tiles: int
    block_n_tiles: int
    block_k_tiles: int
    num_k_blocks: int
    last_block_k_tiles: int
    sb_h: int
    sb_w: int
    epilogue: str = "none"
    # B read once per core and retained across its M-blocks (RESIDENT_SPLIT_BLOCKS); needs one K-block,
    # block_batches == 1 and the core's whole column range in one N-block.
    weight_resident: bool = False
    # A read once per core M-block and retained across its N-blocks (the twin: B streams in N-blocks);
    # needs one K-block and block_batches == 1.
    act_resident: bool = False
    # Output row groups the writer drains per wait / flush / pop (cb_output_tiles holds DEPTH_OUT batches).
    write_batch: int = 1
    # Operand CB depths in K-blocks (cb_act_tiles, cb_weight_tiles): DEPTH_ACT / DEPTH_WEIGHT, or
    # ACT_STREAM_DEPTH on an R1b K-walk. The L1 solve and the CB sizes both read them here.
    depth_act: int = DEPTH_ACT
    depth_weight: int = DEPTH_WEIGHT

    @property
    def use_partials(self):
        # The K-accumulator exists for a multi-K-block walk, and always under an epilogue (the
        # matmul packs its last K-block there for the epilogue to read: LastBlockTarget::Interm).
        return self.num_k_blocks > 1 or self.epilogue != "none"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


_FIDELITY_PHASES = {
    ttnn.MathFidelity.LoFi: 1,
    ttnn.MathFidelity.HiFi2: 2,
    ttnn.MathFidelity.HiFi3: 3,
    ttnn.MathFidelity.HiFi4: 4,
}


def _out_list(out):
    """The output tensor(s): one tensor, or the `chunks` equal-width column chunks of a split output
    (Refinement 13; interleaved placement only). The geometry is the LOGICAL output, the chunks side by side."""
    return list(out) if isinstance(out, (list, tuple)) else [out]


def _a_list(a):
    """The A operand(s): one tensor, or the K segments [A_0, A_1, ...] of A = concat(A_i, dim=-1) (Refinement 14;
    interleaved placement only). The segments share leading dims, M, dtype and memory config, so segment 0 stands
    in for every per-tensor property; the geometry is the LOGICAL A, the segments side by side along K."""
    return list(a) if isinstance(a, (list, tuple)) else [a]


def _a_segment_table(a_segs):
    """The reader / R1b-writer RT segment table (kernels/a_segments.hpp): [kt_0, .., kt_{s-1}, addr_1, ..,
    addr_{s-1}] for s > 1 segments, empty for one tensor (segment 0's address is the existing A-address arg)."""
    if len(a_segs) == 1:
        return []
    return [_cdiv(list(t.shape)[-1], TILE) for t in a_segs] + [t.buffer_address() for t in a_segs[1:]]


def _geometry(a, b, out, fp32_dest_acc_en, epilogue="none", row=None, residual=None, math_fidelity=None):
    outs = _out_list(out)
    out = outs[0]
    a_segs = _a_list(a)
    a = a_segs[0]
    a_shape = list(a.shape)[:-1] + [sum(list(t.shape)[-1] for t in a_segs)]
    b_shape = list(b.shape)
    lead = a_shape[:-2]
    img_m_tiles = _cdiv(a_shape[-2], TILE)
    tensor_k_tiles = _cdiv(a_shape[-1], TILE)
    tensor_n_tiles = _cdiv(b_shape[-1], TILE)
    out_n_tiles = sum(_cdiv(list(o.shape)[-1], TILE) for o in outs)
    lead_prod = 1
    for d in lead:
        lead_prod *= d
    batched = any(d > 1 for d in b_shape[:-2])
    if batched:
        tensor_batches = lead_prod
        tensor_m_tiles = img_m_tiles
        weight_batch_stride = tensor_k_tiles * tensor_n_tiles
    else:
        # Single B: the leading dims fold into M (each image is tile-padded on its own).
        tensor_batches = 1
        tensor_m_tiles = lead_prod * img_m_tiles
        weight_batch_stride = 0
    p_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    return Geometry(
        tensor_batches=tensor_batches,
        tensor_m_tiles=tensor_m_tiles,
        tensor_k_tiles=tensor_k_tiles,
        tensor_n_tiles=tensor_n_tiles,
        out_n_tiles=out_n_tiles,
        weight_batch_stride=weight_batch_stride,
        a_tile_bytes=ttnn.tile_size(a.dtype),
        b_tile_bytes=ttnn.tile_size(b.dtype),
        o_tile_bytes=ttnn.tile_size(out.dtype),
        p_tile_bytes=ttnn.tile_size(p_dtype),
        p_dtype=p_dtype,
        dest_limit=4 if fp32_dest_acc_en else 8,
        block_k_cap=tensor_k_tiles if fp32_dest_acc_en else min(tensor_k_tiles, DEST16_BLOCK_K_MAX),
        epilogue=epilogue,
        row_tile_bytes=ttnn.tile_size(row.dtype) if row is not None else 0,
        r_tile_bytes=ttnn.tile_size(residual.dtype) if residual is not None else 0,
        a_in_l1=a.memory_config().buffer_type == ttnn.BufferType.L1,
        b_in_l1=b.memory_config().buffer_type == ttnn.BufferType.L1,
        fidelity_phases=_FIDELITY_PHASES.get(math_fidelity, 4),
    )


# ---------------------------------------------------------------------------
# Work split (core-assignment knob)
# ---------------------------------------------------------------------------


def _placement_for(gb, gm, gn, gx, gy):
    if gb == 1 and gm <= gy and gn <= gx:
        return "rect"
    if gb == 1 and gm <= gx and gn <= gy:
        return "rect_t"
    if gb > 1 or gm == 1 or gn == 1:
        return "flat"
    return None


def _row_subblock_h(geom: Geometry, gm: int, core_m: int):
    """Q_N_ROW_AWARE: the subblock height a 1-D N split (gm == 1) contributes to the DEST area (the
    largest power of two <= Q_N_MAX dividing core_m), so the N quantum only has to supply the rest."""
    if not Q_N_ROW_AWARE or gm != 1 or geom.tensor_batches != 1:
        return 1
    h = 1
    while h * 2 <= Q_N_MAX and core_m % (h * 2) == 0:
        h *= 2
    return h


def grid_candidates(geom: Geometry, gx: int, gy: int):
    """Every legal (grid_b, grid_m, grid_n) factorization as a GridChoice (used grid, deduplicated)."""
    C = gx * gy
    tb, tm, tn = geom.tensor_batches, geom.tensor_m_tiles, geom.tensor_n_tiles
    seen = {}
    for gb in range(1, min(tb, C) + 1):
        for gm in range(1, min(tm, C // gb) + 1):
            for gn in range(1, min(tn, C // (gb * gm)) + 1):
                core_b = _cdiv(tb, gb)
                core_m = _cdiv(tm, gm)
                n_per = _cdiv(tn, gn)
                q_n = max(geom.q_n_min, min(Q_N_MAX // _row_subblock_h(geom, gm, core_m), _pow2_floor(n_per)))
                # SPLIT_COST_MODEL (Refinement 11): also the epilogue's own quantum, so a column count that the
                # 4-tile subblock quantum would round away stays a candidate (Qwen in_proj M = 1024: 11x10
                # with 3x13 regions instead of 8x11 with 4x12); split_cost prices its narrower subblock.
                for q in (q_n, geom.q_n_min) if SPLIT_COST_MODEL else (q_n,):
                    core_n = _round_up(n_per, q)
                    ub, um, un = _cdiv(tb, core_b), _cdiv(tm, core_m), _cdiv(tn, core_n)
                    placement = _placement_for(ub, um, un, gx, gy)
                    if placement is None:
                        continue
                    grid = GridChoice(ub, um, un, core_b, core_m, core_n, q, placement)
                    seen.setdefault((ub, um, un, core_b, core_m, core_n, q), grid)
    return list(seen.values())


def uses_operand_mcast(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """Regime R1a predicate: a single-batch 2-D split mapped onto a physical rectangle, so every grid
    line (cores sharing a row region, or a column region) is a physical row or column of cores, and a
    K-block large enough to amortize the per-round line handshake (MCAST_MIN_KBLOCK_BYTES)."""
    kblock_bytes = blk.block_k_tiles * max(blk.block_m_tiles * geom.a_tile_bytes, blk.block_n_tiles * geom.b_tile_bytes)
    return (
        OPERAND_MCAST
        and grid.grid_b == 1
        and grid.grid_m > 1
        and grid.grid_n > 1
        and grid.placement in ("rect", "rect_t")
        and kblock_bytes >= MCAST_MIN_KBLOCK_BYTES
    )


def uses_act_stream(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """Regime R1b predicate: a single-B, single-batch 1-D N split over more than one core (every core
    owns all rows, so every core needs every A K-block), with an A K-block large enough to amortize the
    per-round handshake with every core (ACT_STREAM_MIN_KBLOCK_BYTES), on a weight-streaming shape (B
    bytes >= output bytes): the A rounds share the writer RISC with the output, so on a write-bound shape
    they would serialize behind the drains (x_embedder 2048x128x3072: 48.5 -> 62.8 us)."""
    b_bytes = geom.tensor_k_tiles * geom.tensor_n_tiles * geom.b_tile_bytes
    out_bytes = geom.tensor_m_tiles * geom.out_n_tiles * geom.o_tile_bytes
    return (
        ACT_STREAM_MCAST
        and b_bytes >= out_bytes
        and geom.tensor_batches == 1
        and grid.grid_b == 1
        and grid.grid_m == 1
        and grid.grid_n > 1
        and blk.block_batches == 1
        and blk.block_k_tiles * blk.block_m_tiles * geom.a_tile_bytes >= ACT_STREAM_MIN_KBLOCK_BYTES
    )


def k_head_ramp(blk: "Blocking", mode: int, epilogue: str):
    """Perf 2 head ramp (kernels/genify_matmul_kschedule.hpp) -> (ramp, compact_a) for an R1-descriptor walk.

    The ramp is the K-walk's delivery schedule wherever it can be expressed. Carve-outs (inexpressible, not
    unmeasured):
      - R1b: the writer's A stream runs its own K loop over nominal K-blocks;
      - residency walks and one-K-block walks: one delivery by construction (a ramp there would introduce
        packer L1 accumulation where K accumulates wholly in DEST -- a precision change, not a speedup);
      - R1a on the matmul_block helper walk (epilogue none, sb_h > 1): the helper strides A rows by block_k, so A
        cannot be compacted and a short head delivery still multicasts a full A slot (the ramp's mechanism is
        gone: measured flat to +2.3%).
    compact_a: A laid out with the delivery's row stride -- the raw K-walk only (epilogue, 1 x sb_w subblocks).
    """
    raw_walk = epilogue != "none" and blk.sb_h == 1 and not (blk.weight_resident or blk.act_resident)
    if mode == 2 or blk.weight_resident or blk.act_resident or blk.num_k_blocks <= 1:
        return False, False
    if mode == 1 and not raw_walk:
        return False, False
    return True, raw_walk


def _attach_k_head_ramp(readers, compute, blk, mode, epilogue):
    ramp, compact_a = k_head_ramp(blk, mode, epilogue)
    if not ramp:
        return
    defs = [("GM_K_RAMP", "1"), ("GM_COMPACT_A", "1" if compact_a else "0")]
    for k in [*readers, compute]:
        k.defines = list(k.defines) + defs


def r1a_prefetch(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """R1A_INJECTOR_AHEAD: the plain R1a helper walk with rings deep enough for the injector stall-shadow."""
    mode = operand_mode(geom, grid, blk)
    return (
        R1A_INJECTOR_AHEAD > 1
        and mode == 1
        and blk.num_k_blocks > 1
        and blk.block_batches == 1
        and not (blk.weight_resident or blk.act_resident)
        and not k_head_ramp(blk, mode, geom.epilogue)[0]
        and blk.depth_act == blk.depth_weight >= R1A_PREFETCH_DEPTH  # one ring depth addresses both
    )


def operand_mode(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """The operand source of an R1 walk (reader CT arg): 0 = R1 (per-core reads), 1 = R1a (row / column
    line multicast), 2 = R1b (A broadcast stream on the writer, B per core)."""
    if uses_operand_mcast(geom, grid, blk):
        return 1
    if uses_act_stream(geom, grid, blk):
        return 2
    return 0


def injector_offsets(grid: GridChoice):
    """(act_offset, weight_offset) of the R1a line injectors: the act line of row region i injects
    from column region (act_offset + i) % grid_n, the weight line of column region j from row region
    (weight_offset + j) % grid_m (ttnn.Mcast `Staggered` placement). Offsets minimize the number of
    cores that inject both operands (zero whenever possible)."""
    if not STAGGER_INJECTORS:
        return 0, 0
    best = None
    for sa in range(grid.grid_n):
        for sb in range(grid.grid_m):
            both = sum(1 for i in range(grid.grid_m) if (sb + (sa + i) % grid.grid_n) % grid.grid_m == i)
            if best is None or both < best[0]:
                best = (both, sa, sb)
            if both == 0:
                return sa, sb
    return best[1], best[2]


def grid_work(grid: GridChoice):
    """W: per-core output tiles, the critical path (each costs tensor_k_tiles matmul steps)."""
    return grid.core_batches * grid.core_m_tiles * grid.core_n_tiles


def read_traffic(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """Aggregate A + B bytes read over the used cores, including the re-reads the block split adds
    (A once per N-block, B once per M-block of each core's region; a resident operand is read once),
    L1-sourced bytes weighted by L1_READ_COST."""
    nb_m = 1 if blk.weight_resident else _cdiv(grid.core_m_tiles, blk.block_m_tiles)  # B re-reads per M-block
    nb_n = 1 if blk.act_resident else _cdiv(grid.core_n_tiles, blk.block_n_tiles)  # A re-reads per N-block
    a_cost = L1_READ_COST if geom.a_in_l1 else 1.0
    b_cost = L1_READ_COST if geom.b_in_l1 else 1.0
    a_per_core = grid.core_batches * geom.tensor_k_tiles * grid.core_m_tiles * geom.a_tile_bytes * nb_n * a_cost
    b_per_core = grid.core_batches * geom.tensor_k_tiles * grid.core_n_tiles * geom.b_tile_bytes * nb_m * b_cost
    mode = operand_mode(geom, grid, blk)
    if mode == 1:
        # R1a: one A read per grid row-line, one B read per column-line.
        return a_per_core * grid.grid_m + b_per_core * grid.grid_n
    if mode == 2:
        # R1b: A read once (by the injector), B per core; the single injector's throughput caps it.
        return max(a_per_core + b_per_core * grid.grid_n, a_per_core * ACT_STREAM_INJECTOR_COST)
    return (a_per_core + b_per_core) * grid.grid_b * grid.grid_m * grid.grid_n


def tile_k_cycles(geom: Geometry, sb_h: int, sb_w: int):
    """Cycles per output tile per K tile: the FPU's fidelity passes, or the subblock's operand unpacks
    (sb_h + sb_w tiles per sb_h * sb_w products) when they bind."""
    return max(FPU_CYCLES_PER_TILE_K * geom.fidelity_phases, UNPACK_CYCLES_PER_TILE * (sb_h + sb_w) / (sb_h * sb_w))


def padded_work(grid: GridChoice, blk: "Blocking"):
    """Output tiles a core's walk computes: every core walks the nominal blocks of the grid-uniform region
    (a ragged last block still runs its full block_m x block_n)."""
    nb_m = _cdiv(grid.core_m_tiles, blk.block_m_tiles)
    nb_n = _cdiv(grid.core_n_tiles, blk.block_n_tiles)
    return grid.core_batches * nb_m * blk.block_m_tiles * nb_n * blk.block_n_tiles


def split_cost(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """SPLIT_COST_MODEL: estimated cycles of a split (the max of its overlapped streams)."""
    return max(split_cost_terms(geom, grid, blk))


def split_cost_terms(geom: Geometry, grid: GridChoice, blk: "Blocking"):
    """(K-walk, DRAM stream, busiest core's reads, busiest core's writes) in cycles."""
    walk = padded_work(grid, blk) * geom.tensor_k_tiles * tile_k_cycles(geom, blk.sb_h, blk.sb_w)
    out_bytes = geom.tensor_batches * geom.tensor_m_tiles * geom.out_n_tiles * geom.o_tile_bytes
    dram = (read_traffic(geom, grid, blk) + out_bytes) / DRAM_BYTES_PER_CYCLE
    # The busiest reading core: R1 reads its own A (once per N-block) and B (once per M-block); an R1a line
    # injector reads + multicasts one of them for its line; R1b's injector is priced inside read_traffic.
    nb_m = 1 if blk.weight_resident else _cdiv(grid.core_m_tiles, blk.block_m_tiles)
    nb_n = 1 if blk.act_resident else _cdiv(grid.core_n_tiles, blk.block_n_tiles)
    a_core = grid.core_batches * grid.core_m_tiles * geom.tensor_k_tiles * geom.a_tile_bytes * nb_n
    b_core = grid.core_batches * grid.core_n_tiles * geom.tensor_k_tiles * geom.b_tile_bytes * nb_m
    mode = operand_mode(geom, grid, blk)
    core_bytes = max(a_core, b_core) if mode == 1 else (0 if mode == 2 else a_core + b_core)
    core_out = grid.core_batches * grid.core_m_tiles * geom.out_block_n(grid.core_n_tiles) * geom.o_tile_bytes
    return walk, dram, core_bytes / CORE_READ_BYTES_PER_CYCLE, core_out / CORE_WRITE_BYTES_PER_CYCLE


def choose_split(geom: Geometry, gx: int, gy: int, budget: int):
    """Core-assignment knob + extent knobs together. SPLIT_COST_MODEL: solve the blocking of every
    factorization and take the lexicographic min of (split_cost, aggregate read traffic, W, used cores);
    candidates are visited in W order and the walk stops once W alone (the unpadded FPU floor) exceeds the
    best cost. Legacy: among the factorizations whose W is within (1 + GRID_W_SLACK) of the best W, the
    lexicographic min of (aggregate read traffic incl. block re-reads, W, used cores)."""
    candidates = grid_candidates(geom, gx, gy)
    if SPLIT_COST_MODEL:
        floor = geom.tensor_k_tiles * FPU_CYCLES_PER_TILE_K * geom.fidelity_phases
        best_key, best = None, None
        for grid in sorted(candidates, key=grid_work):
            if best_key is not None and grid_work(grid) * floor > best_key[0]:
                break
            try:
                blk = choose_blocking(geom, grid, budget)
            except RuntimeError:
                continue
            key = (
                split_cost(geom, grid, blk),
                read_traffic(geom, grid, blk),
                grid_work(grid),
                grid.grid_b * grid.grid_m * grid.grid_n,
            )
            if best_key is None or key < best_key:
                best_key, best = key, (grid, blk)
        if best is None:
            raise RuntimeError(f"genify_matmul: even the minimum block does not fit the L1 CB budget of {budget} bytes")
        return best
    w_cap = min(grid_work(g) for g in candidates) * (1 + GRID_W_SLACK)
    best_key, best = None, None
    for grid in candidates:
        if grid_work(grid) > w_cap:
            continue
        try:
            blk = choose_blocking(geom, grid, budget)
        except RuntimeError:
            continue
        key = (read_traffic(geom, grid, blk), grid_work(grid), grid.grid_b * grid.grid_m * grid.grid_n)
        if best_key is None or key < best_key:
            best_key, best = key, (grid, blk)
    if best is None:
        raise RuntimeError(f"genify_matmul: even the minimum block does not fit the L1 CB budget of {budget} bytes")
    return best


def _logical_cores(grid: GridChoice, gx: int):
    """[(b, i, j, CoreCoord)] for every used logical region, plus the CoreRangeSet covering them."""
    out = []
    for b in range(grid.grid_b):
        for i in range(grid.grid_m):
            for j in range(grid.grid_n):
                if grid.placement == "rect":
                    x, y = j, i
                elif grid.placement == "rect_t":
                    x, y = i, j
                else:
                    p = (b * grid.grid_m + i) * grid.grid_n + j
                    x, y = p % gx, p // gx
                out.append((b, i, j, ttnn.CoreCoord(x, y)))
    if grid.placement == "rect":
        ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.grid_n - 1, grid.grid_m - 1))]
    elif grid.placement == "rect_t":
        ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.grid_m - 1, grid.grid_n - 1))]
    else:
        n = len(out)
        full_rows, rem = n // gx, n % gx
        ranges = []
        if full_rows:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full_rows - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return out, ttnn.CoreRangeSet(ranges)


# ---------------------------------------------------------------------------
# Blocking (extent knobs, L1 solve)
# ---------------------------------------------------------------------------


def choose_subblock(block_m, block_n, dest_limit):
    """Largest sb_h*sb_w <= dest_limit with sb_h | block_m, sb_w | block_n. Ties: SUBBLOCK_MIN_UNPACK
    prefers the smallest operand-unpack count per subblock (sb_h + sb_w per K step), then larger sb_w;
    otherwise larger sb_w."""

    def key(h, w):
        return (h * w, -(h + w) if SUBBLOCK_MIN_UNPACK else 0, w)

    best = (1, 1)
    for h in _divisors(block_m):
        for w in _divisors(block_n):
            if h * w <= dest_limit and key(h, w) > key(*best):
                best = (h, w)
    return best


def _footprint(geom, bm, bn, bk, partials, out_in_place=False, depth_weight=None, act_cb=True, depth_act=None):
    """F(block_m, block_n, block_k) of l1_ledger.md, epilogue terms included (E_bias, E_addcmul).
    An epilogue always needs cb_partials (the matmul's last K-block lands there). `out_in_place`: the
    output CB is backed on the output tensor's own L1 shard (R4), so it adds no CB bytes. `act_cb` False:
    A is consumed straight from its resident shard (R2 / single-K-shard R3), no cb_act_tiles."""
    depth_act = DEPTH_ACT if depth_act is None else depth_act
    depth_weight = DEPTH_WEIGHT if depth_weight is None else depth_weight
    sb_h, _ = choose_subblock(bm, bn, geom.dest_limit)
    partials = partials or geom.epilogue != "none"
    return (
        (depth_act * bm * bk * geom.a_tile_bytes if act_cb else 0)
        + depth_weight * bk * bn * geom.b_tile_bytes
        + (bm * bn * geom.p_tile_bytes if partials else 0)
        + (0 if out_in_place else DEPTH_OUT * sb_h * geom.out_block_n(bn) * geom.o_tile_bytes)
        + bn * geom.row_tile_bytes  # cb_bias_row / cb_gate_row: one N-block row
        + DEPTH_RESIDUAL * sb_h * bn * geom.r_tile_bytes  # cb_residual_tiles
    )


def _block_k_max(geom, bm, bn, budget, depths=None):
    """Largest block_k with F(bm, bn, block_k, partials=True) <= budget (F is affine in block_k)."""
    da, dw = depths or (DEPTH_ACT, DEPTH_WEIGHT)
    fixed = _footprint(geom, bm, bn, 0, partials=True)
    per_k = da * bm * geom.a_tile_bytes + dw * bn * geom.b_tile_bytes
    if budget < fixed:
        return 0
    return (budget - fixed) // per_k


def _k_split(tk, bk_max):
    """Balanced K split: (block_k_tiles, num_k_blocks, last_block_k_tiles)."""
    bk_max = max(1, min(bk_max, tk))
    num_k_blocks = _cdiv(tk, bk_max)
    block_k = _cdiv(tk, num_k_blocks)
    num_k_blocks = _cdiv(tk, block_k)
    last = tk - (num_k_blocks - 1) * block_k
    assert 0 < last <= block_k
    return block_k, num_k_blocks, last


def _make_blocking(geom, block_b, bm, bn, bk_max, depths=None):
    block_k, nkb, last = _k_split(geom.tensor_k_tiles, min(bk_max, geom.block_k_cap))
    sb_h, sb_w = choose_subblock(bm, bn, geom.dest_limit)
    da, dw = depths or (DEPTH_ACT, DEPTH_WEIGHT)
    return Blocking(block_b, bm, bn, block_k, nkb, last, sb_h, sb_w, geom.epilogue, depth_act=da, depth_weight=dw)


def _resident_split(geom: Geometry, grid: GridChoice):
    """RESIDENT_SPLIT_BLOCKS: the one-block, one-K-block walk of S1 split along one axis with the smaller
    per-core operand resident (it fits: every CB term shrinks with the split extent). None when off or
    the axis is too short to split. B resident: M-blocks, multiples of the whole region's subblock height
    (the subblock, i.e. SRC operand reuse, holds). A resident: N-blocks, multiples of the N quantum."""
    cm, cn, tk = grid.core_m_tiles, grid.core_n_tiles, geom.tensor_k_tiles
    if RESIDENT_SPLIT_BLOCKS <= 1 or grid.core_batches != 1:
        return None
    if cm * geom.a_tile_bytes <= cn * geom.b_tile_bytes:
        bn = min(cn, _round_up(_cdiv(cn, RESIDENT_SPLIT_BLOCKS), grid.q_n))
        if bn < cn:
            return replace(_make_blocking(geom, 1, cm, bn, tk), act_resident=True)
    else:
        sb_h, _ = choose_subblock(cm, cn, geom.dest_limit)
        bm = min(cm, _round_up(_cdiv(cm, RESIDENT_SPLIT_BLOCKS), sb_h))
        if bm < cm:
            return replace(_make_blocking(geom, 1, bm, cn, tk), weight_resident=True)
    return None


def _stream_depths(geom: Geometry, grid: GridChoice):
    """ACT_STREAM_DEPTH for both operand rings when the split can run R1b (its K-walk), else None."""
    probe = Blocking(1, grid.core_m_tiles, 1, geom.tensor_k_tiles, 1, geom.tensor_k_tiles, 1, 1)
    if ACT_STREAM_DEPTH and uses_act_stream(geom, grid, replace(probe, block_k_tiles=geom.tensor_k_tiles)):
        return (ACT_STREAM_DEPTH, ACT_STREAM_DEPTH)
    return None


def choose_blocking(geom: Geometry, grid: GridChoice, budget: int) -> Blocking:
    depths = _stream_depths(geom, grid)
    blk = _choose_blocking(geom, grid, budget, depths)
    if depths is not None and operand_mode(geom, grid, blk) != 2:
        blk = _choose_blocking(geom, grid, budget)  # the solved K-block missed the R1b gate: plain depths
    # MIN_K_BLOCKS: an R1a walk without the head ramp (the ramp already shortens the fill on the raw walk:
    # 384x1536x1024 bias +5% re-split under it).
    mode = operand_mode(geom, grid, blk)
    if 1 < blk.num_k_blocks < MIN_K_BLOCKS and mode == 1 and not k_head_ramp(blk, mode, geom.epilogue)[0]:
        bk, nkb, last = _k_split(geom.tensor_k_tiles, max(BLOCK_K_FLOOR, _cdiv(geom.tensor_k_tiles, MIN_K_BLOCKS)))
        finer = replace(blk, block_k_tiles=bk, num_k_blocks=nkb, last_block_k_tiles=last)
        if bk < blk.block_k_tiles and operand_mode(geom, grid, finer) == 1:
            blk = finer
    # R1A_INJECTOR_AHEAD: deepen the operand rings of a plain R1a helper walk (shrinking block_k when they do
    # not fit), as long as the walk stays R1a.
    d = R1A_PREFETCH_DEPTH
    deeper = replace(blk, depth_act=max(d, blk.depth_act), depth_weight=max(d, blk.depth_weight))
    walk, _, core_read, _ = split_cost_terms(geom, grid, blk)
    if R1A_INJECTOR_AHEAD > 1 and core_read >= R1A_PREFETCH_MIN_RATIO * walk and r1a_prefetch(geom, grid, deeper):
        bm, bn = blk.block_m_tiles, blk.block_n_tiles
        depths = (deeper.depth_act, deeper.depth_weight)
        bk_max = min(blk.block_k_tiles, _block_k_max(geom, bm, bn, budget, depths))
        if bk_max >= min(geom.tensor_k_tiles, BLOCK_K_FLOOR):
            bk, nkb, last = _k_split(geom.tensor_k_tiles, bk_max)
            deeper = replace(deeper, block_k_tiles=bk, num_k_blocks=nkb, last_block_k_tiles=last)
            if r1a_prefetch(geom, grid, deeper):
                blk = deeper
    # Write batching: grows only the output CB term of F (DEPTH_OUT * batch row groups).
    groups = blk.block_m_tiles // blk.sb_h
    row_group_bytes = blk.sb_h * geom.out_block_n(blk.block_n_tiles) * geom.o_tile_bytes
    base = _footprint(
        geom,
        blk.block_m_tiles,
        blk.block_n_tiles,
        blk.block_k_tiles,
        blk.use_partials,
        depth_weight=blk.depth_weight,
        depth_act=blk.depth_act,
    )
    for batch in sorted((d for d in _divisors(groups) if d <= OUT_WRITE_BATCH_ROW_GROUPS), reverse=True):
        if base + DEPTH_OUT * (batch - 1) * row_group_bytes <= budget:
            return replace(blk, write_batch=batch)
    return blk


def _choose_blocking(geom: Geometry, grid: GridChoice, budget: int, depths=None) -> Blocking:
    """`depths`: (depth_act, depth_weight) of a K-walk (S2 / S3), default (DEPTH_ACT, DEPTH_WEIGHT). The
    one-K-block S1 walk keeps the defaults (its depth-2 rings already hold its whole walk)."""
    da, dw = depths or (DEPTH_ACT, DEPTH_WEIGHT)
    tk = geom.tensor_k_tiles
    block_b = grid.core_batches  # costs no L1 (the helper's batch loop reuses per-batch CBs)
    cm, cn = grid.core_m_tiles, grid.core_n_tiles
    k_floor = min(tk, BLOCK_K_FLOOR, geom.block_k_cap)
    one_k_block = geom.block_k_cap >= tk  # the precision cap allows K in a single DEST chain

    # S1: whole per-core region, one resident K-block, no accumulator.
    if one_k_block and _footprint(geom, cm, cn, tk, partials=False) <= budget:
        # A core whose whole walk is this single K-block can neither use the second slot of the
        # depth-2 operand CBs nor overlap its read with compute. Split K so compute starts on the
        # first half while the second lands (the two halves fill exactly the old one-slot footprint).
        if SINGLE_WALK_K_BLOCKS > 1 and block_b == 1 and tk >= SINGLE_WALK_K_BLOCKS * BLOCK_K_FLOOR:
            bk = _cdiv(tk, SINGLE_WALK_K_BLOCKS)
            if _footprint(geom, cm, cn, bk, partials=True) <= budget:
                return _make_blocking(geom, block_b, cm, cn, bk)
        whole = _make_blocking(geom, block_b, cm, cn, tk)
        resident = _resident_split(geom, grid)
        # The split must not cost the R1a / R1b operand multicast: a smaller K-block can drop under
        # MCAST_MIN_KBLOCK_BYTES (2048x256x512 bf8b, 16-bit DEST: 52 -> 26 KB A rounds, +11% as plain R1).
        if resident is not None and operand_mode(geom, grid, resident) == operand_mode(geom, grid, whole):
            return resident
        return whole

    # S2: whole region, K-walk with the accumulator.
    bk_max = _block_k_max(geom, cm, cn, budget, depths)
    if bk_max >= k_floor:
        return _make_blocking(geom, block_b, cm, cn, bk_max, depths)

    # S3: shrink the output block; minimize A/B re-reads, ties to the larger block. SPLIT_COST_MODEL
    # (Refinement 11): minimize the block's split_cost instead -- it prices the padded work of a block that
    # does not divide the region (every core walks nominal blocks: FLUX single out's 7x20 region in 12-wide
    # blocks computed 7x24, 1760 us; 10-wide blocks, 1x2 subblock, 1544 us) against the re-reads -- and let
    # block_n step by the epilogue's own quantum (q_n_min) as well as the split's subblock quantum q_n.
    def solved(bm, bn):
        if one_k_block and _footprint(geom, bm, bn, tk, partials=False) <= budget:
            return _make_blocking(geom, block_b, bm, bn, tk)
        return _make_blocking(geom, block_b, bm, bn, _block_k_max(geom, bm, bn, budget, depths), depths)

    quanta = sorted({grid.q_n, geom.q_n_min}) if SPLIT_COST_MODEL else [grid.q_n]
    best_key, best = None, None
    seen = set()
    for nb_m in range(1, cm + 1):
        bm = _cdiv(cm, nb_m)
        for q in quanta:
            for nb_n in range(1, _cdiv(cn, q) + 1):
                bn = _round_up(_cdiv(cn, nb_n), q)
                if (bm, bn) in seen:
                    continue
                seen.add((bm, bn))
                if _footprint(geom, bm, bn, k_floor, partials=True, depth_weight=dw, depth_act=da) > budget:
                    continue
                eff_nb_m, eff_nb_n = _cdiv(cm, bm), _cdiv(cn, bn)
                reread = tk * (cm * geom.a_tile_bytes * (eff_nb_n - 1) + cn * geom.b_tile_bytes * (eff_nb_m - 1))
                key = (reread, -(bm * bn))
                if SPLIT_COST_MODEL:
                    key = (split_cost(geom, grid, solved(bm, bn)),) + key
                if best_key is None or key < best_key:
                    best_key, best = key, (bm, bn)
    if best is not None:
        return solved(*best)

    # S4
    raise RuntimeError(f"genify_matmul: even the minimum block does not fit the L1 CB budget of {budget} bytes")


def _l1_tensor_bytes_per_core(tensors, num_cores):
    total = 0
    for t in tensors:
        mc = t.memory_config()
        if mc.buffer_type != ttnn.BufferType.L1:
            continue
        if mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
            total += _cdiv(t.buffer_num_pages(), num_cores) * t.buffer_aligned_page_size()
        else:
            shape = mc.shard_spec.shape
            total += _cdiv(shape[0], TILE) * _cdiv(shape[1], TILE) * t.buffer_aligned_page_size()
    return total


def _lowest_allocated_l1_offset(view):
    """Offset (from the bank base, where the CB region starts) of the lowest allocated L1 block, or
    the bank size when nothing is allocated. L1 tensors are placed top-down but freed blocks leave
    holes, so this -- not `total - allocated` -- is where the CB region must end (program.cpp throws
    "circular buffers ... clash with L1 buffers" otherwise)."""
    allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(allocated) if allocated else view.total_bytes_per_bank


def l1_cb_budget(device, op_tensors):
    """Bytes per core the CBs may use: the worker-L1 region from l1_unreserved_base (where CBs start
    and where program.cpp bounds them) up to the lowest allocated L1 block, never more than the bank
    minus this op's own L1-resident tensors per core, minus the reserve knob. Note:
    ttnn.get_max_worker_l1_unreserved_size() counts from the kernel-config ring buffer, below the CB
    base, and over-states the budget."""
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    room = min(
        _lowest_allocated_l1_offset(view),
        view.total_bytes_per_bank - _l1_tensor_bytes_per_core(op_tensors, view.num_banks),
    )
    return room - L1_RESERVE_BYTES


def plan(a, b, out, fp32_dest_acc_en, device, epilogue="none", row=None, residual=None, math_fidelity=None):
    """The full blocking decision (geometry, grid, blocking). Exposed for tests / perf tooling.
    `row` is the bias (bias / bias_relu) or gate (addcmul) tensor, `residual` the addcmul residual;
    `math_fidelity` feeds the split cost model (None = HiFi4, the op's default)."""
    geom = _geometry(a, b, out, fp32_dest_acc_en, epilogue, row, residual, math_fidelity)
    grid_size = device.compute_with_storage_grid_size()
    gx, gy = grid_size.x, grid_size.y
    budget = l1_cb_budget(device, [t for t in (*_a_list(a), b, *_out_list(out), row, residual) if t is not None])
    grid, blocking = choose_split(geom, gx, gy, budget)
    return geom, grid, blocking, (gx, gy)


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


def _cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


# Reader CT args [24, 32): regime R4 role (dram_sharded, worker, K-blocks per A round, act CB depth in
# K-blocks, per-row B slice reads, weight CB depth in K-blocks, B K-blocks in flight, A rounds on the
# reader). Every non-R4 reader passes these defaults.
_DS_ROLE_ABSENT = (0, 0, 1, DEPTH_ACT, 0, DEPTH_WEIGHT, 1, 1)
# Writer CT args [8, 16): the R4 A-round role when it runs on the writer (ds_act, worker, cb_act_tiles,
# block_k, num_k_blocks, K-blocks per round, A tile bytes, act CB depth). Every other writer: absent.
_DS_WRITER_ACT_ABSENT = (0, 0, CB_ACT_TILES, 1, 1, 1, 0, DEPTH_ACT)
# Reader CT args [33, 40): regime R3 / R2 role (block_sharded, row multicast, cb_act_shard id) and the
# R2a / R3a weight multicast (on, chunks per B round, K rows per chunk, K-blocks per B round).
_BS_ROLE_ABSENT = (0, 0, CB_ACT_SHARD, 0, 1, 1, 1)


def _reader_ct(geom, blk, a, b, epilogue, cb_row, row_t, res_t):
    """Reader CT args [0, 22): CB ids, block knobs, page sizes, epilogue operands, operand residency."""
    return [
        CB_ACT_TILES,
        CB_WEIGHT_TILES,
        blk.block_batches,
        blk.block_m_tiles,
        blk.block_n_tiles,
        blk.block_k_tiles,
        blk.num_k_blocks,
        blk.last_block_k_tiles,
        geom.a_tile_bytes,
        geom.b_tile_bytes,
        a.buffer_aligned_page_size(),
        b.buffer_aligned_page_size(),
        EPILOGUES[epilogue],
        cb_row,
        CB_RESIDUAL_TILES,
        blk.sb_h,
        ttnn.tile_size(row_t.dtype),
        row_t.buffer_aligned_page_size(),
        ttnn.tile_size(res_t.dtype),
        res_t.buffer_aligned_page_size(),
        int(blk.weight_resident),
        int(blk.act_resident),
    ]


def _reader_accessor_ct(a, b, row_t, res_t):
    ct = []
    for t in (a, b, row_t, res_t):
        ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    return ct


def _writer_ct(
    cb_out, blk, out_block_n_tiles, geom, out, out_in_place, ds_act_role=_DS_WRITER_ACT_ABSENT, act_stream_a=None
):
    """`ds_act_role`: the R4 A-round role (CT [8, 16)). `act_stream_a`: A, when this writer runs R1b's A
    stream (CT 8 = 2 overrides the absent R4 role; A's accessor follows the output's)."""
    role = list(ds_act_role)
    if act_stream_a is not None:
        role = [2, 1, CB_ACT_TILES, blk.block_k_tiles, blk.num_k_blocks, 1, geom.a_tile_bytes, blk.depth_act]
    a_t = act_stream_a if act_stream_a is not None else out  # stand-in accessor (never dereferenced)
    return [
        cb_out,
        blk.block_batches,
        blk.block_m_tiles,
        out_block_n_tiles,
        blk.sb_h,
        geom.o_tile_bytes,
        out.buffer_aligned_page_size(),
        int(out_in_place),  # the output CB is the output's own L1 shard: retire only, no NoC write
        *role,
        blk.write_batch,  # row groups per writer wait / flush / pop
        blk.last_block_k_tiles,
        a_t.buffer_aligned_page_size(),
        int(blk.act_resident),
        max(1, min(ACT_STREAM_AHEAD, blk.depth_act)) if act_stream_a is not None else 1,  # R1b act_ahead
        *ttnn.TensorAccessorArgs(out).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(a_t).get_compile_time_args(),
    ]


def _compute_ct(cb_out, blk, epilogue, cb_row, out, swiglu_natural_pairs, own_round_slot=False):
    return [
        CB_ACT_TILES,
        CB_WEIGHT_TILES,
        cb_out,
        CB_PARTIALS,
        blk.block_batches,
        blk.block_m_tiles // blk.sb_h,  # in0_num_subblocks
        blk.block_n_tiles // blk.sb_w,  # in1_num_subblocks
        blk.sb_h,
        blk.sb_w,
        blk.block_k_tiles,
        blk.num_k_blocks,
        blk.last_block_k_tiles,
        blk.block_n_tiles,
        EPILOGUES[epilogue],
        cb_row,
        CB_RESIDUAL_TILES,
        blk.block_m_tiles,
        EPILOGUE_BLOCK_TILES,
        0 if (SWIGLU_FAST_SILU and out.dtype != ttnn.float32) else 1,  # silu_fp32_precision
        # swiglu: the block's B columns are in natural (gate, up) tile-pair order (R4 coalesced bank
        # reads) instead of the reader's gate-half / up-half permutation (R1 / R1a).
        int(swiglu_natural_pairs),
        CB_ACT_SHARD,
        # R3a: the reader pushes a lockstep cb_act_tiles slot on the core's own K turn; compute retires it.
        int(own_round_slot),
        int(blk.weight_resident),  # B fronted once, retained across the core's blocks, popped after them
        int(blk.act_resident),  # A fronted once per M-block, retained across its N-blocks
    ]


def create_program_descriptor(
    a, b, out, compute_config, epilogue="none", row=None, residual=None, placement="interleaved"
):
    """`row` = bias (bias / bias_relu) or gate (addcmul); `residual` = addcmul residual. Regime
    selection: R4 for placement dram_sharded, R3 for block_sharded, else R1 (sub-selecting R1a)."""
    if placement == "dram_sharded":
        return create_dram_sharded_program_descriptor(
            a, b, out, compute_config, epilogue=epilogue, row=row, residual=residual
        )
    if placement in ("block_sharded", "height_sharded"):
        return create_block_sharded_program_descriptor(
            a, b, out, compute_config, epilogue=epilogue, row=row, residual=residual, placement=placement
        )
    a_segs = _a_list(a)  # Refinement 14: A as K segments (R1 family only)
    device = a_segs[0].device()
    geom, grid, blk, (gx, gy) = plan(
        a,
        b,
        out,
        bool(compute_config.fp32_dest_acc_en),
        device,
        epilogue=epilogue,
        row=row,
        residual=residual,
        math_fidelity=compute_config.math_fidelity,
    )
    cores, core_ranges = _logical_cores(grid, gx)
    # Split output (Refinement 13): the chunks share one memory config / dtype / shape, so chunk 0 stands in
    # for every per-tensor property; the writer (and the reader's tail share) route each output tile column
    # to its chunk through the address table appended to their runtime args (kernels/out_chunks.hpp).
    outs = _out_list(out)
    out = outs[0]
    out_row_tiles = geom.out_n_tiles // len(outs)  # tiles per tile row of ONE output tensor
    chunk_addrs = [o.buffer_address() for o in outs[1:]]
    # A as K segments (Refinement 14): segment 0 stands in for A's per-tensor properties; the reader (and R1b's
    # A-streaming writer) map each K tile to its segment through the table in their RT args (a_segments.hpp).
    a = a_segs[0]
    a_seg_table = _a_segment_table(a_segs)

    out_block_n_tiles = geom.out_block_n(blk.block_n_tiles)  # swiglu halves the output width
    cb_row = CB_GATE_ROW if epilogue == "addcmul" else CB_BIAS_ROW

    # ---- CBs: every page count derives from the block / depth knobs ----
    cbs = [
        _cb(
            CB_ACT_TILES, a.dtype, geom.a_tile_bytes, blk.depth_act * blk.block_m_tiles * blk.block_k_tiles, core_ranges
        ),
        _cb(
            CB_WEIGHT_TILES,
            b.dtype,
            geom.b_tile_bytes,
            blk.depth_weight * blk.block_k_tiles * blk.block_n_tiles,
            core_ranges,
        ),
        _cb(
            CB_OUTPUT_TILES,
            out.dtype,
            geom.o_tile_bytes,
            DEPTH_OUT * blk.write_batch * blk.sb_h * out_block_n_tiles,
            core_ranges,
        ),
    ]
    if blk.use_partials:
        # Exactly one block (mechanism cap: in-place packer L1 accumulation).
        cbs.append(
            _cb(CB_PARTIALS, geom.p_dtype, geom.p_tile_bytes, blk.block_m_tiles * blk.block_n_tiles, core_ranges)
        )
    if row is not None:
        # One N-block row of the bias / gate, refreshed once per block.
        cbs.append(_cb(cb_row, row.dtype, geom.row_tile_bytes, blk.block_n_tiles, core_ranges))
    if residual is not None:
        cbs.append(
            _cb(
                CB_RESIDUAL_TILES,
                residual.dtype,
                geom.r_tile_bytes,
                DEPTH_RESIDUAL * blk.sb_h * blk.block_n_tiles,
                core_ranges,
            )
        )

    # Absent epilogue tensors: B stands in for the accessor CT args (never dereferenced).
    row_t = row if row is not None else b
    res_t = residual if residual is not None else b

    # ---- Kernels ----
    reader_ct = _reader_ct(geom, blk, a, b, epilogue, cb_row, row_t, res_t)
    # [operand_mcast, a_sends, b_sends] + the R4 role args go between reader_ct and the accessors,
    # set per kernel group below.
    reader_accessor_ct = _reader_accessor_ct(a, b, row_t, res_t)
    mode = operand_mode(geom, grid, blk)
    operand_mcast = mode == 1  # R1a
    act_stream = mode == 2  # R1b
    act_offset, weight_offset = injector_offsets(grid)
    writer_ct = _writer_ct(
        CB_OUTPUT_TILES, blk, out_block_n_tiles, geom, out, out_in_place=False, act_stream_a=a if act_stream else None
    )
    compute_ct = _compute_ct(CB_OUTPUT_TILES, blk, epilogue, cb_row, out, swiglu_natural_pairs=False)

    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    a_addr, b_addr, out_addr = a.buffer_address(), b.buffer_address(), out.buffer_address()
    row_addr = row.buffer_address() if row is not None else 0
    res_addr = residual.buffer_address() if residual is not None else 0
    out_div = 2 if geom.swiglu else 1  # B columns per output column
    reader_roles = {}  # core -> (a_sends, b_sends)
    reader_args = {}  # core -> reader runtime args (placed per role group below)
    for bi, i, j, core in cores:
        b0 = bi * grid.core_batches
        batches_valid = min(grid.core_batches, geom.tensor_batches - b0)
        row0 = i * grid.core_m_tiles
        rows_valid = min(grid.core_m_tiles, geom.tensor_m_tiles - row0)
        col0 = j * grid.core_n_tiles
        cols_valid = min(grid.core_n_tiles, geom.tensor_n_tiles - col0)
        nb_b = _cdiv(batches_valid, blk.block_batches)
        if operand_mcast or act_stream:
            # Every core of a line (R1b: of the split) walks the same nominal block sequence (CB pointers in lockstep, one
            # multicast per K-block); blocks in a ragged edge's padded tail run with 0 valid tiles.
            nb_m = _cdiv(grid.core_m_tiles, blk.block_m_tiles)
            nb_n = _cdiv(grid.core_n_tiles, blk.block_n_tiles)
            reader_roles[(core.x, core.y)] = (
                (int(j == (act_offset + i) % grid.grid_n), int(i == (weight_offset + j) % grid.grid_m))
                if operand_mcast
                else (0, 0)
            )
        else:
            nb_m = _cdiv(rows_valid, blk.block_m_tiles)
            nb_n = _cdiv(cols_valid, blk.block_n_tiles)
            reader_roles[(core.x, core.y)] = (0, 0)
        region = [b0, batches_valid, row0, rows_valid, col0, cols_valid, nb_b, nb_m, nb_n]
        reader_args[(core.x, core.y)] = [
            a_addr,
            b_addr,
            *region,
            geom.tensor_m_tiles,
            geom.tensor_k_tiles,
            geom.tensor_n_tiles,
            geom.weight_batch_stride,
            row_addr,
            res_addr,
            NO_OWN_KB,
            *a_seg_table,
        ]
        # The writer walks output columns: swiglu halves col0 / cols_valid (both even, q_n >= 2).
        out_region = [b0, batches_valid, row0, rows_valid, col0 // out_div, cols_valid // out_div, nb_b, nb_m, nb_n]
        a_stream_rt = [a_addr, geom.tensor_k_tiles, *a_seg_table] if act_stream else []  # R1b: the writer reads A
        wrt = [out_addr, *out_region, geom.tensor_m_tiles, out_row_tiles, *a_stream_rt, *chunk_addrs]
        # Tail write split (Perf 2): the reader serves part of the last block's writes on NoC0.
        reader_args[(core.x, core.y)] += [out_addr, *out_region, geom.tensor_m_tiles, out_row_tiles, *chunk_addrs]
        writer_rt[core.x][core.y] = wrt
        compute_rt[core.x][core.y] = [nb_b * nb_m * nb_n, NO_OWN_KB, nb_n]

    # One reader kernel per (a_sends, b_sends) role group: under R1a the line injectors are CT-
    # specialized, so every core hosts exactly one reader. Plain R1 is a single group (0, 0).
    groups = {}
    for (x, y), role in reader_roles.items():
        groups.setdefault(role, []).append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)))
    readers = []
    for (a_sends, b_sends), ranges in sorted(groups.items()):
        group_cores = ttnn.CoreRangeSet(ranges)
        rt = ttnn.RuntimeArgs()
        for r in ranges:
            rt[r.start.x][r.start.y] = reader_args[(r.start.x, r.start.y)]
        readers.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "genify_matmul_reader.cpp"),
                core_ranges=group_cores,
                compile_time_args=[
                    *reader_ct,
                    mode,
                    a_sends,
                    b_sends,
                    *_DS_ROLE_ABSENT,
                    *_BS_ROLE_ABSENT,
                    *reader_accessor_ct,
                    *ttnn.TensorAccessorArgs(out).get_compile_time_args(),  # tail write split: the output
                ],
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),  # NCRISC / NoC0
            )
        )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),  # BRISC / NoC1
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    _attach_operand_channels(descriptor, readers, grid, core_ranges, device, operand_mcast, act_offset, weight_offset)
    if act_stream:
        _attach_act_stream(descriptor, writer, core_ranges, device)
    else:
        ttnn.attach_absent_mcast(writer, "act")  # the writer's A role (R4 rounds / R1b stream) is absent
    _attach_tail_split(descriptor, readers, writer, core_ranges, blk, out_block_n_tiles, geom, out)
    if len(outs) > 1:
        for k in (*readers, writer):
            k.defines = list(k.defines) + [("GM_OUT_CHUNKS", str(len(outs)))]
    if len(a_segs) > 1:
        for k in (*readers, writer):
            k.defines = list(k.defines) + [("GM_A_SEGMENTS", str(len(a_segs)))]
    _attach_k_head_ramp(readers, compute, blk, mode, epilogue)
    walk, *streams = split_cost_terms(geom, grid, blk)
    if NONE_RAW_WALK and epilogue == "none" and walk >= NONE_RAW_WALK_MIN_RATIO * max(streams):
        compute.defines = list(compute.defines) + [("GM_NONE_RAW_WALK", "1")]
    if r1a_prefetch(geom, grid, blk):
        defs = [("GM_R1A_AHEAD", str(R1A_INJECTOR_AHEAD)), ("GM_R1A_DEPTH", str(min(blk.depth_act, blk.depth_weight)))]
        for k in readers:
            k.defines = list(k.defines) + defs
    descriptor.kernels = [*readers, writer, compute]
    return descriptor


def _attach_tail_split(descriptor, readers, writer, core_ranges, blk, out_block_n_tiles, geom, out):
    """Tail write split: the reader (NCRISC / NoC0) writes TAIL_NC_PCT % of every output row of the core's
    LAST block, the writer (BRISC / NoC1) the rest; a post / done handshake on three local semaphores keeps
    the writer the output CB's only consumer."""
    sems = list(descriptor.semaphores)
    base = max([sm.id for sm in sems], default=-1) + 1
    for i in range(3):  # mailbox (L1 address), post seq, done seq
        sems.append(ttnn.SemaphoreDescriptor(id=base + i, core_ranges=core_ranges, initial_value=0))
    descriptor.semaphores = sems
    common = [
        ("TAIL_SPLIT_MODE", "1"),
        ("TAIL_NC_PCT", str(TAIL_NC_PCT)),
        ("TAIL_SEM_MBOX", str(base)),
        ("TAIL_SEM_POST", str(base + 1)),
        ("TAIL_SEM_DONE", str(base + 2)),
    ]
    rd = common + [
        ("TAIL_OUT_BLOCK_N", str(out_block_n_tiles)),
        ("TAIL_WRITE_BATCH", str(blk.write_batch)),
        ("TAIL_O_TILE_BYTES", str(geom.o_tile_bytes)),
        ("TAIL_O_PAGE_BYTES", str(out.buffer_aligned_page_size())),
    ]
    for r in readers:
        r.defines = list(r.defines) + rd
    writer.defines = list(writer.defines) + common


def _attach_act_stream(descriptor, writer, core_ranges, device):
    """R1b: one channel over the whole split (one group: every core needs every A K-block), a fixed
    in-group injector (ACT_STREAM_INJECTOR in row-wise order) on the writer RISC's NoC. A flat split's
    core set (full rows + a partial row) is an irregular receiver set: one hardware multicast per
    rectangle."""
    config = ttnn.McastConfig(noc=ACT_STREAM_NOC, handshake=True)
    sender = ttnn.McastFixedSenderConfig(sender_index=ACT_STREAM_INJECTOR)
    mc = ttnn.Mcast(device, config, core_ranges, core_ranges.num_cores(), sender, ttnn.McastCoreOrder.RowMajor)
    mc.attach(descriptor, "act", [writer], 0)  # the only channel of an R1b program (R1a's are absent)


def _attach_operand_channels(
    descriptor, readers, grid, core_ranges, device, operand_mcast, act_offset=0, weight_offset=0
):
    """R1a: two line channels on the split rectangle at disjoint semaphore ids -- `act` along the lines
    of cores sharing a row region (logical i), `weight` along those sharing a column region (logical
    j). Groups are enumerated in line order and a line's cores in logical order, so the `Staggered`
    fixed sender (offset + group index, wrapped) is exactly the injector `injector_offsets` picked.
    Under "rect" (x = j, y = i) act lines are physical rows; under "rect_t" (x = i, y = j) columns.
    Plain R1: both channels attach as absent."""
    if not operand_mcast:
        for reader in readers:
            ttnn.attach_absent_mcast(reader, "act")
            ttnn.attach_absent_mcast(reader, "weight")
        return
    row_major, col_major = ttnn.McastCoreOrder.RowMajor, ttnn.McastCoreOrder.ColumnMajor
    act_order, weight_order = (row_major, col_major) if grid.placement == "rect" else (col_major, row_major)
    config = ttnn.McastConfig(handshake=True)
    staggered = ttnn.McastSenderPlacement.Staggered
    act_sender = ttnn.McastFixedSenderConfig(sender_index=act_offset, placement=staggered)
    weight_sender = ttnn.McastFixedSenderConfig(sender_index=weight_offset, placement=staggered)
    mc_act = ttnn.Mcast(device, config, core_ranges, grid.grid_n, act_sender, act_order)
    mc_weight = ttnn.Mcast(device, config, core_ranges, grid.grid_m, weight_sender, weight_order)
    mc_act.attach(descriptor, "act", readers, 0)
    mc_weight.attach(descriptor, "weight", readers, mc_act.next_semaphore_id())


# ---------------------------------------------------------------------------
# Regime R4 `dram_sharded` (Refinement 4)
# ---------------------------------------------------------------------------


class DramShardedLayoutError(ValueError):
    """The A / B shard layout is outside what regime R4 accepts (the op file maps it to a support refusal)."""


@dataclass(frozen=True)
class DramShardedTopology:
    """Who does what under R4. Index w runs over the banks that hold at least one valid B column."""

    a_cores: tuple  # A's shard cores in shard (= K, = round) order; core r holds K tiles [r*kw, (r+1)*kw)
    a_row_major: bool  # A shard orientation (the rotating sender order follows it)
    shard_k_tiles: int  # kw: K tiles per A shard
    bank_n_tiles: int  # sw: B columns (tiles) per bank shard
    primaries: tuple  # bank w's optimal worker (logical CoreCoord), w in bank-shard order: output shard w
    cores_per_bank: int  # C: compute cores sharing each bank's columns (the cores_per_bank split)
    workers: tuple  # every compute core, bank-major: workers[w * C + j] owns bank w's column slice j
    rect: object  # CoreRange: the workers' bounding rectangle = the A broadcast's receiver group

    @property
    def num_shards(self):
        return len(self.a_cores)

    @property
    def core_n_tiles(self):
        # The per-worker block width: one C-th of the bank's columns.
        return self.bank_n_tiles // self.cores_per_bank

    @property
    def out_in_place(self):
        # One core per bank owns its whole output shard and packs it in place; split banks stream it.
        return self.cores_per_bank == 1

    @property
    def worker_ranges(self):
        return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.workers])

    @property
    def primary_ranges(self):
        # Insertion order is kept by CoreRangeSet: output shard w lands on bank w's worker.
        return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.primaries])


def _core_key(c, row_major):
    return (c.y, c.x) if row_major else (c.x, c.y)


def dram_sharded_topology(a, b, swiglu=False) -> DramShardedTopology:
    """Validate the R4 operand layout and derive its topology (pure host, cheap)."""
    a_mc, b_mc = a.memory_config(), b.memory_config()
    if a_mc.buffer_type != ttnn.BufferType.L1 or a_mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        raise DramShardedLayoutError("dram_sharded needs A L1 WIDTH_SHARDED (read A where it lives)")
    if b_mc.buffer_type != ttnn.BufferType.DRAM or b_mc.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        raise DramShardedLayoutError("dram_sharded needs B DRAM WIDTH_SHARDED over the banks")
    a_shape, b_shape = list(a.shape), list(b.shape)
    m_rows = 1
    for d in a_shape[:-2]:
        m_rows *= d
    m_rows *= _round_up(a_shape[-2], TILE)  # every image is tile-padded on its own
    kt = _cdiv(a_shape[-1], TILE)
    nt = _cdiv(b_shape[-1], TILE)

    a_spec = a_mc.shard_spec
    if a_spec.shape[0] != m_rows or a_spec.shape[1] % TILE:
        raise DramShardedLayoutError(
            f"dram_sharded needs A width shards holding all {m_rows} padded rows and whole K tiles; got {a_spec.shape}"
        )
    kw = a_spec.shape[1] // TILE
    num_shards = _cdiv(kt, kw)
    a_row_major = a_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    a_cores = list(ttnn.corerange_to_cores(a_spec.grid, None, a_row_major))[:num_shards]
    # The broadcast's rotating senders take their turns in sorted core order: it must be the K order.
    if [_core_key(c, a_row_major) for c in a_cores] != sorted(_core_key(c, a_row_major) for c in a_cores):
        raise DramShardedLayoutError("dram_sharded needs A's shard order to follow its grid's core order")

    b_spec = b_mc.shard_spec
    b_rows = 1
    for d in b_shape[:-1]:
        b_rows *= d
    if b_spec.shape[0] != b_rows or b_spec.shape[1] % TILE:
        raise DramShardedLayoutError(f"dram_sharded needs B column shards of all {b_rows} rows; got {b_spec.shape}")
    sw = b_spec.shape[1] // TILE
    if swiglu and sw % 2:
        raise DramShardedLayoutError("dram_sharded swiglu needs whole (gate, up) tile pairs per bank shard")
    used_banks = _cdiv(nt, sw)
    bank_cores = list(ttnn.corerange_to_cores(b_spec.grid, None, b_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR))
    optimal = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(a.device(), DS_READ_NOC)
    if used_banks > len(bank_cores) or any(c.x >= len(optimal) for c in bank_cores[:used_banks]):
        raise DramShardedLayoutError("dram_sharded needs one B column shard per DRAM bank")
    primaries = [optimal[c.x] for c in bank_cores[:used_banks]]
    grid = a.device().compute_with_storage_grid_size()
    cores_per_bank = ds_cores_per_bank(sw, used_banks, grid.x * grid.y, swiglu)
    workers = _ds_bank_cores(primaries, cores_per_bank, grid)
    xs, ys = [c.x for c in workers], [c.y for c in workers]
    rect = ttnn.CoreRange(ttnn.CoreCoord(min(xs), min(ys)), ttnn.CoreCoord(max(xs), max(ys)))
    return DramShardedTopology(
        tuple(a_cores), a_row_major, kw, sw, tuple(primaries), cores_per_bank, tuple(workers), rect
    )


def ds_cores_per_bank(bank_n_tiles, used_banks, grid_cores, swiglu=False):
    """The cores_per_bank split C: C divides the bank width (each core a whole, equal column slice; whole
    (gate, up) pairs under swiglu) and C * banks fits the grid. Auto: the smallest slice >= DS_CORE_N_TILES_MIN."""
    granule = 2 if swiglu else 1
    c_max = max(1, grid_cores // used_banks)
    ok = [c for c in _divisors(bank_n_tiles) if c <= c_max and (bank_n_tiles // c) % granule == 0]
    if DS_CORES_PER_BANK is not None:
        if DS_CORES_PER_BANK not in ok:
            raise ValueError(f"DS_CORES_PER_BANK={DS_CORES_PER_BANK} does not split {bank_n_tiles} bank columns")
        return DS_CORES_PER_BANK
    wide = [c for c in ok if bank_n_tiles // c >= min(DS_CORE_N_TILES_MIN, bank_n_tiles)]
    return max(wide) if wide else 1


def _ds_bank_cores(primaries, cores_per_bank, grid):
    """workers[w * C + j]: bank w's optimal worker (j = 0), then its C - 1 nearest unclaimed cores -- the
    same row first (right before left), then the nearest rows -- so the slice readers sit by their bank."""
    taken = {(c.x, c.y) for c in primaries}
    workers = []
    for p in primaries:
        workers.append(p)
        cands = sorted(
            ((x, y) for x in range(grid.x) for y in range(grid.y) if (x, y) not in taken),
            key=lambda xy: (abs(xy[1] - p.y), abs(xy[0] - p.x), xy[0] < p.x, xy[1] < p.y),
        )
        for x, y in cands[: cores_per_bank - 1]:
            taken.add((x, y))
            workers.append(ttnn.CoreCoord(x, y))
    return workers


def dram_sharded_output_memory_config(a, b, swiglu=False):
    """The op-built output placement: L1 width-sharded, one shard per bank worker (shard w on bank w's
    worker), all padded rows, the bank's output columns."""
    topo = dram_sharded_topology(a, b, swiglu)
    m_rows = 1
    for d in list(a.shape)[:-2]:
        m_rows *= d
    m_rows *= _round_up(list(a.shape)[-2], TILE)
    out_sw = topo.bank_n_tiles // 2 if swiglu else topo.bank_n_tiles
    spec = ttnn.ShardSpec(topo.primary_ranges, [m_rows, out_sw * TILE], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec)


def choose_dram_sharded_blocking(geom: Geometry, topo: DramShardedTopology, budget: int) -> Blocking:
    """R4 extent knobs. block_n = the worker's column slice (the bank width / cores_per_bank); the K-walk is
    one A shard per round, split into `s` K-blocks (block_k = kw / s) only when needed (L1, or the
    16-bit-DEST chain cap); block_m = the largest divisor of the rows that fits (the output shard is
    filled in place, so block_m must tile it exactly). An A K-block must be one contiguous run of the
    sender's row-major shard, so a K split (s > 1) requires block_m == 1.
    Preference: fewest B re-reads (largest block_m), then the fewest A rounds (smallest s)."""
    mt, kt, kw, bn = geom.tensor_m_tiles, geom.tensor_k_tiles, topo.shard_k_tiles, topo.core_n_tiles
    ragged_k = kt % kw != 0
    s_all = [s for s in _divisors(kw) if s == 1 or not ragged_k]
    s_capped = [s for s in s_all if kw // s <= geom.block_k_cap] or [s_all[-1]]
    for bm in sorted(_divisors(mt), reverse=True):
        for s in s_capped:
            if s > 1 and bm > 1:
                continue
            bk = kw // s
            nkb = topo.num_shards * s
            if _footprint(geom, bm, bn, bk, partials=nkb > 1, out_in_place=topo.out_in_place) > budget:
                continue
            last = kt - (nkb - 1) * bk
            sb_h, sb_w = choose_subblock(bm, bn, geom.dest_limit)
            return Blocking(1, bm, bn, bk, nkb, last, sb_h, sb_w, geom.epilogue)
    raise RuntimeError(f"genify_matmul: even the minimum dram_sharded block does not fit the L1 CB budget of {budget}")


def ds_weight_depth(geom: Geometry, topo: DramShardedTopology, blk: Blocking, budget: int) -> int:
    """cb_weight_tiles depth (K-blocks) for the R4 B ring: DS_DEPTH_WEIGHT when the block fits L1 with
    it, else the solve's DEPTH_WEIGHT (the block was solved against that). The ring keeps
    min(DS_B_AHEAD, depth) K-blocks in flight."""
    for depth in range(max(DS_DEPTH_WEIGHT, DEPTH_WEIGHT), DEPTH_WEIGHT, -1):
        fp = _footprint(
            geom,
            blk.block_m_tiles,
            blk.block_n_tiles,
            blk.block_k_tiles,
            blk.use_partials,
            out_in_place=topo.out_in_place,
            depth_weight=depth,
        )
        if fp <= budget:
            return depth
    return DEPTH_WEIGHT


def plan_dram_sharded(a, b, out, fp32_dest_acc_en, epilogue="none", row=None, residual=None):
    """R4 blocking decision (geometry, topology, blocking). Exposed for tests / perf tooling."""
    geom = _geometry(a, b, out, fp32_dest_acc_en, epilogue, row, residual)
    topo = dram_sharded_topology(a, b, geom.swiglu)
    budget = l1_cb_budget(a.device(), [t for t in (a, b, out, row, residual) if t is not None])
    return geom, topo, choose_dram_sharded_blocking(geom, topo, budget)


def create_dram_sharded_program_descriptor(a, b, out, compute_config, epilogue="none", row=None, residual=None):
    """Regime R4. Kernel groups: the reader runs on every core the A broadcast touches -- the bank
    workers (ds_worker = 1: B stream + A rounds + epilogue operands) and the other cores of the
    broadcast (ds_worker = 0: A's shard cores that are not workers send their rounds; the rest of the
    receiver rectangle only lands the multicast and exits). Compute and writer run on the workers."""
    device = a.device()
    geom, topo, blk = plan_dram_sharded(
        a, b, out, bool(compute_config.fp32_dest_acc_en), epilogue=epilogue, row=row, residual=residual
    )
    budget = l1_cb_budget(device, [t for t in (a, b, out, row, residual) if t is not None])
    depth_weight = ds_weight_depth(geom, topo, blk, budget)
    b_ahead = max(1, min(DS_B_AHEAD, depth_weight))
    out_block_n_tiles = geom.out_block_n(blk.block_n_tiles)
    cb_row = CB_GATE_ROW if epilogue == "addcmul" else CB_BIAS_ROW
    nb_m = geom.tensor_m_tiles // blk.block_m_tiles
    k_blocks_per_round = blk.num_k_blocks // topo.num_shards

    worker_ranges = topo.worker_ranges
    worker_set = {(c.x, c.y) for c in topo.workers}
    others = sorted(
        {
            (x, y)
            for x in range(topo.rect.start.x, topo.rect.end.x + 1)
            for y in range(topo.rect.start.y, topo.rect.end.y + 1)
        }
        | {(c.x, c.y) for c in topo.a_cores}
    )
    others = [xy for xy in others if xy not in worker_set]
    other_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*xy), ttnn.CoreCoord(*xy)) for xy in others])
    act_ranges = worker_ranges.merge(other_ranges) if others else worker_ranges

    # ---- CBs ----
    cbs = [
        # The A K-block landing zone exists on every core the broadcast writes (same address everywhere).
        _cb(CB_ACT_TILES, a.dtype, geom.a_tile_bytes, DEPTH_ACT * blk.block_m_tiles * blk.block_k_tiles, act_ranges),
        _cb(
            CB_WEIGHT_TILES,
            b.dtype,
            geom.b_tile_bytes,
            depth_weight * blk.block_k_tiles * blk.block_n_tiles,
            worker_ranges,
        ),
    ]
    if topo.out_in_place:
        # Output packed in place into the worker's own L1 shard (block_n == the output shard width).
        cb_out = CB_OUTPUT_SHARD
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_SHARD, out))
    else:
        # cores_per_bank split: each worker streams its column slice; the writer lands it in the bank's
        # output shard (on the bank's primary worker) through the output's sharded accessor.
        cb_out = CB_OUTPUT_TILES
        cbs.append(
            _cb(CB_OUTPUT_TILES, out.dtype, geom.o_tile_bytes, DEPTH_OUT * blk.sb_h * out_block_n_tiles, worker_ranges)
        )
    if blk.use_partials:
        cbs.append(
            _cb(CB_PARTIALS, geom.p_dtype, geom.p_tile_bytes, blk.block_m_tiles * blk.block_n_tiles, worker_ranges)
        )
    if row is not None:
        cbs.append(_cb(cb_row, row.dtype, geom.row_tile_bytes, blk.block_n_tiles, worker_ranges))
    if residual is not None:
        cbs.append(
            _cb(
                CB_RESIDUAL_TILES,
                residual.dtype,
                geom.r_tile_bytes,
                DEPTH_RESIDUAL * blk.sb_h * blk.block_n_tiles,
                worker_ranges,
            )
        )

    row_t = row if row is not None else b
    res_t = residual if residual is not None else b
    reader_ct = _reader_ct(geom, blk, a, b, epilogue, cb_row, row_t, res_t)
    reader_accessor_ct = _reader_accessor_ct(a, b, row_t, res_t)
    act_on_writer = DS_ACT_ON_WRITER

    def writer_ct(worker):
        role = (
            (
                1,
                int(worker),
                CB_ACT_TILES,
                blk.block_k_tiles,
                blk.num_k_blocks,
                k_blocks_per_round,
                geom.a_tile_bytes,
                DEPTH_ACT,
            )
            if act_on_writer
            else _DS_WRITER_ACT_ABSENT
        )
        return _writer_ct(cb_out, blk, out_block_n_tiles, geom, out, out_in_place=topo.out_in_place, ds_act_role=role)

    compute_ct = _compute_ct(cb_out, blk, epilogue, cb_row, out, swiglu_natural_pairs=True)

    a_addr, b_addr, out_addr = a.buffer_address(), b.buffer_address(), out.buffer_address()
    row_addr = row.buffer_address() if row is not None else 0
    res_addr = residual.buffer_address() if residual is not None else 0
    out_div = 2 if geom.swiglu else 1
    sw, cw, cpb = topo.bank_n_tiles, topo.core_n_tiles, topo.cores_per_bank

    def worker_cols(i):
        # Worker i = bank i // C, slice i % C: its first B column and valid column count (0 when the
        # slice lies wholly in the last bank's padding).
        col0 = (i // cpb) * sw + (i % cpb) * cw
        return col0, max(0, min(cw, geom.tensor_n_tiles - col0))

    def reader_args(w):
        # w = None: a non-worker core of the broadcast (no B columns; it reads only nb_m and a_addr).
        col0, cols_valid = worker_cols(w) if w is not None else (0, 0)
        region = [0, 1, 0, geom.tensor_m_tiles, col0, cols_valid, 1, nb_m, 1]
        return [
            a_addr,
            b_addr,
            *region,
            geom.tensor_m_tiles,
            geom.tensor_k_tiles,
            geom.tensor_n_tiles,
            0,
            row_addr,
            res_addr,
            NO_OWN_KB,
        ]

    a_rt = [a_addr] if act_on_writer else []  # the writer's A-round role reads the A shard address
    worker_rt, other_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    writer_rt, other_writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for w, core in enumerate(topo.workers):
        worker_rt[core.x][core.y] = reader_args(w)
        col0, cols_valid = worker_cols(w)
        out_region = [0, 1, 0, geom.tensor_m_tiles, col0 // out_div, cols_valid // out_div, 1, nb_m, 1]
        writer_rt[core.x][core.y] = [out_addr, *out_region, geom.tensor_m_tiles, geom.out_n_tiles, *a_rt]
        compute_rt[core.x][core.y] = [nb_m, NO_OWN_KB]
    for x, y in others:
        other_rt[x][y] = reader_args(None)
        # A non-worker writer reads only nb_m (index 8) and the A shard address.
        other_writer_rt[x][y] = [0, 0, 1, 0, 0, 0, 0, 1, nb_m, 1, 0, 0, *a_rt]

    def reader(ranges, rt, worker):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "genify_matmul_reader.cpp"),
            core_ranges=ranges,
            compile_time_args=[
                *reader_ct,
                0,
                0,
                0,
                1,
                int(worker),
                k_blocks_per_round,
                DEPTH_ACT,
                int(not topo.out_in_place),  # ds_row_reads: the slice is not whole bank rows
                depth_weight,
                b_ahead,
                int(not act_on_writer),  # ds_reader_act
                *_BS_ROLE_ABSENT,
                *reader_accessor_ct,
            ],
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),  # NCRISC / NOC_0 (DS_READ_NOC)
        )

    def writer(ranges, rt, worker):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "genify_matmul_writer.cpp"),
            core_ranges=ranges,
            compile_time_args=writer_ct(worker),
            runtime_args=rt,
            config=ttnn.WriterConfigDescriptor(),  # BRISC / NOC_1 (DS_ACT_WRITER_NOC)
        )

    # The A rounds run on every core of the broadcast: on the writer (DS_ACT_ON_WRITER) or the reader.
    # Non-worker cores of the broadcast run only the kernel that carries them.
    readers = [reader(worker_ranges, worker_rt, True)]
    writers = [writer(worker_ranges, writer_rt, True)]
    if others:
        if act_on_writer:
            writers.append(writer(other_ranges, other_writer_rt, False))
        else:
            readers.append(reader(other_ranges, other_rt, False))
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_compute.cpp"),
        core_ranges=worker_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    if act_on_writer:
        _attach_act_broadcast(descriptor, writers, device, topo, DS_ACT_WRITER_NOC)
        for r in readers:
            ttnn.attach_absent_mcast(r, "act")
    else:
        _attach_act_broadcast(descriptor, readers, device, topo, DS_READ_NOC)
        for wr in writers:
            ttnn.attach_absent_mcast(wr, "act")
    for r in readers:
        ttnn.attach_absent_mcast(r, "weight")
    descriptor.kernels = [*readers, *writers, compute]
    return descriptor


def _attach_act_broadcast(descriptor, kernels, device, topo: DramShardedTopology, noc):
    """R4 A broadcast: one channel whose senders are A's shard cores taking turns in K order (rotating;
    a single shard is one explicit fixed sender). The helper's rotating mode needs a rectangular
    receiver group and its irregular multicast is capped at three rectangles, while the bank workers
    are scattered single cores, so the receiver group is the workers' bounding rectangle with the
    readiness handshake restricted to the workers: the other rectangle cores only land the payload in
    their (reserved, unused) cb_act_tiles slot. `kernels` carry the rounds (the readers or, under
    DS_ACT_ON_WRITER, the writers) on `noc`, their RISC's NoC."""
    order = ttnn.McastCoreOrder.RowMajor if topo.a_row_major else ttnn.McastCoreOrder.ColumnMajor
    if topo.num_shards > 1:
        senders = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in topo.a_cores])
        sender_config = ttnn.McastRotatingSenderConfig(senders, sender_order=order)
    else:
        sender_config = ttnn.McastExplicitFixedSenderConfig([topo.a_cores[0]])
    config = ttnn.McastConfig(noc=noc, handshake=True, handshake_cores=topo.worker_ranges)
    receivers = ttnn.CoreRangeSet([topo.rect])
    mc = ttnn.Mcast(device, config, receivers, receivers.num_cores(), sender_config, ttnn.McastCoreOrder.RowMajor)
    mc.attach(descriptor, "act", kernels, 0)


# ---------------------------------------------------------------------------
# Regime R3 `block_sharded` (Refinement 5)
# ---------------------------------------------------------------------------


class BlockShardedLayoutError(ValueError):
    """The A / output shard layout is outside what regime R3 accepts (the op file maps it to a support refusal)."""


@dataclass(frozen=True)
class BlockShardedTopology:
    """A's row-sharded grid (R3 block_sharded, R2 height_sharded): grid row r holds A rows [r*shard_m, ...),
    grid column c holds K tiles [c*shard_k, ...); output shard (c, r) holds the same rows x output columns
    [c*out_w, ...). Height-sharded A is the single-column case (cols = 1, shard_k = all of K): its shards
    sit on an arbitrary core list (`core_list`, shard order) instead of a rectangle."""

    grid: object  # CoreRange: A's (and the output's) shard grid (None when `core_list` is given)
    rows: int  # grid rows = M shards
    cols: int  # grid columns = K shards = K-blocks per walk
    shard_m_tiles: int
    shard_k_tiles: int
    out_w_tiles: int  # output shard width (tiles)
    core_list: tuple = ()  # R2: the shard cores in shard order (row r = core_list[r], cols = 1)

    def core(self, c, r):
        if self.core_list:
            return self.core_list[r * self.cols + c]
        return ttnn.CoreCoord(self.grid.start.x + c, self.grid.start.y + r)

    @property
    def core_set(self):
        if self.core_list:
            return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.core_list])
        return ttnn.CoreRangeSet([self.grid])

    @property
    def act_resident(self):
        # A single K shard: every K-block is the core's own resident shard, so there are no peer K-slices
        # and cb_act_tiles is not allocated (compute reads cb_act_shard directly).
        return self.cols == 1


@dataclass(frozen=True)
class ShardRows:
    """How a row shard maps onto (batch, image rows) -- batched B needs a block's rows inside one batch.
    Single B: one 'batch' of all M rows. Batched B: either the shard rows divide the image rows (every shard
    lies in one batch; `block_batches` = 1) or they are a multiple of them (a shard holds whole batches;
    `block_batches` = shard rows / image rows and the block is one whole image, nb_m = 1)."""

    block_batches: int
    rows_per_batch: int  # shard rows per batch slice (= the m-extent the blocks divide)

    def region(self, geom, r, shard_m):
        """(b0, batches_valid, row0, rows_valid) of row shard r."""
        g0 = r * shard_m
        tm = geom.tensor_m_tiles
        if geom.tensor_batches == 1:
            return 0, 1, g0, max(0, min(shard_m, tm - g0))
        b0 = g0 // tm
        if self.block_batches == 1:
            return b0, 1, g0 % tm, shard_m
        return b0, max(0, min(self.block_batches, geom.tensor_batches - b0)), 0, tm


def shard_rows(tensor_batches, tensor_m_tiles, shard_m):
    """The ShardRows of a `shard_m`-tile row shard (Geometry's tensor_batches / tensor_m_tiles)."""
    if tensor_batches == 1 or tensor_m_tiles % shard_m == 0:
        return ShardRows(1, shard_m)
    if shard_m % tensor_m_tiles == 0:
        return ShardRows(shard_m // tensor_m_tiles, tensor_m_tiles)
    raise BlockShardedLayoutError(
        f"a batched weight needs A's shard rows ({shard_m} tiles) to divide or be a multiple of the "
        f"image rows ({tensor_m_tiles} tiles), so no block straddles two batches"
    )


def _padded_m_rows(a):
    m_rows = 1
    for d in list(a.shape)[:-2]:
        m_rows *= d
    return m_rows * _round_up(list(a.shape)[-2], TILE)  # every image is tile-padded on its own


def _block_shard_grid(spec):
    box = spec.grid.bounding_box()
    if spec.grid.num_cores() != box.grid_size().x * box.grid_size().y:
        raise BlockShardedLayoutError("block_sharded needs a rectangular shard grid")
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise BlockShardedLayoutError("block_sharded needs ROW_MAJOR shard orientation (K over grid columns)")
    return box


def block_sharded_output_memory_config(a, b, swiglu=False):
    """The default output placement when none is requested: L1 block-sharded on A's grid, A's shard
    height, the output's columns split over the grid columns."""
    spec = a.memory_config().shard_spec
    box = _block_shard_grid(spec)
    out_n_tiles = _cdiv(list(b.shape)[-1] // (2 if swiglu else 1), TILE)
    width = _cdiv(out_n_tiles, box.grid_size().x) * TILE
    out_spec = ttnn.ShardSpec(spec.grid, [spec.shape[0], width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, out_spec)


def block_sharded_topology(a, out_mc, swiglu=False, out_n=None) -> BlockShardedTopology:
    """Validate the R3 layout (A and the requested output placement) and derive its topology."""
    a_mc = a.memory_config()
    if a_mc.buffer_type != ttnn.BufferType.L1 or a_mc.memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise BlockShardedLayoutError("block_sharded needs A L1 BLOCK_SHARDED (read A where it lives)")
    spec = a_mc.shard_spec
    box = _block_shard_grid(spec)
    if spec.shape[0] % TILE or spec.shape[1] % TILE:
        raise BlockShardedLayoutError(f"block_sharded needs whole-tile A shards; got {spec.shape}")
    shard_m, shard_k = spec.shape[0] // TILE, spec.shape[1] // TILE
    mt = _padded_m_rows(a) // TILE
    kt = _cdiv(list(a.shape)[-1], TILE)
    rows, cols = box.grid_size().y, box.grid_size().x
    if _cdiv(mt, shard_m) != rows or _cdiv(kt, shard_k) != cols:
        raise BlockShardedLayoutError(
            f"block_sharded needs A's shard grid to be exactly (K shards) x (M shards) = "
            f"{_cdiv(kt, shard_k)} x {_cdiv(mt, shard_m)}; got {cols} x {rows}"
        )
    if out_mc.buffer_type != ttnn.BufferType.L1 or out_mc.memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise BlockShardedLayoutError(f"block_sharded writes an L1 BLOCK_SHARDED output; got {out_mc}")
    o_spec = out_mc.shard_spec
    if o_spec is None or o_spec.grid != spec.grid or o_spec.shape[0] != spec.shape[0]:
        raise BlockShardedLayoutError(
            "block_sharded writes the output on A's shard grid with A's shard height; " f"got {o_spec}"
        )
    if o_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR or o_spec.shape[1] % TILE:
        raise BlockShardedLayoutError(f"block_sharded needs a ROW_MAJOR whole-tile output shard; got {o_spec}")
    out_w = o_spec.shape[1] // TILE
    if out_n is not None and _cdiv(_cdiv(out_n, TILE), out_w) > cols:
        raise BlockShardedLayoutError(f"block_sharded output shard width {o_spec.shape[1]} leaves columns unowned")
    return BlockShardedTopology(box, rows, cols, shard_m, shard_k, out_w)


def height_sharded_output_memory_config(a, b, swiglu=False):
    """The default R2 output placement when none is requested: L1 height-sharded on A's cores, A's shard
    height, the whole output width."""
    spec = a.memory_config().shard_spec
    out_n_tiles = _cdiv(list(b.shape)[-1] // (2 if swiglu else 1), TILE)
    out_spec = ttnn.ShardSpec(spec.grid, [spec.shape[0], out_n_tiles * TILE], spec.orientation)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_spec)


def height_sharded_topology(a, out_mc, swiglu=False, out_n=None) -> BlockShardedTopology:
    """Validate the R2 layout (A L1 height-sharded with all of K per shard, the output L1 height-sharded on
    A's cores with A's shard height and the whole output width) and derive its topology: R3's single-column
    case on A's shard cores in shard order."""
    a_mc = a.memory_config()
    if a_mc.buffer_type != ttnn.BufferType.L1 or a_mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        raise BlockShardedLayoutError("height_sharded needs A L1 HEIGHT_SHARDED (read A where it lives)")
    spec = a_mc.shard_spec
    kt = _cdiv(list(a.shape)[-1], TILE)
    if spec.shape[0] % TILE or spec.shape[1] != kt * TILE:
        raise BlockShardedLayoutError(f"height_sharded needs whole-tile A shards holding all of K; got {spec.shape}")
    shard_m = spec.shape[0] // TILE
    rows = _cdiv(_padded_m_rows(a) // TILE, shard_m)
    row_wise = spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = ttnn.corerange_to_cores(spec.grid, None, row_wise)
    if rows > len(cores):
        raise BlockShardedLayoutError(f"height_sharded: {rows} A shards on {len(cores)} cores")
    if out_mc.buffer_type != ttnn.BufferType.L1 or out_mc.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        raise BlockShardedLayoutError(f"height_sharded writes an L1 HEIGHT_SHARDED output; got {out_mc}")
    o_spec = out_mc.shard_spec
    if (
        o_spec is None
        or o_spec.grid != spec.grid
        or o_spec.shape[0] != spec.shape[0]
        or o_spec.orientation != spec.orientation
    ):
        raise BlockShardedLayoutError(
            "height_sharded writes the output on A's shard cores with A's shard height and orientation; "
            f"got {o_spec}"
        )
    out_w = o_spec.shape[1] // TILE
    if o_spec.shape[1] % TILE or (out_n is not None and out_w != _cdiv(out_n, TILE)):
        raise BlockShardedLayoutError(f"height_sharded needs a whole-width output shard; got {o_spec.shape}")
    return BlockShardedTopology(None, rows, 1, shard_m, kt, out_w, tuple(cores[:rows]))


def choose_block_sharded_blocking(geom: Geometry, topo: BlockShardedTopology, budget: int, rows=None):
    """R3 / R2 extent knobs -> (Blocking, out_in_place). block_k = the A shard width (one peer shard per
    K-block, num_k_blocks = grid columns; R2: all of K). Preferred: block_n = the output shard's B columns
    (output packed in place) with block_m the largest divisor of the batch slice's shard rows that fits.
    Fallback: the whole slice rows with the N range split (output streamed through cb_output_tiles); the own
    shard is re-pushed per N-block, which wraps cb_act_shard only when nb_m == 1, so the two splits never
    mix. `rows` (ShardRows) carries a batched weight's block_batches; with whole batches per shard
    (block_batches > 1) the shard is walked image by image, so block_m is the whole image (nb_m = 1).
    The weight CB holds a second K-block only when the walk has more than one (B for the next block)."""
    rows = rows or ShardRows(1, topo.shard_m_tiles)
    bk, kt = topo.shard_k_tiles, geom.tensor_k_tiles
    nkb = _cdiv(kt, bk)
    last = kt - (nkb - 1) * bk
    partials = nkb > 1
    bn_full = topo.out_w_tiles * (2 if geom.swiglu else 1)
    act_cb = not topo.act_resident
    m_ext = rows.rows_per_batch

    def blocking(bm, bn):
        sb_h, sb_w = choose_subblock(bm, bn, geom.dest_limit)
        return Blocking(rows.block_batches, bm, bn, bk, nkb, last, sb_h, sb_w, geom.epilogue)

    def fits(bm, bn, in_place, walk_blocks):
        depth_w = _weight_depth(walk_blocks * nkb * rows.block_batches)
        f = _footprint(geom, bm, bn, bk, partials=partials, out_in_place=in_place, depth_weight=depth_w, act_cb=act_cb)
        return f <= budget

    bm_options = sorted(_divisors(m_ext), reverse=True) if rows.block_batches == 1 else [m_ext]
    for bm in bm_options:
        if fits(bm, bn_full, True, m_ext // bm):
            return blocking(bm, bn_full), True
    q = geom.q_n_min
    for nb_n in range(2, _cdiv(bn_full, q) + 1):
        bn = _round_up(_cdiv(bn_full, nb_n), q)
        if fits(m_ext, bn, False, nb_n):
            return blocking(m_ext, bn), False
    raise RuntimeError(f"genify_matmul: even the minimum block_sharded block does not fit the L1 CB budget of {budget}")


def sharded_b_chunks(geom: Geometry, blk: Blocking, round_kblocks=1):
    """R2a / R3a: (chunks, K rows per chunk) of one B round (`round_kblocks` K-blocks of contiguous K rows)."""
    rows = round_kblocks * blk.block_k_tiles
    round_bytes = rows * blk.block_n_tiles * geom.b_tile_bytes
    chunks = max(1, min(SHARDED_B_MAX_CHUNKS, rows, round_bytes // SHARDED_B_CHUNK_BYTES))
    chunk_k = _cdiv(rows, chunks)
    return _cdiv(rows, chunk_k), chunk_k


def _sharded_b_depth(round_kblocks, walk_k_blocks):
    """cb_weight_tiles depth (K-blocks) for B rounds of `round_kblocks`: two rounds in flight, or one when the
    core's whole walk is a single round (round_kblocks == 1 is `_weight_depth`)."""
    return round_kblocks if walk_k_blocks <= round_kblocks else DEPTH_WEIGHT * round_kblocks


def sharded_b_round(geom: Geometry, blk: Blocking, walk_k_blocks, budget, in_place, act_cb):
    """R2a / R3a: K-blocks per B round -- the smallest divisor g of num_k_blocks whose round carries
    SHARDED_B_ROUND_BYTES (else all of them), shrunk while its 2g-deep weight CB does not fit the budget."""
    kblock_bytes = blk.block_k_tiles * blk.block_n_tiles * geom.b_tile_bytes
    divs = sorted(_divisors(blk.num_k_blocks))
    g = next((d for d in divs if d * kblock_bytes >= SHARDED_B_ROUND_BYTES), divs[-1])
    for d in reversed([d for d in divs if d <= g]):
        f = _footprint(
            geom,
            blk.block_m_tiles,
            blk.block_n_tiles,
            blk.block_k_tiles,
            partials=blk.use_partials,
            out_in_place=in_place,
            depth_weight=_sharded_b_depth(d, walk_k_blocks),
            act_cb=act_cb,
        )
        if d == 1 or f <= budget:
            return d
    return 1


def _sharded_b_channel(device, geom: Geometry, topo: BlockShardedTopology):
    """R2a / R3a: the weight channel (an unattached ttnn.Mcast), or None where the per-core reads stay: the
    knob is off, B is batched (it varies along the rows), a group is a single core, R3 saves less than
    BS_B_MCAST_MIN_SAVED_BYTES of B re-reads, or the receiver set is not multicastable (an R2 core list of
    more than three rectangles)."""
    if not SHARDED_B_MCAST or geom.tensor_batches != 1 or topo.rows < 2:
        return None
    saved = (topo.rows - 1) * geom.tensor_k_tiles * geom.tensor_n_tiles * geom.b_tile_bytes
    if topo.cols > 1 and saved < BS_B_MCAST_MIN_SAVED_BYTES:
        return None
    staggered = ttnn.McastFixedSenderConfig(
        sender_index=SHARDED_B_INJECTOR, placement=ttnn.McastSenderPlacement.Staggered
    )
    # Groups = grid columns (R3; R2 is a single column of `rows` shard cores).
    order = ttnn.McastCoreOrder.ColumnMajor if topo.core_list == () else ttnn.McastCoreOrder.RowMajor
    try:
        return ttnn.Mcast(device, ttnn.McastConfig(handshake=True), topo.core_set, topo.rows, staggered, order)
    except RuntimeError:
        return None


def _weight_depth(walk_k_blocks):
    """cb_weight_tiles depth in K-blocks for the R2 / R3 walk: DEPTH_WEIGHT (prefetch the next K-block)
    unless the core's whole walk is a single K-block, whose second slot could never fill."""
    return DEPTH_WEIGHT if walk_k_blocks > 1 else 1


def plan_block_sharded(
    a, b, out, fp32_dest_acc_en, epilogue="none", row=None, residual=None, placement="block_sharded"
):
    """R3 / R2 blocking decision (geometry, topology, shard-row mapping, blocking, out_in_place). Exposed for
    tests / perf tooling. placement "height_sharded" is R2: R3's single-K-shard case on A's shard cores."""
    geom = _geometry(a, b, out, fp32_dest_acc_en, epilogue, row, residual)
    topology = height_sharded_topology if placement == "height_sharded" else block_sharded_topology
    topo = topology(a, out.memory_config(), geom.swiglu)
    rows = shard_rows(geom.tensor_batches, geom.tensor_m_tiles, topo.shard_m_tiles)
    budget = l1_cb_budget(a.device(), [t for t in (a, b, out, row, residual) if t is not None])
    blk, in_place = choose_block_sharded_blocking(geom, topo, budget, rows)
    return geom, topo, rows, blk, in_place


def create_block_sharded_program_descriptor(
    a, b, out, compute_config, epilogue="none", row=None, residual=None, placement="block_sharded"
):
    """Regime R3 (block_sharded) and R2 (height_sharded). One reader / compute / writer per shard core;
    under R3 every core is both a sender (its own K turn) and a receiver (its row peers' turns) of its grid
    row's A exchange. R2 (one K shard) has no exchange: each core's resident shard is its whole A."""
    device = a.device()
    geom, topo, rows, blk, in_place = plan_block_sharded(
        a,
        b,
        out,
        bool(compute_config.fp32_dest_acc_en),
        epilogue=epilogue,
        row=row,
        residual=residual,
        placement=placement,
    )
    grid = topo.core_set
    out_block_n_tiles = geom.out_block_n(blk.block_n_tiles)
    cb_row = CB_GATE_ROW if epilogue == "addcmul" else CB_BIAS_ROW
    cb_out = CB_OUTPUT_SHARD if in_place else CB_OUTPUT_TILES
    row_mcast = BS_ROW_MCAST and topo.cols > 1
    mc_weight = _sharded_b_channel(device, geom, topo)
    out_div = 2 if geom.swiglu else 1
    bn_full = topo.out_w_tiles * out_div  # B columns per grid column
    # Every core of a row walks the same nominal block sequence (lockstep CB pointers for the multicast).
    nb_m = rows.rows_per_batch // blk.block_m_tiles
    nb_n = _cdiv(bn_full, blk.block_n_tiles)
    # A single K shard (R2): compute consumes the resident shard directly; no cb_act_tiles exists.
    cb_act_compute = CB_ACT_SHARD if topo.act_resident else CB_ACT_TILES
    walk_k_blocks = nb_m * nb_n * blk.num_k_blocks * blk.block_batches
    b_round, b_chunks, b_chunk_k = 1, 1, 1
    if mc_weight is not None:
        budget = l1_cb_budget(device, [t for t in (a, b, out, row, residual) if t is not None])
        b_round = sharded_b_round(geom, blk, walk_k_blocks, budget, in_place, act_cb=not topo.act_resident)
        b_chunks, b_chunk_k = sharded_b_chunks(geom, blk, b_round)
    depth_weight = _sharded_b_depth(b_round, walk_k_blocks)

    # ---- CBs ----
    # The own K-slice: backed on A's resident shard (zero-copy).
    act_shard_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_ACT_SHARD, a)
    act_shard_cb.core_ranges = grid
    cbs = [
        act_shard_cb,
        _cb(CB_WEIGHT_TILES, b.dtype, geom.b_tile_bytes, depth_weight * blk.block_k_tiles * blk.block_n_tiles, grid),
    ]
    if not topo.act_resident:
        cbs.append(
            _cb(CB_ACT_TILES, a.dtype, geom.a_tile_bytes, DEPTH_ACT * blk.block_m_tiles * blk.block_k_tiles, grid)
        )
    if in_place:
        # Output packed in place into the core's own L1 shard (block_n == the shard's B columns).
        out_shard_cb = ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT_SHARD, out)
        out_shard_cb.core_ranges = grid
        cbs.append(out_shard_cb)
    else:
        cbs.append(_cb(CB_OUTPUT_TILES, out.dtype, geom.o_tile_bytes, DEPTH_OUT * blk.sb_h * out_block_n_tiles, grid))
    if blk.use_partials:
        cbs.append(_cb(CB_PARTIALS, geom.p_dtype, geom.p_tile_bytes, blk.block_m_tiles * blk.block_n_tiles, grid))
    if row is not None:
        cbs.append(_cb(cb_row, row.dtype, geom.row_tile_bytes, blk.block_n_tiles, grid))
    if residual is not None:
        cbs.append(
            _cb(
                CB_RESIDUAL_TILES,
                residual.dtype,
                geom.r_tile_bytes,
                DEPTH_RESIDUAL * blk.sb_h * blk.block_n_tiles,
                grid,
            )
        )

    row_t = row if row is not None else b
    res_t = residual if residual is not None else b
    reader_ct = [
        *_reader_ct(geom, blk, a, b, epilogue, cb_row, row_t, res_t),
        0,  # operand_mcast
        0,  # a_sends
        0,  # b_sends
        *_DS_ROLE_ABSENT,
        1,  # block_sharded (R3 / R2 walk)
        int(row_mcast),
        CB_ACT_SHARD,
        int(mc_weight is not None),
        b_chunks,
        b_chunk_k,
        b_round,
        *_reader_accessor_ct(a, b, row_t, res_t),
    ]
    writer_ct = _writer_ct(cb_out, blk, out_block_n_tiles, geom, out, out_in_place=in_place)
    compute_ct = _compute_ct(cb_out, blk, epilogue, cb_row, out, swiglu_natural_pairs=False, own_round_slot=row_mcast)
    compute_ct[0] = cb_act_compute

    a_addr, b_addr, out_addr = a.buffer_address(), b.buffer_address(), out.buffer_address()
    row_addr = row.buffer_address() if row is not None else 0
    res_addr = residual.buffer_address() if residual is not None else 0
    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for r in range(topo.rows):
        b0, batches_valid, row0, rows_valid = rows.region(geom, r, topo.shard_m_tiles)
        for c in range(topo.cols):
            core = topo.core(c, r)
            col0 = c * bn_full
            cols_valid = max(0, min(bn_full, geom.tensor_n_tiles - col0))
            region = [b0, batches_valid, row0, rows_valid, col0, cols_valid, 1, nb_m, nb_n]
            reader_rt[core.x][core.y] = [
                a_addr,
                b_addr,
                *region,
                geom.tensor_m_tiles,
                geom.tensor_k_tiles,
                geom.tensor_n_tiles,
                geom.weight_batch_stride,
                row_addr,
                res_addr,
                c,  # own_kb
            ]
            out_region = [b0, batches_valid, row0, rows_valid, col0 // out_div, cols_valid // out_div, 1, nb_m, nb_n]
            writer_rt[core.x][core.y] = [out_addr, *out_region, geom.tensor_m_tiles, geom.out_n_tiles]
            compute_rt[core.x][core.y] = [nb_m * nb_n, c]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_reader.cpp"),
        core_ranges=grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),  # NCRISC / NOC_0
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_writer.cpp"),
        core_ranges=grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "genify_matmul_compute.cpp"),
        core_ranges=grid,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[], cbs=cbs)
    if row_mcast:
        # One channel group per grid row (RowMajor groups of `cols` cores); the senders rotate through
        # the row in column (= K shard) order, so round kb's sender is the core owning K shard kb.
        mc = ttnn.Mcast(
            device,
            ttnn.McastConfig(handshake=True),
            grid,
            topo.cols,
            ttnn.McastRotatingSenderConfig(),
            ttnn.McastCoreOrder.RowMajor,
        )
        mc.attach(descriptor, "act", [reader], 0)
        next_sem = mc.next_semaphore_id()
    else:
        ttnn.attach_absent_mcast(reader, "act")
        next_sem = 0
    if mc_weight is not None:
        # R2a / R3a: B down each grid column (R2: to every shard core) from the group's fixed injector.
        mc_weight.attach(descriptor, "weight", [reader], next_sem)
    else:
        ttnn.attach_absent_mcast(reader, "weight")
    ttnn.attach_absent_mcast(writer, "act")  # the writer's R4 A-round role is absent
    descriptor.kernels = [reader, writer, compute]
    return descriptor
