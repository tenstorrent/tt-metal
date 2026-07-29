# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""``tilize`` — host planner + ProgramDescriptor.

Two dataflow paths (see ``op_design.md`` "Dataflow Strategy"):

* **Path A/C — generic** (``path="generic"``).  RM sticks are read through a
  ``TensorAccessor`` into a tile-page input CB, tilized, and written back as
  whole TILE pages through the output ``TensorAccessor``.  The work unit is a
  *chunk-block* = 32 rows x ``chunk_wt`` tile-columns; each core owns a 2D
  rectangle (contiguous tile-row range x contiguous column-chunk range), so the
  split degenerates to pure-height when height fills the grid and to pure-width
  when ``nt_h == 1``.  Covers interleaved I/O and every
  interleaved<->sharded / cross-spec-sharded combination.  When the input is
  ROW_MAJOR-sharded with ``pages_per_row > 1`` the reader switches to a raw
  strided read (the helper hard-codes one page per logical row).

* **Path B — aliased, zero-copy** (``path="alias"``).  Same-spec L1-sharded in
  and out: both CBs are built with ``cb_descriptor_from_sharded_tensor`` so the
  CB base address *is* the shard base address.  Zero NoC traffic on both sides;
  the reader degenerates to one ``cb_push_back`` and the writer to one
  ``cb_wait_front``/``cb_pop_front``.

Only two CBs in either path — tilize is a single-phase compute with no
intermediate.  Per-core CB L1 is ``depth * chunk_wt * (tile_in + tile_out)``
with ``chunk_wt <= WT_CHUNK_MAX``, i.e. bounded by a constant in ``W``.

``depth`` itself is gated (Refinement 1, lever C16): ``use_double_buffer=None``
— the public default — asks the planner for depth-2 only in the regime where it
was *measured* to pay (``depth2_pays``); ``True``/``False`` force it. See
``A0_KNEE_CORES`` and ``BANDWIDTH_KNEE_CORES`` for the measurements behind both
gates.
"""

from __future__ import annotations

import os
from math import gcd, prod
from pathlib import Path

import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HW = 32

# Caps the reader transaction at 1024 B (bf16) / 2048 B (fp32) and bounds the
# per-core CB footprint independently of W.
WT_CHUNK_MAX = 16
# Conservative literal: there is no `device.l1_size_per_core()` Python binding
# on this build. Both CBs combined.
L1_CB_BUDGET_BYTES = 131072

# Fast-tilize LLK limit (`can_use_fast_tilize`: block_width_tiles < 256).
MAX_BLOCK_WIDTH_TILES = 256

# --- A0 active-core criterion: min(grid, total_tiles, A0_KNEE_CORES) ----------
#
# master.md Part 2 A0 states the criterion as `active == min(grid, total_tiles,
# bandwidth_knee)`, and `examples/dram_saturation/report.md` measures that knee at
# ~16 cores @ 190.9 GB/s for a *large-transaction* DRAM copy (16 -> 64 buys
# +1.5 %). tilize's knee was MEASURED for tilize's own transfer shapes
# (probes/probe_009.py + probe_010.py, Refinement 1) and it is **the whole grid**:
#
#   d_tall_narrow [1,1,2048,32], forced core cap -> device ns (median of 5x10)
#     64c 3 623 | 32c 5 186 | 16c 8 580 | 8c 14 780 | 4c 27 950 | 1c 107 561
#
# i.e. latency is ~linear in tiles-per-core: capping at 16 cores is 2.4x SLOWER,
# not ~2x faster. Two measured reasons the bandwidth knee never binds here:
#   1. a W=32 bf16 ROW_MAJOR input has 64 B DRAM pages, so the reader issues 64 B
#      transactions. The NoC model puts 64 B interleaved DRAM reads at
#      0.68-1.41 B/cyc/core, i.e. 45-90 GB/s aggregate over 64 cores -- the
#      190.9 GB/s knee is UNREACHABLE for this shape at any core count. The op is
#      read-transaction-rate bound, not DRAM-bandwidth bound.
#   2. the sync/dispatch floor scales with BLOCKS PER CORE, not with core count
#      (sync_only: 64c/1blk 1 202 ns, 16c/4blk 3 079 ns, 4c/16blk 10 677 ns
#      ~= 590 + 612*blocks), so shedding cores *adds* sync cost.
#
# Keep the term in the formula (it is A0's criterion, and a future shape family
# with big transactions could re-introduce a real knee) but set it above any
# current compute grid => identity. Changing this constant requires re-running
# probes/probe_009.py.
A0_KNEE_CORES = 64

# --- C16 depth-2 default gate (master.md C16 "but only when it pays") ---------
# Depth-2 buys read/write overlap across a block boundary. Measured
# (probes/probe_010.py + the paired in-run `x_*_depth2` bench rows, 7 rounds,
# CV <= 1.2 %), it pays in exactly two situations and is dead L1 otherwise:
#
#  1. **Below the DRAM bandwidth-saturation knee** the binding resource is the
#     core's OWN NoC issue rate, so overlapping its reader and writer is a large
#     win: c_single_core depth1/depth2 = **1.321**, x_wide_short_1core = **1.360**.
#  2. **At or above the knee** DRAM aggregate bandwidth is the binding resource
#     and both depths already reach it -- but each block boundary still costs one
#     un-overlapped fill/drain, and those add up with the block count:
#       blk/core  depth1/depth2   verdict
#         1       0.995 - 1.010   depth-2 structurally inert (nothing to overlap)
#         4       0.998 / 1.005   free  (a_square, e_square_bf8b_out)
#         8       1.019 - 1.028   costs ~2 %  (e_square_fp32, e_square_fp32_to_bf16,
#                                 g_sharded_to_dram -- 3 independent regimes, same sign)
#     so depth-1 is the default only up to DEPTH1_MAX_BLOCKS_PER_CORE boundaries.
#
# NB this is narrower than the refinement's proposal ("default off once the op is
# DRAM-saturated with large per-core work"): measurement says *large* per-core
# work is precisely where the residual overlap still pays. 4 is measured free and
# 8 measured costly; 5-7 is unmeasured, so the threshold sits at the conservative
# end of that gap.
BANDWIDTH_KNEE_CORES = 16
MIN_BLOCKS_FOR_DEPTH2 = 2
DEPTH1_MAX_BLOCKS_PER_CORE = 4
# Sweep hook: probes set this to force a core cap while re-measuring the knee.
# None => A0_KNEE_CORES decides. Never set in production.
CORE_CAP_OVERRIDE = None
# Sweep hook: force a smaller chunk width, i.e. MORE chunk-blocks per core at the
# same core count. Refinement 2b uses it to sweep the read/write-overlap gate
# (`chunk_blocks_pays`) without changing the shape. Never set in production.
CHUNK_CAP_OVERRIDE = None
# Sweep hook: Refinement 2b's read-rotation modulus. None => TILE_HW (the row-loop
# period). The alternative worth sweeping is NUM_DRAM_BANKS, which makes the per-core
# STARTING bank perfectly uniform instead of uniform-mod-32. Never set in production.
STAGGER_MOD_OVERRIDE = None

CB_RM_INPUT = 0
# Refinement 2b staging buffer -- one `piece_bytes` window, allocated ONLY on the
# fan-in redistribution path (dead L1 otherwise).
CB_STAGE = 1
CB_TILED_OUTPUT = 16

_HEIGHT_SHARDED = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH_SHARDED = ttnn.TensorMemoryLayout.WIDTH_SHARDED

# --- Refinement 1c: the two sub-one-packet read-path levers -------------------
#
# `d_tall_narrow` [1,1,2048,32] reads 32 x 64 B sticks per block (a W=32 bf16
# ROW_MAJOR row IS one 64 B DRAM page), 1 block per core. Refinement 1 priced the
# 3 609 ns by subtraction: 764 ns launch + 437 ns address-gen + 1 504 ns read
# issue/service/barrier + 931 ns tilize LLK. Both levers here attack the middle
# two terms, i.e. the transaction *rate* -- NOT the transaction size, which is
# pinned at 64 B: consecutive DRAM pages of a W=32 row-major tensor land on
# DIFFERENT banks (round-robin), so rows cannot be coalesced without a
# permutation the tilize LLK cannot consume.
#
#   B13 (`stateful_read`) -- arm the NoC command buffer once per bank and then
#       write only the varying addresses. Every row of a block has the same
#       transfer size, and for an interleaved tensor pages p and p+num_banks share
#       a bank one aligned page apart, so one arm covers ~32/12 rows and their
#       source addresses are a running increment. Implemented inside
#       `dataflow_kernel_lib::read_stick_rows_for_tilize` (StickReadMode::Stateful)
#       with a watcher-build ASSERT that re-derives every address through the
#       accessor. Kernel-side it self-disables when the accessor is not
#       interleaved or when there are < 2 rows per bank, so this flag is "offer
#       the lever", not "force it".
#
#   C7 (`split_read`) -- give BRISC half of each block's stick reads. It is
#       otherwise parked in cb_wait_front for the entire read window.
#       Needs depth == 1: BRISC must write into the window NCRISC reserved, and it
#       may not touch the CB pointers (single producer), so the window has to be
#       at the CB base address every block -- which is exactly what depth-1 gives.
#
# Both are gated per plan and both have an env counterfactual switch so the
# Mode-C ledger rows are re-measurable (`_bench_tilize.py` x_* rows).
#
# Both payoff gates are MEASURED, and both turned out to be **read-transaction
# size** gates -- i.e. each lever pays only in the sub-one-packet regime this
# refinement was scoped to, and costs real time outside it. Measured ratio
# lever/none at 64 cores, 1 block per core (7 rounds x 10 launches, CV <= 2.1 %):
#
#   read bytes | shape             | B13   | C7    | verdict
#   -----------|-------------------|-------|-------|------------------------------
#         64 B | [1,1,2048,32]     | 0.980 | 0.956 | both pay  (together 0.948)
#        128 B | [1,1,32,4096]     | 0.968 | 1.018 | B13 only
#        256 B | [1,1,32,8192]     | 1.023 | 1.056 | neither
#        512 B | [1,1,32,16384]    | 1.177 | 1.146 | neither (worst case)
#       1024 B | [1,1,2048,512]->B | 1.057 | 1.045 | neither
#
# and at 64 B with 4 blocks per core: B13 **0.950**, C7 **1.145**.
#
# Why each turns over:
#
#  * B13 forces **bank-major** issue order, because `set_state` pins the NoC
#    coordinate and only pages num_banks apart share a bank. That is free when the
#    read is one 64 B packet (the saved per-read command programming and address
#    arithmetic dominate), but from 256 B on, queueing 2-3 consecutive same-bank
#    reads costs more DRAM-endpoint serialization than the issue saving buys.
#  * C7 costs the read/write overlap across the block boundary (BRISC's read of
#    block i+1 queues behind its write of block i) -- free only at 1 block per
#    core -- and it doubles the number of read issuers per core, which is a win
#    only while each core reads its OWN rows. Every regime above 64 B here has
#    `nt_h == 1`, i.e. all 64 cores read the same 32 source pages, so a second
#    issuer per core just deepens an existing DRAM hot spot.
#
# The two are also mutually exclusive (measured -- see the `stateful_read`
# assignment in `_plan_generic`), so each regime ships exactly one:
#
#   read bytes | blocks/core | ships   | vs no lever
#   -----------|-------------|---------|-------------
#         64 B |           1 | C7      | 0.948 - 0.956
#         64 B |        >= 2 | B13     | 0.950 - 0.957
#        128 B |         any | B13     | 0.950 - 0.968
#     >= 256 B |         any | neither | 1.0 (both measured negative)
STATEFUL_READ_MAX_ROW_BYTES = 128
SPLIT_READ_MAX_ROW_BYTES = 64
SPLIT_READ_MAX_BLOCKS_PER_CORE = 1
SEM_SPLIT_RESERVE = 0
SEM_SPLIT_DONE = 1

# --- Refinement 2: the three remaining per-transaction levers ------------------
#
# All three are `master.md` Part 2 items that Phase 0 pre-classified `deferred`.
# Refinement 1c already refuted B13/C7 on `b_wide_short` (512 B reads), so this
# entry's set is B8 / B10 / A3.
#
#   B8 (`prefetch_blocks == 2`) -- trid double-issue on the READ path. The reader
#       tags each chunk-block's 32 stick reads with one of two transaction ids and
#       barriers only on the PREVIOUS id, so the next block's reads are already in
#       flight while the current block is drained. Structurally requires
#       `blocks_per_core >= 2` (with one block there is no next block to keep in
#       flight) and a THIRD CB window (the reader must own the next window's L1
#       address before it pushes the current one, which `cb_reserve_back`/
#       `get_write_ptr` cannot express at depth 2 -- see the reader kernel).
#       NB the WRITE path already has this property without trids: Phase-0
#       verification fix #1 replaced the per-block `noc_async_write_barrier()`
#       with `noc_async_writes_flushed()`, so writes stay in flight across the
#       block boundary. B8 is the read-side analogue of that fix, and the reader
#       cannot use `flushed` because it needs the bytes PRESENT, not departed.
#
#   B10 (`vc_spread == 1`) -- per-reader / per-writer static unicast VC. Breaks
#       first-come-first-serve serialization when many readers share a route into
#       the same DRAM endpoints. Read side: in DM_DEDICATED_NOC (what
#       Reader/WriterConfigDescriptor select) `noc_async_read`'s `read_req_vc`
#       argument is DEAD -- `ncrisc_noc_fast_read` only writes NOC_CTRL under
#       `DM_DYNAMIC_NOC` (noc_nonblocking_api.h:415-437). The VC therefore has to
#       be programmed into the sticky NOC_CTRL register once per kernel with
#       `noc_async_read_one_packet_set_state<use_vc=true>` and RESTORED at the end
#       (the register survives kernel launches). Write side needs no such dance:
#       `ncrisc_noc_fast_write` writes NOC_CTRL on every call, so the vc argument
#       of `noc_async_write` is live.
#
#   A3 (`bank_placement == 1`) -- assign work units to cores in
#       `get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)` order (bank i's
#       NoC-optimal worker first), instead of plain row-major. Stacks on the
#       already-applied A1 `row_wise=True` line choice. Host-only; no kernel change.
#
# Payoff gates are set by measurement below; every one has an env counterfactual
# switch and a permanent `_bench_tilize.py` row.
PREFETCH_TRIDS = (1, 2)  # 0 is the firmware default tag; leave it alone
PREFETCH_DEPTH = 3
# Lever B8 needs a third CB window. Keeping `chunk_wt` pinned (so the transaction
# shape is identical to the counterfactual -- the rule Refinement 1 established)
# means the footprint grows 1.5x, which does not fit `L1_CB_BUDGET_BYTES`. WH B0
# has 1 499 136 B of L1 per core, so 192 KiB of CB is still a small fraction; this
# budget is only consulted on the B8 path.
L1_CB_BUDGET_PREFETCH_BYTES = 196608
# Unicast VCs are 0-3 (noc_parameters.h). The firmware default is VC 1 for both
# reads (noc_init -> NCRISC_RD_CMD_BUF NOC_CTRL) and unicast writes
# (NOC_UNICAST_WRITE_VC, dataflow_api_common.h:62).
NUM_UNICAST_VCS = 4
DEFAULT_UNICAST_VC = 1
# `vc_spread` bitmask -- the two halves are programmed by different mechanisms and
# measured separately (see the Mode-C ledger).
VC_SPREAD_READ = 1
VC_SPREAD_WRITE = 2


# --- Refinement 2b: the wide-short 64-way partial-page fan-in -------------------
#
# `b_wide_short` [1,1,32,16384] has `nt_h == 1`, so all 64 cores own tile-COLUMNS of
# the same single tile-row and every core reads a `chunk_row_bytes` slice of each of
# the SAME 32 source pages (a bf16 W=16384 ROW_MAJOR row IS one 32 768 B DRAM page).
# That is a 64-way partial-page fan-in, and Refinement 2 priced its cost with two
# bench rows that differ in nothing but WHICH pages the readers hit:
#
#   regime         | shape           | cores | read | GB/s  | source pages
#   ---------------|-----------------|-------|------|-------|-------------------------
#   b_wide_short   | [1,1,32,16384]  |    64 | 512B | 156.9 | the SAME 32, 64-way split
#   p_2blk_512B    | [1,1,4096,256]  |    64 | 512B | 179.3 | 64 PRIVATE pages per core
#   p_2blk_1024B   | [1,1,4096,512]  |    64 |1024B | 188.8 | private, bigger transaction
#
# Every per-transaction lever is already refuted on this regime (B13 +19.5 %, C7
# +14.2 %, B10 +99 %, A3 +1.7 %, B8 structurally inapplicable, finer chunking
# +1.3 %), so what is left is an ALGORITHM change: decouple "which bytes a core
# reads" from "which tiles a core owns".
#
#   phase 1  each core reads ONE contiguous `piece_bytes = TILE_HW * chunk_row_bytes`
#            slice of ONE source page -- 32x fewer and 32x bigger DRAM transactions
#            for exactly the same bytes. Cores form groups of `TILE_HW` (one per
#            source row of the tile-row); group g stages source piece g.
#   phase 2  all-to-all ready handshake inside the group (one posted atomic inc per
#            group-mate).
#   phase 3  each core PULLS its own slice out of every group-mate's staging buffer
#            into its own cb_rm_input window, producing byte-identical sticks.
#
# The trade the gate has to be measured against: the DRAM read gets much bigger
# transactions, but every byte pays ONE extra L1->L1 hop plus a 32-core barrier.
FANIN_GROUP_ROWS = TILE_HW  # one staging core per source row of a tile-row block
SEM_FANIN_READY = 0
# The staging buffer is on top of the two data CBs, so the fan-in path has its own
# budget. `piece_bytes` is bounded by a constant in W
# (TILE_HW * WT_CHUNK_MAX * TILE_HW * 4 = 65 536 B worst case, fp32 chunk 16), which
# is what keeps `PROPERTIES["bounded_cb"]` true on this path too.
L1_CB_BUDGET_FANIN_BYTES = 196608


# --- Refinement 2b, second lever: per-core transaction-order rotation ----------
#
# An interleaved tensor puts page p in DRAM bank `p % NUM_DRAM_BANKS` (12 on WH B0),
# and every core issues its transactions in the same page order. On the wide-short
# regime (`nt_h == 1`) all `ncores` cores read the SAME 32 source pages, so at issue
# step r every core hits ONE bank while the other 11 idle: the requests are spread
# across banks in aggregate but CLUSTERED in time. The write side has the same shape
# for a different reason -- a core writes `chunk_wt` consecutive output tile pages, so
# with chunk_wt = 8 over 12 banks the 64 cores only ever start on 3 distinct banks.
#
# `stagger` rotates each work unit's issue order (`row_rot` on the read side,
# `col_rot` on the write side) so step 0 is spread over the banks. It is a pure index
# permutation -- same transactions, same count, same size, same L1 addresses, no extra
# L1 and no extra state -- which is what makes it counterfactualable at zero risk.
# The payoff gate below is MEASURED.
# `stagger` is a BITMASK, and the two halves are measured separately because they
# de-cluster two different things: the READ rotation only matters when several cores
# read the SAME source pages (`n_w > 1`, i.e. the tile-row is split by column), while
# the WRITE rotation is about a core's own run of consecutive output pages.
STAGGER_READ = 1
STAGGER_WRITE = 2
# Both halves ship TOGETHER, because they are measured SUPERADDITIVE and neither is
# worth much alone (in-run A/B, 7 rounds x 10 launches, CV <= 1.6 %):
#
#   shape            | chunk | read only | write only | BOTH
#   -----------------|-------|-----------|------------|-------
#   [1,1,32,16384]   |     8 |   0.992   |   0.985    | 0.929
#   [1,1,32,8192]    |     4 |   0.993   |   0.924    | 0.897
#
# Mechanism (it explains the interaction): the instantaneous demand on a DRAM bank is
# read demand PLUS write demand. Spreading only the reads leaves the writes piled on a
# few banks, so the busiest bank -- which is what sets the time -- barely moves. Only
# when both streams are spread does the per-bank load flatten.
#
# Where it pays -- the clause is `nt_h == 1`, i.e. EVERY core reads the SAME 32 source
# pages (the width-split fan-in regime), plus a wide enough chunk:
#
#   shape            | nt_h | n_w | chunk | read B |    off ns |    stg ns | ratio
#   -----------------|------|-----|-------|--------|-----------|-----------|-------
#   [1,1,32,4096]    |    1 |  64 |     2 |  128 B |     4 989 |     4 972 | 0.997
#   [1,1,32,8192]    |    1 |  64 |     4 |  256 B |     8 046 |     7 194 | 0.894
#   [1,1,32,16384]   |    1 |  64 |     8 |  512 B |    13 433 |    12 543 | 0.934
#   [1,1,32,32768]   |    1 |  64 |    16 | 1024 B |    25 394 |    23 820 | 0.938
#   [1,1,64,16384]   |    2 |  32 |    16 | 1024 B |    24 669 |    24 447 | 0.991
#   a_square         |   64 |   1 |    16 | 1024 B |    86 058 |    86 591 | 1.006
#   d_tall_narrow    |   64 |   1 |     1 |   64 B |     3 609 |     3 627 | 1.005
#   g_dram_to_sharded|   64 |   1 |    16 | 1024 B |    19 049 |    19 402 | 1.019
#   e_square_fp32    |   64 |   1 |     8 | 1024 B |   182 908 |   183 178 | 1.001
#
# At `nt_h == 2` only half the grid shares each page set, which already halves the
# clustering, and the win is gone (0.991). At `n_w == 1` each core reads its OWN rows,
# so there is nothing to de-cluster and the rotation is neutral-to-slightly-negative.
# `chunk_wt == 2` is measured neutral, so the chunk clause sits at 4.
#
# Also swept and REJECTED: rotating by NUM_DRAM_BANKS (12) instead of TILE_HW (32),
# which makes the starting bank perfectly uniform -- 12 673 vs 12 480 (+1.5 %) and
# 7 249 vs 7 155 (+1.3 %). The row-loop period wins; do not "improve" it to 12.
STAGGER_MIN_CHUNK_WT = 4


def stagger_pays(ncores: int, nt_h: int, chunk_wt: int) -> int:
    """Refinement 2b gate: which halves of the issue-order rotation pay?

    Returns a ``STAGGER_READ | STAGGER_WRITE`` bitmask. Three measured clauses, all
    tabulated above:

    1. more than one core -- a single core cannot cluster against itself.
    2. ``nt_h == 1`` -- the width-split fan-in regime, where every core reads the
       same source pages in the same order. Measured 0.894-0.938 there and
       0.991-1.019 everywhere else.
    3. ``chunk_wt >= STAGGER_MIN_CHUNK_WT`` -- at chunk 2 the rotation is neutral.

    Both halves are returned together or not at all (they are superadditive).
    """
    if ncores <= 1 or nt_h != 1 or chunk_wt < STAGGER_MIN_CHUNK_WT:
        return 0
    return STAGGER_READ | STAGGER_WRITE


# --- Refinement 3: the interleaved <-> sharded crossover -----------------------
#
# Phase 0 routed BOTH sides of a crossover through the generic `TensorAccessor`, so
# the sharded side paid a full NoC leg it does not need. The one-sided DM ablation
# (`TILIZE_BENCH_SPLIT_DM=1`) prices that leg on each direction (7 rounds x 10
# launches, CV <= 1.1 %):
#
#   regime            | shape / spec              | full   | read leg | write leg | floor
#   ------------------|---------------------------|--------|----------|-----------|------
#   g_dram_to_sharded | [1,1,2048,512] -> B-shard | 19 150 |    9 832 |     6 786 | 2 697
#   g_sharded_to_dram | [1,1,2048,512] B-shard -> | 19 734 |   12 749 |    12 812 | 3 685
#
# so the leg the alias deletes is 6 786 ns (35 %) on DRAM->sharded and 12 749 ns
# (65 %) on sharded->DRAM. Two levers, both measured below:
#
#   C14 one-sided aliasing (`path == "alias_out"` / `"alias_in"`) -- the sharded
#       side's CB is built with `cb_descriptor_from_sharded_tensor`, so its base
#       address IS the shard's, and each core owns exactly the shard's own tiles
#       (`_one_sided_shard_split`). On `alias_out` the tilize LLK packs straight
#       into the output shard (zero write traffic); on `alias_in` the unpacker
#       reads straight out of the input shard (zero read traffic).
#
#       The COST, which is why this is a measured lever and not a free win: the
#       work split is no longer free to choose the read transaction size, because a
#       core owns exactly its shard's columns. A BLOCK-sharded [1,1,2048,512] on
#       8x8 has 64-column shards, so `alias_out` reads 128 B rows where the generic
#       2D split reads 1024 B. Measured on the generic path with `chunk_cap=2`
#       (same transaction shape, same cores -- the prediction row
#       `x_g_to_sharded_chunk2*`), `no_write` == what the aliased path must beat:
#
#         plan (chunk 2, 8 blk/core, 64 cores) | no_write ns | vs chunk-16 full
#         -------------------------------------|-------------|------------------
#         depth 1, no read lever               |      16 874 | 0.88
#         depth 1, + B13 stateful reads        |      16 244 | 0.85
#         depth 1, + C7 split reader           |    * 14 259 | 0.74
#         depth 3, + B8 trid double-issue      |      15 866 | 0.83
#
#       -- i.e. the narrow read costs ~0 in DRAM time (2.10 MB / ~10 us = 212 GB/s
#       at BOTH 128 B and 1024 B, which is R2b's finding again) and the whole
#       difference between those rows is per-read ISSUE cost on the RISC.
#
#   C7 split reader ON THE ALIAS PATH -- with the output aliased BRISC has nothing
#       left to do but one `cb_wait_front`/`cb_pop_front`, so handing it half the
#       stick reads is free of the read/write-overlap cost that made C7 negative on
#       the generic path past 1 block per core (R1c). That is the 0.845 above.
#       Refinement 3 also GENERALISES C7 to depth >= 2 (it was depth-1-only,
#       because BRISC read the reserved window out of `get_write_ptr`): BRISC now
#       derives the window from `cb_base + (block % depth) * window_bytes`, the
#       same identity lever B8 already relies on. That matters because at depth 1
#       the reader and the tilize LLK serialize, and on the alias path the LLK is
#       the only thing left to overlap with.
#
#   B5/B6 coalesced sharded read (`coalesce_rows`) -- a ROW_MAJOR-*sharded* source
#       stores one page per logical row and one page COLUMN per shard, so the 32
#       rows of a chunk-block are 32 CONSECUTIVE pages inside a single core's L1
#       (`core_to_host_pages` pages a shard row-major). When the chunk covers the
#       whole source page the L1 destination is contiguous too, so the whole block
#       is ONE read of `32 * page_bytes` instead of 32 reads of `page_bytes`.
#       `g_sharded_to_dram` plans `chunk_wt = 2` => 128 B reads, 4x below the
#       one-packet threshold, and its read leg is 12 749 ns for 2.10 MB (164 GB/s)
#       at ~50 ns per read -- issue-rate bound, which is exactly what this removes.
#       It is the fallback for every sharded-RM input the alias declines (wide
#       shards, ND specs, `use_multicore=False`, cross-spec reshards).
ALIAS_OUT_MIN_CORES = 2
ALIAS_IN_MIN_CORES = 2

# --- Refinement 3, third lever: one barrier per GROUP of blocks (B7') ----------
#
# The ablation of the shipped `alias_out` plan says the read PAYLOAD is already at
# the DRAM read rate (10 199 ns for 2.10 MB = 206 GB/s, vs 214 for the 1024 B
# generic path) while `sync_only` is 5 616 ns — 3 815 ns of it the 256
# `accessor.get_noc_addr` calls and the rest per-block CB + barrier cost. A
# per-block `noc_async_read_barrier()` drains the read pipeline to ZERO once per
# block, so with 8 blocks per core the DRAM round trip is exposed 8 times and the
# next block's address generation cannot hide behind it either.
#
# `read_group = G` reserves G windows, issues 32*G reads, then takes ONE barrier —
# lever B7 ("one barrier per block, not per transaction") applied one level up. It
# costs G+1 CB windows (the +1 is what lets compute drain one window while the next
# G fill), which is why it is only offered on the delegated depth default, exactly
# like lever B8's third window.
READ_GROUP_MAX = 8
# Sweep hook: force a group size while calibrating the gate. Never set in production.
READ_GROUP_OVERRIDE = None


def alias_out_pays(ncores: int) -> bool:
    """C14 gate, DRAM/interleaved RM -> sharded TILE: is the write leg worth the
    narrower read the shard-shaped work split forces?

    One clause: more than one core. On a single core the whole tensor is one
    "shard" only when the grid is 1x1, and `use_multicore=False` deliberately means
    exactly one core on the GENERIC path (Refinement 1), so the alias never
    competes with it.
    """
    return ncores >= ALIAS_OUT_MIN_CORES


def alias_in_pays(ncores: int) -> bool:
    """C14 gate, sharded RM -> DRAM/interleaved TILE: same clause as
    ``alias_out_pays``. The read leg it deletes is 65 % of `g_sharded_to_dram`."""
    return ncores >= ALIAS_IN_MIN_CORES


def read_group_pays(blocks_per_core: int) -> int:
    """B7' gate: how many chunk-blocks should share one read barrier?

    **Measured: NO — 1 always.** In-run A/B on `g_dram_to_sharded`'s aliased plan
    (64 cores, 8 blocks, 128 B reads; 7 rounds x 10 launches, CV <= 1.5 %):

      group | depth | ns     | vs the shipped d2+B13 15 866
      ------|-------|--------|-----------------------------
        1   |   2   | 15 866 | 1.000   (B13 on -- the shipped plan)
        2   |   3   | 16 187 | 1.020
        4   |   5   | 17 706 | 1.116
        8   |   9   | 19 750 | 1.245

    The premise -- that the per-block `noc_async_read_barrier()` drains the read
    pipeline and exposes DRAM latency 8 times -- is FALSE here: the ablation says the
    read payload is already 10 199 ns for 2.10 MB = **206 GB/s**, i.e. at the best
    DRAM read rate this op has ever measured (214 GB/s for its 1024 B reads), so
    there is no latency left to hide. What grouping DOES do is delay the first push
    by G blocks, which serializes compute behind the reads -- monotonically worse
    with G, exactly as measured. Kept identity-false (and with the
    ``READ_GROUP_OVERRIDE`` sweep hook + the `x_g_alias_g*` bench rows) so the
    verdict is re-measurable rather than a changelog claim.
    """
    return 1


def coalesce_rows_pays(chunk_row_bytes: int, source_page_bytes: int) -> bool:
    """B5/B6 gate: fold a chunk-block's 32 same-shard page reads into one.

    The structural half (the source must be a ROW_MAJOR shard whose pages are
    contiguous in one core's L1, and the chunk must cover the whole page) is
    checked by the caller. The payoff half is that the coalesced read is strictly
    bigger and strictly fewer -- 1 x 32*page_bytes instead of 32 x page_bytes -- so
    it pays whenever it is legal; the gate exists to keep the counterfactual
    switchable (`TILIZE_LEVER_COAL=0`) rather than to exclude a regime.
    """
    return chunk_row_bytes == source_page_bytes


def _lever_flags():
    """Env counterfactual switches for the read-path levers (Mode-C ledger).

    Default 1 == the lever is offered to the plan, 0 == off, 2 == forced past its
    payoff gate (structural preconditions still apply). `_bench_tilize.py` flips
    one at a time so each lever has a re-measurable counterfactual row instead of
    a changelog claim. Never set in production.

    ``r2b`` additionally accepts 3 == the *measurement probe*: phase 1 only (the
    whole-piece staged read straight into the CB, no exchange), which prices the
    read-side ceiling on its own. Output is garbage, so it is bench-only.

    ``r3`` is Refinement 3's one-sided CB alias: 0 == off (the Phase-0 generic
    path on both sides of a crossover, i.e. the counterfactual), 1 == gated,
    2 == force past the payoff gate. ``coal`` is its coalesced sharded read.
    """
    return dict(
        b13=int(os.environ.get("TILIZE_LEVER_B13", "1")),
        c7=int(os.environ.get("TILIZE_LEVER_C7", "1")),
        b8=int(os.environ.get("TILIZE_LEVER_B8", "1")),
        b10=int(os.environ.get("TILIZE_LEVER_B10", "1")),
        a3=int(os.environ.get("TILIZE_LEVER_A3", "1")),
        r2b=int(os.environ.get("TILIZE_LEVER_R2B", "1")),
        stg=int(os.environ.get("TILIZE_LEVER_STG", "1")),
        r3=int(os.environ.get("TILIZE_LEVER_R3", "1")),
        coal=int(os.environ.get("TILIZE_LEVER_COAL", "1")),
        bt=int(os.environ.get("TILIZE_LEVER_BT", "1")),
        nw=int(os.environ.get("TILIZE_LEVER_NW", "1")),
        # Refinement 4, lever 1: the compute-only program on Path B.
        #   0 = off (the Phase-0 three-kernel program == the counterfactual)
        #   1 = gated (see `drop_dataflow_pays`)
        #   2 = forced past the payoff gate
        #   3 = compute-only but with the arm/drain FOLDED onto TRISC — i.e. the
        #       `examples/zero_copy_fold` variant, kept as a bench row because it
        #       is the arm that isolates *which half* of "fewer kernels" pays.
        nd=int(os.environ.get("TILIZE_LEVER_ND", "1")),
        # Refinement 4, lever 3: Fp32Mode::Fast on an fp32 input whose OUTPUT is
        # narrower than fp32. 0 = off (always Lossless, the Phase-0 behaviour),
        # 1 = gated, 2 = forced past the payoff gate.
        f32=int(os.environ.get("TILIZE_LEVER_F32", "1")),
        # Refinement 4, lever 2 (measurement, not a lever): issue ONE tilize call
        # per chunk-block, each with its own InitAndUninit pair, instead of one
        # call around the whole block loop. Prices the config-burst pair by
        # subtraction — the ceiling of ANY init/uninit amortisation scheme. Output
        # stays bit-exact (every call is fully inited), so unlike the ablations it
        # measures the real kernel.
        iu=int(os.environ.get("TILIZE_LEVER_IU", "0")),
        addr=int(os.environ.get("TILIZE_LEVER_ADDR", "0")),
    )


# --- Refinement 3b, lever 3: the hoisted interleaved bank table -----------------
#
# B13 (`StickReadMode::Stateful`) already reads bank-major with an armed command
# buffer, but it rebuilds its bank addresses on EVERY band: `num_banks`
# `accessor.get_noc_addr` calls per chunk-block. Refinement 3b's per-RISC Tracy
# timeline is what makes that worth attacking — on `g_dram_to_sharded`'s aliased
# plan it prices address generation at **4 857 of 17 017 NCRISC cycles** while the
# unpack TRISC sits in `cb_wait_front` for 90 % of the kernel and BRISC for 99 %,
# i.e. NCRISC is the critical path and every cycle it spends is the op's.
# (Refinement 3 estimated the ceiling here at ~400 ns from B13's delta; the
# timeline overturns that.)
#
# `dataflow_kernel_lib::InterleavedStickBands` primes the table ONCE per core (per
# chunk) and every band reuses it, because the interleaved map is affine in the
# page index. Everything else is identical to B13: same 32 transactions, same
# bank-major order, one arm per bank group, one barrier per block.
#
# Two clauses, both structural-with-a-payoff-reading:
#   * >= 2 bands per core -- with one band the hoist replaces `num_banks` calls
#     with `num_banks` calls and only adds a table.
#   * the same read-SIZE clause B13 carries. Above it the per-read command
#     programming is a small share of the transaction, so neither the arming nor
#     the address generation is worth the bank-major issue order.
BANK_TABLE_MIN_BLOCKS = 2


def bank_table_pays(blocks_per_core: int, chunk_row_bytes: int) -> bool:
    """Lever-3 gate: hoist B13's bank table out of the block loop?

    **MEASURED: NO — always False.** In-run A/B on the regime it was designed for,
    15 rounds x 10 launches, CV <= 1.7 %:

      variant                        |     ns | vs B13
      -------------------------------|--------|--------
      hoisted table (this lever)     | 16 483 | 1.016
      B13 per-band table (shipped)   | 16 231 | 1.000
      neither                        | 16 719 | 1.030

    reproduced in a second session at 16 062 / 15 916 / 16 827 (1.009). On the two
    interleaved multi-block regimes that meet the structural clause it is
    sign-unstable across sessions (`[1,1,4096,64]` 1.012 then 0.976;
    `[1,1,8192,32]` 1.004 then 0.997), i.e. inside the noise there.

    The hoist removes **84 of 96** `accessor.get_noc_addr` calls per core and gets
    NOTHING, while B13 — which removes 160 of 256 — is 3 % faster than bare. Two
    facts follow, and together they close the address-generation question
    Refinement 3b opened:

      1. `InterleavedAddrGen::get_noc_addr` is already about as cheap as the cached
         table lookup + `bank_page * aligned_page` multiply that replaces it, so
         the hoist trades one arithmetic sequence for another.
      2. What B13 actually buys is not the arithmetic — it is the ARMED command
         buffer (`noc_async_read_with_state` writes 2 registers where
         `noc_async_read` writes 4-5), which the hoist already inherits.

    And the deeper reading, which the per-RISC timeline (lever 1) supplies: the
    read loop is DRAM-SERVICE bound, and `noc_async_read` spins in
    `noc_cmd_buf_ready`, so the address arithmetic of row r+1 overlaps the service
    of row r. Removing it just makes NCRISC spin longer. That is the same
    "~70 % of the address-gen prize is hidden" that Refinements 1c and 3 inferred
    from B13's delta — now measured directly by removing 87 % of the calls and
    getting nothing.

    Kept identity-false (with the `TILIZE_LEVER_BT=2` force flag and the
    `x_*_bt*` bench rows) so the verdict stays re-measurable rather than a
    changelog claim.
    """
    return False


# --- Refinement 3b, lever 2: drop the writer kernel on `alias_out` --------------
#
# With the output CB aliased onto the shard the writer has NO NoC traffic: its
# whole body is one `cb_wait_front(shard_tiles)` / `cb_pop_front(shard_tiles)`.
# The per-RISC timeline prices that at **17 372 of 17 522 BRISC cycles blocked in
# the wait (99 %)** — the kernel exists only to close the CB loop.
#
# It does not even need to close it. The aliased output CB has exactly
# `shard_tiles` pages and compute pushes exactly `shard_tiles`, so after k pushes
# the free space is `shard_tiles - k*chunk_wt >= chunk_wt` for every k < num_blocks:
# `cb_reserve_back` never blocks and the CB never needs recycling. Across launches
# the firmware re-runs `setup_local_cb_read_write_interfaces`, which sets
# `tiles_acked_received_init = 0` (`circular_buffer_init.h`), so the un-popped
# pages of the previous launch cannot leak into the next one.
#
# The program therefore ships TWO kernels instead of three. The one thing that has
# to move with it is the output base-address runtime arg — with the CB aliased
# there is no `Buffer*` arg forcing a re-patch, and `test_alias_program_cache_
# rebinding` is the probe that catches it — so it is emitted on the compute kernel
# instead (read by nobody; its job is to exist).
def drop_writer_pays() -> bool:
    """Lever-2 gate: is removing the do-nothing writer kernel worth it?

    **MEASURED: YES, and it is a FIXED cost, not a proportional one.** The saving
    is BRISC's kernel entry/exit — the wait it replaces was already fully
    overlapped with the TRISCs (the timeline has BRISC-KERNEL ending 52 cycles
    after pack), so nothing on the critical path moves. Two instruments agree on
    the size:

      * duration, in-run A/B on the small `alias_out` shape (`[1,1,512,128]`,
        8 cores), 12-15 rounds x 10 launches, CV 0.3-0.5 %: **6 829 vs 6 864 =
        0.995**, reproduced in four sessions at -0.51 / -0.69 / -0.71 / -0.72 %,
        i.e. a stable **~45 ns**;
      * the per-RISC timeline on the same shape: **BRISC-FW end -42 cycles**, with
        the NCRISC (-12) and TRISC (0) spans unchanged.

    On the big regime (`[1,1,2048,512]`, 16 µs) the same 45 ns is 0.3 % and is not
    resolvable against a 0.7-1.4 % CV — which is the expected signature of a fixed
    per-launch term, not a contradiction. Kept because it is real, it is free (the
    program ships one kernel FEWER: less dispatch, one less JIT unit, no writer
    binary in L1), and it is the precondition Refinement 4 needs for the same move
    on Path B, where the reader must also go.
    """
    return True


# --- Refinement 4, lever 1: the compute-only program on Path B (B0) -------------
#
# Path B (same-spec L1-sharded I/O) aliases BOTH CBs onto the resident shards, so
# the reader's whole body is `cb_reserve_back(shard_tiles); cb_push_back(shard_tiles)`
# and the writer's is `cb_wait_front(shard_tiles); cb_pop_front(shard_tiles)`. Two
# kernel launches whose only job is to publish pages that are ALREADY at the CB
# address. Refinement 3b removed the writer on the one-sided `alias_out` alias for a
# measured ~45 ns/launch; this is the same move with the reader going too.
#
# THE ARM/DRAIN IS NOT MOVED — IT IS DELETED. That distinction is the whole lever,
# and `ttnn/ttnn/operations/examples/zero_copy_fold` is why it has to be made
# explicitly: that example measures a compute-only program that FOLDS the arm/drain
# onto TRISC and finds it 0.74x-0.95x, i.e. SLOWER, because the arm/drain used to
# overlap the tilize on NCRISC/BRISC and now serializes onto the compute thread.
# Deleting it instead is only sound because of the exact-page argument
# (Refinement 3b, `drop_writer_pays`), which Path B satisfies on BOTH sides:
#
#   * INPUT: the aliased input CB has exactly `shard_tiles` pages and the bytes are
#     already there, so nothing has to be published -- `WaitMode::NoWait` drops the
#     per-block `cb_wait_front` and the helper's `cb_pop_front` still walks
#     `fifo_rd_ptr` across the shard exactly as before (it is the read pointer, not
#     the credit, that selects block k's rows). Total pops == `shard_tiles` == the
#     CB size, so `llk_pop_tiles`' fifo-limit LLK_ASSERTs hold with equality.
#   * OUTPUT: the aliased output CB has exactly `shard_tiles` pages and compute
#     pushes exactly `shard_tiles`, so after k pushes the free space is
#     `shard_tiles - k*chunk_wt >= chunk_wt` for every k < num_blocks:
#     `cb_reserve_back` never blocks and the CB never needs recycling.
#
# Across launches the firmware re-runs `setup_local_cb_read_write_interfaces`, which
# sets `tiles_acked_received_init = 0` (`circular_buffer_init.h`), so neither the
# un-popped output pages nor the un-pushed input credits can leak into the next
# launch of a cached program.
#
# Both base-address runtime args move onto the compute kernel (read by nobody; their
# job is to exist), because with both CBs aliased there is no `Buffer*` runtime arg
# forcing a re-patch -- `test_tilize_refinement4.py::test_path_b_program_cache_rebinding`
# is the probe.
DROP_DATAFLOW_MAX_BLOCKS_PER_CORE = 1 << 30  # no structural cap; see the gate


def drop_dataflow_pays(blocks_per_core: int) -> bool:
    """Lever-1 gate: run Path B as a compute-only program?

    **MEASURED: YES, unconditionally on this path.** In-run A/B against the
    three-kernel program (`TILIZE_LEVER_ND=0`), 15 rounds x 10 launches:

      regime                  | 1 kernel |  3 kernels | ratio
      ------------------------|----------|------------|-------
      f_sharded_small  4 blk  |  see changelog.md, Refinement 4, Perf gate
      f_sharded_large  8 blk  |  "
      x_sharded_tiny   1 blk  |  "

    The saving is two kernels' entry/exit plus the two CB handshakes they close,
    and it is a FIXED per-launch term, so it is largest where the work per core is
    smallest -- which is exactly the regime this refinement targets
    (`f_sharded_small` is 50 % dispatch/CB-sync, 50 % tilize LLK, 0 % DM).

    Not to be confused with FOLDING the arm/drain into compute, which
    `examples/zero_copy_fold` measures at 0.74x-0.95x (slower). See the block
    comment above: this deletes the arm/drain, it does not move it. The folded
    variant is retained as the `TILIZE_LEVER_ND=3` bench arm so the two are
    distinguishable by measurement rather than by argument.
    """
    return blocks_per_core <= DROP_DATAFLOW_MAX_BLOCKS_PER_CORE


# --- Refinement 4, lever 3: Fp32Mode::Fast on a narrowing fp32 cast -------------
#
# The compute kernel used to pick `Fp32Mode::Lossless` from the INPUT CB format
# alone, so `fp32 -> bf16` / `fp32 -> bf8b` took the slow (non-fast_tilize) LLK path
# to preserve a precision the narrower output cannot hold. `Lossless` is only
# *required* when the tiled output must be bit-identical to the fp32 input, i.e.
# when the output is itself fp32.
#
# The switch is a HOST+KERNEL pair, not a kernel-only flag: `can_use_fast_tilize`
# carries `static_assert(!has_unpack_to_dest_fp32<input_dfb>())` -- fast tilize
# reads DEST at bf16 stride, so an UnpackToDestFp32 input CB would have the
# unpacker write 32-bit payloads into slots the pack stage reads as tf32 and
# silently corrupt the output. So `_compute_config` must also drop
# `unpack_to_dest_mode[CB_RM_INPUT]` back to `Default` on exactly the cells this
# gate turns on.
def fp32_fast_pays(in_dtype, out_dtype) -> bool:
    """Lever-3 gate: may an fp32 input take the fast tilize path?

    Structural clause (not a payoff question): the output must NOT be fp32.
    `fp32 -> fp32` is a bit-exactness contract (`test_tilize_precision_baseline`
    asserts `exact_frac == 1.0`) and `can_use_fast_tilize` refuses an fp32 output
    anyway, so Lossless + UnpackToDestFp32 is the only correct configuration there.

    Payoff clause: see `changelog.md` Refinement 4 -- measured on the compute-bound
    sharded regime the refinement targets AND on the DM-bound interleaved one, and
    gated to where it was measured to pay.
    """
    if in_dtype != ttnn.float32:
        return False  # not an fp32 input -> `Lossless` was never selected
    return out_dtype != ttnn.float32


def stateful_read_pays(chunk_row_bytes: int) -> bool:
    """B13 gate: is arming the command buffer per bank worth the bank-major order?

    One measured clause: the read must be at most ``STATEFUL_READ_MAX_ROW_BYTES``.
    Measured 0.980 at 64 B and 0.968 at 128 B, then 1.023 / 1.177 / 1.057 at
    256 / 512 / 1024 B — see the table above ``STATEFUL_READ_MAX_ROW_BYTES``.
    """
    return chunk_row_bytes <= STATEFUL_READ_MAX_ROW_BYTES


def split_read_pays(depth: int, blocks_per_core: int, chunk_row_bytes: int) -> bool:
    """C7 gate: is handing BRISC half the stick reads worth its stall + handshake?

    Three clauses:

    1. ``depth == 1`` is **structural**, not a payoff question: BRISC writes into
       the window NCRISC reserved without touching the CB pointers, so the window
       must be at the CB base address on every block.
    2. **one block per core** — measured. The split costs the read/write overlap
       across the block boundary (BRISC's read of block i+1 queues behind its
       write of block i), which is free only when there is no boundary. Measured
       0.956 on `[1,1,2048,32]` (1 block) vs 1.145 on `[1,1,8192,32]` (4 blocks).
    3. **64 B reads only** — measured. 0.956 at 64 B, then 1.018 / 1.056 / 1.146 /
       1.045 at 128 / 256 / 512 / 1024 B.
    """
    return (
        depth == 1 and blocks_per_core <= SPLIT_READ_MAX_BLOCKS_PER_CORE and chunk_row_bytes <= SPLIT_READ_MAX_ROW_BYTES
    )


# --- Refinement 2 payoff gates -- all three MEASURED -------------------------
#
# B8 pays exactly while the plan is **below DRAM bandwidth saturation**, i.e. while
# the binding resource is this core's own read issue/drain rather than the DRAM
# aggregate. That single mechanism explains both clauses, and each clause has its
# own device sweep (7 rounds x 10 launches, in-run A/B pairs, CV <= 1.1 %).
#
# Clause 1 -- CORE COUNT, at a fixed 1024 B read (`[1,1,4096,512]`, chunk 16, the
# `core_cap` hook forcing the count so only `ncores` moves):
#
#   cores | blk | no lever ns |    B8 ns | B8/none | achieved GB/s (no lever)
#   ------|-----|-------------|----------|---------|-------------------------
#       1 | 128 |     218 729 |  190 276 | **0.870** |  38.4
#       2 |  64 |     112 613 |   98 426 | **0.874** |  74.5
#       4 |  32 |      60 983 |   52 957 | **0.868** | 137.6
#       8 |  16 |      45 350 |   45 411 |   1.001 | 185.0   <- DRAM saturates here
#      16 |   8 |      45 037 |   43 577 |   0.968 | 186.3
#      32 |   4 |      45 119 |   43 878 |   0.972 | 185.9
#      64 |   2 |      44 110 |   43 942 |   0.996 | 190.2
#
# 1-4 cores is a flat, reproducible -13 %; from 8 cores the wall-clock stops moving
# with the core count at all (~45 us / ~186 GB/s), which is the saturation the
# mechanism predicts. The residual -3 % at 16/32 cores is NOT monotone with the
# +0.1 % at 8 and -0.4 % at 64, so it is scatter, not an effect: excluded.
#
# Clause 2 -- READ SIZE, at a fixed 64 cores x 2 blocks/core (`[1,1,4096,W]`,
# W = 64/128/256/512 giving chunk 2/4/8/16, i.e. 128/256/512/1024 B reads):
#
#   read B | no lever ns |   B8 ns | B8/none | achieved GB/s
#   -------|-------------|---------|---------|--------------
#     64 B |      14 319 |  11 197 | **0.782** |  73.2   (4 blk/core row)
#    128 B |       9 594 |   7 883 | **0.822** | 109.3
#    256 B |      13 573 |  13 624 |   1.004 | 154.5
#    512 B |      23 375 |  22 700 |   0.971 | 179.4
#   1024 B |      44 110 |  43 942 |   0.996 | 190.2
#
# Same story from the other side: at <= 128 B even the full grid is
# transaction-rate bound (73-109 GB/s, far under the ~190 GB/s achievable copy), so
# the per-block drain is still on the critical path. From 256 B up it is not.
# At 64 B / 4 blocks B8 also BEATS B13 (0.782 vs 0.925 on the same row), which is
# why the two are mutually exclusive with B8 winning whenever it fires.
TRID_PREFETCH_MAX_CORES = 4
TRID_PREFETCH_MAX_ROW_BYTES = 128
TRID_PREFETCH_MIN_BLOCKS = 2
#
# B10 (per-reader/per-writer static unicast VC) is a **measured regression on every
# regime with meaningful traffic**, and the two halves are separable:
#
#   regime                | reads only | writes only |  both
#   ----------------------|------------|-------------|--------
#   b_wide_short (64c)    | 1.084      | **1.779**   | 1.947
#   a_square     (64c)    | 1.085      | **1.957**   |   --
#   d_tall_narrow(64c,1blk)|    --     |     --      | 1.001
#   c_single_core(1c)     |     --     |     --      | 1.011
#
# So the write half nearly doubles the runtime and the read half costs ~8.5 %.
# Mechanism: the firmware picks ONE static VC for a reason -- VC 1 for unicast
# (dataflow_api_common.h:62) -- and rotating requests over VCs 0/2/3 splits the
# per-VC buffering at the DRAM NIU instead of pooling it, so each core's stream
# gets a fraction of the queue depth it had. On one core (nothing to de-serialize)
# it is exactly inert, which is the B0 control this verdict needs. The gate is
# therefore identity-false; the lever and its bench rows stay so the verdict is
# re-measurable rather than a changelog claim.
VC_SPREAD_MIN_CORES = 65  # > any current grid => OFF (measured regression)
#
# A3 (bank-adjacent work->core order) is **measured neutral**: b_wide_short 1.016,
# a_square 1.005, d_tall_narrow 1.000 -- at or inside the noise floor, never a win.
# The structural reason is prior to the measurement: a tilize block needs 32
# CONSECUTIVE source pages and interleaved round-robin puts page p in bank
# p % NUM_DRAM_BANKS, so EVERY core necessarily touches all 12 banks. There is no
# core<->bank affinity for a placement to exploit, so the only thing the
# permutation can move is average hop count on a grid that is already full.
BANK_PLACEMENT_MIN_CORES = 65  # > any current grid => OFF (measured no payoff)


def trid_prefetch_pays(ncores: int, blocks_per_core: int, chunk_row_bytes: int, prefetch_cb_bytes: int) -> bool:
    """B8 gate: is a third CB window + two transaction ids worth 1.5x the CB L1?

    Four clauses (the sweeps behind clauses 2 and 3 are tabulated above):

    1. ``blocks_per_core >= TRID_PREFETCH_MIN_BLOCKS`` is **structural**, not a
       payoff question: with one chunk-block per core there is no next block whose
       reads could stay in flight across the barrier.
    2. ``ncores <= TRID_PREFETCH_MAX_CORES`` — measured. Below DRAM saturation the
       core's own read drain is the bound and removing it is worth ~13 %.
    3. ``chunk_row_bytes <= TRID_PREFETCH_MAX_ROW_BYTES`` — measured. The other way
       to be under saturation: at <= 128 B the op is transaction-rate bound even on
       the full grid, and the drain is worth 18-22 %.
    4. the depth-3 footprint must fit ``L1_CB_BUDGET_PREFETCH_BYTES`` at the
       **unchanged** ``chunk_wt`` — a lever is not allowed to move the transaction
       shape behind the caller's back (Refinement 1's chunk-pin rule).
    """
    if blocks_per_core < TRID_PREFETCH_MIN_BLOCKS:
        return False
    if prefetch_cb_bytes > L1_CB_BUDGET_PREFETCH_BYTES:
        return False
    return ncores <= TRID_PREFETCH_MAX_CORES or chunk_row_bytes <= TRID_PREFETCH_MAX_ROW_BYTES


def vc_spread_pays(ncores: int) -> bool:
    """B10 gate: does spreading readers/writers over the 4 unicast VCs pay?

    No — measured a regression at every core count with real traffic (the table
    above ``VC_SPREAD_MIN_CORES``), so this is identity-false on any current grid.
    """
    return ncores >= VC_SPREAD_MIN_CORES


#
# Refinement 2b payoff gate -- MEASURED, and the answer is NO. Three in-run A/B rows
# on `b_wide_short` (7 rounds x 10 launches, CV <= 1.6 %):
#
#   variant                         |    ns | vs off | what it isolates
#   --------------------------------|-------|--------|-------------------------------
#   off (32 x 512 B strided reads)  | 13 461| 1.000  | the baseline
#   PROBE: 1 x 16 384 B read, no    | 12 736| 0.946  | the read-side CEILING of this
#     exchange (`fanin_mode == 2`)  |       |        | algorithm -- the most it can buy
#   full 3-phase redistribution     | 18 574| 1.380  | + the L1 hop and the barrier
#
# and the one-sided DM ablation (TILIZE_SKIP_DM=2/3) says why, decisively:
#
#   leg                | off      | probe    | verdict
#   -------------------|----------|----------|---------------------------------------
#   read alone         | 5 966 ns | 5 985 ns | IDENTICAL -- a 32x bigger transaction
#                      |          |          | moves the same bytes in the same time
#   write alone        | 7 785 ns | 7 765 ns | untouched (as expected)
#   compute+sync       | 2 226 ns | 1 684 ns | the whole probe gain is the 32 read
#                      |          |          | ISSUES, not DRAM efficiency
#
# So the entry's premise -- that a 64-way *partial-page* fan-in costs DRAM bandwidth
# -- is false on this hardware: 512 B slices of a shared 32 768 B page cost exactly
# what 512 B whole pages cost. The 4.9 % ceiling the probe does show is issue
# overhead, already below this entry's 14 % gate, and the redistribution's own L1 leg
# (+4 676 ns) plus its 32-core barrier (+1 217 ns of sync) spend it three times over.
#
# The code stays so the verdict is re-measurable rather than a changelog claim
# (`TILIZE_LEVER_R2B=2` forces it, `=3` runs the probe; both have permanent
# `_bench_tilize.py` rows). The gate is identity-false on any reachable plan.
FANIN_MIN_READ_BYTES = 1 << 30  # > any reachable chunk_row_bytes => OFF (refuted)
FANIN_MIN_GROUPS = 1


def fanin_pays(chunk_row_bytes: int, fanin_groups: int) -> bool:
    """Refinement 2b gate: is one extra L1 hop worth a 32x bigger DRAM read?

    No — measured 1.380x SLOWER, and its own read-side probe shows the whole-page
    read buys **zero** DRAM time (the table above ``FANIN_MIN_READ_BYTES``). Kept
    identity-false so the counterfactual stays re-measurable.
    """
    return chunk_row_bytes >= FANIN_MIN_READ_BYTES and fanin_groups >= FANIN_MIN_GROUPS


def bank_placement_pays(ncores: int) -> bool:
    """A3 gate: does bank-adjacent work->core assignment pay?

    No — measured neutral-to-slightly-negative, and structurally there is no
    affinity to exploit (see the note above ``BANK_PLACEMENT_MIN_CORES``), so this
    is identity-false on any current grid.
    """
    return ncores >= BANK_PLACEMENT_MIN_CORES


# ---------------------------------------------------------------------------
# Small integer helpers (ttnn.div_up / round_up / find_max_divisor are not
# bound on this build — verified).
# ---------------------------------------------------------------------------


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def _largest_divisor_le(n: int, limit: int) -> int:
    """Largest divisor of ``n`` that is <= ``limit`` (never skips 5 or 7)."""
    limit = max(1, min(limit, n))
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1


def a0_active_cores(grid_cores: int, total_tiles: int) -> int:
    """master.md A0: ``min(grid, total_tiles, bandwidth_knee)``.

    The single place the active-core count is decided for the generic path, so
    the bench / unit-test A0 assert can check the *declared* criterion instead of
    re-deriving it. See ``A0_KNEE_CORES`` for why the knee term is identity on
    this op (measured, not assumed).
    """
    cap = A0_KNEE_CORES if CORE_CAP_OVERRIDE is None else int(CORE_CAP_OVERRIDE)
    return max(1, min(grid_cores, total_tiles, cap))


def depth2_pays(ncores: int, blocks_per_core: int) -> bool:
    """C16 gate: is depth-2 worth 2x the per-core CB L1 on this plan?

    Three measured clauses (numbers in the ``BANDWIDTH_KNEE_CORES`` comment):

    1. fewer than ``MIN_BLOCKS_FOR_DEPTH2`` blocks -> **no**: there is no block
       boundary to overlap, so depth-2 cannot do anything except cost L1.
    2. below the DRAM bandwidth-saturation knee -> **yes**: the core's own NoC
       issue rate is the bound and overlapping its reader/writer is worth 1.3x.
    3. at or above the knee -> **only past ``DEPTH1_MAX_BLOCKS_PER_CORE``**: DRAM
       aggregate bandwidth is the bound, but each un-overlapped block boundary
       still costs, and beyond 4 boundaries that reaches ~2 %.
    """
    if blocks_per_core < MIN_BLOCKS_FOR_DEPTH2:
        return False
    if ncores < BANDWIDTH_KNEE_CORES:
        return True
    return blocks_per_core > DEPTH1_MAX_BLOCKS_PER_CORE


def _bank_ordered_cores(device, grid, ncores: int):
    """Lever A3: ``ncores`` cores ordered bank-adjacent-first, plus their range set.

    ``get_optimal_dram_bank_to_logical_worker_assignment(NOC_0)`` returns, for DRAM
    bank *i*, the worker core that is NoC-optimal for it (one hop where the
    topology allows). Those cores come first, in bank order, so work unit *i* —
    which starts at source page *i* and therefore at bank ``i % NUM_DRAM_BANKS`` —
    lands on that bank's own worker. Remaining cores fill in A1's row-major order.

    Returns ``(cores, core_ranges)``. When the resulting core *set* is the same as
    the default row-major prefix (the common full-grid case) the compact
    ``num_cores_to_corerangeset`` is reused, so only the work->core MAPPING changes
    and the CB / kernel placement is byte-identical to the counterfactual.
    """
    row_major = ttnn.grid_to_cores(ncores, grid.x, grid.y, True)
    try:
        bank_order = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    except Exception:  # pragma: no cover - API not bound on this build
        return row_major, ttnn.num_cores_to_corerangeset(ncores, grid, True)

    inside = {(int(c.x), int(c.y)) for c in row_major}
    ordered, seen = [], set()
    for core in list(bank_order) + list(row_major):
        key = (int(core.x), int(core.y))
        if key in seen or key not in inside:
            continue
        seen.add(key)
        ordered.append(ttnn.CoreCoord(key[0], key[1]))
        if len(ordered) == ncores:
            break
    # A bank's optimal worker can fall outside the active rectangle, so top up.
    for core in row_major:
        if len(ordered) == ncores:
            break
        key = (int(core.x), int(core.y))
        if key not in seen:
            seen.add(key)
            ordered.append(core)

    if seen == inside:
        return ordered, ttnn.num_cores_to_corerangeset(ncores, grid, True)
    return ordered, ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in ordered})


def _split_contiguous(total: int, parts: int):
    """``parts`` contiguous (start, count) ranges covering ``total`` units.

    The first ``total % parts`` partitions get one extra unit.
    """
    base, rem = divmod(total, parts)
    ranges = []
    start = 0
    for i in range(parts):
        count = base + (1 if i < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


# ---------------------------------------------------------------------------
# Shard geometry
# ---------------------------------------------------------------------------


def _shard_geometry(tensor):
    """2D-normalised shard geometry, or None when the tensor is interleaved."""
    memory_config = tensor.memory_config()
    if not memory_config.is_sharded():
        return None

    shard_spec = memory_config.shard_spec
    if shard_spec is not None:
        shard_h = int(shard_spec.shape[0])
        shard_w = int(shard_spec.shape[1])
        grid = shard_spec.grid
        orientation = shard_spec.orientation
    else:
        nd = memory_config.nd_shard_spec
        if nd is None:
            return None
        shard_shape = list(nd.shard_shape)
        shard_h = int(prod(shard_shape[:-1]))
        shard_w = int(shard_shape[-1])
        grid = nd.grid
        orientation = nd.orientation

    return {
        "h": shard_h,
        "w": shard_w,
        "grid": grid,
        "grid_key": str(grid),
        "orientation": orientation,
        "layout": memory_config.memory_layout,
        "buffer": memory_config.buffer_type,
        # True for an NdShardSpec. The shard -> core order of an ND spec is
        # row-major over the *ND* shard grid with a round-robin core assignment
        # (buffer_distribution_spec.cpp:53-86,152-191), which only agrees with the
        # flattened 2D map when the fold is consistent -- so Refinement 3's
        # one-sided alias, which needs that map exactly, declines ND.
        #
        # NB this fires for a spec that STAYS ND, not for the ND *API*: the allocator
        # normalises every 2D-representable ND request to a legacy layout (verified on
        # device, `probes/probe_024.py` -- including the dangerous-looking case where a
        # LEADING dim is split while the row dim is whole, which normalises to
        # HEIGHT_SHARDED over the folded rows), and the buffer then uses the legacy
        # page mapping the 2D map above is derived from. What is left as genuinely ND
        # is e.g. a round-robin with more shards than cores, which the one-shard-per-
        # core clause would decline anyway. Both are pinned by
        # `test_tilize_refinement3.py::test_nd_*`.
        "nd": shard_spec is None or memory_config.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED,
    }


def _alias_eligible(in_geo, out_geo, folded_h: int, width: int) -> bool:
    """True iff the same-spec zero-copy path (Path B) applies."""
    if in_geo is None or out_geo is None:
        return False
    if in_geo["buffer"] != ttnn.BufferType.L1 or out_geo["buffer"] != ttnn.BufferType.L1:
        return False
    for key in ("h", "w", "orientation", "layout", "grid_key"):
        if in_geo[key] != out_geo[key]:
            return False

    shard_h, shard_w = in_geo["h"], in_geo["w"]
    if shard_h % TILE_HW or shard_w % TILE_HW:
        return False
    if folded_h % shard_h or width % shard_w:
        return False
    if (folded_h // shard_h) * (width // shard_w) != in_geo["grid"].num_cores():
        return False
    # Whole shard width is one tilize block, so it must fit the LLK limit.
    return shard_w // TILE_HW < MAX_BLOCK_WIDTH_TILES


# ---------------------------------------------------------------------------
# Refinement 3: the shard -> global-tile map behind one-sided CB aliasing
# ---------------------------------------------------------------------------
#
# `op_design.md` Risk #2 (and the verifier's note on this refinement) is that a
# WRONG map keeps every CB count balanced and silently transposes blocks. The map
# is therefore derived from the tt-metal page mapping rather than guessed:
#
#   * legacy 2D shard specs get their core list from
#     `corerange_to_cores(shard_grid, num_cores, row_wise = orientation == ROW_MAJOR)`
#     (`buffer.cpp:271`), and
#   * `core_to_host_pages` (`buffer.cpp:119-180`) walks shards COLUMN-inner /
#     ROW-outer over the shard grid, and pages ROW-major *within* each shard
#     (`i` over shard rows outer, `j` over shard columns inner).
#
# So shard `(sh, sw)` is linear index `sh * n_sw + sw` and lives on
# `cores[sh * n_sw + sw]`, and its pages are its own tiles in row-major order --
# which is exactly the order `compute_kernel_lib::tilize` pushes them in when the
# reader iterates tile-row-outer / column-chunk-inner. That identity is what makes
# an aliased CB legal: the CB's page k IS the shard's tile k.
#
# ND shard specs are DECLINED (see `_shard_geometry`'s "nd" key): their shard ->
# core order is row-major over the ND shard grid with a round-robin assignment, and
# for a shard that splits a leading dim while leaving an inner one whole the
# flattened row order does not agree. Those cells keep the generic accessor path.


def _one_sided_shard_split(geo, folded_h: int, width: int):
    """(shard_ht, shard_wt, [(core, shard_row, shard_col), ...]) or ``None``.

    ``None`` means "this shard geometry is not one the map above covers" -- the
    caller then keeps the generic ``TensorAccessor`` path, which is correct for
    every geometry.
    """
    if geo is None or geo["nd"]:
        return None
    # A CB can only be aliased onto L1.
    if geo["buffer"] != ttnn.BufferType.L1:
        return None

    shard_h, shard_w = geo["h"], geo["w"]
    if shard_h % TILE_HW or shard_w % TILE_HW:
        return None
    if folded_h % shard_h or width % shard_w:
        return None

    n_sh, n_sw = folded_h // shard_h, width // shard_w
    grid = geo["grid"]
    # One shard per core, exactly -- with fewer shards than cores the legacy map
    # leaves cores empty and with more it wraps, neither of which this map covers.
    if n_sh * n_sw != grid.num_cores():
        return None
    # HEIGHT/WIDTH sharding pin one of the two shard-grid dims by definition; a
    # spec that disagrees would take a different branch of `core_to_host_pages`.
    if geo["layout"] == _HEIGHT_SHARDED and n_sw != 1:
        return None
    if geo["layout"] == _WIDTH_SHARDED and n_sh != 1:
        return None
    # Whole shard width is at most one tilize block, so it must fit the LLK limit.
    if shard_w // TILE_HW >= MAX_BLOCK_WIDTH_TILES:
        return None

    cores = ttnn.corerange_to_cores(grid, n_sh * n_sw, geo["orientation"] == ttnn.ShardOrientation.ROW_MAJOR)
    units = [(cores[sh * n_sw + sw], sh, sw) for sh in range(n_sh) for sw in range(n_sw)]
    return shard_h // TILE_HW, shard_w // TILE_HW, units


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def build_plan(input_tensor, output_tensor, device, *, use_multicore=True, use_double_buffer=None):
    """Evaluate the host planner once per program build.

    ``use_double_buffer=None`` (the public default) means *the planner decides*:
    depth-2 only where it was measured to pay (see ``depth2_pays``). ``True`` /
    ``False`` force depth-2 / depth-1 and keep their documented meaning.

    The tile grid is derived from the **output** tensor's padded shape — that is
    the page grid the writer addresses. A ROW_MAJOR-*sharded* input can carry
    extra padding on its last dim (its width is rounded up to a whole number of
    shard widths, e.g. logical W=160 with shard_W=96 stores a padded W=192), and
    that padding is a source *stride* concern only. Deriving the tile grid from
    the input's padded shape would invent tile columns that do not exist in the
    output and silently corrupt every page index.
    """
    out_padded = list(output_tensor.padded_shape)
    in_padded = list(input_tensor.padded_shape)

    folded_h = int(prod(out_padded[:-1]))
    width = int(out_padded[-1])
    nt_h = folded_h // TILE_HW
    wt = width // TILE_HW
    total_tiles = nt_h * wt

    # Only the last dim may differ between the two padded shapes; anything else
    # means the row fold is not the same on both sides and the plain
    # "flatten the leading dims" mapping below would not hold.
    if in_padded[:-1] != out_padded[:-1]:
        raise UnsupportedAxisValue(
            f"tilize: input padded shape {in_padded} and output padded shape "
            f"{out_padded} disagree on the leading dims — the row fold is not "
            "expressible as a single flatten"
        )
    if int(in_padded[-1]) < width:
        raise UnsupportedAxisValue(
            f"tilize: input padded width {in_padded[-1]} is narrower than the " f"output width {width}"
        )

    elem_in = input_tensor.element_size()
    tile_in = ttnn.tile_size(input_tensor.dtype)
    tile_out = ttnn.tile_size(output_tensor.dtype)
    tile_row_bytes = TILE_HW * elem_in

    in_geo = _shard_geometry(input_tensor)
    out_geo = _shard_geometry(output_tensor)

    plan = {
        "folded_h": folded_h,
        "width": width,
        "in_padded_width": int(in_padded[-1]),
        "nt_h": nt_h,
        "wt": wt,
        "total_tiles": total_tiles,
        "elem_in": elem_in,
        "tile_in": tile_in,
        "tile_out": tile_out,
        "tile_row_bytes": tile_row_bytes,
        "needs_cast": int(output_tensor.dtype != input_tensor.dtype),
        # Refinement 4 lever 3 needs the dtypes themselves (not just `needs_cast`)
        # in both planners, and `_compute_config` has to agree with the kernel's
        # Fp32Mode on the same cell, so the decision lives in the plan.
        "in_dtype": input_tensor.dtype,
        "out_dtype": output_tensor.dtype,
    }

    # Path B is inherently multi-core (one shard per core), so an explicit
    # use_multicore=False request routes to the generic single-core path
    # instead of being refused.
    if use_multicore and _alias_eligible(in_geo, out_geo, folded_h, width):
        return _plan_alias(plan, in_geo)

    # Refinement 3, lever C14: one-sided aliasing on a crossover. The sharded side
    # gets its CB aliased onto the shard and each core owns exactly its own shard's
    # tiles; the interleaved side keeps the generic accessor. Only when the OTHER
    # side is interleaved -- with both sides sharded (a cross-spec reshard) the
    # generic path is already correct on both and this refinement does not touch it.
    levers = _lever_flags()
    shard_split = None
    if use_multicore and levers["r3"]:
        if out_geo is not None and in_geo is None:
            shard_split = _one_sided_shard_split(out_geo, folded_h, width)
            if shard_split is not None:
                shard_split = ("alias_out", out_geo) + shard_split
        elif in_geo is not None and out_geo is None:
            shard_split = _one_sided_shard_split(in_geo, folded_h, width)
            if shard_split is not None:
                shard_split = ("alias_in", in_geo) + shard_split

    chunk_cap = None
    if use_double_buffer is None:
        # C16 gate. The depth feeds the L1 chunk-width budget, which feeds the
        # 2D split, which decides ncores / blocks-per-core -- i.e. the gate's own
        # inputs. Resolve it with a depth-2 trial plan (pure host arithmetic, no
        # device work) and re-plan once at the chosen depth.
        trial = _plan_generic(
            dict(plan),
            input_tensor,
            device,
            in_geo,
            use_multicore=use_multicore,
            depth_request=2,
            shard_split=shard_split,
        )
        if depth2_pays(trial["ncores"], trial["blocks_per_core"]):
            depth_request = 2
        else:
            depth_request = 1
            # Pin the chunk width to the depth-2 plan's, so the *only* difference
            # between the gated plan and the ungated one is that the CB has half
            # the pages. Letting the freed L1 grow the chunk instead would change
            # the reader's transaction size and the work split behind the caller's
            # back -- measured a 1.3 % LOSS on e_square_fp32 (chunk 8 -> 16) with
            # zero L1 saved. Non-regression is then structural, not just measured.
            chunk_cap = trial["chunk_wt"]
    else:
        depth_request = 2 if use_double_buffer else 1

    return _plan_generic(
        plan,
        input_tensor,
        device,
        in_geo,
        use_multicore=use_multicore,
        depth_request=depth_request,
        chunk_cap=chunk_cap,
        # Lever B8 adds a THIRD CB window, so it may only fire on the delegated
        # default. `use_double_buffer=True/False` keep their documented meanings
        # ("depth-2, +L1" / "depth-1, minimal L1") exactly -- a caller who pinned
        # the depth gets the depth they asked for.
        depth_forced=use_double_buffer is not None,
        shard_split=shard_split,
    )


def _fp32_lossless(plan) -> int:
    """Refinement 4 lever 3: 1 == Fp32Mode::Lossless (slow, bit-exact fp32 path).

    Only ever 0 for an fp32 input whose output is narrower than fp32 — every other
    dtype either never selected Lossless (the kernel's own
    `is_fp32_input_format` guard) or needs it for bit-exactness.
    """
    f32 = _lever_flags()["f32"]
    if f32 == 0:
        return 1  # counterfactual: the Phase-0 behaviour
    if f32 == 2 or fp32_fast_pays(plan["in_dtype"], plan["out_dtype"]):
        return 0
    return 1


def _plan_alias(plan, geo):
    """Path B: one resident shard per core, no NoC traffic on either side."""
    shard_h, shard_w = geo["h"], geo["w"]
    chunk_wt = shard_w // TILE_HW
    num_blocks = shard_h // TILE_HW
    shard_tiles = chunk_wt * num_blocks

    grid = geo["grid"]
    cores = []
    for core_range in grid.ranges():
        cores.extend(ttnn.grid_to_cores(core_range.start, core_range.end, True))

    # --- Refinement 4, lever 1: the compute-only program (see `drop_dataflow_pays`)
    # The zone variant instruments the reader's and writer's OWN stages, so it needs
    # both kernels to exist. The DM/compute ablations do not: `TILIZE_SKIP_DM` is a
    # no-op on Path B (there is no NoC traffic to drop, which is the whole point) and
    # `TILIZE_SKIP_COMPUTE` keeps the compute kernel's CB dance, which the
    # compute-only program reproduces under `no_wait`. Keeping the lever ON under
    # ablation is what makes `no_dm == full` a statement about the SHIPPED program;
    # the three-kernel arm (`TILIZE_LEVER_ND=0`) carries the original witness.
    nd = _lever_flags()["nd"]
    drop_dataflow = int((nd == 2 or nd == 3 or (nd == 1 and drop_dataflow_pays(num_blocks))) and _zone_flag() == 0)
    # nd == 3 is the `examples/zero_copy_fold` arm: still one kernel, but the
    # arm/drain is FOLDED onto TRISC instead of deleted.
    self_arm = int(drop_dataflow and nd == 3)

    plan.update(
        {
            "path": "alias",
            # Refinement 3: Path B aliases BOTH sides (the one-sided crossover
            # paths set exactly one of these).
            "alias_in": 1,
            "alias_out": 1,
            "coalesce_rows": 0,
            "blocks_row_major": 0,
            "read_group": 1,
            "addr_probe": 0,
            "core_ranges": grid,
            "cores": cores,
            "chunk_wt": chunk_wt,
            "shard_tiles": shard_tiles,
            "num_blocks": num_blocks,
            "blocks_per_core": num_blocks,
            "chunks_per_core": 1,
            "depth": 1,  # the CB *is* the shard; use_double_buffer is inert here
            "row_page_stride": 1,
            "source_page_bytes": shard_w * plan["elem_in"],
            "chunk_row_bytes": shard_w * plan["elem_in"],
            # Path B has no NoC traffic at all, so no transaction-shaping lever
            # applies (B13 / C7 / B8 / B10 / A3 all move NoC commands around).
            "stateful_read": 0,
            "bank_table": 0,
            # Refinement 4 lever 1: both dataflow kernels go, or neither.
            "drop_writer": drop_dataflow,
            "drop_reader": drop_dataflow,
            "self_arm": self_arm,
            "fp32_lossless": _fp32_lossless(plan),
            "split_read": 0,
            "prefetch_blocks": 1,
            "vc_spread": 0,
            "bank_placement": 0,
            "read_vcs": None,
            "write_vcs": None,
            "stagger": 0,
            "fanin_mode": 0,
            "fanin_groups": 0,
            "fanin_group_rows": 0,
            "fanin_grid_x": 1,
            "fanin_group_axes": None,
            "piece_bytes": 0,
            "ncores": len(cores),
            "cb_bytes_per_core": shard_tiles * (plan["tile_in"] + plan["tile_out"]),
            "alias_cb_bytes": shard_tiles * (plan["tile_in"] + plan["tile_out"]),
        }
    )
    return plan


def _plan_generic(
    plan,
    input_tensor,
    device,
    in_geo,
    *,
    use_multicore,
    depth_request,
    chunk_cap=None,
    depth_forced=False,
    shard_split=None,
):
    """Path A/C: 2D height-first rectangular split over the compute grid.

    ``chunk_cap`` pins the chunk width from a previous (depth-2) pass so the C16
    depth gate cannot change the transaction shape — see ``build_plan``.

    ``shard_split`` (Refinement 3) replaces the 2D split with the sharded side's own
    shard map — ``("alias_out"|"alias_in", geo, shard_ht, shard_wt, units)`` from
    ``_one_sided_shard_split``. Everything downstream (source-page geometry, the CB
    depth budget, every read/write lever) is shared with the generic path on
    purpose: the alias changes *which tiles a core owns* and *which side pays a NoC
    leg*, nothing else.
    """
    nt_h, wt = plan["nt_h"], plan["wt"]
    tile_in, tile_out = plan["tile_in"], plan["tile_out"]
    elem_in = plan["elem_in"]
    width = plan["width"]

    # --- source page geometry (one page == one stick of `page_bytes`) --------
    # NB: the stride is measured against the input's *padded* row, which for a
    # ROW_MAJOR-sharded input may be wider than the logical/tile row.
    in_page_bytes = input_tensor.buffer_page_size()
    in_padded_row_bytes = plan["in_padded_width"] * elem_in
    if in_padded_row_bytes % in_page_bytes:
        raise UnsupportedAxisValue(
            f"tilize: input padded row of {in_padded_row_bytes} B is not a whole " f"number of {in_page_bytes} B pages"
        )
    row_page_stride = in_padded_row_bytes // in_page_bytes

    if in_page_bytes % (TILE_HW * elem_in):
        raise UnsupportedAxisValue(
            f"tilize: input page of {in_page_bytes} B is not a whole number of " f"{TILE_HW * elem_in} B tile-columns"
        )

    # A chunk must never straddle a source page, so when a logical row spans
    # several pages the chunk width has to divide BOTH Wt (for the column split)
    # and the page width in tiles (so `byte_offset` stays inside one page).
    page_wt = in_page_bytes // (TILE_HW * elem_in)
    chunk_unit = wt if row_page_stride == 1 else gcd(wt, page_wt)

    # --- planner (op_design.md "Host planner") ------------------------------
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    if not use_multicore:
        # use_multicore=False means EXACTLY one core (the acceptance test and the
        # c_single_core bench regime depend on it) -- the A0 knee term is a G clamp
        # inside the multicore path, never a new user-visible mode.
        max_cores = 1
    else:
        max_cores = a0_active_cores(grid_cores, plan["total_tiles"])

    bytes_per_chunk_tile = tile_in + tile_out
    levers = _lever_flags()
    path = "generic"
    alias_in = alias_out = 0
    shard_tiles = 0
    alias_cb_bytes = 0

    if shard_split is not None:
        # --- Refinement 3, lever C14: the work unit IS the shard ---------------
        path, shard_geo, shard_ht, shard_wt, units = shard_split
        alias_out = int(path == "alias_out")
        alias_in = 1 - alias_out
        shard_tiles = shard_ht * shard_wt
        # The aliased side costs no *CB* L1 (the CB is the tensor's own shard), so
        # only the plain side is budgeted -- and only IT can be chunked. On
        # `alias_in` the unpacker reads the RM shard in place, so a block must be 32
        # whole shard rows: chunk_wt == shard_wt exactly, no chunking possible.
        plain_tile_bytes = tile_in if alias_out else tile_out
        alias_cb_bytes = shard_tiles * (tile_out if alias_out else tile_in)
        depth = depth_request
        if depth * plain_tile_bytes > L1_CB_BUDGET_BYTES:
            depth = 1
        max_chunk_l1 = max(1, L1_CB_BUDGET_BYTES // (depth * plain_tile_bytes))
        if alias_in:
            chunk_wt = shard_wt
            if depth * chunk_wt * plain_tile_bytes > L1_CB_BUDGET_BYTES:
                return _plan_generic(  # too wide a shard to hold a whole block
                    plan,
                    input_tensor,
                    device,
                    in_geo,
                    use_multicore=use_multicore,
                    depth_request=depth_request,
                    chunk_cap=chunk_cap,
                    depth_forced=depth_forced,
                    shard_split=None,
                )
        else:
            max_chunk = min(WT_CHUNK_MAX, max_chunk_l1, shard_wt)
            if chunk_cap is not None:
                max_chunk = min(max_chunk, chunk_cap)
            if CHUNK_CAP_OVERRIDE is not None:  # sweep hook, never set in production
                max_chunk = min(max_chunk, int(CHUNK_CAP_OVERRIDE))
            # A chunk must also not straddle a source page (the generic invariant).
            chunk_wt = _largest_divisor_le(gcd(shard_wt, chunk_unit), max_chunk)
        chunks_per_shard = shard_wt // chunk_wt
        assert shard_wt % chunk_wt == 0, f"chunk_wt={chunk_wt} must divide shard_wt={shard_wt}"

        ncores = len(units)
        if not (levers["r3"] == 2 or (alias_out and alias_out_pays(ncores)) or (alias_in and alias_in_pays(ncores))):
            return _plan_generic(
                plan,
                input_tensor,
                device,
                in_geo,
                use_multicore=use_multicore,
                depth_request=depth_request,
                chunk_cap=chunk_cap,
                depth_forced=depth_forced,
                shard_split=None,
            )

        bank_placement = 0
        cores = [core for core, _, _ in units]
        core_ranges = shard_geo["grid"]
        # The shard grid IS the work grid: n_w shard columns x n_h shard rows.
        n_w = width // (shard_wt * TILE_HW)
        n_h = ncores // n_w
        n_chunks = wt // chunk_wt
        work = [
            {
                "core": core,
                "row_start": sh * shard_ht,
                "row_count": shard_ht,
                "chunk_start": sw * chunks_per_shard,
                "chunk_count": chunks_per_shard,
            }
            for core, sh, sw in units
        ]
    else:
        # "Depth-2 only if it fits": the smallest possible depth-2 footprint is one
        # chunk tile-pair. If even that exceeds the budget, fall back to depth 1
        # rather than OOM (the ttnn.concat pattern). Decided BEFORE the chunk width
        # so `max_chunk_l1` is computed against the depth actually used; the
        # post-loop assert below is then an invariant, not a second clamp.
        depth = depth_request
        if depth * bytes_per_chunk_tile > L1_CB_BUDGET_BYTES:
            depth = 1
        max_chunk_l1 = max(1, L1_CB_BUDGET_BYTES // (depth * bytes_per_chunk_tile))

        n_h = min(nt_h, max_cores)
        want_chunks = _div_up(max_cores, n_h)
        max_chunk_par = max(1, wt // want_chunks)
        max_chunk = min(WT_CHUNK_MAX, max_chunk_l1, max_chunk_par)
        if chunk_cap is not None:
            max_chunk = min(max_chunk, chunk_cap)
        if CHUNK_CAP_OVERRIDE is not None:  # sweep hook, never set in production
            max_chunk = min(max_chunk, int(CHUNK_CAP_OVERRIDE))

        chunk_wt = _largest_divisor_le(chunk_unit, max_chunk)
        assert wt % chunk_wt == 0, f"chunk_wt={chunk_wt} must divide Wt={wt}"
        assert depth * chunk_wt * bytes_per_chunk_tile <= L1_CB_BUDGET_BYTES, (
            f"CB budget blown: depth={depth} chunk_wt={chunk_wt} "
            f"bytes_per_chunk_tile={bytes_per_chunk_tile} > {L1_CB_BUDGET_BYTES}"
        )
        n_chunks = wt // chunk_wt
        n_w = min(n_chunks, max(1, max_cores // n_h))
        ncores = n_h * n_w

        # --- lever A3: work-unit -> core assignment order --------------------
        # Default is A1's row-major line (`grid_to_cores(..., row_wise=True)`).
        # `bank_placement` re-orders the SAME core set so that work unit
        # 0..NUM_BANKS-1 land on the NoC-optimal worker of DRAM bank 0..NUM_BANKS-1.
        # Decided here because it needs `ncores`; whether it fires is gated below.
        bank_placement = int(
            levers["a3"] == 2 or (levers["a3"] == 1 and bank_placement_pays(ncores)),
        )
        if bank_placement:
            cores, core_ranges = _bank_ordered_cores(device, grid, ncores)
        else:
            cores = ttnn.grid_to_cores(ncores, grid.x, grid.y, True)
            core_ranges = ttnn.num_cores_to_corerangeset(ncores, grid, True)

        row_ranges = _split_contiguous(nt_h, n_h)
        chunk_ranges = _split_contiguous(n_chunks, n_w)

        work = []
        for i in range(n_h):
            row_start, row_count = row_ranges[i]
            for j in range(n_w):
                chunk_start, chunk_count = chunk_ranges[j]
                work.append(
                    {
                        "core": cores[i * n_w + j],
                        "row_start": row_start,
                        "row_count": row_count,
                        "chunk_start": chunk_start,
                        "chunk_count": chunk_count,
                    }
                )

    # --- Refinement 1c read-path levers (see the module header) ---------------
    # Both need one source page per logical row: the helpers' page index advances
    # by exactly 1 per row, and B13's "p and p+num_banks share a bank" identity is
    # only a bank *increment* for consecutive pages.
    b13, c7 = levers["b13"], levers["c7"]
    chunk_row_bytes = chunk_wt * TILE_HW * elem_in
    blocks_per_core = max(u["row_count"] * u["chunk_count"] for u in work)
    chunks_per_core = max(u["chunk_count"] for u in work)
    # Refinement 3: with the OUTPUT CB aliased onto the shard, CB page k IS shard
    # tile k and a shard's tiles are row-major, so a shard wider than one chunk
    # forces tile-row-OUTER / chunk-INNER order on the reader. Every lever that owns
    # the block sequence (B8's flattened trid pipeline, C7's chunk-outer hand-off) is
    # therefore excluded; with one chunk per core the two orders coincide and this is
    # 0, which is what keeps C7 available on the crossover regimes.
    blocks_row_major = int(alias_out and chunks_per_core > 1)
    # --- Refinement 3, lever B5/B6: coalesce a block's same-shard page reads ----
    # Structural: the source must be a ROW_MAJOR shard (so its pages are one page
    # COLUMN wide and therefore contiguous in one core's L1), the chunk must cover
    # the whole source page, the shard must be a whole number of tile-rows tall (so
    # a 32-row block never straddles two shards, which are on different cores), and
    # the page must be NoC-aligned (no inter-page padding to skip).
    coalesce_rows = int(
        alias_in == 0
        and in_geo is not None
        and not in_geo["nd"]
        and in_geo["h"] % TILE_HW == 0
        and chunk_row_bytes == in_page_bytes
        # The accessor strides by the ALIGNED page size (`tensor_accessor.h:310`), so
        # the fold is only byte-identical when page == aligned page. 64 covers every
        # allocator alignment on both architectures (L1 16, WH DRAM 32, BH DRAM 64)
        # and is free for every supported dtype (shard_w % 32 == 0 x elem >= 2), which
        # is what keeps this from silently mis-striding a DRAM-sharded source
        # (`ttnn-static-analyzer`, cleared-with-a-caveat #4).
        and in_page_bytes % 64 == 0
        and (levers["coal"] == 2 or (levers["coal"] == 1 and coalesce_rows_pays(chunk_row_bytes, in_page_bytes)))
    )
    # b13 / c7 == 2 force the lever past its payoff gate (the structural conditions
    # still apply) so the bench can measure it on regimes the gate excludes.
    #
    # Refinement 3 relaxes C7's depth-1 clause: BRISC now derives the reserved
    # window from `cb_base + (block % depth) * window_bytes` instead of reading
    # `get_write_ptr`, so the split works at any depth (the identity lever B8
    # already relies on). Its measured PAYOFF clause still stands on the generic
    # path (`split_read_pays`: 1 block, <= 64 B reads); on `alias_out` the writer
    # has no writes left to overlap with, which is what made C7 negative past one
    # block, so the gate there is just "the split is expressible":
    #   * one source page per logical row (the helper's page index steps by 1), and
    #   * one column chunk per core, so both kernels agree on the block order
    #     without also agreeing on the chunk order.
    # `not alias_in` is a HANG fix, not a payoff clause (`ttnn-static-analyzer`, F1).
    # C7 is a two-party protocol keyed on `alias_mode` being the same on both kernels:
    # on `alias_out` the reader (alias_mode 0) reserves and signals while the writer
    # (alias_mode 1) does the read half inside its alias branch. On `alias_in` the two
    # values are SWAPPED, so the party that signals is compiled out and the party that
    # waits is compiled in -- BRISC spins in `noc_semaphore_wait_min(sem_reserve)`
    # forever. Reachable from a plain public call (a HEIGHT-sharded RM input with a
    # 32-wide shard gives `chunk_row_bytes == 64` and 1 block/core, which is exactly
    # `split_read_pays`), and invisible to a suite whose sharded cells are all 64 wide.
    # Excluded rather than wired up on purpose: on `alias_in` the CB *is* the input
    # shard, so a second reader writing into it would overwrite the source.
    split_read_expressible = row_page_stride == 1 and not coalesce_rows and not blocks_row_major and not alias_in
    if alias_out:
        # MEASURED NO. The freed BRISC really is idle, and C7 really does halve the
        # reads each RISC-V issues -- but the read leg is DRAM-BANDWIDTH bound
        # (10 199 ns for 2.10 MB = 206 GB/s, against a 214 GB/s best), not
        # issue-rate bound, so a second issuer buys nothing and its per-block
        # hand-off costs. In-run A/B at depth 2 on `g_dram_to_sharded`:
        # **17 699 ns with C7 vs 15 866 without (+11.6 %)**; at depth 1,
        # 18 113 vs 16 480 (+9.9 %). The `x_g_alias_d*_c7` bench rows keep it
        # re-measurable. `chunks_per_core == 1` remains the structural clause (both
        # kernels must agree on the block order) for the forced counterfactual.
        c7_pays = False
    else:
        c7_pays = depth == 1 and split_read_pays(depth, blocks_per_core, chunk_row_bytes)
    split_read = int(split_read_expressible and (c7 == 2 or (c7 == 1 and c7_pays)))
    # The two levers are mutually exclusive by measurement, not by construction:
    # C7 already halves the reads each RISC-V issues, so B13's saved command
    # programming is halved too while its bank-major serialization cost is not.
    # Three independent in-run A/B pairs on `[1,1,2048,32]`, 7-12 rounds x 10
    # launches: C7 alone 3411.1 / 3404.1 / 3419.6 ns vs C7+B13 3462.4 / 3431.6 /
    # 3434.2 ns -- B13 on top of C7 is +0.9 % on average and never negative, i.e.
    # it does not pay where C7 runs. Every regime therefore ships exactly ONE of
    # the two levers.
    stateful_read = int(
        row_page_stride == 1
        and not coalesce_rows
        and not alias_in
        and (b13 == 2 or (b13 == 1 and not split_read and stateful_read_pays(chunk_row_bytes)))
    )

    # --- lever B8: trid double-issue on the read path -------------------------
    # Structural preconditions (never overridden by the force flag):
    #   * one source page per logical row -- the prefetch loop calls the same
    #     helper the non-prefetched path does, which hard-codes that.
    #   * not the split reader -- C7 already has BRISC writing into the window
    #     NCRISC reserved, and that hand-off assumes exactly ONE live window.
    #   * >= 2 chunk-blocks on the busiest core -- otherwise there is no next
    #     block to keep in flight.
    #   * the depth-3 footprint has to fit, at the UNCHANGED chunk width.
    prefetch_cb_bytes = PREFETCH_DEPTH * chunk_wt * bytes_per_chunk_tile
    b8 = levers["b8"]
    prefetch_ok = (
        row_page_stride == 1
        and not split_read
        and not coalesce_rows
        and not alias_in
        and not blocks_row_major
        and not depth_forced
        and blocks_per_core >= TRID_PREFETCH_MIN_BLOCKS
        and prefetch_cb_bytes <= L1_CB_BUDGET_PREFETCH_BYTES
    )
    # MEASURED NO on `alias_out`, for the same reason C7 fails there: B8's whole
    # point is to keep reads in flight across the per-block barrier, and this path's
    # read leg is already at the DRAM read rate with 32 reads in flight. Its own
    # size clause (<= 128 B) would otherwise fire on every crossover shard narrower
    # than 8 tiles. In-run A/B on `g_dram_to_sharded`: **17 440 ns with B8 (depth 3)
    # vs 15 866 without (+9.9 %)**; `x_g_alias_d3_b8` keeps it re-measurable.
    b8_pays = (not alias_out) and trid_prefetch_pays(ncores, blocks_per_core, chunk_row_bytes, prefetch_cb_bytes)
    prefetch_blocks = 2 if (prefetch_ok and (b8 == 2 or (b8 == 1 and b8_pays))) else 1
    if prefetch_blocks == 1 and b8 == 3 and prefetch_ok:
        # Isolation row for the Mode-C ledger: B8 bundles TWO changes (a third CB
        # window and the trid pipeline). `TILIZE_LEVER_B8=3` gives the extra window
        # WITHOUT the trid pipeline, so the ledger can attribute the delta.
        depth = PREFETCH_DEPTH
    if prefetch_blocks == 2:
        # B8 owns the read command programming for the whole block sequence, so it
        # is mutually exclusive with B13's bank-major arming (measured -- see the
        # ledger). B13's own gate keeps it under 128 B reads, which only overlaps
        # B8 on the narrow multi-block shapes.
        stateful_read = 0
        depth = PREFETCH_DEPTH
        assert prefetch_cb_bytes <= L1_CB_BUDGET_PREFETCH_BYTES, (
            f"B8 prefetch CB budget blown: {prefetch_cb_bytes} B/core > "
            f"{L1_CB_BUDGET_PREFETCH_BYTES} B (chunk_wt={chunk_wt})"
        )

    # --- Refinement 3, lever B7': one barrier per GROUP of blocks ---------------
    # Structural: only on the `alias_out` read loop (the one whose ablation showed a
    # barrier-drain cost), only where no other lever owns the block sequence, and
    # only on the delegated depth default -- the group needs G+1 CB windows, and a
    # caller who pinned `use_double_buffer` gets the depth they asked for.
    read_group = 1
    if alias_out and not depth_forced and not split_read and prefetch_blocks == 1:
        want = int(READ_GROUP_OVERRIDE) if READ_GROUP_OVERRIDE is not None else read_group_pays(blocks_per_core)
        want = max(1, min(want, blocks_per_core))
        group_cb_bytes = (want + 1) * chunk_wt * (tile_in if alias_out else tile_out)
        if want > 1 and group_cb_bytes <= L1_CB_BUDGET_PREFETCH_BYTES:
            read_group = want
            depth = want + 1
    assert read_group == 1 or depth >= read_group + 1, "B7' needs one CB window more than the group"
    # Address-generation ceiling probe (bench only; output is garbage by design).
    addr_probe = int(levers["addr"] == 1 and alias_out and not split_read and prefetch_blocks == 1)

    # --- Refinement 2b: whole-page staged read + L1 redistribution -------------
    # Structural preconditions (never overridden by the force flag):
    #   * one source page per logical row, and one chunk-block per core -- the
    #     scheme has a single un-flow-controlled staging window per core, so a
    #     second block would need a "window free" round trip per block.
    #   * the whole tile-row is split by COLUMN only (`n_h == 1`, `n_w == ncores`):
    #     that is the fan-in this addresses, and it makes the piece exactly
    #     FANIN_GROUP_ROWS chunks wide on every core.
    #   * `ncores` a whole number of groups of FANIN_GROUP_ROWS, and a group has to
    #     be a core RECTANGLE so the kernel can address mate `r` from the group's
    #     two physical coordinate axes (grid.x words + group_rows/grid.x words)
    #     instead of 2 * group_size runtime args.
    #   * row-major core order (lever A3's permutation would break the rectangle).
    #   * the staging buffer must fit on top of both data CBs.
    r2b = levers["r2b"]
    fanin_groups = ncores // FANIN_GROUP_ROWS
    piece_bytes = FANIN_GROUP_ROWS * chunk_row_bytes
    fanin_cb_bytes = depth * chunk_wt * bytes_per_chunk_tile + piece_bytes
    fanin_ok = (
        shard_split is None
        and row_page_stride == 1
        and n_h == 1
        and n_w == ncores
        and blocks_per_core == 1
        and ncores % FANIN_GROUP_ROWS == 0
        and fanin_groups >= 1
        and FANIN_GROUP_ROWS % grid.x == 0
        and not bank_placement
        and not split_read
        and prefetch_blocks == 1
        and fanin_cb_bytes <= L1_CB_BUDGET_FANIN_BYTES
    )
    if r2b == 3 and fanin_ok:
        fanin_mode = 2  # measurement probe: phase 1 only, output is garbage
    elif fanin_ok and (r2b == 2 or (r2b == 1 and fanin_pays(chunk_row_bytes, fanin_groups))):
        fanin_mode = 1
    else:
        fanin_mode = 0

    fanin_group_axes = None
    if fanin_mode:
        # B13 reshapes the same stick reads this path replaces, so it yields.
        stateful_read = 0
        if fanin_mode == 1:
            # Group g == work units [g*rows, (g+1)*rows) == a logical rectangle
            # `grid.x` wide (guaranteed by `FANIN_GROUP_ROWS % grid.x == 0`).
            group_h = FANIN_GROUP_ROWS // grid.x
            fanin_group_axes = []
            for g in range(fanin_groups):
                y0 = g * group_h
                xs = [int(device.worker_core_from_logical_core(ttnn.CoreCoord(x, y0)).x) for x in range(grid.x)]
                ys = [int(device.worker_core_from_logical_core(ttnn.CoreCoord(0, y0 + dy)).y) for dy in range(group_h)]
                fanin_group_axes.append((xs, ys))
            # The kernel derives mate r as (xs[r % grid.x], ys[r // grid.x]); assert
            # the host's work->core order agrees, so a future change to
            # `grid_to_cores` cannot silently transpose the exchange.
            for g in range(fanin_groups):
                for r in range(FANIN_GROUP_ROWS):
                    core = cores[g * FANIN_GROUP_ROWS + r]
                    want = ttnn.CoreCoord(r % grid.x, g * group_h + r // grid.x)
                    assert (int(core.x), int(core.y)) == (int(want.x), int(want.y)), (
                        f"fan-in group {g} mate {r}: core order is ({core.x},{core.y}) but the "
                        f"kernel's rectangle indexing gives ({want.x},{want.y})"
                    )

    # --- Refinement 2b: per-core transaction-order rotation --------------------
    # Structurally it only needs the row loop to be this kernel's own (so not B8 /
    # C7 / B13 / fan-in, each of which owns it) and more than one core (with one core
    # there is nothing to de-cluster).
    # Force values mirror B10's convention: 2 = both halves, 3 = read only,
    # 4 = write only, so the ledger can attribute the delta to one mechanism.
    stg = levers["stg"]
    stagger_ok = (
        shard_split is None
        and row_page_stride == 1
        and not coalesce_rows
        and fanin_mode == 0
        and not split_read
        and prefetch_blocks == 1
        and not stateful_read
        and ncores > 1
    )
    if not stagger_ok or stg == 0:
        stagger = 0
    elif stg == 2:
        stagger = STAGGER_READ | STAGGER_WRITE
    elif stg == 3:
        stagger = STAGGER_READ
    elif stg == 4:
        stagger = STAGGER_WRITE
    else:
        stagger = stagger_pays(ncores, nt_h, chunk_wt)
    # The write rotation is a no-op with a single page per block; keep the plan value
    # honest so the bench column and the tests report what actually happens.
    if chunk_wt == 1:
        stagger &= ~STAGGER_WRITE

    # --- Refinement 3b, lever 3: hoisted interleaved bank table ------------------
    # B13's address generation, hoisted out of the block loop (see `bank_table_pays`
    # and `InterleavedStickBands`). Decided LAST among the read-path levers because
    # it is exclusive with every one of them -- it owns the whole 32-read row loop --
    # and it SUPERSEDES B13, whose per-band table it replaces.
    bt = levers["bt"]
    bank_table_ok = (
        row_page_stride == 1
        # The affine page->bank map only exists on an INTERLEAVED source.
        and in_geo is None
        and not alias_in
        and not coalesce_rows
        and not blocks_row_major
        and not split_read
        and prefetch_blocks == 1
        and read_group == 1
        and not addr_probe
        and not fanin_mode
        and not stagger
    )
    bank_table = int(bank_table_ok and (bt == 2 or (bt == 1 and bank_table_pays(blocks_per_core, chunk_row_bytes))))
    if bank_table:
        # One or the other, never both: the hoisted table IS the bank-major arming.
        stateful_read = 0

    # --- Refinement 3b, lever 2: drop the do-nothing writer kernel ---------------
    # Only on `alias_out`, where the writer has no NoC traffic at all, and only
    # when nothing else needs it: lever C7 uses the writer as the SECOND read
    # issuer, and the zone variant instruments its wait/pop.
    nw = levers["nw"]
    drop_writer = int(
        alias_out and not split_read and (nw == 2 or (nw == 1 and drop_writer_pays())) and _zone_flag() == 0
    )

    # --- lever B10: per-reader / per-writer static unicast VC -----------------
    # `vc_spread` is a BITMASK -- bit 0 = spread the reads, bit 1 = spread the
    # writes. Read and write live on different NoCs (B9) and are programmed by
    # completely different mechanisms (sticky NOC_CTRL vs per-call), so the ledger
    # needs to attribute the delta to one half or the other rather than to "B10".
    # Force values: 2 = both, 3 = reads only, 4 = writes only.
    b10 = levers["b10"]
    if b10 == 2:
        vc_spread = VC_SPREAD_READ | VC_SPREAD_WRITE
    elif b10 == 3:
        vc_spread = VC_SPREAD_READ
    elif b10 == 4:
        vc_spread = VC_SPREAD_WRITE
    elif b10 == 1 and vc_spread_pays(ncores):
        vc_spread = VC_SPREAD_READ | VC_SPREAD_WRITE
    else:
        vc_spread = 0
    # Rotate over the 4 unicast VCs by work-unit index. Reads and writes are on
    # different NoCs (B9), so they get independent rotations; offsetting the write
    # rotation by half the VC count keeps a core's read and write VCs different.
    read_vcs = [i % NUM_UNICAST_VCS for i in range(len(work))] if vc_spread & VC_SPREAD_READ else None
    write_vcs = (
        [(i + NUM_UNICAST_VCS // 2) % NUM_UNICAST_VCS for i in range(len(work))]
        if vc_spread & VC_SPREAD_WRITE
        else None
    )

    plan.update(
        {
            "path": path,
            # Refinement 3: which SIDE is aliased onto its shard. "generic" has
            # neither, Path B ("alias") has both, and a crossover has exactly one.
            "alias_in": alias_in,
            "alias_out": alias_out,
            "coalesce_rows": coalesce_rows,
            "blocks_row_major": blocks_row_major,
            "read_group": read_group,
            "addr_probe": addr_probe,
            "core_ranges": core_ranges,
            "cores": cores,
            "work": work,
            "chunk_wt": chunk_wt,
            "chunk_row_bytes": chunk_row_bytes,
            "row_page_stride": row_page_stride,
            "source_page_bytes": in_page_bytes,
            "shard_tiles": shard_tiles,
            "stateful_read": stateful_read,
            "bank_table": bank_table,  # Refinement 3b lever 3
            "drop_writer": drop_writer,  # Refinement 3b lever 2
            # Refinement 4 lever 1 is Path-B only: on the generic and one-sided-alias
            # paths the reader is the op's whole data movement.
            "drop_reader": 0,
            "self_arm": 0,
            "fp32_lossless": _fp32_lossless(plan),  # Refinement 4 lever 3
            "split_read": split_read,
            "prefetch_blocks": prefetch_blocks,
            "vc_spread": vc_spread,
            "bank_placement": bank_placement,
            "read_vcs": read_vcs,
            "write_vcs": write_vcs,
            "stagger": stagger,
            "fanin_mode": fanin_mode,
            "fanin_groups": fanin_groups if fanin_mode else 0,
            "fanin_group_rows": FANIN_GROUP_ROWS if fanin_mode else 0,
            # The kernel divides by this to index the group rectangle, so it must be
            # non-zero even when the path is off (`grp_h = group_rows / grp_w`).
            "fanin_grid_x": grid.x if fanin_mode else 1,
            "fanin_group_axes": fanin_group_axes,
            "piece_bytes": piece_bytes if fanin_mode else 0,
            "depth": depth,
            "n_h": n_h,
            "n_w": n_w,
            "ncores": ncores,
            # Busiest core's chunk-block count -- the C16 gate's "is there
            # anything to pipeline?" input, and the per-block sync cost's
            # multiplier (measured ~612 ns/block, see A0_KNEE_CORES).
            "blocks_per_core": blocks_per_core,
            "chunks_per_core": chunks_per_core,
            # ALLOCATED CB bytes per core. On a one-sided alias the aliased side is
            # the tensor's own shard (not CB L1), so only the plain side counts here
            # and `alias_cb_bytes` reports the shard it points at -- which is what
            # keeps `PROPERTIES["bounded_cb"]` a statement about the op's own sizing.
            "cb_bytes_per_core": (
                fanin_cb_bytes
                if fanin_mode == 1
                else depth * chunk_wt * ((tile_in if alias_out else tile_out) if shard_split else bytes_per_chunk_tile)
            ),
            "alias_cb_bytes": alias_cb_bytes,
        }
    )
    return plan


# ---------------------------------------------------------------------------
# ComputeConfigDescriptor
# ---------------------------------------------------------------------------

_FP32_DEST_IN = (ttnn.float32, ttnn.uint32, ttnn.int32)
_FP32_DEST_OUT = (ttnn.float32, ttnn.bfloat8_b, ttnn.uint32, ttnn.int32)


def _compute_config(in_dtype, out_dtype, fp32_lossless=1):
    """ComputeConfig for the tilize kernel.

    ``fp32_lossless`` (Refinement 4 lever 3) MUST agree with the compute kernel's
    ``Fp32Mode`` on the same cell — the two are one decision split across the host
    and the kernel, and the helper enforces the pairing with a ``static_assert`` in
    both directions:

    * ``Fp32Mode::Lossless`` requires ``unpack_to_dest_mode[CB_RM_INPUT] =
      UnpackToDestFp32`` (otherwise the slow path still truncates fp32 to tf32 on
      its way into DEST and the "lossless" promise is silently broken); and
    * ``Fp32Mode::Fast`` on an fp32 input requires ``Default`` (fast tilize's pack
      stage steps DEST at bf16 stride, so a 32-bit unpack payload would be read as
      tf32 and the output would be silently corrupt).

    Both are compile-time, so a mismatch fails the JIT build rather than the test.
    """
    fp32_dest_acc_en = in_dtype in _FP32_DEST_IN or out_dtype in _FP32_DEST_OUT

    config = ttnn.ComputeConfigDescriptor()
    config.fp32_dest_acc_en = fp32_dest_acc_en
    # `can_use_fast_tilize` requires !get_dst_full_sync_enabled().
    config.dst_full_sync_en = False
    if fp32_dest_acc_en:
        # Must be assigned wholesale: nanobind's bound vector copies on
        # __getitem__, so in-place element assignment is silently dropped.
        modes = [ttnn.UnpackToDestMode.Default] * 32
        # An fp32 input on the FAST path must stay Default; every other input format
        # either needs the fp32 unpack (Lossless) or ignores it (uint32/int32/bf16,
        # where `is_fp32_input_format` is false so the kernel never asks for
        # Lossless and `has_supported_fast_tilize_format` never selects fast).
        if fp32_lossless or in_dtype != ttnn.float32:
            modes[CB_RM_INPUT] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    return config


# ---------------------------------------------------------------------------
# CB descriptors
# ---------------------------------------------------------------------------


def _plain_cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _aliased_cb(index, tensor, page_size, num_pages, core_ranges):
    """CB whose L1 base address *is* the tensor's shard base address."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(
        index, tensor, total_size=num_pages * page_size, core_ranges=core_ranges
    )
    # Read-modify-write-back: the bound vector copies on __getitem__.
    format_descriptors = cb.format_descriptors
    format_descriptors[0].page_size = page_size
    cb.format_descriptors = format_descriptors
    return cb


# ---------------------------------------------------------------------------
# ProgramDescriptor
# ---------------------------------------------------------------------------


def _ablation_flags():
    """Perf-ablation compile-time flags (/perf-measure stage attribution).

    ``TILIZE_SKIP_DM=1`` drops the noc_async_read/write payload, ``TILIZE_SKIP_COMPUTE=1``
    drops the tilize LLK; both keep every CB op, barrier and loop trip count so the
    synchronization structure — and therefore the timing structure — is unchanged.
    Output is garbage by design; only ``_bench_tilize.py`` sets these.

    ``TILIZE_SKIP_DM`` also has two ONE-SIDED values, which is what decomposes a
    serialized (depth-1, one-block) regime into its read and write legs — the
    measurement Refinement 2b needs to know how much of `b_wide_short` is the read:

    ==== ==================================================================
    1    drop both legs (the classic `no_dm` ablation)
    2    drop the READ leg only  -> the remaining time is write + compute + sync
    3    drop the WRITE leg only -> the remaining time is read + compute + sync
    ==== ==================================================================
    """
    skip_dm = int(os.environ.get("TILIZE_SKIP_DM", "0"))
    return (
        1 if skip_dm in (1, 2) else 0,  # reader
        1 if skip_dm in (1, 3) else 0,  # writer
        int(os.environ.get("TILIZE_SKIP_COMPUTE", "0")),
    )


def _zone_flag():
    """Refinement 3b lever 1 -- the per-RISC Tracy timeline (MEASUREMENT ONLY).

    ``TILIZE_ZONES=1`` selects an instrumented copy of the per-block loop in each
    of the three kernels, with a ``DeviceZoneScopedN`` around every stage
    (reader: reserve / issue / barrier / push; compute: wait / LLK; writer:
    wait / pop). The output is still CORRECT -- unlike the ablations this changes
    no read, no CB count and no trip count -- but the zone writes perturb the
    timing, so it is never on in a shipped plan. It answers the one question no
    ablation variant can: which RISC is blocked, and on what.

    Off unless the bench asks for it. Cannot be combined with the ablation flags
    (the instrumented loop replaces the branch they patch), nor with the levers
    that own the read loop -- the reader carries a ``static_assert`` for that.
    """
    return int(os.environ.get("TILIZE_ZONES", "0"))


def create_program_descriptor(input_tensor, output_tensor, plan) -> ttnn.ProgramDescriptor:
    alias = plan["path"] == "alias"
    # Refinement 3: each side is aliased independently. Path B has both, a crossover
    # exactly one, the generic path neither. The aliased side's CB base address IS
    # the shard's, and the work split guarantees this core owns exactly that shard's
    # tiles in the CB's page order (`_one_sided_shard_split`).
    alias_in = bool(plan["alias_in"])
    alias_out = bool(plan["alias_out"])
    core_ranges = plan["core_ranges"]
    chunk_wt = plan["chunk_wt"]
    reader_skip_dm, writer_skip_dm, skip_compute = _ablation_flags()
    zones = _zone_flag()

    # ---------------- circular buffers ----------------
    shard_tiles = plan["shard_tiles"]
    plain_pages = plan["depth"] * chunk_wt
    if alias_in:
        cb_rm_input = _aliased_cb(CB_RM_INPUT, input_tensor, plan["tile_in"], shard_tiles, core_ranges)
    else:
        cb_rm_input = _plain_cb(CB_RM_INPUT, input_tensor.dtype, plan["tile_in"], plain_pages, core_ranges)
    if alias_out:
        cb_tiled_output = _aliased_cb(CB_TILED_OUTPUT, output_tensor, plan["tile_out"], shard_tiles, core_ranges)
    else:
        cb_tiled_output = _plain_cb(CB_TILED_OUTPUT, output_tensor.dtype, plan["tile_out"], plain_pages, core_ranges)

    # Refinement 2b staging buffer. One page of `piece_bytes`, never reserved or
    # pushed -- the reader owns it as scratch (`get_write_ptr` is the base address
    # every launch, because the firmware re-inits the CB interfaces per launch), and
    # group-mates read it at the SAME L1 offset on their own cores.
    cb_stage = None
    if plan["fanin_mode"] == 1:
        cb_stage = _plain_cb(CB_STAGE, input_tensor.dtype, plan["piece_bytes"], 1, core_ranges)

    split_read = plan["split_read"]
    stateful_read = plan["stateful_read"]
    prefetch_blocks = plan["prefetch_blocks"]
    vc_spread = plan["vc_spread"]

    # ---------------- reader ----------------
    reader_ct_args = [
        1 if alias_in else 0,
        chunk_wt,
        plan["chunk_row_bytes"],
        plan["row_page_stride"],
        plan["source_page_bytes"],
        plan["shard_tiles"],
        reader_skip_dm,
        stateful_read,
        split_read,
        SEM_SPLIT_RESERVE,
        SEM_SPLIT_DONE,
        prefetch_blocks,  # lever B8: 1 = off, 2 = trid double-issue
        vc_spread,  # lever B10: read VC comes from runtime arg 5
        plan["depth"],  # CB windows -- the B8 prefetch's own window arithmetic
        PREFETCH_TRIDS[0],
        PREFETCH_TRIDS[1],
        DEFAULT_UNICAST_VC,
        plan["fanin_mode"],  # Refinement 2b: 0 off, 1 redistribute, 2 read-probe
        plan["piece_bytes"],
        plan["fanin_group_rows"],
        plan["fanin_grid_x"],
        SEM_FANIN_READY,
        CB_STAGE,
        plan["stagger"] & STAGGER_READ,  # Refinement 2b: rotate the read issue order
        # Refinement 3: coalesce a block's 32 same-shard page reads into one, and
        # iterate tile-row-outer / chunk-inner (which is the order the aliased
        # OUTPUT CB's pages are in -- see `_one_sided_shard_split`).
        plan["coalesce_rows"],
        plan["blocks_row_major"],
        plan["read_group"],  # lever B7': blocks per read barrier
        plan["addr_probe"],  # measurement probe (bench only, garbage output)
        zones,  # Refinement 3b lever 1: per-RISC Tracy timeline (bench only)
        plan["bank_table"],  # Refinement 3b lever 3: hoisted interleaved bank table
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---------------- writer ----------------
    # On the split-read path the writer also reads, so it carries the INPUT
    # accessor as well. Both TensorAccessorArgs are emitted unconditionally (the
    # kernel declares them outside any `if constexpr`), so the CT arg layout does
    # not depend on the lever.
    writer_ct_args = [
        1 if alias_out else 0,
        chunk_wt,
        plan["tile_out"],
        plan["wt"],
        plan["shard_tiles"],
        writer_skip_dm,
        split_read,
        plan["chunk_row_bytes"],
        stateful_read,
        SEM_SPLIT_RESERVE,
        SEM_SPLIT_DONE,
        vc_spread,  # lever B10: write VC comes from runtime arg 6
        1 if (plan["stagger"] & STAGGER_WRITE) else 0,  # Refinement 2b: write order
        # Refinement 3: the input CB's window count, so lever C7's read half can
        # derive the window NCRISC reserved at any depth (it was depth-1 only).
        plan["depth"],
        zones,  # Refinement 3b lever 1: per-RISC Tracy timeline (bench only)
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---------------- compute ----------------
    # Refinement 4 lever 1. `no_wait` == nobody publishes the input CB (the bytes are
    # already at its address), so the helper's per-block `cb_wait_front` must go;
    # `self_arm` == the `zero_copy_fold` counterfactual, where the arm/drain is folded
    # onto TRISC instead of deleted (so the wait stays and is satisfied by the kernel's
    # own push).
    drop_reader = bool(plan["drop_reader"])
    self_arm = bool(plan["self_arm"])
    compute_ct_args = [
        chunk_wt,
        plan["needs_cast"],
        skip_compute,
        zones,
        int(drop_reader and not self_arm),  # no_wait
        int(self_arm),
        plan["shard_tiles"],
        plan["fp32_lossless"],  # Refinement 4 lever 3
        _lever_flags()["iu"],  # Refinement 4 lever 2 (measurement)
    ]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    # Refinement 3b lever 2 / Refinement 4 lever 1. With a dataflow kernel gone the
    # base address it carried has no other carrier, and an aliased CB is not a
    # `Buffer*` runtime arg that would force a re-patch -- so it rides on the compute
    # kernel. Nothing reads them; their job is to exist
    # (`test_alias_program_cache_rebinding` / `test_path_b_program_cache_rebinding`).
    drop_writer = bool(plan["drop_writer"])
    compute_tail = ([src_addr] if drop_reader else []) + ([dst_addr] if drop_writer else [])

    if alias:
        for core in plan["cores"]:
            reader_rt[core.x][core.y] = [src_addr]
            writer_rt[core.x][core.y] = [dst_addr]
            compute_rt[core.x][core.y] = [plan["num_blocks"]] + compute_tail
    else:
        read_vcs = plan["read_vcs"]
        write_vcs = plan["write_vcs"]
        fanin_mode = plan["fanin_mode"]
        group_rows = plan["fanin_group_rows"]
        group_axes = plan["fanin_group_axes"]
        for index, unit in enumerate(plan["work"]):
            core = unit["core"]
            row_start = unit["row_start"]
            row_count = unit["row_count"]
            chunk_start = unit["chunk_start"]
            chunk_count = unit["chunk_count"]
            reader_rt[core.x][core.y] = [
                src_addr,
                row_start * TILE_HW,
                row_count * TILE_HW,
                chunk_start,
                chunk_count,
                # lever B10 (arg 5). Always emitted so the arg layout does not
                # depend on the lever; only read when `vc_spread`.
                read_vcs[index] if read_vcs is not None else DEFAULT_UNICAST_VC,
                # Refinement 2b (arg 6): rotation of this core's 32 stick reads.
                # Rotating by the work-unit index de-clusters the instantaneous bank
                # demand; TILE_HW is the period of the row loop.
                index % (STAGGER_MOD_OVERRIDE or TILE_HW) % TILE_HW,
            ]
            if fanin_mode:
                # Refinement 2b. Work unit `index` lives in group `g` at slot `slot`;
                # it STAGES source row `slot` of piece `g` and OWNS column slice
                # `slot` of that piece, so one number does both jobs.
                g, slot = divmod(index, group_rows)
                reader_rt[core.x][core.y].extend(
                    [
                        row_start * TILE_HW + slot,  # source page (row) to stage
                        g * plan["piece_bytes"],  # byte offset of this piece
                        slot,  # this core's slice inside the piece
                    ]
                )
                if fanin_mode == 1:
                    xs, ys = group_axes[g]
                    reader_rt[core.x][core.y].extend(xs)
                    reader_rt[core.x][core.y].extend(ys)
            # src_addr is appended after the existing indices; the writer only
            # reads it on the split-read path. Then the B10 write VC (arg 6).
            writer_rt[core.x][core.y] = [
                dst_addr,
                row_start,
                row_count,
                chunk_start,
                chunk_count,
                src_addr,
                write_vcs[index] if write_vcs is not None else DEFAULT_UNICAST_VC,
                # Refinement 2b (arg 7): rotation of this core's chunk_wt tile writes.
                index % chunk_wt,
            ]
            compute_rt[core.x][core.y] = [row_count * chunk_count] + compute_tail

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=_compute_config(input_tensor.dtype, output_tensor.dtype, plan["fp32_lossless"]),
    )

    # Lever C7's NCRISC <-> BRISC handshake. Two monotonic per-launch counters in
    # this core's own L1 (set/wait are local loads and stores, no NoC round trip);
    # the dispatcher re-writes the initial values on every launch, which is what
    # makes a monotonic counter safe across launches.
    semaphores = []
    if split_read:
        semaphores = [
            ttnn.SemaphoreDescriptor(id=SEM_SPLIT_RESERVE, core_ranges=core_ranges, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_SPLIT_DONE, core_ranges=core_ranges, initial_value=0),
        ]
    elif plan["fanin_mode"] == 1:
        # Refinement 2b's group-ready counter. The dispatcher re-writes the initial
        # value on every launch, which is what makes a monotonic counter safe across
        # launches of a cached program.
        semaphores = [ttnn.SemaphoreDescriptor(id=SEM_FANIN_READY, core_ranges=core_ranges, initial_value=0)]

    cbs = [cb_rm_input, cb_tiled_output]
    if cb_stage is not None:
        cbs.append(cb_stage)

    # Refinement 3b lever 2: the do-nothing writer is not launched at all on
    # `alias_out` (see `drop_writer_pays`). The aliased output CB has exactly as
    # many pages as compute pushes, so nothing ever has to recycle it.
    # Refinement 4 lever 1: on Path B the reader goes too, leaving a COMPUTE-ONLY
    # program (see `drop_dataflow_pays`).
    kernels = []
    if not drop_reader:
        kernels.append(reader_kernel)
    if not drop_writer:
        kernels.append(writer_kernel)
    kernels.append(compute_kernel)

    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=semaphores,
        cbs=cbs,
    )
