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

CB_RM_INPUT = 0
CB_TILED_OUTPUT = 16

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


def _lever_flags():
    """Env counterfactual switches for the read-path levers (Mode-C ledger).

    Default 1 == the lever is offered to the plan, 0 == off, 2 == forced past its
    payoff gate (structural preconditions still apply). `_bench_tilize.py` flips
    one at a time so each lever has a re-measurable counterfactual row instead of
    a changelog claim. Never set in production.
    """
    return dict(
        b13=int(os.environ.get("TILIZE_LEVER_B13", "1")),
        c7=int(os.environ.get("TILIZE_LEVER_C7", "1")),
        b8=int(os.environ.get("TILIZE_LEVER_B8", "1")),
        b10=int(os.environ.get("TILIZE_LEVER_B10", "1")),
        a3=int(os.environ.get("TILIZE_LEVER_A3", "1")),
    )


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
    }

    # Path B is inherently multi-core (one shard per core), so an explicit
    # use_multicore=False request routes to the generic single-core path
    # instead of being refused.
    if use_multicore and _alias_eligible(in_geo, out_geo, folded_h, width):
        return _plan_alias(plan, in_geo)

    chunk_cap = None
    if use_double_buffer is None:
        # C16 gate. The depth feeds the L1 chunk-width budget, which feeds the
        # 2D split, which decides ncores / blocks-per-core -- i.e. the gate's own
        # inputs. Resolve it with a depth-2 trial plan (pure host arithmetic, no
        # device work) and re-plan once at the chosen depth.
        trial = _plan_generic(dict(plan), input_tensor, device, in_geo, use_multicore=use_multicore, depth_request=2)
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
    )


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

    plan.update(
        {
            "path": "alias",
            "core_ranges": grid,
            "cores": cores,
            "chunk_wt": chunk_wt,
            "shard_tiles": shard_tiles,
            "num_blocks": num_blocks,
            "blocks_per_core": num_blocks,
            "depth": 1,  # the CB *is* the shard; use_double_buffer is inert here
            "row_page_stride": 1,
            "source_page_bytes": shard_w * plan["elem_in"],
            "chunk_row_bytes": shard_w * plan["elem_in"],
            # Path B has no NoC traffic at all, so no transaction-shaping lever
            # applies (B13 / C7 / B8 / B10 / A3 all move NoC commands around).
            "stateful_read": 0,
            "split_read": 0,
            "prefetch_blocks": 1,
            "vc_spread": 0,
            "bank_placement": 0,
            "read_vcs": None,
            "write_vcs": None,
            "ncores": len(cores),
            "cb_bytes_per_core": shard_tiles * (plan["tile_in"] + plan["tile_out"]),
        }
    )
    return plan


def _plan_generic(
    plan, input_tensor, device, in_geo, *, use_multicore, depth_request, chunk_cap=None, depth_forced=False
):
    """Path A/C: 2D height-first rectangular split over the compute grid.

    ``chunk_cap`` pins the chunk width from a previous (depth-2) pass so the C16
    depth gate cannot change the transaction shape — see ``build_plan``.
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

    chunk_wt = _largest_divisor_le(chunk_unit, max_chunk)
    assert wt % chunk_wt == 0, f"chunk_wt={chunk_wt} must divide Wt={wt}"
    assert depth * chunk_wt * bytes_per_chunk_tile <= L1_CB_BUDGET_BYTES, (
        f"CB budget blown: depth={depth} chunk_wt={chunk_wt} "
        f"bytes_per_chunk_tile={bytes_per_chunk_tile} > {L1_CB_BUDGET_BYTES}"
    )
    n_chunks = wt // chunk_wt
    n_w = min(n_chunks, max(1, max_cores // n_h))
    ncores = n_h * n_w

    # --- lever A3: work-unit -> core assignment order ------------------------
    # Default is A1's row-major line (`grid_to_cores(..., row_wise=True)`).
    # `bank_placement` re-orders the SAME core set so that work unit 0..NUM_BANKS-1
    # land on the NoC-optimal worker of DRAM bank 0..NUM_BANKS-1. Decided here
    # because it needs `ncores`; whether it fires is gated below.
    levers = _lever_flags()
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
    # b13 / c7 == 2 force the lever past its payoff gate (the structural conditions
    # still apply) so the bench can measure it on regimes the gate excludes.
    split_read = int(
        row_page_stride == 1
        and depth == 1
        and (c7 == 2 or (c7 == 1 and split_read_pays(depth, blocks_per_core, chunk_row_bytes)))
    )
    # The two levers are mutually exclusive by measurement, not by construction:
    # C7 already halves the reads each RISC-V issues, so B13's saved command
    # programming is halved too while its bank-major serialization cost is not.
    # Three independent in-run A/B pairs on `[1,1,2048,32]`, 7-12 rounds x 10
    # launches: C7 alone 3411.1 / 3404.1 / 3419.6 ns vs C7+B13 3462.4 / 3431.6 /
    # 3434.2 ns -- B13 on top of C7 is +0.9 % on average and never negative, i.e.
    # it does not pay where C7 runs. Every regime therefore ships exactly ONE of
    # the two levers.
    stateful_read = int(
        row_page_stride == 1 and (b13 == 2 or (b13 == 1 and not split_read and stateful_read_pays(chunk_row_bytes)))
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
        and not depth_forced
        and blocks_per_core >= TRID_PREFETCH_MIN_BLOCKS
        and prefetch_cb_bytes <= L1_CB_BUDGET_PREFETCH_BYTES
    )
    prefetch_blocks = (
        2
        if (
            prefetch_ok
            and (
                b8 == 2 or (b8 == 1 and trid_prefetch_pays(ncores, blocks_per_core, chunk_row_bytes, prefetch_cb_bytes))
            )
        )
        else 1
    )
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
            "path": "generic",
            "core_ranges": core_ranges,
            "cores": cores,
            "work": work,
            "chunk_wt": chunk_wt,
            "chunk_row_bytes": chunk_row_bytes,
            "row_page_stride": row_page_stride,
            "source_page_bytes": in_page_bytes,
            "shard_tiles": 0,
            "stateful_read": stateful_read,
            "split_read": split_read,
            "prefetch_blocks": prefetch_blocks,
            "vc_spread": vc_spread,
            "bank_placement": bank_placement,
            "read_vcs": read_vcs,
            "write_vcs": write_vcs,
            "depth": depth,
            "n_h": n_h,
            "n_w": n_w,
            "ncores": ncores,
            # Busiest core's chunk-block count -- the C16 gate's "is there
            # anything to pipeline?" input, and the per-block sync cost's
            # multiplier (measured ~612 ns/block, see A0_KNEE_CORES).
            "blocks_per_core": blocks_per_core,
            "cb_bytes_per_core": depth * chunk_wt * bytes_per_chunk_tile,
        }
    )
    return plan


# ---------------------------------------------------------------------------
# ComputeConfigDescriptor
# ---------------------------------------------------------------------------

_FP32_DEST_IN = (ttnn.float32, ttnn.uint32, ttnn.int32)
_FP32_DEST_OUT = (ttnn.float32, ttnn.bfloat8_b, ttnn.uint32, ttnn.int32)


def _compute_config(in_dtype, out_dtype):
    fp32_dest_acc_en = in_dtype in _FP32_DEST_IN or out_dtype in _FP32_DEST_OUT

    config = ttnn.ComputeConfigDescriptor()
    config.fp32_dest_acc_en = fp32_dest_acc_en
    # `can_use_fast_tilize` requires !get_dst_full_sync_enabled().
    config.dst_full_sync_en = False
    if fp32_dest_acc_en:
        # Must be assigned wholesale: nanobind's bound vector copies on
        # __getitem__, so in-place element assignment is silently dropped.
        modes = [ttnn.UnpackToDestMode.Default] * 32
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
    """
    return (
        int(os.environ.get("TILIZE_SKIP_DM", "0")),
        int(os.environ.get("TILIZE_SKIP_COMPUTE", "0")),
    )


def create_program_descriptor(input_tensor, output_tensor, plan) -> ttnn.ProgramDescriptor:
    alias = plan["path"] == "alias"
    core_ranges = plan["core_ranges"]
    chunk_wt = plan["chunk_wt"]
    skip_dm, skip_compute = _ablation_flags()

    # ---------------- circular buffers ----------------
    if alias:
        pages = plan["shard_tiles"]
        cb_rm_input = _aliased_cb(CB_RM_INPUT, input_tensor, plan["tile_in"], pages, core_ranges)
        cb_tiled_output = _aliased_cb(CB_TILED_OUTPUT, output_tensor, plan["tile_out"], pages, core_ranges)
    else:
        pages = plan["depth"] * chunk_wt
        cb_rm_input = _plain_cb(CB_RM_INPUT, input_tensor.dtype, plan["tile_in"], pages, core_ranges)
        cb_tiled_output = _plain_cb(CB_TILED_OUTPUT, output_tensor.dtype, plan["tile_out"], pages, core_ranges)

    alias_flag = 1 if alias else 0

    split_read = plan["split_read"]
    stateful_read = plan["stateful_read"]
    prefetch_blocks = plan["prefetch_blocks"]
    vc_spread = plan["vc_spread"]

    # ---------------- reader ----------------
    reader_ct_args = [
        alias_flag,
        chunk_wt,
        plan["chunk_row_bytes"],
        plan["row_page_stride"],
        plan["source_page_bytes"],
        plan["shard_tiles"],
        skip_dm,
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
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---------------- writer ----------------
    # On the split-read path the writer also reads, so it carries the INPUT
    # accessor as well. Both TensorAccessorArgs are emitted unconditionally (the
    # kernel declares them outside any `if constexpr`), so the CT arg layout does
    # not depend on the lever.
    writer_ct_args = [
        alias_flag,
        chunk_wt,
        plan["tile_out"],
        plan["wt"],
        plan["shard_tiles"],
        skip_dm,
        split_read,
        plan["chunk_row_bytes"],
        stateful_read,
        SEM_SPLIT_RESERVE,
        SEM_SPLIT_DONE,
        vc_spread,  # lever B10: write VC comes from runtime arg 6
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---------------- compute ----------------
    compute_ct_args = [chunk_wt, plan["needs_cast"], skip_compute]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    if alias:
        for core in plan["cores"]:
            reader_rt[core.x][core.y] = [src_addr]
            writer_rt[core.x][core.y] = [dst_addr]
            compute_rt[core.x][core.y] = [plan["num_blocks"]]
    else:
        read_vcs = plan["read_vcs"]
        write_vcs = plan["write_vcs"]
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
            ]
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
            ]
            compute_rt[core.x][core.y] = [row_count * chunk_count]

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
        config=_compute_config(input_tensor.dtype, output_tensor.dtype),
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

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=semaphores,
        cbs=[cb_rm_input, cb_tiled_output],
    )
