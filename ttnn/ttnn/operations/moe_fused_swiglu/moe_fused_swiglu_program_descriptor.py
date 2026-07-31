# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""moe_fused_swiglu — ProgramDescriptor.

Realises the Blocking Model of ``op_design.md`` §1 on the full HGROUPS x KGROUPS worker grid:

  * **Hn** (hidden, gate/up output axis) is split across the grid COLUMNS  -> ``HN_PAD``
  * **Kg** (emb, gate/up contraction) is split across the grid ROWS        -> ``KR_PAD`` / ``kr(y)``
  * **Ne** (emb, ``down`` output axis) is split across ALL cores           -> ``ec(i)``
  * **Kh** (hidden, ``down`` contraction) stays sequential per core        -> ``HGROUPS`` K-blocks
  * **M**  (tokens) is the sequential outer loop                          -> ``M_BLOCK``

The dependent axis (Kg) is combined by a binary reduce tree down each column; the two
reuse-shared operands are broadcast (``x`` along the row, ``h`` across the whole grid).

EVERY block factor, buffer depth and core assignment below is a named parameter with ONE
definition. Every CB page count, loop trip count and grid formula is derived from those
parameters — none is a whole-op dimension (``EMB_T``, ``HID_T``, ``capacity``) and none is a
magic literal.
"""

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
HIDDEN = 2048

# ---------------------------------------------------------------------------
# BLOCKING KNOBS — the single source of truth. Each is defined exactly once.
# ---------------------------------------------------------------------------
#: Token tile-rows per M-block — the CB SIZING bound. The sequential outer loop is
#: ceil(M_t / M_BLOCK); the graded counts 128 and 256 fit ONE block, so the weight stream is read
#: exactly once for them. Raising it (to fold count=512 into one block as well) is a knob turn — it
#: costs (M_BLOCK - 8) * KR_PAD tiles of resident-x L1 plus the same scaling on every per-block CB.
#:
#: The WORK per block is the runtime `m_eff = m_tiles_eff(M_t, b, M_BLOCK, M_EFF_MIN)` derived on
#: device (kernels/moe_fused_swiglu_common.hpp), NOT M_BLOCK: the tail block shrinks to a power of
#: two >= its real tile-row count, so count 128 does 4 tile-rows of work in a block sized for 8.
#: MUST be a power of two, so every `m_eff` divides it and no shrunk CB reserve can straddle a FIFO
#: end (the reserve granularity has to divide the DEPTH * M_BLOCK * W total).
M_BLOCK = 8

#: DEST tile budget for one output sub-block: 8 at half sync (dst_full_sync_en=False) with
#: fp32_dest_acc_en=False — `dest_helpers.hpp`'s DEST_AUTO_LIMIT. Named once; every sub-block
#: assertion below reads it.
DEST_AUTO_LIMIT_TILES = 8

#: PERF 1 — eltwise DEST-WINDOW BLOCK SIZE. Tiles processed per `eltwise_chain` outer iteration
#: (i.e. per `tile_regs_acquire/commit/wait/release` cycle) in every eltwise pass of the compute
#: kernel: the cross-column reduce adds, the non-root partial copies, the SwiGLU multiply and the
#: bf16->bfp8 output pack.
#:
#: WHY THIS IS A KNOB AND NOT A LITERAL 1. `eltwise_convenience`'s `input(cb)` defaults to
#: `WaitPolicy::PerTile` / `PopPolicy::PerTile`, and `eltwise_chain`'s `input_supports_block()`
#: (`eltwise_chain.inl:1511`) accepts a block only for `Upfront` / `Cumulative` / `None+None` /
#: `PerChunk+PerChunk`. A per-tile policy therefore makes `chain_supports_block_v` false and the
#: chain CLAMPS `block_size` to 1 at runtime (`eltwise_chain.inl:3054`) — silently, with no
#: diagnostic. So every `add<>` / `mul<>` / `copy<>` written the convenient way runs ONE tile per
#: DEST window against a DEST budget of 8, paying a full DEST sync round trip per tile.
#:
#: The fast path pairs `PerChunk` wait/pop/reserve/push with `EltwiseShape::tiles(n, blk)`. Numeric
#: shapes use `BlockTailSync::ValidTiles`, so a ragged final window synchronizes only its valid
#: remainder and the totals still match the producer's pushes exactly — which is what keeps the
#: cross-core reduce's CB accounting intact at every `m_eff` (6/12/24/48 tiles at HN_PAD 6).
#:
#: MEASURED (isolated root-epilogue bake-off, `perf_experiments/root_epilogue_fusion/`): the
#: baseline runs 104 DEST windows for the root's 144 tile-ops; blocking them collapses that to 16
#: and measures **1.05x**, and hoisting the bias-add's per-row reconfig/init on top gives **1.07x**
#: — uniform across tile counts 6/24/48 and HN_PAD 4/6, PCC bit-unchanged. Setting this to 1
#: reproduces the pre-Perf-1 per-tile shape exactly, which is the A/B for a re-measurement.
ELTWISE_BLK = int(os.environ.get("MOE_SWIGLU_ELTWISE_BLK", DEST_AUTO_LIMIT_TILES))

#: Token tiles per gate/up output sub-block (the matmul's `out_subblock_h`). PINNED AT 1 and NOT a
#: knob: the gate/up output must stay m-major with HN_PAD consecutive hidden tiles per token row,
#: because that IS the `cb_h` in0 layout phase 2 reads. `OutputCBLayout::SubblockMajor` walks
#: in0_subblock outer / in1_subblock inner, so height 1 keeps the order `m*HN_PAD + n` for ANY
#: in1_num_subblocks — but height 2 would emit `sb*2*HN_PAD + off*2*w + h*w + n`, i.e. `h` and
#: `off` swapped, which silently permutes h. DEST is 1 x HN_PAD = 6 of 8 either way.
OUT_SUBBLOCK_H_GU = 1

#: Token tiles per `down` output sub-block — Refinement 2 lever 3. SEPARATE from the gate/up knob
#: above: `down`'s out_subblock_w is `ec` (2-3), so at height 1 its sub-block is 2-3 tiles of a
#: DEST budget of 8, and `matmul_output_subblock` measures the win tracking sub-block SIZE
#: (1x1 -> any 8-tile shape = 1.46x; a 4-tile shape = 1.40x). The `down` matmul packs
#: `OutputCBLayout::TileRowMajor` at `out_row_width = EC_MAX`, which places every tile at its true
#: row-major offset, so its output layout is height-AGNOSTIC and this really is a pure knob turn.
#:
#: The value is DERIVED (largest power of two whose sub-block still fits DEST) and clamped by this
#: cap, so it tracks EC_MAX across emb widths instead of being a literal: cap 4 gives 2 at EC_MAX 3
#: (emb 7168, 6 DEST tiles) and 4 at EC_MAX 2 (emb 6144, 8 tiles). The kernels additionally take
#: `min(this, m_eff)` at runtime, so the runtime M shrink is untouched: M_EFF_MIN stays
#: pow2_ceil(OUT_SUBBLOCK_H_GU) = 1 and `count 32` still does m_eff 1.
#:
#: MEASURED (Refinement 2, bf16_rm, count 256 / 128 / emb 6144 count 256):
#:   cap 1 (Phase-0 shape) 226 645 / 151 109 / 212 222 ns
#:   cap 4 (derived 2/2/4) 228 310 / 151 434 / 215 124 ns   = +0.7 % / +0.2 % / +1.4 %
#: A consistent small REGRESSION, so the knob is PARKED AT 1 — byte-identical to Phase 0, zero cost,
#: still live. `matmul_output_subblock`'s 1.46x is measured on an L1-resident, compute-bound matmul
#: with Kt=1; this `down` matmul is gated on the h all-gather and on a per-round DRAM read, so a
#: bigger DEST sub-block buys nothing and the extra DEST pressure costs. Worth re-turning only if a
#: later refinement makes phase 2 compute-bound.
OUT_SUBBLOCK_H_DN_MAX = int(os.environ.get("MOE_SWIGLU_SBH_DN", 1))

#: Hidden tiles per gate/up output SUB-BLOCK (the matmul's `out_subblock_w`) — Refinement 2 lever 2.
#: 0 means "the whole HN_PAD", i.e. `in1_num_subblocks == 1`, which is the Phase-0 shape and the
#: thing op_design.md §4.3's per-sub-block reduce pipelining needs broken up. A smaller value splits
#: the gate/up block into `HN_PAD / HN_BLOCK` in1 sub-blocks.
#:
#: Layout-safe at ANY value because `OUT_SUBBLOCK_H_GU == 1`: `OutputCBLayout::SubblockMajor` walks
#: in0_subblock outer / in1_subblock inner, so the emitted order stays `m*HN_PAD + n` — the exact
#: `cb_h` in0 order phase 2 reads. Must DIVIDE HN_PAD, and the ragged last column (hn < HN_PAD) must
#: still land inside the LAST sub-block, since `last_in1_subblock_w_valid` can narrow that sub-block
#: but cannot skip a whole one; both are asserted below.
#:
#: MEASURED (Refinement 2, bf16_rm, count 256 / 128 / emb 6144 count 256):
#:   HN_BLOCK 6 == HN_PAD (default) 222 446 / 144 500 / 208 500 ns (medians)
#:   HN_BLOCK 3 (2 in1 sub-blocks) 224 296 / 143 516 / 210 307 ns  = +0.8 % / -0.7 % / +0.9 %
#: A wash, and for a understood reason: the split HALVES the DEST sub-block (6 -> 3 tiles) — which
#: `matmul_output_subblock` measures as a real slowdown — while the thing it is supposed to buy,
#: op_design.md §4.3's "sub-block `off`'s reduce overlaps `off+1`'s matmul", needs the REDUCE to be
#: split per sub-block too, which this knob alone does not do. That pipelining is bounded by the
#: measured cost of the whole reduce transport (4.2 % of count 256 — see changelog), and the
#: parallel-fan-in experiment below showed that cost is bandwidth, not serialisation. PARKED at
#: HN_PAD (byte-identical), still live for a future per-sub-block reduce.
HN_BLOCK = int(os.environ.get("MOE_SWIGLU_HN_BLOCK", 0))

#: K tiles per gate/up matmul K-block, expressed as a fraction of the per-row K extent.
#: 1 == the coarsest correct block (num_k_blocks == 1), which is what lets the gate and the up
#: matmul share ONE resident `x` block (op_design.md §6 "the cb_x_tiles-consumed-twice
#: contract"). Splitting it further would need the documented second-CB copy of x.
KB1_FRACTION = 1

#: Buffer depths (per streaming CB).
DEPTH_W = 2  # gate/up weight CBs: overlap the next K-block's DRAM read with compute
#: Resident-x slots in `cb_x_tiles` — Refinement 3's named lever. 1 (the Phase-0 shape) makes the
#: reader's `cb_reserve_back(cb_x_tiles)` for M-block b+1 block until compute has popped block b's
#: resident x, i.e. until block b is COMPLETELY finished: nothing of b+1's x staging or row
#: multicast can start under b's phase 2. A second slot decouples them.
#:
#: Costs `M_BLOCK * KR_PAD` bfp8 tiles (195.5 KB at M_BLOCK 8 / KR_PAD 23), which only fits because
#: W_RESIDENT below collapses `DEPTH_W` to 1 and frees 155 KB.
#:
#: MEASURED (Refinement 3, against the resident-weight configuration, one fresh-cache run each).
#: It can only act where `m_blocks > 1`, and that is exactly where it moved:
#:   `count 512` (2 blocks)      397 954 -> 396 078 ns  = -0.47 %
#:   `count = capacity` (20)   3 569 484 -> 3 516 405 ns = -1.49 %
#: while the five SINGLE-block cells — which provably never reserve the second slot — moved
#: +1.03 / +0.73 / +0.69 / -0.92 / -0.54 %, i.e. this op's per-cell noise floor, measured for free.
#: SHIPPED at 2, path-gated below to programs whose SIZED M extent can reach a second block.
#: On the bf16_rm path it currently hides the x STICK READ but not the row multicast: the reader's
#: staging blocks on `cb_wait_front(cb_x_stage)`, i.e. on compute's fused tilize, which still sits
#: at the top of block b+1's compute iteration. Hoisting that tilize is the complementary step —
#: see the Refinement 3 Outcome note in `op_requirements.md`.
#:
#: SAFE AT ANY DEPTH for the multicast: the landing address must be identical on every core in the
#: row, and it is — the write pointer after `b` blocks is `(sum of pushed m_eff) * KR_PAD` pages,
#: a pure function of the mailbox words, hence the same number on all 110 cores. And no reserve can
#: straddle the FIFO end: only the LAST block shrinks, so every earlier write pointer is a multiple
#: of `M_BLOCK * KR_PAD`, and `m_eff | M_BLOCK` (Refinement 1's power-of-two invariant).
DEPTH_X = int(os.environ.get("MOE_SWIGLU_DEPTH_X", 2))
#: phase-2 (W_down) weight CB depth, in K-blocks — see the derivation of `depth_wd` below.
#: Overridable for `/perf-measure` A/B via MOE_SWIGLU_DEPTH_WD; HGROUPS reproduces the
#: whole-hidden-extent sizing this CB used to have.
DEPTH_WD = int(os.environ.get("MOE_SWIGLU_DEPTH_WD", 5))
DEPTH_H = 3  # h all-gather: 3 so a late round's producer is not flow-controlled by itself
DEPTH_OUT = 2
DEPTH_XSTAGE = 1  # tilized x staging slots (a core injects <= ceil(M_BLOCK/HGROUPS) rows/block)
XSTICK_ROWS = 1  # tile-rows of row-major x sticks held in flight

#: Read-coalescing knob: max BANK-CONTIGUOUS weight/output tiles fetched per NoC transaction.
#: 1 reproduces the naive one-transaction-per-tile read (the ablation baseline).
#: Overridable for `/perf-measure` A/B via MOE_SWIGLU_WRUN.
WRUN = int(os.environ.get("MOE_SWIGLU_WRUN", 8))

#: `/perf-measure` ablation hook (payload stubbed, ALL synchronisation scaffolding intact).
#: MOE_SWIGLU_ABLATE=skip_compute defines SKIP_COMPUTE in the compute TU, which drops the inner
#: matmul LLK call while keeping every CB wait/push, reload and L1_ACC toggle — the documented
#: way to separate the dataflow ceiling from the compute ceiling. NOT a correctness mode.
#:
#: The COLLECTIVE ablations (Refinement 2) stub one transport at a time in the dataflow TUs while
#: keeping every CB reserve/push/pop and every loop trip count, so the diff against the baseline is
#: that collective's exposed cost:
#:   no_reduce_xfer — parent skips invite + data wait, child skips the two unicasts + the signal
#:   no_h_xfer      — the h all-gather send/receive is dropped (cb_h still cycles)
#:   no_x_xfer      — the x row-multicast send/receive is dropped (cb_x_tiles still cycles)
#:   no_w_xfer      — the three bfp4 weight DRAM streams are dropped (CBs + barriers still cycle)
#: None is a correctness mode; each answers "how much of the 85% is THIS collective?".
ABLATE = os.environ.get("MOE_SWIGLU_ABLATE", "")
_DM_ABLATIONS = ("no_reduce_xfer", "no_h_xfer", "no_x_xfer", "no_w_xfer")

#: PER-STAGE ZONES ARE PERMANENT AND UNCONDITIONAL — there is no knob here any more.
#: Every serial stage of the per-M-block chain in all three kernels is bracketed by
#: `MaybeDeviceZoneScope("<stage>")` (`ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp`), which
#: emits NO instructions when the device profiler is off, so the shipped kernels are byte-identical
#: to unzoned ones and the observability can never be "optimised away". Enable the numbers with
#: `scripts/run_safe_pytest.sh --profile` and read
#: `generated/profiler/.logs/profile_log_device.csv`.
#:
#: The old `MOE_SWIGLU_ZONES=1` opt-in is GONE. It was measured (Refinement 2 note (c)) to produce an
#: EMPTY device CSV, and the cause was structural: the compute TU used `DeviceZoneScopedN` without
#: including the profiler header (a compute kernel does not get it through `dataflow_api.h`), so the
#: hook could never have worked as written.
#:
#: ZONE BUDGET: 8 (reader) / 5 (writer) / 6 (compute) records per M-BLOCK against the profiler's
#: 125-records-per-core cap, so per-stage numbers are resolvable up to m_blocks = 15 (count 3840 at
#: M_BLOCK 8). Beyond that the tail is silently dropped and only the whole-kernel duration is valid.

#: W_down prefetch depth, in phase-2 K-blocks kept in flight ahead of the round that consumes
#: them. Clamped to [1, HGROUPS].
#:
#: MEASURED AT PHASE 0 (emb 7168, count 256, bf16_rm): 1 -> 227.8 us, 4 -> 228.7 us, 11 -> 240.1 us.
#: That null had a CAUSE, found in Refinement 2: the round issued its block and then immediately ran
#: `noc_async_read_barrier()`, which drains EVERY outstanding read — so however deep the prefetch,
#: the latency was still paid on the spot. The reader now defers that barrier past the round's
#: collective (see "DEFERRED READ BARRIER" in the reader), which is what makes this knob live: at 1
#: the block published in a round is the one that round consumes, so >= 2 is what actually decouples
#: compute from the DRAM read.
WD_AHEAD = int(os.environ.get("MOE_SWIGLU_WD_AHEAD", 1))

#: Refinement 3 — CROSS-M-BLOCK WEIGHT RESIDENCY. The three bfp4 weight streams are read once per
#: M-BLOCK today, but every one of those reads is BYTE-IDENTICAL: `BR::read(wg_acc, (kstart+k)*HID_T,
#: hstart, ...)` (reader), the W_up twin (writer) and the W_down K-blocks (reader) are all pure
#: functions of this core's `kstart`/`hstart`/`jstart`, with NO dependence on the M-block index. So
#: `count 512` reads 26 MB of weights TWICE and `count = capacity` reads it TEN times — which is
#: exactly the `m_blocks > 1` cliff (count 512 measured at 431 030 ns vs count 256's 220 576, i.e.
#: 1.95x for 2x the tokens), and it is also why the graded target is reachable at all: the harness
#: grades `read_bytes` with the weights counted ONCE (feature_spec.read_bytes), so a re-read is pure
#: loss against the metric.
#:
#: The mechanism needs NO compute-side change and no new CB, because the CB cycle already returns
#: each weight block to the same L1 slot every M-block:
#:   * `cb_w_gate` / `cb_w_up` hold ONE block (depth_w == 1 below), so the reserve/push cycle always
#:     lands at the CB base;
#:   * `cb_w_down` holds exactly `HGROUPS` K-blocks (depth_wd forced to HGROUPS below) and the
#:     reader pushes exactly `HGROUPS` per M-block, so K-block r always occupies slot r.
#: `cb_pop_front` only advances a read pointer — it does not clear the bytes — and these CBs have a
#: single producer, so the block a later M-block re-reserves still holds the data it read at b == 0.
#: The reader/writer therefore keep the FULL reserve/push handshake (compute's waits and pops are
#: untouched, bit-for-bit) and skip only the `BR::read` DRAM loops for b > 0.
#:
#: MEASURED (Refinement 3, one fresh-cache run each over the 9 graded loose cells). Gate/up
#: residency alone, i.e. this knob with WD_RESIDENT=0:
#:   `count 512`  (2 M-blocks)  431 030 -> 404 078 ns   = -6.25 %
#:   `count = capacity` (20)  4 200 686 -> 3 676 894 ns = -12.47 %
#:   9-cell sum               5 873 831 -> 5 323 776 ns = -9.36 %
#: and every SINGLE-M-block cell inside +-0.53 %, i.e. free — as it must be, since b == 0 still
#: reads. SHIPPED AT 1.
W_RESIDENT = int(os.environ.get("MOE_SWIGLU_W_RESIDENT", 1))  # gate/up (NoC0 + NoC1 halves)
#: The W_down half of the same lever. Kept as a SEPARATE knob because it is the one with an L1
#: price: residency requires `depth_wd == HGROUPS` (the slot-r-holds-K-block-r invariant above),
#: which is +60.8 KB over the measured-optimal depth 5. Gate/up residency is free — it SAVES 155 KB
#: by collapsing DEPTH_W to 1.
#:
#: `depth_wd == HGROUPS` is FORCED, not chosen: the invariant is that the reader's HGROUPS pushes
#: per M-block bring the write pointer exactly back to the CB base, i.e. `HGROUPS % depth_wd == 0`,
#: and HGROUPS is 11 here — prime — so 11 is the only depth above `wd_ahead + 2` that qualifies.
#:
#: MEASURED (Refinement 3), as the DELTA on top of gate/up residency:
#:   `count 512`               404 078 -> 397 954 ns   = a further -1.52 %
#:   `count = capacity`      3 676 894 -> 3 569 484 ns = a further -2.92 %
#:   9-cell sum              5 323 776 -> 5 215 058 ns = a further -2.04 %
#: The single-block cells drifted +0.3 to +1.5 % against the gate/up-only run, which reads as this
#: op's ~1 % per-cell noise (the two other count-256 cells, cap 1024 and cap 2048, moved the other
#: way at -0.51 / -0.26 % against the same reference, and the DEPTH_X run below independently
#: measured +-1.0 % on cells it cannot affect at all). SHIPPED AT 1.
WD_RESIDENT = int(os.environ.get("MOE_SWIGLU_WD_RESIDENT", 1))

#: Reduce-tree fan-in cap (>= ceil(log2(KGROUPS)) for the binary tree below).
MAX_CHILDREN = 5

#: Refinement 2 lever 1 — how many child partials the reduce-tree LANDING CBs
#: (`cb_reduce_gate_in` / `cb_reduce_up_in`) can hold AT ONCE, i.e. how many children a parent can
#: have in flight concurrently.
#:
#: 1 (the Phase-0 shape) forces the parent to invite child `c`, wait for its data, hand it to
#: compute and only THEN invite `c + 1` — up to 4 SEQUENTIAL ~102 KB round trips per M-block at the
#: root. Raising it lets the parent invite every child at once and wait for `num_children` arrivals.
#:
#: The slot count is DERIVED as min(real max fan-in, this cap) so it tracks the tree instead of
#: being a literal; the cap exists because each extra slot costs `M_BLOCK * HN_PAD` tiles of bfp8
#: L1 in EACH of the two CBs (2 x 51 KB = 102 KB per extra slot at M_BLOCK 8 / HN_PAD 6) and the
#: bf16_rm path has ~159 KB free, so 2 is the most that fits today.
#:
#: MEASURED (Refinement 2, bf16_rm, count 256 / 128 / emb 6144 count 256):
#:   cap 1 (Phase-0, one child at a time) 226 009 / 153 883 / 211 220 ns
#:   cap 2 (two children concurrent)      230 482 / 152 454 / 217 121 ns  = +2.0 % / -0.9 % / +2.8 %
#: A REGRESSION at count 256, and the ablation says why: the ENTIRE reduce transport is only 4.2 %
#: of count 256 (baseline 226 134 ns vs 216 651 with the transport stubbed), and roughly half of
#: that is destination-port BANDWIDTH — four children write 4 x 102 KB into one core's L1, which
#: concurrency cannot speed up. What the one-slot protocol did buy is an interleave that the wave
#: protocol gives up: child `c`'s ~102 KB transfer overlapped child `c-1`'s in-place `add`. PARKED
#: AT 1 — byte-identical to Phase 0, no L1 cost (so Refinement 3 keeps the space), still live and
#: already fan-in-general if a later structure makes the transport latency-bound.
REDUCE_SLOTS_CAP = int(os.environ.get("MOE_SWIGLU_REDUCE_SLOTS", 1))

#: Mailbox handshake word — the reader publishes {count, M_t, m_blocks} plus this flag.
MAILBOX_MAGIC = 0xC0FFEE01
MAILBOX_WORDS = 16

# ---------------------------------------------------------------------------
# Circular buffers (semantic names; the numeric slot is only the buffer index)
# ---------------------------------------------------------------------------
CB_X_IN = 0  # row-major x stick slices (bf16) or bfp8 tiles
CB_X_TILES = 1  # resident bfp8 in0 block, filled by the row multicast
CB_X_STAGE = 2  # tilized x tile-row awaiting its multicast turn
CB_W_GATE = 3
CB_W_UP = 4
CB_W_DOWN = 5
CB_REDUCE_GATE_IN = 6  # incoming child partials (gate)
CB_REDUCE_UP_IN = 7  # incoming child partials (up)
CB_H = 8  # gathered h, one phase-2 K-block per round
CB_IDX_SCRATCH = 9
CB_COUNTS_SCRATCH = 10
CB_OUT_TILES = 16
CB_GATE_ACC = 24  # gate partial accumulator (matmul out + in-place reduce adds)
CB_UP_ACC = 25
CB_GATE_SEND = 26  # partials handed to the writer for the unicast to the tree parent
CB_UP_SEND = 27
CB_GATE_SILU = 28  # root only: SiLU(sum(gate))
CB_H_LOCAL = 29  # root only: this column's h slice, awaiting its all-gather round
CB_OUT_INTERM = 30  # phase-2 packer-L1 accumulation region

# ---------------------------------------------------------------------------
# Semaphores
# ---------------------------------------------------------------------------
SEM_X_BASE = 0  # x row multicast (data_ready, consumer_ready)
SEM_H_BASE = 2  # h all-gather   (data_ready, consumer_ready)
SEM_GO = 4  # reduce tree: parent -> child "your slot is free, send"
SEM_DATA = 5  # reduce tree: child -> parent "partials landed"


def _pow2_ceil(v):
    """Smallest power of two >= v (v >= 1)."""
    p = 1
    while p < v:
        p <<= 1
    return p


def _split(total, groups):
    """`base + (i < rem)` split — alignment-aware, no floor on a tile count."""
    base, rem = total // groups, total % groups
    sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    starts, acc = [], 0
    for s in sizes:
        starts.append(acc)
        acc += s
    return sizes, starts


def _reduce_tree(kgroups, hgroups):
    """Binary reduce tree per grid column.

    Column ``x``'s root is row ``x % kgroups`` so the 13 roots (which additionally carry the
    SwiGLU and the h-multicast injection) spread over all rows. Relative index
    ``r = (y - root) % kgroups``; node ``r`` receives from ``r + 2^(l-1)`` at level ``l`` when
    ``r % 2^l == 0``, and sends to ``r - lowbit(r)``. Depth = ceil(log2(kgroups)).
    """
    info = {}
    for x in range(hgroups):
        root_y = x % kgroups
        for y in range(kgroups):
            r = (y - root_y) % kgroups
            children = []
            s = 1
            while s < kgroups:
                if r % (2 * s) == 0 and r + s < kgroups:
                    children.append((x, (root_y + r + s) % kgroups))
                s *= 2
            parent = None
            if r != 0:
                low = r & (-r)
                parent = (x, (root_y + r - low) % kgroups)
            info[(x, y)] = {"is_root": r == 0, "parent": parent, "children": children}
    return info


def _virt(device, x, y):
    c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return int(c.x), int(c.y)


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def make_mailbox(device, num_cores):
    """Zeroed L1 scratch, one page per L1 bank, used as the token-count mailbox.

    The reader publishes the DEVICE-resident count here; compute (all three TRISCs) and the
    writer spin on the magic word. Zeroed host-side so a stale magic from a previous dispatch
    cannot be mistaken for a fresh publish.
    """
    import torch  # local: ttnn must not carry a global torch import

    return ttnn.from_torch(
        torch.zeros((num_cores, MAILBOX_WORDS), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def create_program_descriptor(
    input_tensor,
    w_gate,
    w_up,
    w_down,
    counts,
    global_expert_idx_table,
    output_tensor,
    mailbox,
    *,
    local_expert_id,
    input_m_tiles,
    compute_kernel_config,
):
    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()

    # ---- grid / core assignment (every core count derives from the device grid) ----
    HGROUPS = int(grid.x)  # hidden groups == grid columns
    KGROUPS = int(grid.y)  # emb-contraction groups == grid rows
    num_cores = HGROUPS * KGROUPS
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(HGROUPS - 1, KGROUPS - 1))])
    if KGROUPS < 2:
        raise RuntimeError(f"moe_fused_swiglu needs a grid at least 2 rows tall (got {HGROUPS}x{KGROUPS})")

    emb = int(input_tensor.shape[-1])
    capacity = int(input_tensor.shape[-2])
    hidden = int(w_gate.shape[-1])
    EMB_T = emb // TILE
    HID_T = hidden // TILE
    M_T_MAX = input_m_tiles

    # ---- block factors, derived from the knobs ----
    kr_sizes, kr_starts = _split(EMB_T, KGROUPS)
    KR_PAD = max(kr_sizes)  # uniform in0 K stride; per-row kr shrinks the FMA loop
    KB1 = max(1, (KR_PAD * KB1_FRACTION))
    num_k_blocks_gu = (KR_PAD + KB1 - 1) // KB1
    if num_k_blocks_gu != 1:
        raise RuntimeError(
            "moe_fused_swiglu: gate/up K-blocking > 1 needs the second-CB copy of the resident x "
            "block (op_design.md §6); set KB1_FRACTION = 1."
        )

    HN_PAD = (HID_T + HGROUPS - 1) // HGROUPS  # uniform hidden width per column group
    hn_sizes = [max(0, min(HN_PAD, HID_T - x * HN_PAD)) for x in range(HGROUPS)]
    if min(hn_sizes) == 0:
        raise RuntimeError(f"moe_fused_swiglu: hidden {hidden} cannot fill {HGROUPS} column groups")

    wd_ahead = max(1, min(WD_AHEAD, HGROUPS))
    # cb_w_down depth, in phase-2 K-blocks. It must hold the block being consumed PLUS the
    # `wd_ahead` blocks the reader keeps in flight, hence >= wd_ahead + 1; DEPTH_WD raises that for
    # extra reader/compute slack. Sized by a DEPTH KNOB x ONE K-block — NOT by the whole hidden
    # extent (`HGROUPS * HN_PAD == HID_T`), which is what it used to be and which put 111 KB of L1
    # behind a whole-op dimension.
    #
    # FIFO-wrap safety: single-block pushes can never straddle the end (the total is a multiple of
    # one block), but the `wd_ahead`-block BATCH reserve at the top of each M-block starts
    # ((b * HGROUPS) % depth_wd) blocks in, so it only stays inside the buffer when the depth
    # divides the per-M-block push count. When it does not (only reachable via the
    # MOE_SWIGLU_WD_AHEAD ablation), fall back to the whole stream so the knob stays legal at any
    # value.
    # >= wd_ahead + 2: the blocks in flight, PLUS the one the reader has reserved but not yet
    # published (the deferred-barrier restructure carries exactly one such block across a round
    # boundary), PLUS the one compute is consuming.
    depth_wd = max(DEPTH_WD, wd_ahead + 2)
    if wd_ahead > 1 and HGROUPS % depth_wd != 0:
        depth_wd = HGROUPS
    # W_down residency (Refinement 3) needs the CB to hold the WHOLE phase-2 K stream, so that the
    # reader's `HGROUPS` pushes per M-block bring the write pointer exactly back to the base and
    # K-block r always occupies slot r. Then b > 0 can re-push slot r without re-reading it.
    if WD_RESIDENT:
        depth_wd = HGROUPS
    # The residency PRECONDITION, asserted rather than assumed: the reader pushes exactly HGROUPS
    # W_down K-blocks per M-block, so slot r holds K-block r on EVERY M-block only while the CB
    # capacity divides that push count. Break it (a future depth knob, a different per-block push
    # count) and M-blocks b > 0 silently matmul against the WRONG weight block — no hang, no
    # compile error, just wrong numbers on the multi-M-block path alone.
    if WD_RESIDENT and (depth_wd == 0 or HGROUPS % depth_wd != 0):
        raise RuntimeError(
            f"moe_fused_swiglu: WD_RESIDENT needs cb_w_down's depth ({depth_wd} K-blocks) to divide "
            f"the {HGROUPS} blocks pushed per M-block, so the write pointer returns to the CB base "
            f"every M-block; set MOE_SWIGLU_WD_RESIDENT=0 or fix the depth"
        )
    # gate/up residency makes the second weight slot dead by construction (the CB is filled once and
    # every later M-block re-pushes the same bytes), so it FREES DEPTH_W's 155 KB rather than
    # costing L1 — which is what pays for DEPTH_X's resident-x slot below.
    depth_w = 1 if W_RESIDENT else DEPTH_W
    ec_sizes, ec_starts = _split(EMB_T, num_cores)
    # EC_MAX is the phase-2 N *stride*: every phase-2 CB reserves/pushes in EC_MAX-wide units so
    # its page count is a multiple of the increment (a CB whose total is not a multiple of its
    # reserve granularity walks its FIFO pointer off the end). Cores with ec < EC_MAX leave the
    # tail columns unread — out_subblock_w stays `ec`, so no extra FMA work is done.
    EC_MAX = max(ec_sizes)

    # DEST budget: out_subblock_h * out_subblock_w <= DEST_AUTO_LIMIT, per matmul.
    #   gate/up  out_subblock_w = HN_PAD, height pinned at OUT_SUBBLOCK_H_GU (see the knob).
    #   down     out_subblock_w = ec (<= EC_MAX), height DERIVED here — Refinement 2 lever 3.
    # gate/up in1 sub-block width — Refinement 2 lever 2. 0 (default) = the whole HN_PAD.
    hn_block = HN_PAD if HN_BLOCK <= 0 or HN_BLOCK >= HN_PAD else HN_BLOCK
    if HN_PAD % hn_block != 0:
        raise RuntimeError(f"moe_fused_swiglu: HN_BLOCK {hn_block} must divide HN_PAD {HN_PAD}")
    gu_in1_subblocks = HN_PAD // hn_block
    # The ragged column's real width must reach INTO the last sub-block: the helper can narrow the
    # last in1 sub-block's FMA width (`last_in1_subblock_w_valid`) but cannot drop one entirely.
    if min(hn_sizes) <= (gu_in1_subblocks - 1) * hn_block:
        raise RuntimeError(
            f"moe_fused_swiglu: HN_BLOCK {hn_block} leaves the ragged column (hn={min(hn_sizes)}) "
            f"with an entirely empty last in1 sub-block; use a wider HN_BLOCK"
        )
    if OUT_SUBBLOCK_H_GU * hn_block > DEST_AUTO_LIMIT_TILES:
        raise RuntimeError(
            f"moe_fused_swiglu: gate/up sub-block {OUT_SUBBLOCK_H_GU}x{hn_block} exceeds the "
            f"DEST budget of {DEST_AUTO_LIMIT_TILES} tiles"
        )
    # Largest POWER OF TWO height whose `down` sub-block still fits DEST, capped by the knob and by
    # M_BLOCK. Power of two so that `min(h, m_eff)` divides m_eff exactly for every runtime m_eff
    # (which is itself a power of two) — that is what keeps the M shrink and this lever orthogonal.
    OUT_SUBBLOCK_H_DN = 1
    while (
        OUT_SUBBLOCK_H_DN * 2 <= min(OUT_SUBBLOCK_H_DN_MAX, M_BLOCK)
        and OUT_SUBBLOCK_H_DN * 2 * EC_MAX <= DEST_AUTO_LIMIT_TILES
    ):
        OUT_SUBBLOCK_H_DN *= 2
    if M_BLOCK % OUT_SUBBLOCK_H_GU != 0 or M_BLOCK % OUT_SUBBLOCK_H_DN != 0:
        raise RuntimeError(
            f"moe_fused_swiglu: M_BLOCK {M_BLOCK} must be a multiple of both sub-block heights "
            f"({OUT_SUBBLOCK_H_GU} gate/up, {OUT_SUBBLOCK_H_DN} down)"
        )

    # ---- the runtime M shrink (op_design.md §3's `m_tiles`) ----
    # The kernels work `m_eff = m_tiles_eff(M_t, b, M_BLOCK, M_EFF_MIN)` token tile-rows per block,
    # not M_BLOCK. Two host-side preconditions make that safe, and both are asserted here rather
    # than assumed device-side:
    #   1. M_BLOCK is a power of two, so every m_eff (a power of two <= M_BLOCK) DIVIDES it. Each
    #      M-scaled CB total is DEPTH * M_BLOCK * W, so a m_eff * W reserve can never straddle the
    #      FIFO end whatever order the blocks push in.
    #   2. M_EFF_MIN keeps m_eff a multiple of the gate/up out_subblock_h, so the kernels'
    #      `m_eff / OUT_SUBBLOCK_H_GU` sub-block count stays exact. The `down` height does NOT
    #      enter M_EFF_MIN: the kernel takes `min(OUT_SUBBLOCK_H_DN, m_eff)` at runtime, so raising
    #      it never forces a larger m_eff (i.e. never re-adds work at `count <= 32`).
    if _pow2_ceil(M_BLOCK) != M_BLOCK:
        raise RuntimeError(
            f"moe_fused_swiglu: M_BLOCK {M_BLOCK} must be a power of two so every runtime m_eff "
            f"divides it (see m_tiles_eff in kernels/moe_fused_swiglu_common.hpp)"
        )
    M_EFF_MIN = _pow2_ceil(OUT_SUBBLOCK_H_GU)
    if M_EFF_MIN > M_BLOCK:
        raise RuntimeError(f"moe_fused_swiglu: OUT_SUBBLOCK_H_GU {OUT_SUBBLOCK_H_GU} exceeds M_BLOCK {M_BLOCK}")

    # ---- DRAM bank-run coalescing precondition (op_design.md §1.5) ----
    dram_align = ttnn.get_dram_alignment()
    bfp4_tile = ttnn.tile_size(ttnn.bfloat4_b)
    bfp8_tile = ttnn.tile_size(ttnn.bfloat8_b)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    try:
        num_banks = int(ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.DRAM).num_banks)
    except Exception:  # pragma: no cover - defensive
        num_banks = 0
    remap = int(
        num_banks > 1
        and WRUN > 1
        and HID_T % num_banks == 0
        and EMB_T % num_banks == 0
        and bfp4_tile % dram_align == 0
        and bfp8_tile % dram_align == 0
    )
    if not remap:
        num_banks = max(num_banks, 1)

    # ---- reduce tree ----
    tree = _reduce_tree(KGROUPS, HGROUPS)
    max_fan_in = max(len(n["children"]) for n in tree.values())
    if max_fan_in > MAX_CHILDREN:
        raise RuntimeError(f"moe_fused_swiglu: tree fan-in {max_fan_in} exceeds MAX_CHILDREN {MAX_CHILDREN}")
    # Concurrent child landing slots — Refinement 2 lever 1. Derived from the REAL fan-in, capped by
    # the L1 knob; 1 reproduces the Phase-0 invite-one-child-at-a-time protocol byte-for-byte.
    reduce_slots = max(1, min(max_fan_in, REDUCE_SLOTS_CAP))
    # A parent invites its children in WAVES of `reduce_slots`, so child `c`'s landing slot is
    # `c % reduce_slots` — the one number the child needs and the only new runtime arg. Waves keep
    # the whole-CB reserve/push granularity (and therefore the "every core's write pointer is the CB
    # base" invariant the child's address proxy depends on) at ANY fan-in, and `reduce_slots == 1`
    # degenerates to the Phase-0 one-child-at-a-time protocol byte-for-byte.
    _slot_of = {}
    for _node in tree.values():
        for c, ch in enumerate(_node["children"]):
            _slot_of[ch] = c % reduce_slots

    # ---- collectives: emit the mcast wire (Mcast1D / Mcast2D own the coord + rect math) ----
    # Reader runs on NOC_0 (ReaderDataMovementConfig -> preferred_noc_for_dram_read).
    #
    # DataReadySignal: Flag, NOT the Counter op_design.md §4.2/§4.4 asks for. `mcast_pipe`'s
    # Counter path is UNUSABLE for any sender whose data write is a loopback/linked multicast, for
    # TWO independent reasons (both re-confirmed on device in the verifier pass; see
    # `verification_report.md` and the warning now carried at `mcast_pipe.hpp`'s DataReadySignal):
    #   1. the multicast atomic increment was handed the INCLUDE-source fan-out while
    #      `noc_semaphore_inc_multicast` is unconditionally EXCLUDE-source, so `fence_()`'s
    #      non-posted atomic barrier waited for an ack from a destination never addressed.
    #      FIXED in `mcast_pipe.inl::signal_ready_` (it now always passes `num_dests_excl_`).
    #   2. STILL OPEN: `send_data_` issues the data multicast with `NOC_CMD_VC_LINKED` and relies
    #      on the *signal* to terminate the chain — but the Flag signal is a multicast WRITE on the
    #      same command buffer (terminates it) while the Counter signal is a multicast ATOMIC on
    #      `write_at_cmd_buf` (a DIFFERENT command buffer), so the link is never released and the
    #      next write on `NCRISC_WR_CMD_BUF` blocks in `noc_cmd_buf_ready` forever. Observed
    #      exactly that: round-0's h sender stuck in `noc_async_write_multicast_loopback_src` while
    #      all 110 readers sat in the h `wait_min`. Fixing it needs the Counter path to send
    #      UNLINKED plus an acked write barrier before the atomic (data-before-signal ordering) —
    #      a helper-level change with its own measurement, filed as a perf refinement.
    # Flag is correct but forces the sender of round r+1 to wait for every receiver to reset round
    # r's flag, which is part of the measured collective serialisation.
    #
    # `handshake` is the receiver->sender "my cb slot is free" ack, i.e. the CB flow control, and
    # stays on (MOE_SWIGLU_ABLATE=no_handshake drops it for MEASUREMENT only — not correct).
    handshake = ABLATE != "no_handshake"
    data_ready_signal = ttnn._ttnn.mcast_host.McastDataReady.Flag
    x_mcast = ttnn.Mcast1D(
        device,
        all_cores,
        ttnn.Mcast1DShape.PerRow,
        0,
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=handshake,
            data_ready=data_ready_signal,
            rotating_sender=True,
            base_sem_id=SEM_X_BASE,
        ),
    )
    h_mcast = ttnn.Mcast2D(
        device,
        all_cores,
        ttnn.CoreCoord(0, 0),
        ttnn.McastConfig(
            noc=ttnn.NOC.NOC_0,
            handshake=handshake,
            data_ready=data_ready_signal,
            rotating_sender=True,
            base_sem_id=SEM_H_BASE,
        ),
    )
    assert x_mcast.num_senders() == HGROUPS, x_mcast.num_senders()
    assert h_mcast.num_senders() == num_cores, h_mcast.num_senders()

    # ---- page sizes ----
    x_is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    input_format = 0 if x_is_rm else 1
    x_stick_slice = KR_PAD * TILE * input_tensor.element_size() if x_is_rm else bfp8_tile
    x_page = int(input_tensor.buffer_page_size())
    counts_page = int(counts.buffer_aligned_page_size())
    idx_page = int(global_expert_idx_table.buffer_aligned_page_size())

    # ---- CB page counts: functions of the knobs only ----
    # DEPTH_X slots of M_BLOCK*KR_PAD tiles. The mcast landing address stays identical on every core
    # at any depth (see the DEPTH_X knob); depth 2 is what lets the reader stage M-block b+1's x
    # while compute is still in block b's phase 2.
    #
    # PATH-GATED to programs that can actually reach a second M-block. `input_m_tiles` is the
    # host-time SIZED token extent, so `ceil(M_T_MAX / M_BLOCK)` is the maximum `m_blocks` any
    # runtime count can produce; at 1 the extra slot could never be reserved, so allocating it would
    # be 195.5 KB of provably dead L1. The runtime count is device-resident and cannot gate this —
    # the sized extent can, and it is the tightest host-time bound available.
    max_m_blocks = (M_T_MAX + M_BLOCK - 1) // M_BLOCK
    depth_x = DEPTH_X if max_m_blocks > 1 else 1
    n_x_tiles = depth_x * M_BLOCK * KR_PAD
    n_gu_block = M_BLOCK * HN_PAD
    n_w_gu = KR_PAD * HN_PAD  # one gate/up K-block (num_k_blocks == 1)
    n_w_down = depth_wd * HN_PAD * EC_MAX  # DEPTH knob x one phase-2 K-block (see depth_wd above)
    n_out_block = M_BLOCK * EC_MAX
    # The row-major staging path (read sticks -> fused tilize -> mcast) exists ONLY for the bf16 RM
    # activation; the bfp8 TILE path lands tiles straight in the resident slot and never touches
    # either CB (`if constexpr (INPUT_FORMAT == 0)` in the reader and in compute). Give them one
    # page each in that configuration instead of ~48 KB of L1 that no kernel can reach.
    n_x_in = XSTICK_ROWS * TILE if x_is_rm else 1
    n_x_stage = DEPTH_XSTAGE * KR_PAD if x_is_rm else 1

    cbs = [
        _cb(
            CB_X_IN,
            all_cores,
            n_x_in,
            x_stick_slice,
            ttnn.bfloat16 if x_is_rm else ttnn.bfloat8_b,
        ),
        _cb(CB_X_TILES, all_cores, n_x_tiles, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_X_STAGE, all_cores, n_x_stage, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_W_GATE, all_cores, depth_w * n_w_gu, bfp4_tile, ttnn.bfloat4_b),
        _cb(CB_W_UP, all_cores, depth_w * n_w_gu, bfp4_tile, ttnn.bfloat4_b),
        _cb(CB_W_DOWN, all_cores, n_w_down, bfp4_tile, ttnn.bfloat4_b),  # depth_wd K-blocks
        _cb(CB_REDUCE_GATE_IN, all_cores, reduce_slots * n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_REDUCE_UP_IN, all_cores, reduce_slots * n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_H, all_cores, DEPTH_H * n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_IDX_SCRATCH, all_cores, 1, max(idx_page, dram_align), ttnn.uint32),
        _cb(CB_COUNTS_SCRATCH, all_cores, 1, max(counts_page, dram_align), ttnn.uint32),
        _cb(CB_OUT_TILES, all_cores, DEPTH_OUT * n_out_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATE_ACC, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_UP_ACC, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATE_SEND, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_UP_SEND, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATE_SILU, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_H_LOCAL, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_OUT_INTERM, all_cores, n_out_block, bf16_tile, ttnn.bfloat16),
    ]

    # -----------------------------------------------------------------------
    # Reader
    # -----------------------------------------------------------------------
    reader_ct = [
        input_format,
        M_T_MAX,
        local_expert_id,
        EMB_T,
        HID_T,
        KR_PAD,
        HN_PAD,
        EC_MAX,
        M_BLOCK,
        HGROUPS,
        KGROUPS,
        num_banks,
        WRUN,
        SEM_GO,
        SEM_DATA,
        x_page,
        x_stick_slice,
        max(counts_page, dram_align),
        max(idx_page, dram_align),
        bfp4_tile,
        bfp8_tile,
        MAX_CHILDREN,
        remap,
        MAILBOX_MAGIC,
        wd_ahead,
        M_EFF_MIN,
        reduce_slots,
        W_RESIDENT,
        WD_RESIDENT,
        CB_X_IN,
        CB_X_TILES,
        CB_X_STAGE,
        CB_W_GATE,
        CB_W_DOWN,
        CB_REDUCE_GATE_IN,
        CB_REDUCE_UP_IN,
        CB_H,
        CB_H_LOCAL,
        CB_IDX_SCRATCH,
        CB_COUNTS_SCRATCH,
    ]
    reader_ct.extend(x_mcast.compile_time_args())
    reader_ct.extend(h_mcast.compile_time_args())
    for t in (input_tensor, w_gate, w_down, counts, global_expert_idx_table):
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    writer_ct = [
        EMB_T,
        HID_T,
        KR_PAD,
        HN_PAD,
        EC_MAX,
        M_BLOCK,
        HGROUPS,
        KGROUPS,
        num_banks,
        WRUN,
        SEM_GO,
        SEM_DATA,
        bfp4_tile,
        bfp8_tile,
        remap,
        MAILBOX_MAGIC,
        M_EFF_MIN,
        reduce_slots,
        W_RESIDENT,
        CB_W_UP,
        CB_OUT_TILES,
        CB_GATE_SEND,
        CB_UP_SEND,
        CB_REDUCE_GATE_IN,
        CB_REDUCE_UP_IN,
    ]
    for t in (w_up, output_tensor):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    compute_ct = [
        M_BLOCK,
        KR_PAD,
        HN_PAD,
        EC_MAX,
        HGROUPS,
        HID_T,
        input_format,
        OUT_SUBBLOCK_H_GU,
        MAILBOX_MAGIC,
        M_EFF_MIN,
        OUT_SUBBLOCK_H_DN,
        reduce_slots,
        hn_block,
        ELTWISE_BLK,
        CB_X_IN,
        CB_X_TILES,
        CB_X_STAGE,
        CB_W_GATE,
        CB_W_UP,
        CB_W_DOWN,
        CB_GATE_ACC,
        CB_UP_ACC,
        CB_GATE_SEND,
        CB_UP_SEND,
        CB_GATE_SILU,
        CB_REDUCE_GATE_IN,
        CB_REDUCE_UP_IN,
        CB_H_LOCAL,
        CB_H,
        CB_OUT_INTERM,
        CB_OUT_TILES,
    ]

    mailbox_addr = mailbox.buffer_address()
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    for y in range(KGROUPS):
        for x in range(HGROUPS):
            core = ttnn.CoreCoord(x, y)
            i = y * HGROUPS + x
            node = tree[(x, y)]
            kr, kstart = kr_sizes[y], kr_starts[y]
            hn, hstart = hn_sizes[x], x * HN_PAD
            ec, jstart = ec_sizes[i], ec_starts[i]
            # The tile-rows this core injects into the row multicast are NO LONGER a host constant:
            # they depend on the runtime m_eff, so compute derives them from `my_col` with the same
            # shared `inject_rows()` the reader's staging loop implements.

            args = [
                mailbox_addr,
                input_tensor.buffer_address(),
                w_gate.buffer_address(),
                w_down.buffer_address(),
                counts.buffer_address(),
                global_expert_idx_table.buffer_address(),
                kr,
                kstart,
                hstart,
                hn,
                ec,
                jstart,
                1 if node["is_root"] else 0,
                len(node["children"]),
                x,
            ]
            for c in range(MAX_CHILDREN):
                if c < len(node["children"]):
                    cx, cy = _virt(device, *node["children"][c])
                else:
                    cx, cy = 0, 0
                args.extend([cx, cy])
            args.extend(x_mcast.runtime_args(core))
            args.extend(h_mcast.runtime_args(core))
            reader_rt[x][y] = args

            px, py = _virt(device, *node["parent"]) if node["parent"] is not None else (0, 0)
            # This core's landing slot in its parent's reduce CBs (Refinement 2 lever 1).
            my_slot = _slot_of.get((x, y), 0)
            writer_rt[x][y] = [
                mailbox_addr,
                w_up.buffer_address(),
                output_tensor.buffer_address(),
                kr,
                kstart,
                hstart,
                hn,
                ec,
                jstart,
                1 if node["is_root"] else 0,
                px,
                py,
                my_slot,
            ]

            compute_rt[x][y] = [
                mailbox_addr,
                kr,
                hn,
                ec,
                1 if node["is_root"] else 0,
                len(node["children"]),
                x,  # my_col — the x-multicast injection slot (round t's injector is column t % HGROUPS)
            ]

    # `/perf-measure` collective ablations: one transport stubbed, all CB scaffolding intact.
    # `+`-separated so stages can be peeled off CUMULATIVELY (they overlap, so removing one alone
    # under-counts it — the documented ablation methodology).
    dm_defines = [("ABLATE_" + a.upper(), "1") for a in ABLATE.split("+") if a in _DM_ABLATIONS]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
            defines=dm_defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
            defines=dm_defines,
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "moe_fused_swiglu_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_kernel_config,
            defines=[("SKIP_COMPUTE", "1")] if "skip_compute" in ABLATE.split("+") else [],
        ),
    ]

    semaphores = list(x_mcast.owned_semaphores()) + list(h_mcast.owned_semaphores())
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=all_cores, initial_value=0))
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_DATA, core_ranges=all_cores, initial_value=0))

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
