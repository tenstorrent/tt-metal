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
M_BLOCK = int(os.environ.get("MOE_SWIGLU_M_BLOCK", 8))

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
#: h all-gather slots. 3 = "a late round's producer is not flow-controlled by itself". Now a knob,
#: because phase 2 measures as EXACTLY 11 serialised rendezvous (43 us with every payload removed
#: ~= 11 x (1.2 us mcast + 1.7 us `down` K-block)) and the CB depth is the first thing that could be
#: forcing that lockstep. One extra slot costs M_BLOCK * HN_PAD bfp8 tiles = 52 224 B/core.
DEPTH_H = int(os.environ.get("MOE_SWIGLU_DEPTH_H", 3))
DEPTH_OUT = 2
DEPTH_XSTAGE = 1  # tilized x staging slots (a core injects <= ceil(M_BLOCK/HGROUPS) rows/block)
XSTICK_ROWS = 1  # tile-rows of row-major x sticks held in flight

#: Read-coalescing knob: max BANK-CONTIGUOUS weight/output tiles fetched per NoC transaction.
#: 1 reproduces the naive one-transaction-per-tile read (the ablation baseline).
#: Overridable for `/perf-measure` A/B via MOE_SWIGLU_WRUN.
#: PERF 10 — SHIPPED AT 1, i.e. bank-run coalescing AND the N-axis bank remap are OFF.
#: `remap` is gated on `WRUN > 1`, so this knob switches both together, and measured over two full
#: guard-set samples the pair is a NET NEGATIVE at the graded shapes: 6 of 12 cells are 3.7-7.2 %
#: FASTER without it (including both worst-gap cells, bf16_rm 256 -7.2 % and 512 -3.7 %), 4 flat,
#: and the only real loss is bf16_rm count 128 at +3.5 % -- which is about the size of that cell's
#: own run-to-run band (97 986..100 627 measured on the shipped binary). Consistent with the
#: RISC-issue-bound reading of Perf 9: the remap buys DRAM-side locality and pays for it in NoC
#: command count, and at the graded shapes the second term dominates. 8 restores the pre-PERF-10
#: coalesced+remapped stream.
WRUN = int(os.environ.get("MOE_SWIGLU_WRUN", 1))

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
#:   no_xstage_xfer — the ACTIVATION DRAM stream is dropped (Perf 2): the bf16 stick reads / bfp8
#:                    tile reads inside `reader_xstage` go away while the cb_x_in reserve/push, the
#:                    fused tilize, the cb_x_stage wait/pop and the self-copy into the resident slot
#:                    all stay. `no_x_xfer` drops the row MULTICAST of x, which is a DIFFERENT
#:                    stage — the two together are the whole activation path.
#: None is a correctness mode; each answers "how much of the 85% is THIS collective?".
ABLATE = os.environ.get("MOE_SWIGLU_ABLATE", "")
#: PERF 7 — `no_owrite` was the LAST hole in the peel. Without it the "all payloads stubbed" floor
#: still carried the op's 1.95 MB output DRAM stream (count 256), which is exactly how the reference
#: op's `down` peel bottomed out in a phantom 47 % floor. `writer_out_issue` measured 17.4 us inside
#: what was being reported as pure synchronisation.
_DM_ABLATIONS = ("no_reduce_xfer", "no_h_xfer", "no_x_xfer", "no_w_xfer", "no_xstage_xfer", "no_owrite")

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

#: PERF 2 — THE CROSS-COLUMN REDUCE STRUCTURE. The one knob of this round, and the biggest single
#: lever the op has had.
#:
#:   "tree"    — the shipped binary (Hillis-Steele) reduce tree of `_reduce_tree` below: every core
#:               funnels its WHOLE `m_eff * HN_PAD` gate AND up block up the column to the root, and
#:               the root alone runs the SwiGLU epilogue (the SiLU-fused bias add walked `m_eff`
#:               times, then the multiply). Reproduced BYTE-IDENTICALLY by this value — it is the A/B
#:               for any re-measurement and the honest fallback for any geometry the scatter's
#:               preconditions do not hold on (see `_scatter_plan` below).
#:   "scatter" — a two-phase REDUCE-SCATTER down each column with a DISTRIBUTED epilogue. Every core
#:               owns a disjoint slice of the T = m_eff*HN_PAD tile block; all KGROUPS cores push
#:               their slice of gate and up straight into every worker's landing CBs; each worker
#:               reduces ONLY its own slice over all contributors, runs the SiLU + SwiGLU epilogue on
#:               it, and unicasts the finished `h` slice straight into the ROOT's cb_h_local at its
#:               tile offset — the gather IS the assembly, so the root does no copy and no add.
#:
#: WHY IT WINS, and the number that says it is the epilogue and not the adds. MEASURED in the
#: isolated bake-off (`perf_experiments/reduce_scatter_swiglu/`, one grid column, KGROUPS 10,
#: m_eff 8, HN_PAD 6, T = 48 bfp8 tiles, identical precision contract on every variant):
#:     tree (shipped)             78 011 ns   1.00x   PCC 0.999823   417 792 B/core
#:     scatter (this knob)        27 853 ns   2.80x   PCC 0.999777   313 344 B/core (-104 448)
#:     scatter, epilogue at root  71 615 ns   1.09x   PCC 0.999775   +91 392 B/core
#: i.e. ~85 % of the win is the EPILOGUE, not the adds — independently corroborated by a per-pass
#: ablation of the shipped 58 830 ns stage: ~44 100 ns (75 %) is the 48-tile SFPU SiLU, ~10 700 (18 %)
#: the plain adds, ~2 700 (4.5 %) the up-add + multiply, ~800 (1.4 %) the whole bias walk overhead.
#: Scattering the epilogue parallelises the DOMINANT term, and a worker's slice is <= DEST (8 tiles),
#: so the `m_eff`-call SiLU bias walk collapses to ONE call for free. On the real 11x10 grid with all
#: 11 column collectives concurrent the isolated 3.08x eroded only 0.7 %.
#:
#: PREDICATE: UNCONDITIONAL. Every cell of KGROUPS {2,4,8,10} x m_eff {1,2,4,8} x HN_PAD {4,6} wins,
#: 1.65x to 3.11x, zero regressions — including both degenerate ends (m_eff 1 = 6 tiles over 10 cores
#: is still 1.65x; KGROUPS 2 is 1.76x). Two sub-predicates from the same sweep are honoured in the
#: code rather than left as folklore: the slice axis is the flat TILE INDEX and never the token (M)
#: axis (that axis caps the worker count at m_eff and REGRESSES to 0.79x at m_eff 1), and the `h`
#: gather is fused straight into cb_h_local rather than through a landing CB the root copies (worth
#: 8.6 % and 52 224 B).
#:
#: MEASURED IN SITU (Perf 2, the 12-cell guard set, one fresh-cache profiled run per value of THIS
#: knob with everything else at its shipped setting, whole-op DEVICE KERNEL DURATION — a true A/B of
#: one binary, not of two branches):
#:   bf16_rm  7168/5120/128    139 229 -> 109 314 ns   -21.5 %   1.274x
#:            7168/5120/256    205 107 -> 150 479      -26.6 %   1.363x   (the focus cell)
#:            7168/5120/512    359 411 -> 251 915      -29.9 %   1.427x
#:            7168/1024/256    203 590 -> 149 036      -26.8 %   1.366x
#:            6144/5120/256    191 797 -> 136 059      -29.1 %   1.410x
#:            7168/5120/5120 3 175 661 -> 2 037 988    -35.8 %   1.558x
#:   bfp8     7168/5120/128    133 143 -> 106 483      -20.0 %   1.250x
#:            7168/5120/256    201 622 -> 147 253      -27.0 %   1.369x
#:            7168/5120/512    349 710 -> 239 979      -31.4 %   1.457x
#:            7168/1024/256    201 927 -> 147 400      -27.0 %   1.370x
#:            6144/5120/256    187 784 -> 131 600      -29.9 %   1.427x
#:            7168/5120/5120 3 051 287 -> 1 910 670    -37.4 %   1.597x
#:   12-cell sum            8 400 268 -> 5 518 176     -34.3 %   1.522x
#: ZERO regressions; 1.25x to 1.60x, monotone in the M extent (more M-blocks = more repetitions of the
#: collective, so more of the op is this stage). And it costs ~82.8 KB/core LESS L1.
#:
#: WHY 1.36x IN SITU AND NOT THE ISOLATED 3.08x — the honest ceiling, and it is AMDAHL, not erosion.
#: The stage really did collapse: over the guard set the critical core's `compute_reduce` went
#: 4 418 040 -> 1 351 719 ns (-69 %) and the whole-op sum moved 2 804 Ms of that 3 066 Ms, i.e. ~91 %
#: of the stage saving reached the wall clock. What bounds it is that the reduce+epilogue was ~33 % of
#: the op, so removing 69 % of it cannot exceed ~1.5x. After this change the focus cell's top stages
#: are `reader_phase2` 78 kns / `compute_down` 98 kns / `writer_out_issue` 87 kns against
#: `compute_reduce` 39 kns — the h all-gather and the phase-2 `down` matmul are the critical path now,
#: and the reduce is a distant fourth whose remaining time is mostly WAITING for the all-to-all, not
#: math. The next round belongs to phase 2.
REDUCE = os.environ.get("MOE_SWIGLU_REDUCE", "scatter")

#: PERF 2 — which NoC carries the two halves of the column all-to-all. `DM_DEDICATED_NOC` binds the
#: reader to NOC_0 and the writer to NOC_1, so "which kernel issues the write" IS the NoC choice.
#:   "one"   — both the gate and the up gather legs go out on the WRITER (NOC_1), the shape the
#:             bake-off measured.
#:   "split" — the GATE legs stay on the writer (NOC_1) and the UP legs move to the reader (NOC_0),
#:             halving each NoC's payload for the same bytes. Split by PAYLOAD, not by direction: a
#:             per-DESTINATION direction split (the ideal, since a sibling measured a bidirectional
#:             all-to-all on one NoC paying up to 1.9x in torus-wrapped HOP COUNT) would need BOTH
#:             data-movement RISC-Vs to consume ONE CB, and `cb_pop_front` writes the shared
#:             `tiles_acked` word with the popping RISC-V's own local count — the exact single-owner
#:             hazard that produced round 1's "in-place add hangs". Splitting by payload gives each
#:             RISC-V its OWN CB (writer pops cb_gate_acc, reader pops cb_up_acc) and stays legal.
#:
#: MEASURED, and deliberately measured TWICE because the effect sits ON the noise band — the 12-cell
#: guard-set sum, two independent fresh-cache runs per value:
#:   "one"    5 596 297 / 5 600 029 ns   (the two runs agree to 0.07 %)
#:   "split"  5 509 590 / 5 524 843 ns   (0.28 %)   = -1.45 % on the median, 12/12 cells faster in
#:                                                    run 1 and 8/12 in run 2
#: The between-group gap is 5x the within-group spread, and it CONCENTRATES exactly where the
#: mechanism predicts: the two `count 5120` cells (20 M-blocks, so 20 repetitions of the collective)
#: move -1.5 to -2.0 % in BOTH runs, while a single-dispatch focus-cell run is a coin flip
#: (150 627 "one" vs 151 427 "split"). SHIPPED AT "split"; "one" reproduces the bake-off's shape.
#:
#: This is NOT the ideal split. A sibling measured a bidirectional all-to-all confined to one NoC
#: paying up to 1.9x in torus-wrapped HOP COUNT (NOC_1 routes modular-decreasing, NOC_0
#: modular-increasing), which says the right split is per-DESTINATION — "below me" on one NoC, "above
#: me" on the other. That is NOT expressible here: both halves would then read the SAME accumulator
#: CB from BOTH data-movement RISC-Vs, and `cb_pop_front` writes the shared `tiles_acked` word with
#: the popping RISC-V's own local count. Getting it would need a second copy of the accumulator
#: (+52 KB/core, giving back half the path's L1 win) or a private ready-signal so the second RISC-V
#: can address the CB without owning its lifecycle — a real experiment, not a knob turn. The
#: payload split captures the bandwidth half of the idea and none of the hop-count half.
SCATTER_NOC = os.environ.get("MOE_SWIGLU_SCATTER_NOC", "split")

#: Worker-grid override, "<HGROUPS>x<KGROUPS>", e.g. "11x8" (88 cores) or "11x9" (99). Empty = the
#: device's full `compute_with_storage_grid_size()`. The graded PERF_MEASURED_NS baselines this op is
#: stretched against were taken on 88/99-core grids, so the op has to be tuned and reported at those
#: core counts too, not only at this box's 110. Clamped to the device grid host-side.
GRID = os.environ.get("MOE_SWIGLU_GRID", "")

#: PERF 3 — READ ORDER inside an M-block: 1 = stage `x` BEFORE issuing the W_gate prefetch, 0 = the
#: Phase-0 order (W_gate first). `noc_async_read_barrier()` is all-or-nothing, so whichever stream is
#: issued first is the one the other's barrier waits for. See the reader's comment for the measured
#: 33 us row spread this decides.
#:
#: MEASURED NULL, kept as the A/B that PROVED the diagnosis rather than as a win. Alone it is -0.2 to
#: +2.8 % (150 260 vs 151 233 ns at count 256), and it is WORSE in every combination with XPRIO
#: (152 791 vs 146 945). That is the same lesson XPRIO encodes: re-ordering ONE core's own two issue
#: streams cannot move x forward when the queue x is stuck behind belongs to the other 109 cores and
#: to the writer's NoC1 twin. 0 (the Phase-0 order) is the default.
XSTAGE_FIRST = int(os.environ.get("MOE_SWIGLU_XSTAGE_FIRST", 0))

#: PERF 3 — N-CHUNKED GATE/UP WEIGHT STREAM. The bfp4 gate/up block is issued, published and
#: consumed in this many HIDDEN-axis chunks instead of one, so the matmul runs on chunk c while
#: chunk c+1 is still in DRAM. 1 == the Phase-0 whole-block shape, byte-identical.
#:
#: WHY N AND NOT K. Measured on the focus cell, the weight DRAM stream (38 163 ns exposed) and the
#: matmul math (22 801 ns) are STRICTLY ADDITIVE — ablating both together lands at 85 266 ns against
#: a 148 742 ns baseline, i.e. 85 266 + 38 163 + 22 801 = 146 230, so nothing overlaps. K-chunking
#: would overlap them too, but a K-chunk is a partial sum: it costs `m_eff` extra L1-ACCUMULATING
#: packs per extra K-block per matrix, and at this shape that is roughly the whole overlap it buys.
#: N-chunks are INDEPENDENT matmuls — no partial sums, no extra packs — and the block is K-major in
#: the CB, so an N-chunk is contiguous on both the DRAM read side and the matmul in1 side.
#:
#: CONSTRAINED, and the host clamps rather than trusts: must divide HN_PAD, must be a multiple of
#: HN_BLOCK, and must leave the RAGGED column group (hn < HN_PAD) at least one real column in every
#: chunk. At HID_T 64 / HGROUPS 11 that is HN_PAD 6, hn_min 4 -> 2 is the largest legal value.
GU_CHUNKS = int(os.environ.get("MOE_SWIGLU_GU_CHUNKS", 3))

#: PERF 3 — ACTIVATION-FIRST DRAM PRIORITY. 1 = the writer holds its W_up stream until this core's
#: reader has finished staging `x` (SEM_XSTAGED); 0 = both streams race from t=0 (Phase-0 behaviour).
#:
#: WHY IT IS A GRID-WIDE ORDERING PROBLEM, not a local one. Profiled at GU_CHUNKS=2: the gate/up
#: matmul does not wait for weights at all — `reader_wg_wait` returns in ~4 us — it waits for
#: `cb_x_tiles`, which the x row-multicast publishes at 56 us. x is only 3.67 MB of phase 1's
#: 20.2 MB, but every core's matmul needs ALL of its row's x before it can start, whereas the
#: weights are consumed per column. With both streams issued at t=0 on 110 cores, DRAM is saturated
#: and x's last byte simply arrives at the END of the mixed stream. Re-ordering the READER's own
#: issues (XSTAGE_FIRST) cannot fix that — the competition is the other 109 cores and the writer's
#: 8.7 MB W_up twin. Holding every core's W_up until its x is read is what actually moves x to the
#: front of the grid-wide queue; the weight stream then runs UNDER the matmul instead of before it.
XPRIO = int(os.environ.get("MOE_SWIGLU_XPRIO", 1))

#: PERF 3 — the h all-gather's data-ready signal: "counter" (monotone, never reset) or "flag" (the
#: reset-per-round signal). Counter was UNUSABLE (a guaranteed sender hang) until the
#: `NOC_CMD_VC_LINKED` bug in `mcast_pipe.inl::send_data_` was fixed this round; it is now correct
#: (golden 45/45 on it) and MEASURED SLOWER: 112 087 / 159 968 / 278 493 ns against Flag's
#: 105 076 / 145 625 / 247 181 (+7 to +13 %).
#:
#: WHY, and it is the useful part: the link that Flag terminates for free is what ordered
#: data-before-signal. Unlinked, the Counter path has to buy that ordering with an ACKED
#: `async_write_barrier()` (SENT is not enough — a receiver could see the counter and read a
#: half-written block) plus a non-posted atomic barrier, and that pair costs MORE than the flag-reset
#: round trip it removes. So the flag reset was never the keystone: what actually serialises phase 2
#: is the LOOP ORDER — the sender of round r+1 must first RECEIVE rounds 0..r — and no signal change
#: touches that. `flag` is the default.
HSIG = os.environ.get("MOE_SWIGLU_HSIG", "flag")

#: PERF 3 — WHO SENDS the h all-gather: "reader" (Phase 0) or "writer" (the dual-RISC split).
#:
#: THE KEYSTONE. Phase 2 measures as HGROUPS SERIALISED rendezvous (43 us with every payload ablated
#: ~= 11 x 3.9 us) and every cheaper explanation has been eliminated by measurement: not the CB depth
#: (DEPTH_H 2/3/4/5 -> 149.9/145.9/147.4/147.6k), not the flag reset (the now-fixed monotone Counter
#: is 7-13 % SLOWER), not the grid shape (HGROUPS 8/10 overflow L1 and break the scatter plan). What
#: is left is the LOOP ORDER: a core sends its own column at iteration `r == my_col` of the same loop
#: in which it receives every other column, so the sender of round r+1 must first RECEIVE rounds
#: 0..r — an 11-long chain. Absolute per-slot addressing would break it but needs the whole of `h`
#: resident: +417 792 B against 121 728 B free, over even after the 104 448 B aliasing enabler.
#:
#: The split moves the SEND to the writer (NoC1), which is not in the receive loop, so a root sends as
#: soon as the grid has freed its slot. `consume(r)` then needs only `consume(r - DEPTH_H)` — chain
#: depth HGROUPS/DEPTH_H ~= 4 instead of 11 — and it costs NO extra L1 because the rolling window is
#: unchanged. Pairs with the per-slot arrival counters below (one monotone counter per cb_h slot, so
#: out-of-order senders are distinguishable, which a single counter cannot do).
#: BUILT AND MEASURED: correct (golden 45/45) and a NULL — 110 299 / 152 919 / 264 829 ns against
#: reader's 104 813 / 146 741 / 246 790 (+5 to +7 %). Phase 1 got FASTER (compute_gateup 2->81 us
#: vs 2->85) but compute_down stretched to 144, because the rounds were never waiting on the loop
#: order: they wait on ROOT READINESS (compute_reduce spans 40->96 us, so the eleven roots finish
#: ~56 us apart), and that spread is the weight stream's. `reader` is the default.
#:
#: PERF 4 CORRECTION, from the tt-npe NoC trace (88 cores, count 256): the "root readiness" reading
#: above is WRONG, and it was wrong because it was inferred from a 110-core zone span instead of
#: measured. Correlating each root's `compute_reduce` END against the timestamp of its own h
#: multicast: every one of the 11 roots is finished by t = 101.2 us, and the multicasts go out at
#: 102.9, 107.2, ... 145.9 us on a rigid 4.3 us cadence. Round 10's root sits on ready data for
#: 57.2 us. The rounds are NOT data-gated. See HACK_AHEAD below for what actually gates them.
HSEND = os.environ.get("MOE_SWIGLU_HSEND", "reader")

#: PERF 4 — how many rounds' senders a receiver acks in one `cb_reserve_back`. THE round-cost lever.
#:
#: Fitting the measured h-multicast cadence across two NoC traces (m_eff 4 -> 3.66 us/round,
#: m_eff 8 -> 4.30 us/round) gives
#:
#:      round period = 3.12 us FIXED + 0.147 us per m-tile
#:
#: against ~2.06 us of real work at m_eff 8 (52 224 B of h ingest at ~43 GB/s = 1.21 us, plus a
#: 144 tile-MAC `down` block at the 8 cycles/tile-MAC LoFi roofline = 0.85 us). So 3.12 us x 11
#: rounds = 34.3 us per M-BLOCK of pure rendezvous, INDEPENDENT of M — and count 512 runs two
#: M-blocks, i.e. 68.6 us, which is 27 % of that cell's wall clock and exactly why 512 is the worst
#: cell. The op's own ablation floor (42.8 us with every payload stubbed) is the same term.
#:
#: The mechanism, from the trace: round r's sender waits for ONE ack from each of NUM_CORES
#: receivers, and a receiver only acks after `cb_reserve_back` for round r proves its slot free.
#: So each round is [NUM_CORES-way ack incast] -> [52 KB multicast] -> [ready signal] -> [every
#: core wakes], four grid traversals that cannot start until the previous round finished on the
#: SLOWEST core. HSEND=writer removed the shared-flag half of that chain and measured null because
#: the ack half — the expensive half — was left just-in-time.
#:
#: A > 1 reserves A blocks at once, which proves A slots free, so a core acks senders r..r+A-1
#: together and every ack lands A-1 rounds early. The 11 senders then overlap up to the CB depth.
#: Clamped to DEPTH_H - 1: reserving the whole CB would demand zero blocks in flight and
#: re-serialise the reader against compute every round. 1 = the pre-PERF-4 path, byte for byte.
#: Only meaningful with HSEND=writer (the per-slot monotone counters are what let senders finish
#: out of order); at HSEND=reader the shared VALID flag re-imposes the chain and A is forced to 1.
HACK_AHEAD = int(os.environ.get("MOE_SWIGLU_HACK_AHEAD", 2))

#: PERF 8 — transaction-id ring on the gate/up weight stream. See the reader kernel for the
#: mechanism. 0 = the pre-PERF-8 one-chunk-in-flight stream, byte for byte.
WG_TRID = int(os.environ.get("MOE_SWIGLU_WG_TRID", 0))

#: PERF 9 — issue the W_down batch AFTER the reduce invite instead of before the reduce, so it lands
#: in the 45 us DRAM-idle window the NoC trace found instead of on top of the W_gate/W_up stream.
#: Only meaningful with the scatter reduce (the tree path has no invite at the same point).
WD_LATE = int(os.environ.get("MOE_SWIGLU_WD_LATE", 0))

#: PERF 12 — DRAM ND-SHARDED WEIGHTS. Not a knob but a DETECTION: the shard width is read off the
#: weight tensors the CALLER handed in, so an interleaved weight produces the byte-identical
#: pre-PERF-12 stream and a sharded one produces the coalesced stream. `MOE_SWIGLU_WSHARD=0` forces
#: the interleaved read path even on a sharded tensor, which is the A/B for a re-measurement (the
#: accessor still addresses the shard correctly — only the RUN LENGTH changes).
#:
#: WHY SHARDING IS THE ONLY WAY TO GET A BIG WEIGHT TRANSACTION. Interleaved page -> bank is
#: `page_id % num_banks`, so for `W_gate [emb, hidden]` the tile `(k, n)` at page `k*HID_T + n` puts
#: CONSECUTIVE n in DIFFERENT banks — one NoC request per tile, structurally. PERF 10 already
#: measured the only interleaved escape (remap the N axis so a bank's slots are consecutive, then
#: coalesce) as a NET NEGATIVE: it buys DRAM-side locality and pays in NoC command count. A DRAM ND
#: shard of `[TILE, W*TILE]` instead makes a shard's W tiles physically contiguous in ONE bank, so
#: the same coalescing costs no remap and no extra commands.
#:
#: THE SHARD MUST BE ONE TILE-ROW TALL, and that is a MEASURED constraint, not a formality: a core
#: pinned to a single DRAM bank saturates near 30 GB/s no matter how big the request, while the same
#: bytes with the bank rotating reach ~370 GB/s. A one-tile-row shard is distributed ROUND_ROBIN_1D
#: as `shard_id = k * shards_per_row + gx`, so consecutive K-rows land in DIFFERENT banks; a TALLER
#: shard would make a core's whole K-run one request but would pin it to one bank.
WSHARD = int(os.environ.get("MOE_SWIGLU_WSHARD", 1))

#: PERF 11 — x-staging placement. DIAG rotates which tile-row a column stages by grid row (so the 8
#: rows of a column read 8 different DRAM pages instead of the same one); STAGGER rotates the stick
#: walk start (the only one that changes which BANK a core is on at a given instant, since the bank
#: is `s % 8` whatever the tile-row). Both 0 = the pre-PERF-11 lockstep column shape.
XSTAGE_DIAG = int(os.environ.get("MOE_SWIGLU_XSTAGE_DIAG", 0))
XSTAGE_STAGGER = int(os.environ.get("MOE_SWIGLU_XSTAGE_STAGGER", 0))

#: PERF 4 — one VALID cell per `cb_h` slot instead of one shared by every round. The other half of
#: the round-cost lever, and the one that makes HACK_AHEAD legal on the FAST (reader-send) path.
#:
#: mcast_pipe offers exactly two data-ready signals and each gives up something this op needs:
#:   Flag    — data and signal ride ONE NOC_CMD_VC_LINKED chain, so ordering is free (no acked
#:             write barrier). But there is ONE cell, so round r+1's sender cannot set VALID until
#:             every core has cleared round r's -- a per-round grid-wide serialisation, which is
#:             mcast_pipe's own documented caveat.
#:   Counter — monotone per-slot, so rounds can overlap. But the signal is an ATOMIC on a different
#:             command buffer and therefore cannot terminate the link (the Perf-3 addendum-2 fix),
#:             so the data must go UNLINKED and the sender pays an ACKED write barrier every round:
#:             a second NUM_CORES-way incast, measured 11 % worse end to end.
#: Per-SLOT FLAGS take the union: the link survives (no barrier) and rounds r and r+1 touch
#: different cells (no reset chain). Rounds r and r+DEPTH_H share a cell and are ordered by the ack
#: itself. Reuses the DEPTH_H `SEM_H_RDY_BASE` cells the Counter path already allocates, so it costs
#: no semaphore and no L1. 0 = the pre-PERF-4 shared-flag pipe, byte for byte.
#: SHIPPED ON (with HACK_AHEAD 2): 99 981 / 146 650 / 244 536 ns at 88 cores against the shared
#: flag's 101 870 / 152 710 / 254 113 -- -1.9 % / -4.0 % / -3.8 %, no cell regressed.
HSLOT = int(os.environ.get("MOE_SWIGLU_HSLOT", 1))

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
CB_REDUCE_GATE_IN = 6  # `tree`: incoming child partials (gate)
CB_REDUCE_UP_IN = 7  # `tree`: incoming child partials (up)
CB_H = 8  # gathered h, one phase-2 K-block per round
CB_IDX_SCRATCH = 9
CB_COUNTS_SCRATCH = 10
# `scatter` (PERF 2): the reduce-scatter's landing + slice CBs. One slot per CONTRIBUTOR in the two
# landing CBs; the three slice CBs are only `slice_pages` deep, which is what makes this path
# ~100 KB/core SMALLER than the tree it replaces.
CB_GATHER_GATE = 11  # every contributor's gate slice, slot `row`
CB_GATHER_UP = 12
CB_SLICE_GATE = 13  # this worker's gate-slice accumulator (in-place)
CB_SLICE_UP = 14
CB_H_SLICE = 15  # this worker's finished h slice, unicast into the root's cb_h_local
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
# `tree`: parent -> child "your slot is free, send". `scatter`: the peer INVITE — every core tells
# every core in its column "my landing CBs are reserved", which is the same flow control generalised
# from one parent to K peers.
SEM_GO = 4
SEM_DATA = 5  # `tree`: child -> parent "partials landed". `scatter`: contributor -> worker, same.
SEM_HSLICE = 6  # `scatter`: worker -> column root "my finished h slice landed in your cb_h_local"
# PERF 3 — reader -> writer, SAME CORE: "this core's `x` tile-rows are read and staged". The only
# intra-core semaphore in the op: no NoC traffic, just an L1 word the reader stores and the writer
# polls. It exists to ORDER THE TWO DRAM STREAMS against each other — see XPRIO.
SEM_XSTAGED = 7
# PERF 3 (HSEND=writer) — ONE monotone arrival counter PER cb_h SLOT. Per-slot, not one shared
# counter: with the senders decoupled from the receive loop their increments can interleave, and a
# single counter says HOW MANY blocks arrived but not WHICH — the same "wave push" defect Perf 2 #4
# root-caused in the reduce. DEPTH_H counters suffice because only DEPTH_H rounds are ever in flight.
SEM_H_RDY_BASE = 8
# ... and the window ack: receiver -> round r's sender, "I have reserved slot r, you may write it".
SEM_H_FREE = SEM_H_RDY_BASE + DEPTH_H


def _pow2_ceil(v):
    """Smallest power of two >= v (v >= 1)."""
    p = 1
    while p < v:
        p <<= 1
    return p


def _largest_divisor_le(n, cap):
    """The `scatter` worker count — largest divisor of `n` that is <= cap. The HOST twin of
    `slice_workers()` in kernels/moe_fused_swiglu_common.hpp; the two must agree exactly, so the
    kernel-side one is the definition and this one only SIZES the CBs it implies."""
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _scatter_plan(m_block, m_eff_min, hn_pad, kgroups):
    """CB sizing for the `scatter` reduce path, or (None, reason) if this geometry cannot express it.

    The whole point of this function is that it enumerates EVERY runtime m_eff the kernels can reach
    (`m_tiles_eff` returns a power of two in [m_eff_min, m_block]) and sizes the CBs for all of them
    at once, because the slice plan SHRINKS with m_eff while the CBs are allocated once.

    TWO HARD PRECONDITIONS, both silent-wrong-answer class if broken — asserted here, never assumed
    device-side, and a failure falls back to `tree` rather than shipping a wrong answer:

      1. `P % B == 0` for every CB the kernels cycle in blocks of `B` pages. A CB's write pointer
         advances by the pages pushed and wraps only at the CB END, so a block that starts mid-CB and
         runs past the end does not wrap — it OVERRUNS INTO THE NEXT CB. Measured in the bake-off:
         a plan violating this scored PCC 0.709-0.886 where every legal plan scored >= 0.9955.
         Here `B` is the per-m_eff slice size `a`, so the slice CBs are `lcm(a)` pages and the
         landing CBs (`KGROUPS * max(a)`, pushed WHOLE every block) must also be a multiple of every
         `a`, since compute consumes them `a` pages at a time.
      2. `KGROUPS >= 2`. The slice reduce is `copy` + (NC-2) in-place adds + one SiLU-fused final
         add over NC == KGROUPS contributors, so a column of one has no reduce to scatter.
    """
    if kgroups < 2:
        return None, f"KGROUPS {kgroups} < 2: a column of one has no cross-column reduce to scatter"
    sizes, m = [], m_eff_min
    while m <= m_block:
        t = m * hn_pad
        sizes.append(t // _largest_divisor_le(t, kgroups))
        m *= 2
    slice_pages = 1
    for a in sizes:
        slice_pages = slice_pages * a // _gcd(slice_pages, a)
    gather_pages = kgroups * max(sizes)
    for a in sizes:
        if slice_pages % a or gather_pages % a:
            return None, (
                f"slice sizes {sorted(set(sizes))} over the reachable m_eff need slice CBs of "
                f"{slice_pages} and landing CBs of {gather_pages} pages, and {a} divides neither"
            )
    return {"slice_pages": slice_pages, "gather_pages": gather_pages, "sizes": sizes}, None


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def nd_shard_n_tiles(t):
    """N-axis TILES per DRAM ND shard of `t`, or 0 if `t` is not usable for the coalesced read.

    This is the ONE place the op learns a weight's placement; everything downstream is a run length.
    Returns 0 (the interleaved stream, byte-identical to pre-PERF-12) unless ALL of:
      * DRAM ND-sharded (`created_with_nd_shard_spec`), so `TensorAccessor` lays a shard's pages out
        contiguously in one bank at `aligned_page_size` stride;
      * rank-2 shard whose N extent is a whole number of tiles.
    The shard HEIGHT is deliberately NOT constrained — `page_offset_within_shard` is
    `(k % SH) * SHARD_W + (n % SHARD_W)`, so for a fixed K-row consecutive n are contiguous at any
    height. Height is a BANDWIDTH choice (one tile-row rotates banks across K, see the WSHARD knob),
    and the caller owns it.
    """
    if not WSHARD:
        return 0
    try:
        mc = t.memory_config()
        spec = mc.nd_shard_spec
        if spec is None or mc.buffer_type != ttnn.BufferType.DRAM:
            return 0
        shape = list(spec.shard_shape)
    except Exception:  # pragma: no cover - defensive: an older tensor type without the attribute
        return 0
    if len(shape) != 2 or int(shape[-1]) % TILE != 0:
        return 0
    return int(shape[-1]) // TILE


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


def worker_grid(device):
    """The (HGROUPS, KGROUPS) this op will actually use on `device` — the ONE definition, shared by
    `create_program_descriptor` and by `weight_memory_configs` below so a caller's shard width cannot
    drift from the op's own N split."""
    grid = device.compute_with_storage_grid_size()
    hgroups, kgroups = int(grid.x), int(grid.y)
    if GRID:
        gx, gy = (int(v) for v in GRID.lower().split("x"))
        hgroups, kgroups = min(gx, hgroups), min(gy, kgroups)
    return hgroups, kgroups


def weight_memory_configs(device, emb, hidden):
    """PERF 12 — the DRAM ND shard the coalesced weight stream wants, as `(gate_up, down)` memory
    configs. PUBLIC because the placement is the CALLER's to choose: the op reads whatever width it
    is handed (`nd_shard_n_tiles`) and an interleaved weight is still correct, just uncoalesced.

    Each shard is exactly the N slice ONE core consumes for ONE K-row:
      * gate/up — `HN_PAD = ceil(HID_T / HGROUPS)` tiles, the hidden split across grid COLUMNS, so
        column x's `[hstart, hstart + hn)` IS shard x and its whole K-row read is one request.
      * down    — `EC_MAX = max(ec)` tiles, the emb-output split across ALL cores. `_split` packs the
        wider groups first, so a narrow core's `[jstart, jstart + ec)` can straddle one boundary and
        cost two requests instead of one; `run()` splits it correctly either way.
    Height is ONE TILE-ROW so shards rotate DRAM banks across K — see the WSHARD knob for the
    measurement that makes that the load-bearing part.
    """
    hgroups, kgroups = worker_grid(device)
    hid_t, emb_t = hidden // TILE, emb // TILE
    hn_pad = (hid_t + hgroups - 1) // hgroups
    ec_max = max(_split(emb_t, hgroups * kgroups)[0])
    dram = device.dram_grid_size()
    banks = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])

    def mc(n_tiles):
        return ttnn.MemoryConfig(
            ttnn.BufferType.DRAM,
            ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([TILE, n_tiles * TILE]),
                grid=banks,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    return mc(hn_pad), mc(ec_max)


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
    if GRID:
        gx, gy = (int(v) for v in GRID.lower().split("x"))
        HGROUPS, KGROUPS = min(gx, HGROUPS), min(gy, KGROUPS)
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
    # PERF 3 — the N-chunk the weight stream is published in. CLAMPED, not trusted: it must divide
    # HN_PAD, and — the constraint that actually binds — every chunk of the RAGGED column group must
    # still hold a real column, because the helper can narrow an in1 sub-block's FMA width but a chunk
    # with no real column at all would have nothing to read. Any illegal request falls back to the
    # largest legal divisor (1 in the worst case = the Phase-0 whole-block stream).
    # The ragged column may now have chunks with NO real column: the kernels skip that chunk's DRAM
    # read and its matmul but still push the CB (so the residency wrap is unchanged), and the pad
    # columns it leaves are the same pad columns `down` already never contracts over — `HnSteps`
    # narrows the last K-block's FMA to `hn_last`. So the only remaining constraint is divisibility.
    gu_chunks = max(1, GU_CHUNKS)
    while gu_chunks > 1 and HN_PAD % gu_chunks != 0:
        gu_chunks -= 1
    if gu_chunks != GU_CHUNKS:
        print(
            f"moe_fused_swiglu: GU_CHUNKS {GU_CHUNKS} illegal at HN_PAD {HN_PAD} / ragged hn "
            f"{min(hn_sizes)}; using {gu_chunks}"
        )
    gu_chunk_w = HN_PAD // gu_chunks

    # gate/up in1 sub-block width — Refinement 2 lever 2. Now a sub-division of ONE chunk, so it is
    # additionally clamped to gu_chunk_w; at gu_chunks == 1 this is the Phase-0 expression unchanged.
    hn_block = gu_chunk_w if HN_BLOCK <= 0 or HN_BLOCK >= gu_chunk_w else HN_BLOCK
    if gu_chunk_w % hn_block != 0:
        raise RuntimeError(f"moe_fused_swiglu: HN_BLOCK {hn_block} must divide the chunk width {gu_chunk_w}")
    gu_in1_subblocks = gu_chunk_w // hn_block
    # The ragged column's real width must reach INTO the last sub-block OF ITS LAST CHUNK: the helper
    # can narrow the last in1 sub-block's FMA width (`last_in1_subblock_w_valid`) but cannot drop one.
    # Only chunks that HAVE a real column must reach into their last in1 sub-block; an empty chunk is
    # skipped wholesale by the kernels, so it imposes nothing.
    _ragged_last = min(hn_sizes) - ((min(hn_sizes) - 1) // gu_chunk_w) * gu_chunk_w
    if _ragged_last <= (gu_in1_subblocks - 1) * hn_block:
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

    # ---- PERF 12: the DRAM ND-sharded weight stream ----
    # W_gate and W_up share one width because the reader and the writer read the SAME [k, n] slice of
    # two identically-shaped tensors; a disagreement would silently give the up matmul a different
    # coalescing than the gate's, so it is resolved here to the common value (0 = interleaved) rather
    # than per-tensor.
    wg_shard_w = min(nd_shard_n_tiles(w_gate), nd_shard_n_tiles(w_up))
    wd_shard_w = nd_shard_n_tiles(w_down)
    # The N-axis REMAP and the shard are two different answers to the same question and cannot both be
    # live: remap re-indexes N for the interleaved bank layout, which would scatter the shard's own
    # contiguous run. The shard wins (it is the measured one) and the remap is dropped grid-wide --
    # including the W_down ROW index, which pairs with the gate/up N axis.
    if wg_shard_w or wd_shard_w:
        remap = 0
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

    # ---- PERF 2: the reduce-scatter plan, or an honest fall back to the tree ----
    # The tree is retained BYTE-IDENTICALLY on the `tree` value of the knob AND as the automatic
    # fallback here, so a geometry whose slice plan is not expressible still produces a correct
    # program instead of a raise (the scatter is a pure perf restructure; it owes the caller nothing).
    scatter_plan, scatter_why_not = (
        (None, "MOE_SWIGLU_REDUCE=tree") if REDUCE != "scatter" else _scatter_plan(M_BLOCK, M_EFF_MIN, HN_PAD, KGROUPS)
    )
    if REDUCE not in ("tree", "scatter"):
        raise RuntimeError(f"moe_fused_swiglu: MOE_SWIGLU_REDUCE must be 'tree' or 'scatter', got {REDUCE!r}")
    scatter = scatter_plan is not None
    if not scatter and REDUCE == "scatter":
        print(f"moe_fused_swiglu: MOE_SWIGLU_REDUCE=scatter unavailable, using the reduce tree — {scatter_why_not}")
    # Pages per CB on each path. The inactive path's CBs get ONE page instead of ~50 KB each — the
    # same trick the row-major staging CBs use on the bfp8 activation path, so both paths' kernels
    # can name every CB index unconditionally while only the live path costs L1.
    n_gather = scatter_plan["gather_pages"] if scatter else 1
    n_slice = scatter_plan["slice_pages"] if scatter else 1
    scatter_noc_split = 1 if (scatter and SCATTER_NOC == "split") else 0

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
    # PERF 3 — the h all-gather's data-ready signal. Flag is RESET by each receiver, so the sender of
    # round r+1 cannot proceed until every receiver has cleared round r's flag: a per-round grid-wide
    # serialisation, and measurably the op's keystone (phase 2 = 43 us with every payload ablated
    # ~= 11 rounds x ~3.9 us). Counter is MONOTONE — nothing is ever reset — which removes that chain.
    # It was unusable until the `NOC_CMD_VC_LINKED` bug in `mcast_pipe.inl::send_data_` was fixed
    # (the Counter signal is an atomic on a different command buffer and so could never terminate the
    # link, hanging the sender); see the comments there.
    h_data_ready_signal = (
        ttnn._ttnn.mcast_host.McastDataReady.Counter if HSIG == "counter" else ttnn._ttnn.mcast_host.McastDataReady.Flag
    )
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
            data_ready=h_data_ready_signal,
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
    # PERF 14 — the resident x block is the WHOLE K extent, because gate/up runs one K-block
    # (KB1_FRACTION == 1). A knob that shrank ONLY this CB was tried and DELETED: the kernels read
    # the full KR_PAD regardless, so it resized the buffer without resizing the access and was a
    # silent wrong answer waiting to happen. Shrinking this needs the K-blocked x multicast, which
    # Perf 14 measured as unreachable (the mandatory bf16 accumulator alone is 98 KB against
    # 10 560 B free).
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
        _cb(CB_REDUCE_GATE_IN, all_cores, 1 if scatter else reduce_slots * n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_REDUCE_UP_IN, all_cores, 1 if scatter else reduce_slots * n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        # PERF 2 `scatter`: the landing CBs are pushed WHOLE every M-block (so every contributor's
        # own-write-pointer proxy is the CB base on every core, at any m_eff), hence `gather_pages`
        # must hold the LARGEST plan — KGROUPS slots of max(a).
        _cb(CB_GATHER_GATE, all_cores, n_gather, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATHER_UP, all_cores, n_gather, bfp8_tile, ttnn.bfloat8_b),
        # THE SLICE ACCUMULATORS ARE bfloat16, NOT bfp8 — and that is a CORRECTNESS choice, measured.
        # The scatter chains KGROUPS contributors through ONE accumulator, so a value is re-packed
        # KGROUPS times where the tree re-packed it ceil(log2(KGROUPS)) times, and the bfp8 pack's
        # rounding is a BIASED half-LSB (every partial here is positive), so the error is LINEAR in the
        # chain length: `test_emb_contraction`'s max relative error measured 0.0204 on the tree and
        # 0.0580 with bfp8 slice accumulators, against that test's 0.05 gate.
        # DEST is bf16 at fp32_dest_acc_en=False, so packing DEST -> a bf16 CB is EXACT: it removes
        # the per-step bfp8 quantisation and leaves only the FPU's own DEST rounding — HALF the
        # rounding steps. Measured back at 0.0204 — BIT-IDENTICAL to the tree's, i.e. the scatter's
        # longer accumulation chain costs NOTHING once the intermediate format stops quantising.
        # Costs 3 * n_slice * (2048 - 1088) B/core (17 280 B here) against the path's ~100 KB saving.
        # `cb_h_slice` STAYS bfp8: it is unicast into the bfp8 cb_h_local, so the ONE genuine dtype
        # boundary of the epilogue is the SwiGLU multiply's pack, exactly as the tree has it.
        _cb(CB_SLICE_GATE, all_cores, n_slice, bf16_tile, ttnn.bfloat16),
        _cb(CB_SLICE_UP, all_cores, n_slice, bf16_tile, ttnn.bfloat16),
        _cb(CB_H_SLICE, all_cores, n_slice, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_H, all_cores, DEPTH_H * n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_IDX_SCRATCH, all_cores, 1, max(idx_page, dram_align), ttnn.uint32),
        _cb(CB_COUNTS_SCRATCH, all_cores, 1, max(counts_page, dram_align), ttnn.uint32),
        _cb(CB_OUT_TILES, all_cores, DEPTH_OUT * n_out_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_GATE_ACC, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_UP_ACC, all_cores, n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        # `scatter` deletes the send pair outright: a dataflow kernel is a first-class CB consumer, so
        # the gather reads cb_gate_acc / cb_up_acc DIRECTLY and the two full-block copies every
        # non-root core pays today go away with them.
        _cb(CB_GATE_SEND, all_cores, 1 if scatter else n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        _cb(CB_UP_SEND, all_cores, 1 if scatter else n_gu_block, bfp8_tile, ttnn.bfloat8_b),
        # SiLU(sum(gate)) — one SLICE on the scatter path (and bf16, for the reason above: it is the
        # output of the LAST accumulation step), the whole bfp8 block on the tree path.
        _cb(
            CB_GATE_SILU,
            all_cores,
            n_slice if scatter else n_gu_block,
            bf16_tile if scatter else bfp8_tile,
            ttnn.bfloat16 if scatter else ttnn.bfloat8_b,
        ),
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
        # PERF 2 — the scatter path. These sit BEFORE the mcast/accessor blocks because the reader's
        # CT_XMCAST / CT_HMCAST / TA_BASE offsets are derived from the length of this scalar block.
        1 if scatter else 0,
        scatter_noc_split,
        n_gather,
        SEM_HSLICE,
        CB_GATHER_GATE,
        CB_GATHER_UP,
        CB_UP_ACC,
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
        # PERF 2 — the scatter path (before the accessor block, which derives TA_BASE from this
        # block's length).
        1 if scatter else 0,
        scatter_noc_split,
        SEM_HSLICE,
        CB_GATE_ACC,
        CB_UP_ACC,
        CB_GATHER_GATE,
        CB_GATHER_UP,
        CB_H_SLICE,
        CB_H_LOCAL,
        # PERF 3 (HSEND=writer) — the writer derives the h landing slot from this CB's BASE address.
        CB_H,
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
        # PERF 2 — the scatter path.
        1 if scatter else 0,
        KGROUPS,
        n_gather,
        DEST_AUTO_LIMIT_TILES,
        CB_GATHER_GATE,
        CB_GATHER_UP,
        CB_SLICE_GATE,
        CB_SLICE_UP,
        CB_H_SLICE,
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
                # PERF 2 — this core's ROW in its grid column. The scatter's whole plan (which slice
                # I own, where it lands in the root's block, which slot I write in every peer's
                # landing CB) is a pure function of this and the runtime m_eff, so there is no
                # host-side slice table to keep in sync with the kernels.
                y,
            ]
            for c in range(MAX_CHILDREN):
                if c < len(node["children"]):
                    cx, cy = _virt(device, *node["children"][c])
                else:
                    cx, cy = 0, 0
                args.extend([cx, cy])
            # The whole COLUMN, row 0..KGROUPS-1, in virtual coordinates: the scatter's peer list
            # (invite fan-out + gather destinations). Row `r` is at index `r` on every core in the
            # column, which is what makes "worker r owns tiles [r*a, (r+1)*a)" agree grid-wide.
            for r in range(KGROUPS):
                args.extend(_virt(device, x, r))
            args.extend(x_mcast.runtime_args(core))
            args.extend(h_mcast.runtime_args(core))
            reader_rt[x][y] = args

            px, py = _virt(device, *node["parent"]) if node["parent"] is not None else (0, 0)
            # This core's landing slot in its parent's reduce CBs (Refinement 2 lever 1).
            my_slot = _slot_of.get((x, y), 0)
            wargs = [
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
                y,  # PERF 2 — my row in the column (see the reader's note)
                x % KGROUPS,  # PERF 2 — the row of THIS column's reduce root == the h-gather target
            ]
            # PERF 2 — the column peer list, in the SAME row order the reader uses. The scatter's
            # root is column x's reduce root, i.e. row `x % KGROUPS`, so its coordinates are just
            # entry `x % KGROUPS` of this list; no separate root argument.
            for r in range(KGROUPS):
                wargs.extend(_virt(device, x, r))
            # PERF 3 (HSEND=writer) — the h multicast rect, in NOC1 ROUTING ORDER (start = the far
            # corner; the mcast hardware walks from `start` in the NoC's own direction, which is the
            # reverse of NOC0's). The writer sends the whole-grid broadcast, so the rect is the whole
            # worker grid. Appended AFTER the column peer list, at RT_PEERS + 2*KGROUPS.
            far = _virt(device, HGROUPS - 1, KGROUPS - 1)
            near = _virt(device, 0, 0)
            wargs.extend([far[0], far[1], near[0], near[1]])
            wargs.append(x)  # PERF 3 — my column == the h round this core broadcasts
            writer_rt[x][y] = wargs

            compute_rt[x][y] = [
                mailbox_addr,
                kr,
                hn,
                ec,
                1 if node["is_root"] else 0,
                len(node["children"]),
                x,  # my_col — the x-multicast injection slot (round t's injector is column t % HGROUPS)
                y,  # PERF 2 — my row in the column: which slice of the scatter I own
            ]

    # `/perf-measure` collective ablations: one transport stubbed, all CB scaffolding intact.
    # `+`-separated so stages can be peeled off CUMULATIVELY (they overlap, so removing one alone
    # under-counts it — the documented ablation methodology).
    dm_defines = [("ABLATE_" + a.upper(), "1") for a in ABLATE.split("+") if a in _DM_ABLATIONS]
    dm_defines.append(("XSTAGE_FIRST", str(XSTAGE_FIRST)))
    dm_defines.append(("GU_CHUNKS", str(gu_chunks)))
    dm_defines.append(("XPRIO", str(XPRIO)))
    dm_defines.append(("SEM_XSTAGED", str(SEM_XSTAGED)))
    dm_defines.append(("HSEND_WRITER", "1" if HSEND == "writer" else "0"))
    # PERF 4 — the deadlock bound is `blocks_cap - 1` where `blocks_cap = DEPTH_H * M_BLOCK / m_eff`,
    # and `m_eff` is a RUNTIME value, so the real clamp lives in the reader. All the host enforces is
    # the floor and the one thing it alone knows: the shared-VALID-flag `reader` send path cannot
    # tolerate out-of-order senders at all, so it is pinned to the byte-identical 1.
    # HSLOT gives the reader-send path per-slot VALID cells, which is exactly what the shared flag
    # denied it, so ack-ahead becomes legal there too. Without either, the shared cell re-imposes
    # the chain and A is pinned to the byte-identical 1.
    hack_ahead = max(1, HACK_AHEAD) if (HSEND == "writer" or HSLOT) else 1
    dm_defines.append(("HSLOT", "1" if (HSLOT and HSEND != "writer") else "0"))
    dm_defines.append(("WG_TRID", str(int(WG_TRID))))
    dm_defines.append(("WD_LATE", "1" if (WD_LATE and REDUCE == "scatter") else "0"))
    # PERF 12 — the two weight streams' DRAM ND shard widths, in N-axis TILES. 0 == interleaved, which
    # makes `BankRuns::run` return 1 exactly as before, so an interleaved caller gets the byte-identical
    # pre-PERF-12 kernel. Passed as DEFINES, not compile-time args, so the reader's CT_XMCAST /
    # CT_HMCAST / TA_BASE offset arithmetic is untouched.
    dm_defines.append(("WG_SHARD_W", str(wg_shard_w)))
    dm_defines.append(("WD_SHARD_W", str(wd_shard_w)))
    dm_defines.append(("XSTAGE_DIAG", str(int(XSTAGE_DIAG))))
    dm_defines.append(("XSTAGE_STAGGER", str(int(XSTAGE_STAGGER))))
    dm_defines.append(("HACK_AHEAD", str(hack_ahead)))
    dm_defines.append(("SEM_H_RDY_BASE", str(SEM_H_RDY_BASE)))
    dm_defines.append(("SEM_H_FREE", str(SEM_H_FREE)))
    dm_defines.append(("DEPTH_H", str(DEPTH_H)))
    dm_defines.append(("NUM_CORES", str(num_cores)))
    compute_defines = [("GU_CHUNKS", str(gu_chunks)), ("XSTAGE_DIAG", str(int(XSTAGE_DIAG)))]
    if "skip_compute" in ABLATE.split("+"):
        compute_defines.append(("SKIP_COMPUTE", "1"))
    # PERF 7 — the peel had a HOLE. `SKIP_COMPUTE` is a matmul_block_helpers define and elides ONLY
    # the inner `ckernel::matmul_block` LLK call; every `eltwise_chain` in the TU keeps running. So
    # the "all payloads stubbed" floor silently contained the whole combine add chain, the SiLU, the
    # SwiGLU multiply and the fused tilize -- the same trap the reference op hit when its `down` peel
    # bottomed out in a 47 % floor that turned out to hold its DRAM output stream. `skip_eltwise`
    # closes it: `CKL_ELTWISE_CHAIN_SKIP_COMPUTE` keeps every CB reserve/wait/push/pop, DEST sync and
    # trip count and drops only the math.
    if "skip_eltwise" in ABLATE.split("+"):
        compute_defines.append(("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1"))

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
            defines=compute_defines,
        ),
    ]

    semaphores = list(x_mcast.owned_semaphores()) + list(h_mcast.owned_semaphores())
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=all_cores, initial_value=0))
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_DATA, core_ranges=all_cores, initial_value=0))
    # PERF 2 — worker -> root "my h slice landed". MONOTONE like every other semaphore in this op
    # (never reset, always compared with `wait_min` against a running total), which is what keeps it
    # race-free across M-blocks. Allocated on both paths so the two programs differ only in kernels.
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_HSLICE, core_ranges=all_cores, initial_value=0))
    # PERF 3 — reader -> writer on the SAME core. Monotone like the rest (incremented once per
    # M-block, always compared with wait_min against a running total), so it needs no reset and
    # cannot race across M-blocks. Allocated unconditionally so both XPRIO settings share a program
    # shape; at XPRIO=0 nothing reads it.
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_XSTAGED, core_ranges=all_cores, initial_value=0))
    # PERF 3 — the dual-RISC h split's counters. Allocated on BOTH settings of HSEND so the two
    # programs differ only in kernel code; at HSEND=reader nothing touches them. Monotone, zero-init.
    for s in range(DEPTH_H):
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_H_RDY_BASE + s, core_ranges=all_cores, initial_value=0))
    semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_H_FREE, core_ranges=all_cores, initial_value=0))

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
