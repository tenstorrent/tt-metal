# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off (Perf 2): where the group root's running sum LIVES.

THE STAGE, in isolation
-----------------------
`compute_root_sum`.  On a width-/block-sharded rms_norm every core reduces its own width
slice to a partial `sum(x^2)` tile (a REDUCE_ROW result: one live COLUMN per tile-row),
ships it to its group root, and the ROOT sums GROUP_SIZE partials per tile-row into one
accumulator tile.  That group SUM is this bench's entire subject: `rows` output tiles, each
the elementwise sum of GROUP_SIZE fp32 tiles already resident in the root's L1.

Everything else is held trivial per /perf-lab's concept-isolation table: ONE Tensix core,
compute only, no NoC in the fast path (the partials are a resident L1 shard exactly as the
real gather leaves them), and the drain out of the accumulator is byte-identical in every
variant.  The `floor` variant keeps the CB contract and ablates the payload, so the fold
cost is a clean subtraction.

WHY THIS IS A RE-MEASUREMENT, NOT A REPEAT
------------------------------------------
`perf_experiments/root_sum_accumulate/` (Perf 1) measured this same mechanism family, but
its headline speedups were quoted against the PRE-D16 spelling (a copy plus GROUP_SIZE-1
separate in-place `add` calls per row) -- the variant named `rmw` there.  D16 then graduated
and the in-tree fold became the single pack-L1-accumulate chain per row.  Here the BASELINE
is that in-tree spelling (`pack_l1_acc`), and the focus geometry is the CURRENT one
(BLOCK_ROWS = 8, not 10).  Every ratio printed by report.py is vs `pack_l1_acc`.

THE PRECISION CONTRACT IS FIXED FOR EVERY VARIANT
-------------------------------------------------
bf16-derived fp32 partials, math_fidelity=HiFi2, fp32_dest_acc_en=False,
math_approx_mode=False.  DEST is therefore 16-bit in EVERY variant -- which is exactly what
makes "where does the running sum live" a precision question as well as a speed question,
and why every variant reports pcc / rel-RMS against an fp64 reference.

THE VARIANTS (the menu)
-----------------------
`acc` = the running per-tile-row sum.  Where it lives, and how many times it crosses L1, is
the only thing that changes.

  pack_l1_acc          THE BASELINE == the op's current in-tree fold (D16).  ONE chain call
                       per tile-row over tiles(GROUP_SIZE): CopyTile brings each partial
                       into DEST and the PACKER folds it onto the resident fp32 acc
                       (`L1Accumulation::SeedFirst` -> pack_reconfig_l1_acc).  acc is only
                       ever PACKED, never unpacked -- but it crosses L1 GROUP_SIZE times per
                       tile-row and every contributor is rounded into 16-bit DEST first.
  pack_l1_acc_pairs    THE HYBRID the precision question asks for: the FPU adds partials in
                       PAIRS in DEST (srcA = slot c, srcB = slot c + GROUP_SIZE/2 of the same
                       row) and the packer fp32-folds each pair-sum onto the L1 acc.  HALF
                       the packs of the baseline, same fp32-L1 accumulator.  DEST depth 2.
  dest_acc_wide        DEST depth GROUP_SIZE: ONE chain call for the WHOLE row-block,
                       grid(rows, GROUP_SIZE/2) with `DestAccumulation::PerRow`, so the
                       running sum is a STICKY DEST tile for a whole tile-row and acc is
                       packed exactly ONCE per row.  The two operands are the row's two
                       halves, addressed by `TileOffset::Strided` so the walk lands on the
                       gather's D16 row-major page layout.  NEEDS AN EVEN GROUP_SIZE.
                       Requires NO zero seed and NO extra CB: `tile_regs_acquire` hands back
                       a zeroed DEST (the op's own `SQ_FOLD` already depends on this), and
                       both operands come from the gather CB itself.  L1 delta ZERO.
  dest_acc_wide_pad    dest_acc_wide made universal by PADDING the gather to an even slot
                       count (`GP = GROUP_SIZE + GROUP_SIZE % 2`): at an odd GROUP_SIZE the
                       writer lands one extra ZERO page per tile-row, so the pairwise walk
                       is exact with no parity predicate in the compute kernel.  At an even
                       GROUP_SIZE `GP == GROUP_SIZE` and this is dest_acc_wide verbatim.
                       Costs `rows` extra fp32 pages of gather L1 at odd GROUP_SIZE only.
  dest_pairs_tail_raw  NEW THIS ROUND, and the one that needs NOTHING: raw-LLK pairwise
                       accumulation into a sticky DEST for `GROUP_SIZE / 2` pairs, plus -- at
                       an ODD GROUP_SIZE only -- ONE `binary_dest_reuse_tiles` step that
                       folds the leftover slot straight into that same DEST.  Universal at
                       any GROUP_SIZE with ZERO extra L1, ZERO gather-layout change and ZERO
                       descriptor change.  See the kernel-head RAW-LLK NOTE for why the odd
                       tail is inexpressible through eltwise_chain.
  dest_acc_any         the helper-expressible sticky-DEST fold that needs no pairing: ONE
                       chain call over grid(rows, GROUP_SIZE) where each step is
                       `DEST += partial + 0` (operand B pinned on a one-page zero CB).  Works
                       at any GROUP_SIZE but costs GROUP_SIZE FPU ops instead of
                       GROUP_SIZE/2, and needs a new 1-page fp32 CB (a constant 4 kB).
  floor                ABLATION, not an option: the fold's payload is removed and only its CB
                       contract is kept.  Its output is undefined by construction, so it
                       carries no correctness gate -- it prices launch + CB publish + drain.

Layouts: the gather CB's page order is the WRITER's free choice in the real op (it computes
the landing address per (row, slot)), and every variant here consumes the D16 row-major order
the op already lands (`page = r * GP + slot`).  No variant needs the layout changed.

MEASURED (blackhole p150b, 1350 MHz, single core, ONE fresh-cache profiled run per variant,
bf16-derived fp32 partials / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False).
`fold_ns` = the launch's DEVICE KERNEL DURATION minus the `floor` ablation for the same
geometry; `x` is vs `pack_l1_acc`, the CURRENT in-tree fold.  Full table: measurements.txt.

    G,rows   pack_l1_acc  pack_l1_acc_pairs  dest_acc_wide(_pad)  dest_pairs_tail_raw  dest_acc_any
    8,8         3715 ns      2014  1.84x         776  4.79x           804  4.62x       1650  2.25x  <- FOCUS
    32,1        1705         1018  1.67x         325  5.25x           299  5.70x        783  2.18x
    28,1        1524          907  1.68x         297  5.13x           269  5.67x        694  2.20x
    9,1          549      (inexpressible)        136  4.04x            39 14.08x        183  3.00x  <- ODD
    8,1          511          262  1.95x         158  3.23x            50 10.22x        185  2.76x
    4,32        7658         5402  1.42x        1784  4.29x          1821  4.21x       3411  2.25x
    16,8        7073         4107  1.72x        1624  4.36x          1580  4.48x       3364  2.10x
    32,8       13968         8004  1.75x        3326  4.20x          3285  4.25x       6779  2.06x

Every one of the 18 geometries measured is a WIN for every candidate; the smallest is
`dest_acc_wide` 1.48x at (G=4, rows=1) -- the geometry with the least work to move.  At
rows == 1 the fold is only ~30-350 ns against a ~730 ns launch floor, so ratios there are
directional and carry ~+-20 ns of absolute noise (the focus/sweep duplicate launches of the
identical geometry agree to <=1.1% everywhere else).

PRECISION -- the question answered by measurement, not argument.  At
`fp32_dest_acc_en=False` a DEST word is 16-bit, and a sticky-DEST fold accumulates
GROUP_SIZE all-positive addends of sum(x^2) there.  It does NOT cost precision; it BUYS
precision.  Every variant clears the op's soft gates (pcc >= 0.9995, rel-RMS <= 0.04) by
three-to-four orders of margin (worst pcc_out 0.999996), and on the RAW SUM the pairwise
sticky-DEST folds are the MOST accurate of the whole menu:

    G=32, rows=1 raw-sum rel-RMS:  pack_l1_acc (in-tree) 6.58e-3
                                   pack_l1_acc_pairs     4.94e-3
                                   dest_acc_wide(_pad)   2.91e-3   <- 2.3x MORE accurate
                                   dest_pairs_tail_raw   2.91e-3
                                   dest_acc_any          7.37e-3   (linear DEST chain)

The reason is that the in-tree fold's "fp32 L1 accumulator" was never lossless: it rounds
EVERY contributor into 16-bit DEST before the exact fp32 L1 add, so it pays GROUP_SIZE
roundings.  The pairwise DEST walk pays the same per-addend rounding but sums as a partial
pairwise TREE (halving), which shortens the error chain from GROUP_SIZE to
log2(GROUP_SIZE)+1.  `dest_acc_any` is the one that does NOT halve -- a linear
DEST-resident chain -- and it lands right where the in-tree fold does (7.37e-3), which
isolates the tree, not the residency, as the accuracy term.  The HYBRID the precision
question asks for (`pack_l1_acc_pairs`: DEST in pairs, fp32-pack across pairs) sits between
the two on both axes and is not needed -- nothing has to be recovered.

DESCRIPTOR / L1 COST of each option (the focus shape's BLOCK_ROWS is L1-bound, so this is
part of the answer):
  dest_acc_wide         ZERO.  No new CB, no seed page, no gather-layout change, no extra
                        gather pages.  Both operands are the gather CB itself, and
                        `tile_regs_acquire` hands back a ZEROED DEST -- verified here (the
                        variant is correct with no seeding step at all) and independently by
                        Perf 1's `dest_reuse_nozero`, and already depended on by the op's own
                        `SQ_FOLD`.  So the 1-page fp32 `cb_zero_tile` Perf 1 thought this
                        needed is NOT needed: that page is `dest_acc_any`'s cost, and
                        `dest_acc_any` is the SLOWEST candidate on the menu.  Restriction:
                        GROUP_SIZE must be EVEN.
  dest_acc_wide_pad     ZERO at even GROUP_SIZE (it is dest_acc_wide verbatim -- measured
                        within 0.4%).  At ODD GROUP_SIZE: `rows` extra fp32 gather pages
                        (GROUP_SIZE=9 -> +1 page = 4 kB at rows=1, +8 pages = 32 kB at
                        rows=8).  The op's only odd-GROUP_SIZE live profile is
                        `(1,1,32,2304)` WIDTH 9c at rows=1, i.e. 4 kB.
  dest_pairs_tail_raw   ZERO at ANY GROUP_SIZE.  No pad, no new CB, no descriptor change at
                        all.  Costs one extra LLK init per row on odd groups (measured: at
                        (9,8) it is 1222 ns vs the pad's 991 -- still 3.37x over the
                        baseline, so this is an intra-candidate tradeoff, not a regression).
  dest_acc_any          +1 fp32 CB page (4 kB, constant -- does not perturb the BLOCK_ROWS
                        solve), and ~2x the fold time of the pairwise forms.  Dominated.

IS THE PAD's BOOT-ZEROING RACE-FREE?  Yes, and by exactly the argument the writer's existing
`writer_gather_zero` comment already rests on.  That comment rules out zeroing the WHOLE
gather CB because a member's partial can land at any time and the zeroing would wipe a
member that already arrived (measured there as pcc 0.87-0.99).  What it does instead is zero
only the bytes the gather NEVER writes -- faces 1 and 3, unshipped at GATHER_FACES=2.  A PAD
slot is in that same category: the writer's landing address is
`(r * GP + my_slot) * stat_bytes` with `my_slot < GROUP_SIZE <= GP - 1`, so NO member ever
writes a pad page, and zeroing all four of its faces at boot cannot race anything.  The
existing loop already zeroes the pad page's faces 1 and 3; the pad only adds faces 0 and 2
of the pages where `p % GP >= GROUP_SIZE`.
"""

import ttnn

TILE = 32
CB_PART = 0  # fp32 partials, resident L1 shard  == cb_partials_gathered
CB_ACC = 16  # fp32 accumulator CB               == cb_row_stat
CB_OUT = 17  # fp32 drained stat, resident shard == cb_stat_handoff
CB_ZERO = 18  # fp32 one-page zero tile (dest_acc_any's pinned operand B)

# Ring depth of the accumulator, in units of `rows` — mirrors the op's CB_ROW_STAT_DEPTH=2.
ACC_DEPTH = 2

VARIANTS = (
    "pack_l1_acc",  # 0 == the in-tree baseline
    "pack_l1_acc_pairs",  # 1
    "dest_acc_wide",  # 2
    "dest_acc_wide_pad",  # 3
    "dest_pairs_tail_raw",  # 4
    "dest_acc_any",  # 5
    "floor",  # 6
)
_METHOD = {name: i for i, name in enumerate(VARIANTS)}

BASELINE = "pack_l1_acc"

# Variants that need an EVEN group size (they add the row's two halves pairwise and carry
# no tail step).
_NEEDS_EVEN_GROUP = ("pack_l1_acc_pairs", "dest_acc_wide")
# Variants that ask the gather for one EXTRA zero slot per tile-row at odd GROUP_SIZE.
_PADS_TO_EVEN = ("dest_acc_wide_pad",)
# No correctness gate: the payload is ablated on purpose.
_ABLATIONS = ("floor",)


_KERNEL = r"""
// =============================================================================
// rms_norm perf experiment: root_sum_dest_accumulate  (ISOLATED BENCH KERNEL)
// =============================================================================
// Sums GROUP_SIZE resident fp32 partial tiles per tile-row into one fp32 accumulator
// tile, `rows` tile-rows per launch, then drains the accumulator.  METHOD selects the
// accumulation mechanism; every other line is byte-identical across variants, so the
// measured delta is the mechanism.
//
// RAW-LLK NOTE (method 4, dest_pairs_tail_raw).  This bypasses `eltwise_chain` and calls
// add_tiles / binary_dest_reuse_tiles / pack_tile directly.  The bypass buys ONE thing
// that the helper family cannot express: a sticky DEST accumulator whose contributor
// count is ODD.
//   * `DestAccumulation` (the helper's DEST-resident accumulator) is a property of
//     `BinaryFpu` only, and a BinaryFpu step consumes TWO operands per step -- so a
//     helper-expressed sticky-DEST fold walks GROUP_SIZE/2 pairs and structurally cannot
//     reach a leftover slot.  `DestReuseBinary` declares no accumulation at all
//     (eltwise_chain.inl: the `if constexpr (!dest_accumulation)` per-iteration DEST
//     lifecycle wraps EVERY step in acquire/commit/release), so a chain built from it
//     cannot retain DEST between steps either.
//   * The two helper-expressible ways to make an odd group even are therefore (a) pad the
//     gather with a zero slot -- method 3, which costs `rows` fp32 pages and a boot-zeroing
//     rule the writer must own; or (b) pin a zero operand B and walk GROUP_SIZE single
//     steps -- method 5, which costs a new CB and 2x the FPU ops.  The raw tail costs
//     NOTHING but one extra LLK init on odd groups, which is why it is measured here.
//   * `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>(cb, idx, 0)` is the primitive: it
//     moves the live DEST tile into SrcB, unpacks the new operand into SrcA, adds, and
//     writes back to the SAME DEST slot -- the exact one-operand accumulate step the
//     helper has no spelling for.
// =============================================================================
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);      // GROUP_SIZE partials per tile-row
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);   // tile-rows in this row-block
    constexpr uint32_t METHOD = get_compile_time_arg_val(2);
    constexpr uint32_t ITERS = get_compile_time_arg_val(3);  // in-kernel repeats of the stage
    // Pages the gather lands PER TILE-ROW.  == G everywhere except dest_acc_wide_pad, where
    // an odd group is padded with one zero slot so the pairwise walk needs no predicate.
    constexpr uint32_t GP = get_compile_time_arg_val(4);

    constexpr uint32_t cb_part = 0, cb_acc = 16, cb_out = 17, cb_zero = 18;
    constexpr uint32_t HALF = G / 2;
    constexpr uint32_t HALF_P = GP / 2;
    constexpr bool ODD = (G % 2) != 0;

    compute_kernel_hw_startup(cb_part, cb_part, cb_acc);

    // ---- the op's ROOT_FOLD_OUT (Perf 1 / D16), verbatim ----------------------
    // (OneUpfront, OneAtEnd) is the policy pair L1 accumulation requires: the whole call
    // pins ONE output tile.  SeedFirst makes the first pack plain and every later one a
    // pack-add, so there is nothing to zero and no separate seed call.
    constexpr auto FOLD_OUT_L1ACC = ckl::output(
        cb_acc,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::SeedFirst);
    // DEST-resident accumulator: one acquire + one pack per tile-row of the grid.
    constexpr auto FOLD_OUT_DESTACC = ckl::output(
        cb_acc,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::Disabled,
        ckl::DestAccumulation::PerRow);

    // Caller-managed views of the partials CB: the two-operand forms read two distinct
    // tiles of the SAME CB, which no per-tile wait/pop schedule can express.
    constexpr auto PART_SET = ckl::input(
        cb_part,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileOffset::Set);
    constexpr auto PART_STRIDED = ckl::input(
        cb_part,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileOffset::Strided);
    // dest_acc_any's pinned zero operand: one page, read every step, never popped.
    constexpr auto ZERO_PIN = ckl::input(
        cb_zero, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Scalar);

    if constexpr (METHOD == 5) {
        cb_reserve_back(cb_zero, 1);
        cb_push_back(cb_zero, 1);
    }

    for (uint32_t iter = 0; iter < ITERS; ++iter) {
        // (Re-)expose the resident partials shard as this round's gather window.
        cb_reserve_back(cb_part, GP * ROWS);
        cb_push_back(cb_part, GP * ROWS);

        {
            MaybeDeviceZoneScope("fold");
            if constexpr (METHOD == 0) {
                // ---- pack_l1_acc: THE BASELINE, the op's current in-tree fold ------
                for (uint32_t r = 0; r < ROWS; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(G),
                        ckl::CopyTile<ckl::input(cb_part)>{},
                        ckl::PackTile<FOLD_OUT_L1ACC>{});
                }
            } else if constexpr (METHOD == 1) {
                // ---- pack_l1_acc_pairs: FPU pairs in DEST, packer fp32-folds them ---
                static_assert(METHOD != 1 || G % 2 == 0, "pack_l1_acc_pairs needs an even GROUP_SIZE");
                cb_wait_front(cb_part, GP * ROWS);
                for (uint32_t r = 0; r < ROWS; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(HALF),
                        ckl::BinaryFpu<PART_SET, PART_SET, ckl::BinaryFpuOp::Add, ckl::BroadcastDim::None>{
                            r * G, r * G + HALF},
                        ckl::PackTile<FOLD_OUT_L1ACC>{});
                }
                cb_pop_front(cb_part, GP * ROWS);
            } else if constexpr (METHOD == 2 || METHOD == 3) {
                // ---- dest_acc_wide / dest_acc_wide_pad: ONE call, sticky DEST -------
                // Identical mechanism; the ONLY difference is that method 3 walks GP (the
                // PADDED slot count) so an odd group needs no parity predicate here.  With
                // an even group GP == G and the two are the same code.
                static_assert(METHOD != 2 || G % 2 == 0, "dest_acc_wide needs an even GROUP_SIZE");
                static_assert(METHOD != 3 || GP % 2 == 0, "dest_acc_wide_pad: GP must be even");
                cb_wait_front(cb_part, GP * ROWS);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS, HALF_P),
                    ckl::BinaryFpu<
                        PART_STRIDED,
                        PART_STRIDED,
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::None,
                        ckl::Dst::D0,
                        ckl::DestAccumulation::PerRow>{
                        ckl::StridedTileRange{0, GP}, ckl::StridedTileRange{HALF_P, GP}},
                    ckl::PackTile<FOLD_OUT_DESTACC>{});
                cb_pop_front(cb_part, GP * ROWS);
            } else if constexpr (METHOD == 4) {
                // ---- dest_pairs_tail_raw: sticky DEST at ANY GROUP_SIZE, zero L1 ----
                // HALF pairwise accumulating add_tiles into DEST slot 0, then (odd G only)
                // ONE dest-reuse step for the leftover slot.  See the RAW-LLK NOTE above.
                cb_wait_front(cb_part, GP * ROWS);
                // The pair loop's LLK mode is invariant, so its init is hoisted out of the
                // row loop -- EXCEPT on an odd group, where the tail's dest-reuse init
                // clobbers it and it has to be re-emitted per row.
                if constexpr (!ODD) {
                    add_tiles_init(cb_part, cb_part, /*acc_to_dest=*/true);
                }
                for (uint32_t r = 0; r < ROWS; ++r) {
                    const uint32_t base = r * G;
                    cb_reserve_back(cb_acc, 1);
                    tile_regs_acquire();  // hands back a ZEROED DEST -- see the note
                    if constexpr (ODD) {
                        add_tiles_init(cb_part, cb_part, /*acc_to_dest=*/true);
                    }
                    for (uint32_t j = 0; j < HALF; ++j) {
                        add_tiles(cb_part, cb_part, base + j, base + HALF + j, 0);
                    }
                    if constexpr (ODD) {
                        binary_dest_reuse_tiles_init<
                            ckernel::EltwiseBinaryType::ELWADD,
                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_part);
                        binary_dest_reuse_tiles<
                            ckernel::EltwiseBinaryType::ELWADD,
                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_part, base + G - 1, 0);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_acc);
                    tile_regs_release();
                    cb_push_back(cb_acc, 1);
                }
                cb_pop_front(cb_part, GP * ROWS);
            } else if constexpr (METHOD == 5) {
                // ---- dest_acc_any: sticky DEST, no pairing, ANY GROUP_SIZE ---------
                // Each step is DEST += (partial + 0): operand B is pinned on a one-page
                // zero CB, which is what lets a DEST-resident fold walk ONE contributor
                // per step instead of a pair -- so odd GROUP_SIZEs need no tail step.
                cb_wait_front(cb_part, GP * ROWS);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS, G),
                    ckl::BinaryFpu<
                        PART_STRIDED,
                        ZERO_PIN,
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::None,
                        ckl::Dst::D0,
                        ckl::DestAccumulation::PerRow>{ckl::StridedTileRange{0, G}},
                    ckl::PackTile<FOLD_OUT_DESTACC>{});
                cb_pop_front(cb_part, GP * ROWS);
            } else {
                // ---- floor: payload ablated, CB contract preserved -----------------
                cb_wait_front(cb_part, GP * ROWS);
                cb_pop_front(cb_part, GP * ROWS);
                cb_reserve_back(cb_acc, ROWS);
                cb_push_back(cb_acc, ROWS);
            }
        }

        // The drain out of the accumulator — IDENTICAL in every variant.
        {
            MaybeDeviceZoneScope("drain");
            ckl::copy<ckl::input(cb_acc), ckl::output(cb_out)>(ckl::EltwiseShape::tiles(ROWS));
        }

        if (iter + 1 < ITERS) {
            cb_wait_front(cb_out, ROWS);
            cb_pop_front(cb_out, ROWS);
        }
    }
}
"""


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded(h_tiles, w_tiles=1):
    """The whole [h_tiles x w_tiles] tile matrix as one shard on one core (tiles row-major)."""
    return ttnn.create_sharded_memory_config(
        shape=(h_tiles * TILE, w_tiles * TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _cb(index, page_size, num_pages, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=_single_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def perf_case_config():
    """The op's `_perf_case` pinned compute config — FIXED for every variant."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def pages_per_row(variant, group_size):
    """Slots the gather lands per tile-row (padded to even for `dest_acc_wide_pad`)."""
    if variant in _PADS_TO_EVEN:
        return group_size + (group_size % 2)
    return group_size


def l1_pages(group_size, rows):
    """fp32 tile pages this geometry pins in L1 (gather + acc ring + drain + the zero page)."""
    return (group_size + 1) * rows + ACC_DEPTH * rows + rows + 1


def gather_l1_delta_pages(variant, group_size, rows):
    """Extra fp32 gather pages this variant asks the DESCRIPTOR for, vs the in-tree fold."""
    return (pages_per_row(variant, group_size) - group_size) * rows


def extra_cb_pages(variant):
    """Extra whole CBs (constant pages) this variant needs in the descriptor."""
    return 1 if variant == "dest_acc_any" else 0


def is_expressible(variant, group_size, rows):
    if variant in _NEEDS_EVEN_GROUP and group_size % 2 != 0:
        return False, f"{variant} adds the row's two halves pairwise; GROUP_SIZE={group_size} is odd"
    return True, ""


# ---------------------------------------------------------------------------
# host-side reference: realistic partials, and the op-level precision transfer
# ---------------------------------------------------------------------------


def reference_sum(group_size, rows, w_per_core, seed):
    """Realistic partials + the exact group sum.

    Each partial is what a member core's REDUCE_ROW actually produces: `sum(x^2)` over that
    core's `w_per_core` columns of a bf16 activation row, held in COLUMN 0 of an fp32 tile
    (the only lanes the op's BroadcastDim::Col consumer ever reads, and — matching the op's
    GATHER_FACES=2 compact gather — the only lanes the gather even ships).  Returns
    (x_bf16_as_f64 [rows*32, W], partials [group_size, rows, 32], exact_sum [rows, 32]),
    all in float64 so the reference carries no error of its own.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    W = group_size * w_per_core
    x = torch.randn(rows * TILE, W, generator=gen).to(torch.bfloat16).to(torch.float64)
    partials = torch.empty(group_size, rows, TILE, dtype=torch.float64)
    for g in range(group_size):
        s = (x[:, g * w_per_core : (g + 1) * w_per_core] ** 2).sum(dim=1)  # [rows*32]
        partials[g] = s.reshape(rows, TILE)
    return x, partials, partials.sum(dim=0)


def _pages(partials, group_size, rows, gp):
    """Lay the partials out in the gather CB's D16 ROW-MAJOR page order (`r * gp + slot`).

    `gp` slots per tile-row; any slot >= group_size stays ZERO (the pad slot, which is what
    makes the pairwise DEST fold exact at an odd group size).
    """
    import torch

    pages = torch.zeros(gp * rows, TILE, TILE, dtype=torch.float32)
    for g in range(group_size):
        for r in range(rows):
            pages[r * gp + g, :, 0] = partials[g, r].to(torch.float32)
    return pages.reshape(gp * rows * TILE, TILE)


def _pcc(a, b):
    import torch

    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _rel_rms(got, ref):
    return float((got - ref).norm() / ref.norm())


def run_variant(device, variant, group_size, rows, *, w_per_core=128, iters=1, seed=1234, eps=1e-5):
    """Run one variant once on device and return its metrics.

    Perf is MEASURED, never asserted, and the profiler CSV is joined by launch order —
    this function only reports correctness / precision and the launch identity.
    """
    import torch

    if variant not in VARIANTS:
        raise ValueError(f"root_sum_dest_accumulate: variant must be one of {VARIANTS}, got {variant!r}")
    ok, why = is_expressible(variant, group_size, rows)
    if not ok:
        raise ValueError(f"root_sum_dest_accumulate: {why}")

    x, partials, exact = reference_sum(group_size, rows, w_per_core, seed)
    gp = pages_per_row(variant, group_size)
    pages = _pages(partials, group_size, rows, gp)

    part_dev = ttnn.from_torch(
        pages,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(gp * rows),
    )
    out_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows * TILE, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, _sharded(rows)
    )
    zero_dev = ttnn.from_torch(
        torch.zeros(TILE, TILE),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(1),
    )
    ft = ttnn.tile_size(ttnn.float32)
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[group_size, rows, _METHOD[variant], iters, gp],
        config=perf_case_config(),
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[compute],
        semaphores=[],
        cbs=[
            ttnn.cb_descriptor_from_sharded_tensor(CB_PART, part_dev),
            _cb(CB_ACC, ft, ACC_DEPTH * rows, ttnn.float32),
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out_dev),
            # Present in EVERY variant (4 KB) so the CB set is identical across the menu.
            ttnn.cb_descriptor_from_sharded_tensor(CB_ZERO, zero_dev),
        ],
    )
    ttnn.generic_op([part_dev, out_dev, zero_dev], descriptor)
    got = ttnn.to_torch(out_dev).to(torch.float64)  # [rows*32, 32]

    result = {
        "variant": variant,
        "gather_pages_per_row": gp,
        "gather_l1_delta_pages": gather_l1_delta_pages(variant, group_size, rows),
        "extra_cb_pages": extra_cb_pages(variant),
        "group_size": group_size,
        "rows": rows,
        "iters": iters,
        "ablation": variant in _ABLATIONS,
    }
    if variant in _ABLATIONS:
        return result

    dev_sum = torch.stack([got[r * TILE : (r + 1) * TILE, 0] for r in range(rows)])  # [rows, 32]
    result["rel_rms_sum"] = _rel_rms(dev_sum, exact)
    result["max_rel_sum"] = float(((dev_sum - exact).abs() / exact.abs()).max())
    result["pcc_sum"] = _pcc(dev_sum, exact)

    # ---- the op-level precision transfer -----------------------------------
    # The stage's ONLY consumer is rsqrt(sum/W + eps) times x, so propagate the measured
    # sum through the rest of the op EXACTLY and report the op's own gate metrics.  Every
    # other error source is common to all variants and is deliberately absent here.
    W = group_size * w_per_core
    s_dev = dev_sum.reshape(-1)
    s_ref = exact.reshape(-1)
    out_d = x * torch.rsqrt(s_dev / W + eps).unsqueeze(1)
    out_r = x * torch.rsqrt(s_ref / W + eps).unsqueeze(1)
    result["rel_rms_out"] = _rel_rms(out_d, out_r)
    result["pcc_out"] = _pcc(out_d, out_r)
    return result
