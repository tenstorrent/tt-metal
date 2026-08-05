# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for rms_norm's cross-core combine stage `compute_root_sum`.

THE STAGE, in isolation
-----------------------
On a width-sharded / block-sharded rms_norm, every core reduces its own width slice to a
partial `sum(x^2)` tile (a REDUCE_ROW result: one column of `rows` tile-rows), ships it to
its group root, and the ROOT sums GROUP_SIZE partials per tile-row into one accumulator tile
before finalizing (`rsqrt(sum/W + eps)`) and multicasting the stat back. That group SUM is
this bench's entire subject: `rows` output tiles, each the elementwise sum of GROUP_SIZE
fp32 input tiles that are already resident in L1.

Everything else is held trivial per /perf-lab's concept-isolation table:
  * ONE Tensix core, compute only. No NoC in the fast path — the partials are a resident
    L1 shard (exactly as the real gather leaves them), the accumulator is an L1 CB, and the
    result is drained into a resident L1 output shard.
  * The drain (`cb_acc -> cb_out`, the op's `compute_stat_handoff`) is IDENTICAL in every
    variant, and the `floor` variant measures it (plus launch + CB publish) with the fold's
    payload ablated away, so the fold cost is a clean subtraction.
  * The user's precision contract is FIXED for every variant: bf16-derived fp32 partials,
    math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False. DEST is therefore
    16-bit in EVERY variant — which is exactly what makes the accumulation mechanism a
    precision question and not only a speed question.

THE VARIANTS (the menu)
-----------------------
`acc` = the running per-tile-row sum. Where it lives, and how many times it crosses L1, is
the only thing that changes.

  rmw                (the op's HEAD baseline) `ckl::copy` seeds acc from slot 0, then
                     GROUP_SIZE-1 streaming `ckl::add<acc, part, acc>` calls fold the rest.
                     acc is UNPACKED and PACKED every contributor, and the per-tile cb_acc
                     push->wait is what synchronizes PACK->UNPACK. GROUP_SIZE helper calls.
  pack_l1_acc        ONE chain call per tile-row over tiles(GROUP_SIZE): CopyTile brings each
                     partial into DEST and the PACKER folds it onto the resident fp32 acc
                     (`L1Accumulation::SeedFirst` -> pack_reconfig_l1_acc). acc is only ever
                     PACKED, never unpacked.
  pack_l1_acc_hoist  pack_l1_acc with the chain's one-time setup hoisted out of the per-row
                     loop (`SetupOwner::Caller`) — isolates the per-call init/reconfig cost.
  pack_l1_acc_pairs  the FPU adds partials in PAIRS in DEST (srcA = slot c, srcB = slot
                     c+GROUP_SIZE/2 of the same row) and the packer folds each pair-sum onto
                     acc — HALF the packs of pack_l1_acc, same L1-resident fp32 accumulator.
  dest_acc_wide      ONE chain call for the WHOLE row-block: grid(rows, GROUP_SIZE/2) with
                     `DestAccumulation::PerRow`, i.e. the running sum is a STICKY DEST tile
                     for a whole row and acc is packed exactly ONCE per row. The two operands
                     are the row's two halves, addressed by `TileOffset::Strided` so the walk
                     lands on the gather's natural row-major page layout.
  dest_reuse_raw     RAW LLK: `copy_tile` seeds DEST, then GROUP_SIZE-1
                     `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>` steps add each partial
                     straight into that DEST slot (one unpack per contributor, no pair trick),
                     one `pack_tile` per row. Not expressible through eltwise_chain (see the
                     kernel-head note). Works at ANY GROUP_SIZE.
  dest_reuse_nozero  dest_reuse_raw without the copy seed — GROUP_SIZE dest-reuse adds onto a
                     freshly acquired DEST. Tests whether `tile_regs_acquire` hands back a
                     ZEROED DEST (the op's own `DestAccumulation::PerRow` square fold relies
                     on it); if it does not, this variant is simply INCORRECT and is reported
                     as such.
  dest_acc_any       the helper-expressible sticky-DEST fold that does NOT need pairing: ONE
                     chain call over grid(rows, GROUP_SIZE) where each step is
                     `DEST += partial + 0` (operand B pinned on a one-page zero CB), so the
                     accumulator stays in DEST and acc is packed once per row at ANY
                     GROUP_SIZE — odd ones included. Costs GROUP_SIZE FPU ops instead of
                     GROUP_SIZE/2, which is exactly what comparing it to dest_acc_wide prices.
  dest_acc_wide_pad  THE WINNER. dest_acc_wide made universal: when GROUP_SIZE is ODD the
                     gather lands one extra ZERO slot per tile-row, so the pairwise walk is
                     exact at ANY GROUP_SIZE and the compute kernel carries no parity
                     predicate (`GP = GROUP_SIZE + GROUP_SIZE % 2`). At even GROUP_SIZE
                     `GP == GROUP_SIZE` and it is dest_acc_wide verbatim (measured within
                     0.2%, i.e. the same code).
  floor              ABLATION, not an option: the fold's payload is removed and only its CB
                     contract is kept (`cb_pop_front` the partials, publish `rows` acc pages).
                     Its output is undefined by construction, so it carries no correctness
                     gate — it exists to price launch + publish + drain.

Layouts: the gather CB's page order is the WRITER's free choice in the real op (it computes
the landing address per (row, slot)), so each variant is fed the page order it wants:
`rmw` needs slot-major (`page = g*rows + r`, one page per row per call), everything else
needs the row-major order the op's Perf-1 fold already uses (`page = r*GROUP_SIZE + g`).
Both are one address expression in the writer; neither is a cost.

MEASURED (blackhole p150b, 1350 MHz, single core, ONE fresh-cache profiled run per variant,
bf16-derived fp32 partials / HiFi2 / fp32_dest_acc_en=False).  `fold_ns` = the launch's
DEVICE KERNEL DURATION minus the `floor` ablation for the same geometry; `x` vs the `rmw`
HEAD baseline.  Full table + raw per-launch metrics: measurements_run2.txt / results_run2.jsonl.

    G,rows      rmw   pack_l1_acc   pack_l1_acc_pairs   dest_acc_wide_pad
    (8,10)     4694    4627 1.01x        2509 1.87x          997  4.71x   <- focus 1
    (32,1)     4803    1700 2.83x        1020 4.71x          346 13.88x   <- focus 2
    (9,10)     5269    5137 1.03x        (inexpressible)     1266  4.16x   <- ODD group
    (4,32)     7409    7661 0.97x        5400 1.37x         1801  4.11x
    (16,1)     2425     897 2.70x         531 4.57x          142 17.08x

Precision (op-level, i.e. propagated through rsqrt: gates PCC >= 0.9995, rel-RMS <= 0.04):
EVERY variant passes with three-to-four orders of margin (worst pcc_out 0.999996, worst
rel_rms_out 3.7e-3).  On the RAW SUM the ordering is the opposite of the folklore: the
sticky-DEST folds are BIT-IDENTICAL to the `rmw` L1 chain (both round the running sum through
16-bit DEST on every add — the "fp32 L1 accumulator" was never lossless), the PAIRWISE DEST
fold is the MOST accurate of the menu because halving is a partial pairwise-tree summation
(rel-RMS 2.9e-3 at GROUP_SIZE=32 vs 7.4e-3 for `rmw` and 6.6e-3 for the op's current
`pack_l1_acc`), and the op's current fold is the LEAST accurate (it rounds every contributor
into 16-bit DEST before the exact fp32 L1 add).  Deeper DEST accumulation did NOT cost
precision here; GROUP_SIZE=32 (the deepest) is where the winner's margin is largest.
"""

import ttnn

TILE = 32
CB_PART = 0  # fp32 partials, resident L1 shard  == cb_partials_gathered
CB_ACC = 16  # fp32 accumulator CB               == cb_row_stat
CB_OUT = 17  # fp32 drained stat, resident shard == cb_stat_handoff
CB_ZERO = 18  # fp32 one-page zero tile (dest_acc_any's pinned operand B)

# Ring depth of the accumulator, in units of `rows` — mirrors the op's CB_ROW_STAT_DEPTH=2.
# The `rmw` baseline NEEDS >= 2 (it holds `rows` pages while reserving `rows` more).
ACC_DEPTH = 2

VARIANTS = (
    "rmw",
    "pack_l1_acc",
    "pack_l1_acc_hoist",
    "pack_l1_acc_pairs",
    "dest_acc_wide",
    "dest_reuse_raw",
    "dest_reuse_nozero",
    "dest_acc_any",
    "dest_acc_wide_pad",
    "floor",
)
_METHOD = {name: i for i, name in enumerate(VARIANTS)}

# Variants that need an EVEN group size (they add the row's two halves pairwise).
_NEEDS_EVEN_GROUP = ("pack_l1_acc_pairs", "dest_acc_wide")
# `dest_acc_wide_pad` is dest_acc_wide made universal by PADDING the gather to an even slot
# count: at odd GROUP_SIZE the writer lands one extra ZERO page per tile-row, so the pairwise
# DEST fold walks (GROUP_SIZE+1)/2 pairs with no special case in the compute kernel and no
# predicate on parity.  The pad page costs `rows` fp32 pages of L1 and one wasted FPU add.
_PADS_TO_EVEN = ("dest_acc_wide_pad",)
# `rmw` walks one page per row per call, so its pages are slot-major; every other variant
# consumes a row's GROUP_SIZE partials contiguously (row-major), which is what the op's
# writer already lands.
_SLOT_MAJOR = ("rmw",)
# No correctness gate: the payload is ablated on purpose.
_ABLATIONS = ("floor",)


_KERNEL = r"""
// =============================================================================
// rms_norm perf experiment: root_sum_accumulate  (ISOLATED BENCH KERNEL)
// =============================================================================
// Sums GROUP_SIZE resident fp32 partial tiles per tile-row into one fp32 accumulator
// tile, `rows` tile-rows per launch, then drains the accumulator (the op's
// compute_stat_handoff).  METHOD selects the accumulation mechanism; every other line
// is byte-identical across variants, so the measured delta is the mechanism.
//
// RAW-LLK NOTE (methods 5 / 6, dest_reuse_*).  These bypass `eltwise_chain` and call
// copy_tile / binary_dest_reuse_tiles / pack_tile directly.  The bypass is not a style
// choice: keeping ONE DEST slot live across GROUP_SIZE *unary* accumulation steps is
// INEXPRESSIBLE in the helper family.  `DestAccumulation` (the helper's DEST-resident
// accumulator) is a property of `BinaryFpu` only -- `DestReuseBinary` declares no
// accumulation, so a chain built from it gets a tile_regs_acquire/commit/release around
// EVERY step and cannot retain DEST between them (eltwise_chain.inl: the
// `if constexpr (!dest_accumulation)` per-iteration DEST lifecycle).  A helper-expressible
// DEST-resident fold therefore has to go through the pairwise BinaryFpu form, which is
// method 4 (dest_acc_wide) and IS in this bake-off for exactly that comparison.
// =============================================================================
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);       // GROUP_SIZE partials per tile-row
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);    // tile-rows in this row-block
    constexpr uint32_t METHOD = get_compile_time_arg_val(2);
    constexpr uint32_t ITERS = get_compile_time_arg_val(3);   // in-kernel repeats of the stage
    // Pages the gather lands PER TILE-ROW.  == G everywhere except dest_acc_wide_pad, where an
    // odd group is padded with one zero slot so the pairwise walk needs no parity predicate.
    constexpr uint32_t GP = get_compile_time_arg_val(4);

    constexpr uint32_t cb_part = 0, cb_acc = 16, cb_out = 17, cb_zero = 18;
    constexpr uint32_t HALF = G / 2;
    constexpr uint32_t HALF_P = GP / 2;

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
    // Same, with reconfig off: SetupOwner::Caller forbids requested-but-inert reconfig.
    constexpr auto FOLD_OUT_L1ACC_NR = ckl::output(
        cb_acc,
        ckl::ReservePolicy::OneUpfront,
        ckl::PushPolicy::OneAtEnd,
        ckl::DataFormatReconfig::Disabled,
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

    if constexpr (METHOD == 7) {
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
                // ---- rmw: the op's HEAD baseline ----------------------------------
                ckl::copy<ckl::input(cb_part), ckl::output(cb_acc)>(ckl::EltwiseShape::tiles(ROWS));
                for (uint32_t g = 1; g < G; ++g) {
                    ckl::add<ckl::input(cb_acc), ckl::input(cb_part), ckl::output(cb_acc)>(
                        ckl::EltwiseShape::tiles(ROWS));
                }
            } else if constexpr (METHOD == 1) {
                // ---- pack_l1_acc: the op's current in-tree fold -------------------
                for (uint32_t r = 0; r < ROWS; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(G),
                        ckl::CopyTile<ckl::input(cb_part)>{},
                        ckl::PackTile<FOLD_OUT_L1ACC>{});
                }
            } else if constexpr (METHOD == 2) {
                // ---- pack_l1_acc_hoist: same fold, setup hoisted over the rows ----
                copy_tile_init(cb_part);
                for (uint32_t r = 0; r < ROWS; ++r) {
                    ckl::eltwise_chain<ckl::SetupOwner::Caller>(
                        ckl::EltwiseShape::tiles(G),
                        ckl::CopyTile<ckl::input(
                            cb_part,
                            ckl::WaitPolicy::PerTile,
                            ckl::PopPolicy::PerTile,
                            ckl::OperandKind::Scalar,
                            ckl::DataFormatReconfig::Disabled)>{},
                        ckl::PackTile<FOLD_OUT_L1ACC_NR>{});
                }
            } else if constexpr (METHOD == 3) {
                // ---- pack_l1_acc_pairs: FPU pairs in DEST, packer folds onto acc ---
                static_assert(METHOD != 3 || G % 2 == 0, "pack_l1_acc_pairs needs an even GROUP_SIZE");
                cb_wait_front(cb_part, GP * ROWS);
                for (uint32_t r = 0; r < ROWS; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(HALF),
                        ckl::BinaryFpu<PART_SET, PART_SET, ckl::BinaryFpuOp::Add, ckl::BroadcastDim::None>{
                            r * G, r * G + HALF},
                        ckl::PackTile<FOLD_OUT_L1ACC>{});
                }
                cb_pop_front(cb_part, GP * ROWS);
            } else if constexpr (METHOD == 4) {
                // ---- dest_acc_wide: ONE call, sticky DEST per row ------------------
                static_assert(METHOD != 4 || G % 2 == 0, "dest_acc_wide needs an even GROUP_SIZE");
                cb_wait_front(cb_part, GP * ROWS);
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(ROWS, HALF),
                    ckl::BinaryFpu<
                        PART_STRIDED,
                        PART_STRIDED,
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::None,
                        ckl::Dst::D0,
                        ckl::DestAccumulation::PerRow>{
                        ckl::StridedTileRange{0, G}, ckl::StridedTileRange{HALF, G}},
                    ckl::PackTile<FOLD_OUT_DESTACC>{});
                cb_pop_front(cb_part, GP * ROWS);
            } else if constexpr (METHOD == 5 || METHOD == 6) {
                // ---- dest_reuse_raw / dest_reuse_nozero: raw-LLK sticky DEST -------
                constexpr bool SEED = (METHOD == 5);
                cb_wait_front(cb_part, GP * ROWS);
                cb_reserve_back(cb_acc, ROWS);
                if constexpr (!SEED) {
                    binary_dest_reuse_tiles_init<
                        ckernel::EltwiseBinaryType::ELWADD,
                        ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_part);
                }
                for (uint32_t r = 0; r < ROWS; ++r) {
                    const uint32_t base = r * G;
                    tile_regs_acquire();
                    if constexpr (SEED) {
                        copy_tile_init(cb_part);
                        copy_tile(cb_part, base, 0);
                        binary_dest_reuse_tiles_init<
                            ckernel::EltwiseBinaryType::ELWADD,
                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_part);
                    }
                    for (uint32_t g = SEED ? 1 : 0; g < G; ++g) {
                        binary_dest_reuse_tiles<
                            ckernel::EltwiseBinaryType::ELWADD,
                            ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_part, base + g, 0);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile</*out_of_order_output=*/true>(0, cb_acc, r);
                    tile_regs_release();
                }
                cb_push_back(cb_acc, ROWS);
                cb_pop_front(cb_part, GP * ROWS);
            } else if constexpr (METHOD == 7) {
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
            } else if constexpr (METHOD == 8) {
                // ---- dest_acc_wide_pad: dest_acc_wide at ANY GROUP_SIZE -------------
                // Identical mechanism to method 4; the ONLY difference is that the walk is
                // over GP (the PADDED slot count) instead of G, so an odd group needs no
                // parity predicate here -- the writer's extra zero slot makes the pair walk
                // exact.  With an even group GP == G and this is method 4 verbatim.
                static_assert(METHOD != 8 || GP % 2 == 0, "dest_acc_wide_pad: GP must be even");
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
            } else {
                // ---- floor: payload ablated, CB contract preserved -----------------
                cb_wait_front(cb_part, GP * ROWS);
                cb_pop_front(cb_part, GP * ROWS);
                cb_reserve_back(cb_acc, ROWS);
                cb_push_back(cb_acc, ROWS);
            }
        }

        // The op's compute_stat_handoff — IDENTICAL in every variant.
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
    """fp32 tile pages this geometry pins in L1 (the same budget the op's BLOCK_ROWS solve respects)."""
    return (group_size + 1) * rows + ACC_DEPTH * rows + rows + 1


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
    (the only lanes the op's BroadcastDim::Col consumer ever reads).  Returns
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


def _pages(partials, group_size, rows, slot_major, gp):
    """Lay the partials out in the gather CB's page order for this variant.

    `gp` slots per tile-row; any slot >= group_size stays ZERO (the pad slot, which is what
    makes the pairwise DEST fold exact at an odd group size).
    """
    import torch

    pages = torch.zeros(gp * rows, TILE, TILE, dtype=torch.float32)
    for g in range(group_size):
        for r in range(rows):
            p = (g * rows + r) if slot_major else (r * gp + g)
            pages[p, :, 0] = partials[g, r].to(torch.float32)
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
        raise ValueError(f"root_sum_accumulate: variant must be one of {VARIANTS}, got {variant!r}")
    ok, why = is_expressible(variant, group_size, rows)
    if not ok:
        raise ValueError(f"root_sum_accumulate: {why}")

    x, partials, exact = reference_sum(group_size, rows, w_per_core, seed)
    gp = pages_per_row(variant, group_size)
    pages = _pages(partials, group_size, rows, slot_major=variant in _SLOT_MAJOR, gp=gp)

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
