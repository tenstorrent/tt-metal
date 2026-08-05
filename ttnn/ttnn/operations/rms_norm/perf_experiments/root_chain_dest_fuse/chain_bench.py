# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: FUSE rms_norm's `compute_root_sum` + `compute_root_finalize` into
ONE DEST window with a DEST-RESIDENT accumulator.

THE STAGE PAIR, in isolation
----------------------------
On a width/block-sharded rms_norm the group ROOT does, per combine round:
    (1) sum GROUP_SIZE gathered fp32 partial `sum(x^2)` tiles per tile-row  -> cb_row_stat
    (2) finalize that row total, `rsqrt(sum*(1/W) + eps)`                   -> cb_stat_handoff
Today those are two separate passes over L1: the fold PACKS the running sum into the
fp32 cb_row_stat (packer L1 accumulation, `L1Accumulation::SeedFirst`), and the finalize
UNPACKS it back into DEST, applies the raw-sfpi column-scoped chain, and packs into
cb_stat_handoff.  This bench prices collapsing them: accumulate the group sum IN DEST,
apply the finalize IN THE SAME DEST WINDOW, and pack ONCE -- deleting cb_row_stat's whole
L1 round trip (one fp32 pack + one fp32 unpack per tile-row) as well as the fold's
per-contributor pack.

Everything else is held trivial per /perf-lab's concept-isolation table:
  * ONE Tensix core, compute only.  No NoC in the fast path -- the gathered partials are
    a resident L1 shard laid out exactly as the real gather leaves them (row-major:
    page = r * GP + g, D16), and the finalized stat is drained into a resident L1 shard.
  * The FINALIZE SPELLING IS HELD CONSTANT.  Every variant runs the op's exact
    `StatFinalize` (Perf 1 / D17 `cskip2`: `*(1/W)` and `+eps` in one raw-sfpi pass, then
    rsqrt, both on the even-parity column stride).  So the measured delta is the FUSION
    and the ACCUMULATION MECHANISM, never the SFPU spelling.
  * The user's precision contract is FIXED for every variant: fp32 partials / fp32 stat
    CBs, bf16 activations, math_fidelity = HiFi2, fp32_dest_acc_en = False,
    math_approx_mode = False.  DEST is therefore 16-bit in EVERY variant, which is what
    makes DEST-resident accumulation a precision question and not only a speed question.

THE MENU
--------
  baseline            THE OP TODAY.  Per tile-row, ONE chain call over tiles(GROUP_SIZE)
                      copies each partial into DEST and the PACKER folds it onto the
                      resident fp32 cb_acc (`L1Accumulation::SeedFirst`); then a SECOND
                      chain over tiles(rows) unpacks cb_acc, runs StatFinalize, and packs
                      cb_out.  GROUP_SIZE packs + 1 unpack + 1 pack per tile-row.
  destacc_split       The accumulate mechanism swapped for a DEST-resident one
                      (`DestAccumulation::PerRow`, pairwise over the row's two halves --
                      i.e. `root_sum_accumulate`'s `dest_acc_wide_pad` winner) but the
                      finalize STILL a separate pass over cb_acc.  This is what
                      graduating the previous round's deferred fold win alone would buy,
                      and it is the reference the fusion's MARGINAL value is priced
                      against.
  pairs_l1acc_split   The PRECISION HEDGE.  FPU adds the row's partials in PAIRS in DEST
                      and the PACKER folds each pair-sum onto the fp32 L1 accumulator, so
                      the running sum never lives in a 16-bit DEST word; finalize stays a
                      separate pass.  Fusing is impossible on this form by construction
                      (the completed sum is in L1, not in DEST), so it is the fallback if
                      DEST accumulation loses precision.
  fused_pairs         THE CANDIDATE.  RAW LLK, one DEST window per tile-row:
                      `add_tiles(..., acc_to_dest=true)` accumulates the row's two halves
                      pairwise into DEST slot 0, StatFinalize runs on that same slot, and
                      ONE `pack_tile` writes cb_out.  cb_acc is never touched.
  fused_pairs_dst2    fused_pairs with TWO tile-rows per DEST window (slots 0 and 1), to
                      price the acquire/commit/wait/release scaffolding itself.
  fused_pairs_stream  fused_pairs that RESERVES AND PUSHES cb_out PER TILE-ROW instead of
                      once for the whole block.  fused_pairs' block-granular publish would
                      make the writer's stat multicast wait for the block's LAST tile-row;
                      this variant keeps the op's current per-tile publish granularity, so
                      the CB-handshake cost of preserving that overlap is priced apart
                      from the fusion.
  fused_reuse         RAW LLK, fusion without the pairwise walk: `copy_tile` seeds DEST,
                      then GROUP_SIZE-1 `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>`
                      steps add each partial straight into that slot, then StatFinalize,
                      then one pack.  Needs NO padded gather slot at odd GROUP_SIZE.
  fused_reuse_nozero  fused_reuse without the copy seed -- GROUP_SIZE dest-reuse adds onto
                      a freshly acquired DEST.  Correct only if `tile_regs_acquire` hands
                      back a ZEROED DEST (which the op's own square fold already relies
                      on); if not, it is simply INCORRECT and reported as such.
  floor               ABLATION, not an option: both payloads removed, every CB
                      reserve/wait/push/pop and trip count kept.  Its output is undefined
                      by construction so it carries no correctness gate -- it prices
                      launch + CB publish so the stage cost is a clean subtraction.

GATHER LAYOUT.  Every variant consumes a tile-row's partials CONTIGUOUSLY (`page =
r * GP + g`), which is what the op's writer already lands (D16).  The pairwise variants
walk GP = GROUP_SIZE + GROUP_SIZE % 2 slots per row, i.e. at ODD GROUP_SIZE the writer
lands one extra ZERO slot so the pair walk is exact with no parity predicate in the
compute kernel (`root_sum_accumulate`'s `dest_acc_wide_pad` rule).  `fused_reuse*` need
no pad at all.

THE FINALIZE'S LANE INVARIANT, and how the fusion keeps it.  `StatFinalize` is raw sfpi
with an even-parity column stride: `<STRIDE=2, ITERS=4>` in BOTH bodies, at
`VectorMode::C`, so `*(1/W) + eps` and `rsqrt` run on EXACTLY the same lane set (columns
0,2,..,14 of faces 0 and 2) and the `+eps` guard covers every lane the rsqrt touches --
an all-zero row can never become `rsqrt(0) = inf`.  The fusion does not touch that: the
DEST accumulation fills the WHOLE tile (the FPU has no lane scope), so the finalize sees
the completed sum on every lane it visits, exactly as it does when it unpacks cb_row_stat.
The unvisited lanes hold the raw finite group sum -- the same defined-but-meaningless
datum the current path leaves there, and pass B's `mul<BroadcastDim::Col>` reads column 0
only (measured, `perf_experiments/root_finalize_scope`).

------------------------------------------------------------------------------
MEASURED  (blackhole p150b, 1350 MHz, single core, ONE fresh-cache profiled run per
variant, ITERS=1 so one launch == one real combine ROUND; bf16-derived fp32 partials /
HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False -- UNCHANGED for every variant).
`stage_ns` = the launch's DEVICE KERNEL DURATION minus the `floor` ablation for the same
geometry, i.e. (root_sum + root_finalize) for ONE round; `x` vs the `baseline` HEAD.
Full tables: measurements_run2.txt / launches.jsonl.
------------------------------------------------------------------------------
FOCUS  (1,1,8192,1024) BLOCK_SHARDED [1024,128] 8x8 -> GROUP_SIZE=8, BLOCK_ROWS=8, 4 rounds

    variant              stage_ns      x   rel-RMS(stat)   pcc_out
    baseline                 5874  1.00x       3.38e-03   0.999998   <- THE OP TODAY
    destacc_split            3048  1.93x       2.42e-03   0.999998   (fold swap only)
    pairs_l1acc_split        4142  1.42x       2.91e-03   0.999998   (precision hedge)
    fused_pairs              2701  2.17x       2.42e-03   0.999998
    fused_pairs_stream       2698  2.18x       2.42e-03   0.999998   <- RECOMMENDED
    fused_pairs_dst2         2674  2.20x       2.42e-03   0.999998   (== fused_pairs)
    fused_reuse              4543  1.29x       2.30e-03   0.999997
    fused_reuse_nozero       4563  1.29x       2.30e-03   0.999997

Sweep, `fused_pairs_stream` vs `baseline` stage_ns (GROUP_SIZE x rows-per-block):

    rows\G       4              8              9             16             28             32
    1      578->162 3.57x  822->218 3.77x  866->230 3.77x 1234->320 3.86x 1800->471 3.82x 2020->529 3.82x
    8     4121->2271 1.81x 5868->2698 2.17x 6204->2842 2.18x 9188->3548 2.59x 14215->4765 2.98x 15953->5246 3.04x
    32   16359->9480 1.73x        --             --             --             --             --

A WIN at every geometry measured, 1.73x-3.87x, no regression anywhere.  ODD GROUP_SIZE
(9) behaves exactly like the even ones -- the padded gather slot carries no cost.

PRECISION: the fused form is MORE accurate than the path it replaces at EVERY geometry
(focus: rel-RMS 2.42e-3 vs 3.38e-3; worst case measured 3.36e-3 vs 5.09e-3 at
GROUP_SIZE=28).  The reason is arithmetic, not luck: the current fold rounds EVERY
contributor into a 16-bit DEST word before its exact fp32 L1 add, whereas the pairwise
DEST fold performs one level of a pairwise summation TREE (GROUP_SIZE/2 pair sums, then
GROUP_SIZE/2 accumulation steps) -- so the "fp32 L1 accumulator is what makes it
accurate" premise does not survive measurement.  `pairs_l1acc_split`, built as the
precision hedge in case DEST accumulation lost accuracy, is BOTH slower (1.42x vs 2.18x)
AND less accurate (2.91e-3 vs 2.42e-3) than the fused form, so the hedge is not needed.
All op-level gates (pcc >= 0.9995, rel-RMS <= 0.04) are met with 3-4 orders of margin by
every variant.

CORRECT AT fp32_dest_acc_en=True TOO (correctness-only slice, never compared to the perf
menu): at (8,8) / (32,1) / (9,8) the fused form is again the more accurate one
(rel-RMS 1.52e-4 vs the baseline's 3.62e-4).

`fused_pairs_dst2` (two tile-rows per DEST window) is a measured NULL -- 2674 vs 2701 ns
is 1.0%, inside the run-to-run band (identical geometries launched twice in the same run
reproduce to <= 0.4%).  The acquire/commit/wait/release scaffolding is NOT what costs;
there is no reason to spend a second DEST slot.

`fused_reuse` / `fused_reuse_nozero` fuse just as well but fold GROUP_SIZE serially
dependent FPU ops instead of GROUP_SIZE/2 pair sums, and land at 1.24x-1.99x.  They need
no padded gather slot, which is their only advantage; `fused_reuse_nozero` also confirms
(again) that `tile_regs_acquire` hands back a ZEROED DEST.

SECONDARY, NOT IN THE ns: the fused form makes `cb_row_stat` COMPLETELY DEAD on the
COMBINE path (post-D18 its only remaining uses are the root fold and the root finalize --
under CROSS_CORE the reduce accumulates in `cb_sum_handoff` and pass B reads
`cb_row_final`).  That frees CB_ROW_STAT_DEPTH(2) x BLOCK_ROWS fp32 pages on EVERY core
of every group -- 64 kB/core on the focus shape -- which loosens the BLOCK_ROWS L1 solve.
This bench allocates cb_row_stat in every variant so the measured ns are not flattered by
the L1 difference.
"""

import ttnn

TILE = 32

CB_PART = 0  # fp32 gathered partials, resident L1 shard == cb_partials_gathered
CB_ACC = 16  # fp32 accumulator CB                       == cb_row_stat
CB_OUT = 17  # fp32 finalized stat, resident shard       == cb_stat_handoff

# Ring depth of the accumulator, in units of `rows` -- mirrors the op's CB_ROW_STAT_DEPTH=2.
ACC_DEPTH = 2

VARIANTS = (
    "baseline",
    "destacc_split",
    "pairs_l1acc_split",
    "fused_pairs",
    "fused_pairs_dst2",
    "fused_reuse",
    "fused_reuse_nozero",
    "fused_pairs_stream",
    "floor",
)
_METHOD = {name: i for i, name in enumerate(VARIANTS)}
BASELINE = "baseline"

# Variants whose walk is PAIRWISE over the row's two halves: they read GP = GROUP_SIZE +
# GROUP_SIZE % 2 slots per tile-row, the odd tail slot being a zero the writer lands.
_PADS_TO_EVEN = (
    "destacc_split",
    "pairs_l1acc_split",
    "fused_pairs",
    "fused_pairs_dst2",
    "fused_pairs_stream",
)
# Variants that FUSE the two stages (no cb_acc traffic at all).
FUSED = ("fused_pairs", "fused_pairs_dst2", "fused_reuse", "fused_reuse_nozero", "fused_pairs_stream")
# No correctness gate: the payload is ablated on purpose.
_ABLATIONS = ("floor",)


_KERNEL = r"""
// =============================================================================
// rms_norm perf experiment: root_chain_dest_fuse  (ISOLATED BENCH KERNEL)
// =============================================================================
// Per tile-row: sum GROUP_SIZE resident fp32 partials, then finalize
// rsqrt(sum*(1/W)+eps).  METHOD selects whether those are two passes over an L1
// accumulator (the op today) or ONE DEST window; every other line is byte-identical
// across variants, so the measured delta is the mechanism.
//
// RAW-LLK NOTE (methods 3..6, the fused forms).  These bypass `eltwise_chain` and call
// add_tiles / binary_dest_reuse_tiles / the finalize / pack_tile directly.  The bypass
// is FORCED, not a style choice.  `eltwise_chain`'s DEST-resident accumulator
// (`DestAccumulation::PerRow`) acquires DEST once per grid ROW and packs once at the end
// of that row -- exactly the window this fusion needs -- but EVERY chain element's
// `elem_apply_compute` is invoked on EVERY inner (Wt) iteration of that row
// (eltwise_chain.inl:3097-3124).  So a `StatFinalize` element sitting after the
// accumulating `BinaryFpu` would run once per CONTRIBUTOR instead of once per completed
// sum -- it would rsqrt a partial sum GROUP_SIZE/2 times.  There is no
// "apply-after-the-accumulation-only" element kind and no per-row tail hook: the chain's
// only per-row tail is the pack itself.  A fused root chain is therefore INEXPRESSIBLE
// in the helper family, and the raw form is measured here against the helper-expressible
// split forms (methods 0..2) that ARE in this bake-off for exactly that comparison.
// =============================================================================
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"              // ckernel::sfpu::_calculate_sqrt_body_
#include "ckernel_sfpu_binop_with_unary.h"  // ckernel::sfpu::Converter::as_float
#endif

namespace ckl = compute_kernel_lib;
using ckernel::VectorMode;

// =============================================================================
// The op's finalize, VERBATIM from rms_norm_compute.cpp (Perf 1 / D17 `cskip2`).  It is
// copied rather than re-derived so this bench's finalize spelling is bit-identical to
// the op's in EVERY variant -- the fusion is the only thing under test.
//
// INVARIANT (carried over): STRIDE/ITERS are <2,4> in BOTH bodies and their product is
// 8.  The rsqrt must never run on a lane the scale body skipped, or an all-zero row
// becomes rsqrt(0) = inf -- the +eps guard only exists on the lanes the scale body
// visited.
// =============================================================================
#ifdef TRISC_MATH
template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_scale_body(uint32_t inv_w_bits, uint32_t eps_bits) {
    const sfpi::vFloat iw = ckernel::sfpu::Converter::as_float(inv_w_bits);
    const sfpi::vFloat ep = ckernel::sfpu::Converter::as_float(eps_bits);
    for (int i = 0; i < ITERS; ++i) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * iw + ep;
        sfpi::dst_reg += STRIDE;
    }
}

template <int STRIDE, int ITERS>
sfpi_inline void rms_stat_rsqrt_body() {
    for (int i = 0; i < ITERS; ++i) {
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}

ALWI void stat_scale_col_skip(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2, 4>, idst, VectorMode::C, inv_w_bits, eps_bits);
}
ALWI void rsqrt_tile_col_skip(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<2, 4>, idst, VectorMode::C);
}
#endif  // TRISC_MATH

template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
ALWI void stat_finalize_payload(uint32_t dst) {
    MATH((stat_scale_col_skip(dst, RMS_INV_W, RMS_EPS)));
    MATH((rsqrt_tile_col_skip(dst)));
}

template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
struct StatFinalize : ckl::UnaryOp<StatFinalize<RMS_INV_W, RMS_EPS>, ckl::Dst::D0> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { stat_finalize_payload<RMS_INV_W, RMS_EPS>(slot_offset); }
};

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);      // GROUP_SIZE partials per tile-row
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);   // tile-rows in this row-block
    constexpr uint32_t METHOD = get_compile_time_arg_val(2);
    constexpr uint32_t ITERS = get_compile_time_arg_val(3);  // in-kernel repeats of the stage pair
    constexpr uint32_t GP = get_compile_time_arg_val(4);     // gather slots landed PER TILE-ROW
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(5);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(6);

    constexpr uint32_t cb_part = 0, cb_acc = 16, cb_out = 17;
    constexpr uint32_t HALF_P = GP / 2;
    constexpr bool FUSED = (METHOD >= 3 && METHOD <= 7);

    compute_kernel_hw_startup(cb_part, cb_part, FUSED ? cb_out : cb_acc);

    // ---- the op's ROOT_FOLD_OUT (Perf 1 / D16), verbatim ----------------------
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
    // Caller-managed strided view of the partials CB: the pairwise forms read two
    // distinct tiles of the SAME CB, which no per-tile wait/pop schedule can express.
    constexpr auto PART_STRIDED = ckl::input(
        cb_part,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileOffset::Strided);
    constexpr auto PART_SET = ckl::input(
        cb_part,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileOffset::Set);

    for (uint32_t iter = 0; iter < ITERS; ++iter) {
        // (Re-)expose the resident partials shard as this round's gather window.
        cb_reserve_back(cb_part, GP * ROWS);
        cb_push_back(cb_part, GP * ROWS);

        if constexpr (METHOD == 0) {
            // ================= baseline: THE OP TODAY, two passes =================
            {
                MaybeDeviceZoneScope("bench_root_sum");
                for (uint32_t r = 0; r < ROWS; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(G),
                        ckl::CopyTile<ckl::input(cb_part)>{},
                        ckl::PackTile<FOLD_OUT_L1ACC>{});
                }
            }
            {
                MaybeDeviceZoneScope("bench_root_finalize");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ROWS),
                    ckl::CopyTile<ckl::input(cb_acc)>{},
                    StatFinalize<INV_W_BITS, EPS_BITS>{},
                    ckl::PackTile<ckl::output(cb_out)>{});
            }
        } else if constexpr (METHOD == 1) {
            // ===== destacc_split: DEST-resident fold, finalize still separate =====
            {
                MaybeDeviceZoneScope("bench_root_sum");
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
            }
            {
                MaybeDeviceZoneScope("bench_root_finalize");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ROWS),
                    ckl::CopyTile<ckl::input(cb_acc)>{},
                    StatFinalize<INV_W_BITS, EPS_BITS>{},
                    ckl::PackTile<ckl::output(cb_out)>{});
            }
        } else if constexpr (METHOD == 2) {
            // ===== pairs_l1acc_split: pair in DEST, ACCUMULATE in fp32 L1 =========
            // The running sum never sits in a 16-bit DEST word, so this is the
            // precision hedge; fusing is impossible on it by construction.
            {
                MaybeDeviceZoneScope("bench_root_sum");
                cb_wait_front(cb_part, GP * ROWS);
                for (uint32_t r = 0; r < ROWS; ++r) {
                    ckl::eltwise_chain(
                        ckl::EltwiseShape::tiles(HALF_P),
                        ckl::BinaryFpu<PART_SET, PART_SET, ckl::BinaryFpuOp::Add, ckl::BroadcastDim::None>{
                            r * GP, r * GP + HALF_P},
                        ckl::PackTile<FOLD_OUT_L1ACC>{});
                }
                cb_pop_front(cb_part, GP * ROWS);
            }
            {
                MaybeDeviceZoneScope("bench_root_finalize");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ROWS),
                    ckl::CopyTile<ckl::input(cb_acc)>{},
                    StatFinalize<INV_W_BITS, EPS_BITS>{},
                    ckl::PackTile<ckl::output(cb_out)>{});
            }
        } else if constexpr (METHOD == 3 || METHOD == 4) {
            // ===== fused_pairs / fused_pairs_dst2: ONE DEST window per tile-row ===
            // acc_to_dest = true makes every add ACCUMULATE into the DEST slot, so the
            // row's whole group sum lands there with ONE acquire and NO pack; the
            // finalize then runs on that same slot and a single pack_tile writes the
            // handoff CB.  cb_acc does not exist on this path.
            static_assert(METHOD < 3 || METHOD > 4 || GP % 2 == 0, "pairwise DEST fold: GP must be even");
            constexpr uint32_t DSLOTS = (METHOD == 4) ? 2 : 1;
            MaybeDeviceZoneScope("bench_root_fused");
            cb_wait_front(cb_part, GP * ROWS);
            cb_reserve_back(cb_out, ROWS);
            add_tiles_init(cb_part, cb_part, /*acc_to_dest=*/true);
            rsqrt_tile_init();
            for (uint32_t r = 0; r < ROWS; r += DSLOTS) {
                const uint32_t n = (r + DSLOTS <= ROWS) ? DSLOTS : (ROWS - r);
                tile_regs_acquire();
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t base = (r + k) * GP;
                    for (uint32_t p = 0; p < HALF_P; ++p) {
                        add_tiles(cb_part, cb_part, base + p, base + HALF_P + p, k);
                    }
                    stat_finalize_payload<INV_W_BITS, EPS_BITS>(k);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t k = 0; k < n; ++k) {
                    pack_tile</*out_of_order_output=*/true>(k, cb_out, r + k);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, ROWS);
            cb_pop_front(cb_part, GP * ROWS);
        } else if constexpr (METHOD == 5 || METHOD == 6) {
            // ===== fused_reuse / fused_reuse_nozero: fusion at ANY GROUP_SIZE =====
            // One contributor per step through binary_dest_reuse (DEST is srcB), so no
            // pairing and no padded gather slot.  METHOD 6 drops the copy seed and
            // relies on tile_regs_acquire handing back a ZEROED DEST.
            constexpr bool SEED = (METHOD == 5);
            MaybeDeviceZoneScope("bench_root_fused");
            cb_wait_front(cb_part, GP * ROWS);
            cb_reserve_back(cb_out, ROWS);
            rsqrt_tile_init();
            if constexpr (!SEED) {
                binary_dest_reuse_tiles_init<
                    ckernel::EltwiseBinaryType::ELWADD,
                    ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_part);
            }
            for (uint32_t r = 0; r < ROWS; ++r) {
                const uint32_t base = r * GP;
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
                stat_finalize_payload<INV_W_BITS, EPS_BITS>(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile</*out_of_order_output=*/true>(0, cb_out, r);
                tile_regs_release();
            }
            cb_push_back(cb_out, ROWS);
            cb_pop_front(cb_part, GP * ROWS);
        } else if constexpr (METHOD == 7) {
            // ===== fused_pairs_stream: fused_pairs, but PUBLISHED PER TILE-ROW =====
            // fused_pairs reserves the whole row-block upfront and pushes at the end,
            // which would make the writer's stat multicast wait for the LAST tile-row of
            // the block instead of starting after the first.  This variant keeps the
            // op's current per-tile publish granularity (the finalize chain's default
            // output policy) so the coordinator can price that overlap separately from
            // the fusion itself.
            static_assert(METHOD != 7 || GP % 2 == 0, "pairwise DEST fold: GP must be even");
            MaybeDeviceZoneScope("bench_root_fused");
            cb_wait_front(cb_part, GP * ROWS);
            add_tiles_init(cb_part, cb_part, /*acc_to_dest=*/true);
            rsqrt_tile_init();
            for (uint32_t r = 0; r < ROWS; ++r) {
                const uint32_t base = r * GP;
                tile_regs_acquire();
                for (uint32_t p = 0; p < HALF_P; ++p) {
                    add_tiles(cb_part, cb_part, base + p, base + HALF_P + p, 0);
                }
                stat_finalize_payload<INV_W_BITS, EPS_BITS>(0);
                tile_regs_commit();
                cb_reserve_back(cb_out, 1);
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }
            cb_pop_front(cb_part, GP * ROWS);
        } else {
            // ===== floor: both payloads ablated, CB contract preserved ============
            MaybeDeviceZoneScope("bench_root_fused");
            cb_wait_front(cb_part, GP * ROWS);
            cb_pop_front(cb_part, GP * ROWS);
            cb_reserve_back(cb_out, ROWS);
            cb_push_back(cb_out, ROWS);
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


def perf_case_config(dest_fp32=False):
    """The op's `_perf_case` pinned compute config -- FIXED for every variant.

    `dest_fp32` is NOT a perf lever and is never varied inside a comparison: it exists
    only so the DOMAIN sweep can prove the fused form is CORRECT at the other
    `fp32_dest_acc_en` the op supports.  The perf menu is measured at
    `fp32_dest_acc_en=False` throughout, which is what the focus case pins.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = bool(dest_fp32)
    cfg.math_approx_mode = False
    return cfg


def pages_per_row(variant, group_size):
    """Slots the gather lands per tile-row (padded to even for the pairwise walks)."""
    if variant in _PADS_TO_EVEN:
        return group_size + (group_size % 2)
    return group_size


def l1_pages(group_size, rows):
    """fp32 tile pages this geometry pins in L1 (the budget the op's BLOCK_ROWS solve respects)."""
    return (group_size + 1) * rows + ACC_DEPTH * rows + rows


def is_expressible(variant, group_size, rows):
    del variant, group_size, rows
    return True, ""


def _f32_bits(x):
    import struct

    return int(struct.unpack("<I", struct.pack("<f", float(x)))[0])


# ---------------------------------------------------------------------------
# host-side reference: realistic partials, exact sum, exact finalized stat
# ---------------------------------------------------------------------------


def reference(group_size, rows, w_per_core, seed, eps):
    """Realistic partials plus the exact group sum and the exact finalized stat.

    Each partial is what a member core's REDUCE_ROW actually produces: `sum(x^2)` over
    that core's `w_per_core` columns of a bf16 activation row, held in COLUMN 0 of an
    fp32 tile.  Everything is float64 so the reference carries no error of its own.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    W = group_size * w_per_core
    x = torch.randn(rows * TILE, W, generator=gen).to(torch.bfloat16).to(torch.float64)
    partials = torch.empty(group_size, rows, TILE, dtype=torch.float64)
    for g in range(group_size):
        s = (x[:, g * w_per_core : (g + 1) * w_per_core] ** 2).sum(dim=1)
        partials[g] = s.reshape(rows, TILE)
    exact_sum = partials.sum(dim=0)
    exact_stat = torch.rsqrt(exact_sum / W + eps)
    return x, partials, exact_sum, exact_stat


def _pages(partials, group_size, rows, gp):
    """Lay the partials out in the gather CB's ROW-MAJOR page order (`page = r*gp + g`).

    Any slot >= group_size stays ZERO -- the pad slot that makes the pairwise DEST fold
    exact at an odd group size.
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


def run_variant(device, variant, group_size, rows, *, w_per_core=128, iters=1, seed=1234, eps=1e-5, dest_fp32=False):
    """Run one variant once on device and return its metrics.

    Perf is MEASURED, never asserted: this function reports correctness / precision and
    the launch identity, and the profiler CSV is joined by launch order in report.py.
    """
    import torch

    if variant not in VARIANTS:
        raise ValueError(f"root_chain_dest_fuse: variant must be one of {VARIANTS}, got {variant!r}")

    W = group_size * w_per_core
    x, partials, exact_sum, exact_stat = reference(group_size, rows, w_per_core, seed, eps)
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
    ft = ttnn.tile_size(ttnn.float32)
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[
            group_size,
            rows,
            _METHOD[variant],
            iters,
            gp,
            _f32_bits(1.0 / float(W)),
            _f32_bits(eps),
        ],
        config=perf_case_config(dest_fp32),
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[compute],
        semaphores=[],
        cbs=[
            ttnn.cb_descriptor_from_sharded_tensor(CB_PART, part_dev),
            # Present in EVERY variant so the CB set (and the L1 footprint) is identical
            # across the menu -- the fused forms simply never touch it.  Its DELETION on
            # the fused path is a real L1 saving and is reported separately, not folded
            # into the measured ns.
            _cb(CB_ACC, ft, ACC_DEPTH * rows, ttnn.float32),
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out_dev),
        ],
    )
    ttnn.generic_op([part_dev, out_dev], descriptor)
    got = ttnn.to_torch(out_dev).to(torch.float64)  # [rows*32, 32]

    result = {
        "variant": variant,
        "gather_pages_per_row": gp,
        "group_size": group_size,
        "rows": rows,
        "iters": iters,
        "w_per_core": w_per_core,
        "dest_fp32": bool(dest_fp32),
        "fused": variant in FUSED,
        "ablation": variant in _ABLATIONS,
    }
    if variant in _ABLATIONS:
        return result

    # StatFinalize is column-scoped: COLUMN 0 is the only lane the op's consumer reads.
    dev_stat = torch.stack([got[r * TILE : (r + 1) * TILE, 0] for r in range(rows)])  # [rows, 32]
    result["rel_rms_stat"] = _rel_rms(dev_stat, exact_stat)
    result["max_rel_stat"] = float(((dev_stat - exact_stat).abs() / exact_stat.abs()).max())
    result["pcc_stat"] = _pcc(dev_stat, exact_stat)

    # ---- the op-level precision transfer -----------------------------------
    # The stage pair's ONLY consumer is pass B's x * (1/rms), so propagate the measured
    # stat through it EXACTLY and report the op's own gate metrics.  Every other error
    # source is common to all variants and is deliberately absent here.
    out_d = x * dev_stat.reshape(-1).unsqueeze(1)
    out_r = x * exact_stat.reshape(-1).unsqueeze(1)
    result["rel_rms_out"] = _rel_rms(out_d, out_r)
    result["pcc_out"] = _pcc(out_d, out_r)
    return result
