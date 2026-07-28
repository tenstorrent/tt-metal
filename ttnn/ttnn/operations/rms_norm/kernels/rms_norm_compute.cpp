// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm compute — op_design.md §8.
//
// Per core, per row-block of HT_BLOCK tile-rows:
//
//   pass A (all NW chunks):  [tilize] -> square -> reduce<SUM,REDUCE_ROW,
//                            AccumulateViaAdd> into cb_partials, finalizing the
//                            last chunk with reduce_mean(n_reduced = W)
//   phase 4 (once):          AddUnary(eps) -> Rsqrt  => cb_rms_recip (1/rms)
//   pass B (all NW chunks):  [tilize] -> mul<Col>(x, 1/rms) -> mul<Row>(., gamma)
//                            -> [untilize]
//
// FUSE_SQ (Refinement 5) collapses pass A's two FPU passes into one whenever the
// chunking allows it (NW == 1, no partial-W mask): the FPU's accumulate-into-DEST
// mode turns `square` into `mul_tiles(x, x, acc_to_dest)` over a sticky D0, so
// Sum_w x_w^2 falls out of the SAME pass that computed the squares and no x^2
// block is ever staged through L1. What it publishes is the identical raw
// elementwise accumulator the pairwise-add datapath publishes, so phase 4, the
// finalize and the whole cross-core combine are untouched by it. Measured
// motive: the sharded geometries are MATH-bound (BLOCK_SHARDED (1,1,8192,1024)
// spends 63.0 of 85.2 us on TRISCs alone), so op count is the only lever left.
//
// Perf 2 adds the COLUMN-PACKED gather payload (COLPACK). Where it engages, pass
// A's raw accumulator lands in cb_partials, a fold+column-pack pass turns the
// row-block's `ht` accumulators into ONE tile whose column h holds tile-row h's
// row-sum, and the root's fold becomes CW1 elementwise adds + `ht` column-selects
// instead of `ht * CW1` tile-reduces. Measured 1.507x on the focus profile; see
// the block comment at the pack, and COLPACK_MIN_HT_BLOCK in the descriptor for
// the predicate.
//
// Under the cross-core W-split (W_SPLIT, §4.2) this core owns only a SLICE of W,
// so pass A stops one step earlier: no chunk finalizes, cb_partials keeps the
// RAW elementwise x^2 accumulator, and a copy publishes it for the writer's
// gather leg. The combine then folds the gathered accumulators with the SAME
// reduce — the local chunk-accumulate, done across cores instead of across
// chunks — and n_reduced stays the grand total W. Its shape is the CW1 x CW2
// topology knob (Refinement 3): CW2 == 1 is one flat fold on the root over all
// CW tiles; CW2 > 1 stages it, so each row LEADER folds CW1 tiles raw (again
// never finalizing) and the root finalizes over CW2 row-sums. Phase 4 onwards is
// byte identical on every core; only the producer of cb_rms_sum changes (the
// reader's multicast receive instead of phase 3).
//
// Every loop trip count and every helper block shape is a function of the block
// knobs (HT_BLOCK / WT_CHUNK / NW / CW) — never of a whole-op dimension.
//
// PER-STAGE INSTRUMENTATION (permanent, Perf 1; extended in Perf 2). Every stage
// boundary carries a MaybeDeviceZoneScope: cmp_gamma_tilize / cmp_tilize_a /
// cmp_wait_x / cmp_square / cmp_rowsum / cmp_publish / cmp_colpack /
// cmp_combine / cmp_rsqrt / cmp_scale / cmp_gamma_mul / cmp_tilize_b /
// cmp_untilize (and, on the writer, wtr_selectors). The macro is free when the
// profiler is off (perf_instrumentation.hpp's durability contract) — NEVER
// remove one, and extend the set to any new predicate-guarded path so per-stage
// observability never regresses. The compute kernel is compiled three times, so
// each zone reports separately on UNPACK / MATH / PACK: that split is what tells
// an FPU-op-count problem apart from an unpack/pack-throughput problem.
//
// All compute goes through ttnn/cpp/ttnn/kernel_lib helpers. Phases 2, 5 and 6
// drop from the `square`/`mul` convenience wrappers to `eltwise_chain` directly
// in the resident regimes, because only the chain surface exposes the
// per-operand TileOffset the resident-block fast path needs (the convenience
// wrappers do not forward it). That is a helper *overload* choice, not a raw-LLK
// substitution: the same BinaryFpu + PackTile elements the wrappers emit.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Perf 2 — the column-pack / column-select mechanism is raw compute API
// (reduce_tile with a NON-canonical scaler + an explicit packer-mask clear).
// No kernel_lib reduce helper can express either; see the block comment at the
// fold below for the full measured justification.
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_input_rm = 3;
constexpr uint32_t cb_gamma_rm = 4;
constexpr uint32_t cb_ones = 5;
constexpr uint32_t cb_group_partials = 6;
constexpr uint32_t cb_rms_mean = 7;
constexpr uint32_t cb_partial_out = 8;
constexpr uint32_t cb_group_partials2 = 9;
// Perf 2 — the column-packed gather payload.
constexpr uint32_t cb_packsel = 10;  // scaler h: 1.0 across face-row h -> pack into column h
constexpr uint32_t cb_colsel = 11;   // scaler h: 1/W one-hot at column h -> select column h
constexpr uint32_t cb_rootsum = 12;  // root's elementwise sum of the CW1 packed tiles
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t cb_x_squared = 24;
constexpr uint32_t cb_partials = 25;
constexpr uint32_t cb_rms_sum = 26;
constexpr uint32_t cb_rms_recip = 27;
constexpr uint32_t cb_scaled = 28;
}  // namespace

namespace ckl = compute_kernel_lib;

// ===========================================================================
// Perf 1 — phase 4's SFPU scoped to the lanes phase 5 actually reads.
// ===========================================================================
//
// MEASURED MOTIVE. Phase 4 (`AddUnary(eps)` then `Rsqrt` over the REDUCE_ROW
// statistic) was the op's single largest real compute item: 906 ns per tile,
// 29.0 us per core on the BLOCK_SHARDED (1,1,8192,1024) profile = 38 % of that
// kernel — measured with the cross-core combine ABLATED AWAY, so it is work, not
// waiting. The tournament bench (perf_experiments/rsqrt_lane_and_window)
// established by ablation that the 906 ns is ~100 % per-tile SFPU LANE work and
// ~0 % per-window overhead: blocking the chain is a measured NULL (1.00x), and
// the fitted cost is 23 ns per 32-lane accurate-rsqrt vector against an 81 ns
// copy+pack floor.
//
// THE WASTE. `cb_rms_sum` holds a REDUCE_ROW result, so only COLUMN 0 is
// meaningful (op_design.md §4.1: "1 tile per tile-row, col-0 valid"). Its sole
// consumer is phase 5's `BinaryFpu<..., cb_rms_recip, Mul, BroadcastDim::Col>`,
// i.e. `mul_tiles_bcast<BroadcastType::COL>`, which reproduces column 0 across
// the row and never reads columns 1..31. The stock chain nevertheless computed
// all 32 vector ops per tile per pass, twice (two separate elements = two
// separate SFPU passes). 8 vector ops in ONE pass produce the identical
// column 0: 912.7 -> 258.3 ns/tile, 3.53x, measured at every ht in {1,2,4,8,16}
// (2.95x-3.60x), at every CB format pair and at BOTH DEST modes.
//
// SAFETY PRECONDITION, verified on device rather than argued: the bench fed
// `mul_tiles_bcast<COL>` a tile whose columns 1..31 held poison (7.5e3 / 3e-4)
// and asserted the product over the WHOLE output tile — max rel-err 0.0078,
// pure bf16 rounding. The broadcast reads column 0 only, so leaving columns
// 1..31 of `cb_rms_recip` unwritten provably cannot change the op's output. If a
// future refinement ever gives `cb_rms_recip` a second consumer that reads other
// lanes, this element must revert to a full-tile scope.
//
// RAW-LLK JUSTIFICATION (required so a later helper-usage pass does not "fix"
// this back and undo the win). Two mechanisms are not reachable through the
// stock helpers today:
//   (a) SFPU WORK SCOPE — `rsqrt_tile` hardcodes `VectorMode::RC` and
//       `ITERATIONS = 8`, and `add_unary_tile` likewise; neither the compute-API
//       wrapper nor the `Rsqrt`/`AddUnary` chain elements expose a vector-mode,
//       iteration-count or DEST-address-stride knob.
//   (b) PASS FUSION — `AddUnary` and `Rsqrt` are separate chain elements, hence
//       separate SFPU passes with separate DEST-address setup + STALLWAIT and
//       separate full walks. There is no "unary op with a pre-added scalar"
//       element to compose.
// The BODY is the stock accurate rsqrt kernel verbatim:
// `_calculate_sqrt_body_<APPROX, RECIPROCAL=true, FAST_APPROX=false>` plus the
// `!fp32_dest_acc_en` round-to-nearest store — exactly what
// `calculate_rsqrt<APPROX, 8, DST_ACCUM_MODE, false, false>` runs. Same
// function, same precision, fewer lanes. The precision contract
// (`fp32_dest_acc_en` / `math_fidelity` / `math_approx_mode` / dtypes) is
// untouched; folding `+eps` into the body's argument in fact removes one bf16
// DEST round trip, so the fused result is marginally MORE accurate (measured
// col-0 PCC 0.9999968 vs the stock chain's 0.9999967).
// Everything else stays on the helper surface: this is a `ckl::UnaryOp` CRTP
// element, so `eltwise_chain` still owns the CB lifecycle, the dtype reconfig
// and the dst-sync window — only the SFPU body is ours.
//
// PREDICATE. Compile-time and architectural only: `_calculate_sqrt_body_` and
// the SFPU `Converter` exist on Wormhole B0 and Blackhole but not on Quasar, so
// Quasar keeps the stock two-element chain, byte-identical. There is no
// shape/dtype/layout guard because the enabling condition is a structural
// invariant of the op, not a property of the input.
#if defined(ARCH_QUASAR)
#define RMS_NORM_COL0_RSQRT 0
#else
#define RMS_NORM_COL0_RSQRT 1
#endif

#if RMS_NORM_COL0_RSQRT
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"            // _calculate_sqrt_body_
#include "sfpu/ckernel_sfpu_converter.h"  // Converter::as_float
#endif

namespace {
#ifdef TRISC_MATH
// The SFPU walks a face as [rg0-even, rg0-odd, rg1-even, rg1-odd, ...]; column 0
// lives only in the EVEN-parity vectors, so visit offsets 0,2,4,6 and skip the
// odd ones. Net dst_reg advance is +8 == the stock ITERATIONS=8, so
// `VectorMode::C`'s face-0 -> face-2 stepping composes unchanged and column 0 is
// covered for all 32 rows in 4 vector ops per face instead of 8.
template <int NVEC, int STRIDE>
sfpi_inline void rms_norm_rsqrt_add_col0_body(uint32_t eps_bits, uint32_t scale_bits) {
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
    const sfpi::vFloat scale = ckernel::sfpu::Converter::as_float(scale_bits);
    for (int d = 0; d < NVEC; d++) {
        // `scale` is 1.0f on the per-tile-row path (the mean was already applied
        // by the producing reduce) and 1/N on the PACKED path, where the tile
        // carries a raw cross-core SUM and the mean has to be applied here. It
        // costs one fp32 SFPU multiply on the same vectors, ahead of the +eps —
        // strictly BEFORE any narrowing, so it cannot lose a bit the old path
        // kept.
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(
            sfpi::dst_reg[0] * scale + eps);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}
#endif

/// `rsqrt(x + eps)` in ONE SFPU pass, scoped to the vector ops holding COLUMN 0.
/// For a REDUCE_ROW statistic consumed through `BroadcastDim::Col` only.
template <ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtAddUnaryColZero : ckl::UnaryOp<RsqrtAddUnaryColZero<Slot>, Slot> {
    uint32_t eps_bits;
    constexpr explicit RsqrtAddUnaryColZero(uint32_t e) noexcept : eps_bits(e) {}
    static ALWI void init() { rsqrt_tile_init(); }  // programs the shared sqrt vConst*Prgm constants
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t idst = ckl::to_u32(Slot) + slot_offset;
        const uint32_t eps = eps_bits;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps]() { rms_norm_rsqrt_add_col0_body<4, 2>(eps, 0x3f800000u /*1.0f*/); }, idst, ckernel::VectorMode::C)));
    }
};

/// Perf 2 — `rsqrt(x * (1/N) + eps)` in ONE SFPU pass over a COLUMN-PACKED
/// statistic tile, i.e. one whose column h holds tile-row h's raw cross-core
/// SUM(x^2) (see COLPACK). Same body, same precision; the scope widens from
/// "column 0 only" to "columns 0..15" because that is where the packed columns
/// are, and HT_BLOCK <= 16 is already a hard precondition of the pack.
///
/// MEASURED MOTIVE. Phase 4 was the op's #2 real-compute item after the combine
/// collapsed: 245 ns/tile x `ht` tiles per row-block, against a measured 81
/// ns/tile floor for the per-tile copy+pack scaffolding alone — i.e. most of the
/// cost was paying that per-tile scaffolding `ht` times over for 32 live datums
/// each. One pass over the packed tile pays it ONCE: the isolated bench measured
/// the stage at 8_679 -> 4_497 ns (1.930x) in exactly this `pack_given` form.
///
/// `NVEC = 8, STRIDE = 1` is the stock full-face walk (the shipped ITERATIONS=8),
/// so under `VectorMode::C` this covers all 16 rows x 16 columns of faces 0 and 2
/// — every packed column, for all 32 tile rows. The column-0-only variant above
/// can skip the odd-parity vectors; a packed tile cannot, because its live data
/// is spread across columns.
template <ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtMeanColPacked : ckl::UnaryOp<RsqrtMeanColPacked<Slot>, Slot> {
    uint32_t eps_bits;
    uint32_t inv_n_bits;
    constexpr RsqrtMeanColPacked(uint32_t e, uint32_t inv_n) noexcept : eps_bits(e), inv_n_bits(inv_n) {}
    static ALWI void init() { rsqrt_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t idst = ckl::to_u32(Slot) + slot_offset;
        const uint32_t eps = eps_bits;
        const uint32_t inv_n = inv_n_bits;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps, inv_n]() { rms_norm_rsqrt_add_col0_body<8, 1>(eps, inv_n); }, idst, ckernel::VectorMode::C)));
    }
};
}  // namespace
#endif  // RMS_NORM_COL0_RSQRT

void kernel_main() {
    // ---- regime flags (§5.2) ----
    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(1) != 0;
    constexpr bool IS_RM_GAMMA = get_compile_time_arg_val(2) != 0;
    constexpr bool X_RESIDENT = get_compile_time_arg_val(3) != 0;
    constexpr bool GAMMA_RESIDENT = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(5) != 0;
    // ---- block knobs (§1.2) ----
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t WT_LAST = get_compile_time_arg_val(8);
    constexpr uint32_t NW = get_compile_time_arg_val(9);
    constexpr uint32_t HT_BLOCK = get_compile_time_arg_val(10);
    // W-chunks the reader coalesces per push on the resident TILE path (see
    // rms_norm_program_descriptor._x_read_chunks). The cumulative wait below has
    // to be quantized to this, since that is the granularity data becomes
    // visible at. Always 1 on the RM path (compute's own tilize is the producer).
    constexpr uint32_t X_READ_CHUNKS = IS_RM ? 1u : get_compile_time_arg_val(11);
    // ---- geometry ----
    constexpr uint32_t W_VALID_LAST = get_compile_time_arg_val(12);
    constexpr uint32_t N_REDUCED = get_compile_time_arg_val(13);  // true element count == W
    // ---- cross-core W-split (§4.2) ----
    constexpr bool W_SPLIT = get_compile_time_arg_val(14) != 0;
    constexpr uint32_t CW = get_compile_time_arg_val(15);   // cores per combine group
    constexpr uint32_t CW1 = get_compile_time_arg_val(16);  // stage-1 fan-in (row -> leader)
    constexpr uint32_t CW2 = get_compile_time_arg_val(17);  // stage-2 fan-in (leaders -> root)
    constexpr bool TWO_STAGE = CW2 > 1;
    static_assert(CW1 * CW2 == CW, "combine stages must tile CW");
    // ---- fused square-accumulate (Refinement 5) ----
    // Collapses phases 2 and 3 into one FPU pass: mul_tiles(x, x, acc_to_dest)
    // over a tile-row leaves Sum_w x_w^2 in DEST, which IS the raw elementwise
    // accumulator the finalize (and the cross-core combine) already consume.
    // The host owns the predicate — see FUSE_SQUARE_ACCUM in the program
    // descriptor for the two structural preconditions it enforces (NW == 1, so
    // the accumulator never has to survive a tile_regs_acquire; and no
    // partial-W mask, which lives on the reduce helper's scaler hook).
    constexpr bool FUSE_SQ = get_compile_time_arg_val(18) != 0;
    static_assert(!FUSE_SQ || NW == 1, "fused square-accumulate requires NW == 1");
    static_assert(!FUSE_SQ || !HAS_PARTIAL_W, "fused square-accumulate cannot apply the partial-W mask");
    // ---- Perf 2: the gather PAYLOAD guard (the SAME flag word the reader and
    // the writer read; bit0 = colpack, bit1 = bf16 wire datum). COLPACK collapses
    // the payload from `ht` full tiles to ONE tile whose COLUMN h holds tile-row
    // h's row-sum, which also turns the root's fold from `ht * CW1` tile-reduces
    // into `CW1 + ht`. The host owns the predicate — see COLPACK_MIN_HT_BLOCK.
    constexpr uint32_t PAYLOAD_FLAGS = get_compile_time_arg_val(19);
    constexpr bool COLPACK = W_SPLIT && ((PAYLOAD_FLAGS & 0x1u) != 0);
    static_assert(!COLPACK || NW == 1, "colpack requires NW == 1 (the fold reads one settled tile/row)");
    static_assert(!COLPACK || HT_BLOCK <= 16, "colpack packs into face-rows 0..15");
    static_assert(!COLPACK || !TWO_STAGE, "colpack requires a flat gather (a staged one would pack twice)");

    static_assert(WT_LAST == WT_CHUNK, "compute assumes uniform chunk widths");
    static_assert(NW * WT_CHUNK == WT, "chunking must tile Wt exactly");
    static_assert(!(NW > 1 && HT_BLOCK > 1), "R7: NW > 1 requires HT_BLOCK == 1");
    static_assert(X_READ_CHUNKS >= 1 && NW % X_READ_CHUNKS == 0, "read batch must tile NW");

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);
    // Perf 2: 1/N as float bits, for the COLPACK phase-4 path — the packed
    // broadcast tile carries a raw cross-core SUM, so the mean is applied inside
    // the rsqrt body instead of by the root's (now deleted) column-select.
    [[maybe_unused]] const uint32_t inv_n_bits = get_arg_val<uint32_t>(5);
    const uint32_t is_root = get_arg_val<uint32_t>(2);
    const uint32_t is_last_w_core = get_arg_val<uint32_t>(3);
    const uint32_t is_leader = get_arg_val<uint32_t>(4);

    // Filler core (inside a group's multicast rectangle, owns no data).
    if (num_tile_rows == 0) {
        return;
    }

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    // ---- phase 0a: RM gamma tilized once and held resident -----------------
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT && IS_RM_GAMMA) {
        MaybeDeviceZoneScope("cmp_gamma_tilize");
        for (uint32_t wc = 0; wc < NW; ++wc) {
            ckl::tilize<WT_CHUNK, cb_gamma_rm, cb_gamma>(/*num_blocks=*/1, /*total_input_pages=*/1);
        }
    }
    if constexpr (HAS_GAMMA && GAMMA_RESIDENT) {
        // R8: waited once, held for the whole kernel, never popped.
        cb_wait_front(cb_gamma, WT);
    }

    // Non-tile-aligned W: the 0/1 mask tile the reader filled zeroes the padded
    // lanes of the LAST reduce-dim tile. n_reduced stays the true count (W).
    // Under a W-split only the core whose slice ENDS on the tensor's last
    // W-tile owns that tile, so only it applies the mask.
    const auto partial = (HAS_PARTIAL_W && (!W_SPLIT || is_last_w_core != 0))
                             ? ckl::ReducePartialScaler::partial_mask(W_VALID_LAST, 0)
                             : ckl::ReducePartialScaler::none();

    constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_scaled : cb_output_tiles;
    // Where the fused square-accumulate publishes its raw Sum_w x^2. Under a
    // W-split that tile is the writer's gather payload and goes straight to
    // cb_partial_out (one producer, one consumer, exactly as the NW == 1
    // pairwise path already did); otherwise it stays compute-internal and the
    // fold below reads it back out of cb_partials.
    // Perf 2: COLPACK inserts the fold+column-pack pass between the square and
    // the ship, so the raw accumulator has to land in a compute-internal CB first
    // instead of straight into the writer's gather source.
    constexpr uint32_t cb_accum = (W_SPLIT && !COLPACK) ? cb_partial_out : cb_partials;

    // Tiles per DEST-sync window (Refinement 5). `EltwiseShape`'s block_size
    // defaults to 1, and at 1 the chain runs a WHOLE
    // tile_regs_acquire/commit/wait/release round — plus a pack phase — around
    // every single tile. examples/compute_block_size measures that fixed
    // per-window cost at ~1.6 us per extra pass, 1.65x end to end.
    //
    // This is not a guessed constant: eltwise_chain clamps block_size to the
    // chain's OWN compile-time DEST capacity (chain_max_block_v =
    // DEST_AUTO_LIMIT / lane_width) and to 1 for any chain whose operand
    // policies are per-tile, so asking for the register file's size is exactly
    // "the coarsest block that fits DEST", re-derived per chain and per
    // fp32_dest_acc_en / dst_full_sync_en setting.
    constexpr uint32_t DEST_BLOCK = ckl::DEST_AUTO_LIMIT;
    // A REDUCE_ROW result is column-shaped, so it broadcasts back across
    // columns via BroadcastDim::Col (eltwise_chain.hpp:526-528).
    constexpr auto rms_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Col : ckl::OperandKind::Scalar;
    constexpr auto gamma_kind = (HT_BLOCK > 1) ? ckl::OperandKind::Row : ckl::OperandKind::Block;
    constexpr auto x_life = X_RESIDENT ? ckl::InputLifecycle::CallerManaged : ckl::InputLifecycle::Bulk;

    const uint32_t num_row_blocks = (num_tile_rows + HT_BLOCK - 1) / HT_BLOCK;
    for (uint32_t hb = 0; hb < num_row_blocks; ++hb) {
        uint32_t ht = num_tile_rows - hb * HT_BLOCK;
        if (ht > HT_BLOCK) {
            ht = HT_BLOCK;
        }
        const auto blk = ckl::EltwiseShape::grid(ht, WT_CHUNK, DEST_BLOCK);

        // ================= pass A: mean(x^2) over the whole W ==============
        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (IS_RM) {
                // Resident regime: this fills the strip in place, chunk by chunk,
                // so pass B needs no re-tilize. Streaming: one block at a time.
                MaybeDeviceZoneScope("cmp_tilize_a");
                ckl::tilize<WT_CHUNK, cb_input_rm, cb_input_tiles>(ht, ht * 32u);
            }
            const uint32_t x_base = wc * WT_CHUNK;
            if constexpr (X_RESIDENT) {
                MaybeDeviceZoneScope("cmp_wait_x");
                // R8: CallerManaged — the chain neither waits nor pops; we do
                // both. Waiting CUMULATIVELY (rather than for the whole strip
                // upfront) is what lets the producer stay a batch ahead of
                // compute. Rounded UP to the producer's push granularity:
                // X_READ_CHUNKS == NW collapses to one wait for the full strip.
                // NW > 1 => HT_BLOCK == 1 (R7), so the strip is one flat Wt
                // strip and chunk wc occupies [wc*WT_CHUNK, +WT_CHUNK).
                const uint32_t batches_ready = (wc / X_READ_CHUNKS) + 1u;
                cb_wait_front(cb_input_tiles, batches_ready * X_READ_CHUNKS * ht * WT_CHUNK);
            }

            // ---- phases 2+3 FUSED: DEST-accumulated x^2 --------------------
            //
            // One pass instead of two. `DestAccumulation::Enabled` pins the
            // BinaryFpu to a sticky D0 that eltwise_chain holds across an outer
            // row's whole Wt, so the chunk's W-tiles accumulate in DEST and are
            // packed ONCE per tile-row. What lands is the raw elementwise
            // Sum_w x_w^2 — byte-for-byte the object the pairwise-add datapath
            // publishes with `Accumulate::at` (never `at_last`), so neither the
            // local finalize below nor the cross-core combine can tell the
            // difference. Under a W-split that raw tile is the gather payload
            // and NOTHING else is owed here; otherwise the within-tile fold and
            // the 1/N still have to run, now over `ht` tiles instead of
            // `ht * WT_CHUNK`.
            if constexpr (FUSE_SQ) {
                {
                    MaybeDeviceZoneScope("cmp_square");
                    if constexpr (X_RESIDENT) {
                        ckl::eltwise_chain(
                            blk,
                            ckl::BinaryFpu<
                                cb_input_tiles,
                                cb_input_tiles,
                                ckl::BinaryFpuOp::Mul,
                                ckl::BroadcastDim::None,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::Input,
                                ckl::Dst::D0,
                                ckl::OperandKind::Block,
                                ckl::OperandKind::Block,
                                ckl::TileOffset::Set,
                                ckl::TileOffset::Set,
                                ckl::DestAccumulation::Enabled>{x_base, x_base},
                            ckl::PackTile<cb_accum, ckl::OutputLifecycle::DestAccumulation>{});
                    } else {
                        ckl::eltwise_chain(
                            blk,
                            ckl::BinaryFpu<
                                cb_input_tiles,
                                cb_input_tiles,
                                ckl::BinaryFpuOp::Mul,
                                ckl::BroadcastDim::None,
                                ckl::InputLifecycle::Bulk,
                                ckl::InputLifecycle::Bulk,
                                ckl::BinaryDataFormatReconfig::Input,
                                ckl::Dst::D0,
                                ckl::OperandKind::Block,
                                ckl::OperandKind::Block,
                                ckl::TileOffset::Unset,
                                ckl::TileOffset::Unset,
                                ckl::DestAccumulation::Enabled>{},
                            ckl::PackTile<cb_accum, ckl::OutputLifecycle::DestAccumulation>{});
                    }
                }
                if constexpr (!W_SPLIT) {
                    MaybeDeviceZoneScope("cmp_rowsum");
                    // The fold the accumulate deliberately skipped: one
                    // within-tile REDUCE_ROW per tile-row, then 1/n_reduced.
                    // of(ht, 1, 1) — the W extent is already summed away.
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_partials,
                        cb_scaler,
                        cb_rms_sum,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, 1, 1),
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::ReducePartialScaler::none());
                }
            } else {
                // ---- phase 2: x^2 ----
                {
                    MaybeDeviceZoneScope("cmp_square");
                    if constexpr (X_RESIDENT) {
                        ckl::eltwise_chain(
                            blk,
                            ckl::BinaryFpu<
                                cb_input_tiles,
                                cb_input_tiles,
                                ckl::BinaryFpuOp::Mul,
                                ckl::BroadcastDim::None,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::InputLifecycle::CallerManaged,
                                ckl::BinaryDataFormatReconfig::Input,
                                ckl::Dst::D0,
                                ckl::OperandKind::Block,
                                ckl::OperandKind::Block,
                                ckl::TileOffset::Set,
                                ckl::TileOffset::Set>{x_base, x_base},
                            ckl::PackTile<cb_x_squared, ckl::OutputLifecycle::Chunked>{});
                    } else {
                        ckl::square<
                            cb_input_tiles,
                            cb_x_squared,
                            ckl::InputLifecycle::Bulk,
                            ckl::OutputLifecycle::Chunked,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::PackTileReconfig::Output,
                            ckl::OperandKind::Block>(blk);
                    }
                }

                MaybeDeviceZoneScope("cmp_rowsum");
                // ---- phase 3: chunked SUM -> mean on the finalizing chunk ----
                //
                // Under a W-split NO chunk finalizes: this core owns only a SLICE of
                // W, so both the within-tile fold and the 1/N are premature. Every
                // chunk therefore uses Accumulate::at (never at_last), which leaves
                // cb_partials holding the RAW elementwise-accumulated x^2 tile — the
                // exact object the cross-core combine needs. Shipping the *reduced*
                // tile instead would be wrong: AccumulateViaAdd's finalize writes the
                // row sum into column 0 and leaves the surviving x^2 lanes in columns
                // 1..31, so a second REDUCE_ROW over such tiles double-counts them
                // (measured: mean(x^2) of an all-ones W=64 came out 8.75, not 1.0).
                const auto rshape = ckl::ReduceInputBlockShape::of(ht, WT_CHUNK, 1);
                const bool finalize_here = !W_SPLIT && (wc + 1 == NW);
                if constexpr (NW == 1) {
                    if constexpr (W_SPLIT) {
                        // Single chunk, so the accumulator is written ONCE and never
                        // reloaded: pack it straight into the writer's gather CB and
                        // skip the republishing copy below. cb_partial_out still has
                        // exactly one producer (this) and one consumer (the writer).
                        // Under COLPACK the target is cb_partials instead — the
                        // column-pack pass below reads it and produces the ONE tile
                        // the writer ships (see cb_accum).
                        ckl::reduce<
                            ckernel::PoolType::SUM,
                            ckernel::ReduceDim::REDUCE_ROW,
                            cb_x_squared,
                            cb_scaler,
                            cb_accum,
                            ckl::ReduceInputPolicy::BulkWaitBulkPop,
                            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                            ckl::ReduceAlgorithm::AccumulateViaAdd>(
                            rshape,
                            ckl::ReduceInputMemoryLayout::contiguous(),
                            ckl::Accumulate::at(cb_accum, wc),
                            ckl::NoOp{},
                            partial);
                    } else {
                        ckl::reduce_mean<
                            ckernel::ReduceDim::REDUCE_ROW,
                            cb_x_squared,
                            cb_scaler,
                            cb_rms_sum,
                            ckl::ReduceInputPolicy::BulkWaitBulkPop,
                            ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                            ckl::ReduceAlgorithm::AccumulateViaAdd>(
                            rshape,
                            N_REDUCED,
                            ckl::ReduceInputMemoryLayout::contiguous(),
                            ckl::NoAccumulation{},
                            partial);
                    }
                } else if (finalize_here) {
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_x_squared,
                        cb_scaler,
                        cb_rms_sum,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        rshape,
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at_last(cb_partials, wc),
                        partial);
                } else {
                    // Non-finalizing chunk. The partial-W mask rides the chunk that
                    // owns the tensor's last W-tile, which under a W-split is this
                    // core's last chunk (and only on the last-W core).
                    const bool last_chunk = (wc + 1 == NW);
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_x_squared,
                        cb_scaler,
                        cb_partials,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        rshape,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_partials, wc),
                        ckl::NoOp{},
                        last_chunk ? partial : ckl::ReducePartialScaler::none());
                }
            }
        }

        // Hand the raw accumulator to the writer's gather leg. With NW > 1
        // cb_partials is a compute->compute read-modify-write across the chunk
        // loop, so it cannot ALSO be the writer's CB (single producer / single
        // consumer) and one copy publishes the settled tile per tile-row. With
        // NW == 1 there is no read-modify-write, so the reduce above already
        // packed into cb_partial_out and this whole pass is gone — worth having
        // as its own case because a compute pass costs ~320 ns of fixed
        // overhead (examples/compute_block_size) and this one sits on the
        // combine's serial path, ahead of the gather.
        if constexpr (W_SPLIT && NW > 1) {
            MaybeDeviceZoneScope("cmp_publish");
            ckl::copy<cb_partials, cb_partial_out>(ckl::EltwiseShape::tiles(ht));
        }

        // ===== Perf 2: FOLD + COLUMN-PACK the gather payload ==================
        //
        // The pre-Perf-2 spelling shipped `ht` RAW 4 KB Float32 tiles per
        // row-block, in which all 32 columns still held live x^2 partial sums. The
        // only information in each is its 32 row-sums — 128 B of 4096. This pass
        // folds each tile-row within-tile AND lands the `ht` results in `ht`
        // DISTINCT COLUMNS of ONE tile, so the payload becomes 1 tile instead of
        // `ht` (8x on the focus profile) for `ht` FPU ops and one pack. It also
        // moves the group's fold OFF the root as a side effect: the root then
        // folds CW1 packed tiles + `ht` column-selects (16 ops on the focus
        // profile) instead of `ht * CW1` (64).
        //
        // MECHANISM (raw compute API, verified on device). Blackhole's REDUCE_ROW
        // SUM is an MVMUL with the scaler in SrcA (transposed on unpack) and the
        // data in SrcB, so `dest[i,j] = sum_k data[i,k] * scaler[j,k]`: the
        // scaler's FACE-ROW index j picks the output COLUMN. `cb_packsel[h]` is
        // 1.0 across face-row h, so `ht` reduce_tiles accumulating into ONE dest
        // tile column-pack the row-sums (the bench measured live columns {0,1,3}
        // at 32.0 each from scalers at face-rows 0,1,3).
        //
        // RAW-LLK JUSTIFICATION (required so a later helper-usage pass does not
        // "fix" this back and undo the win). Measured authorization: focus profile
        // 54_270 -> 36_014 ns, 1.507x, leaving only 988 ns above the
        // combine-fully-ablated floor of 35_026. Two mechanisms are unreachable
        // through the stock helpers:
        //   (a) SCALER SHAPE — `ckl::reduce` / `ckl::reduce_mean` take ONE scaler
        //       tile with no per-output-column index, and cannot express a
        //       non-canonical scaler at all. A per-output-column scaler IS the
        //       whole mechanism.
        //   (b) PACKER MASK — `reduce_init` programs a packer EDGE MASK that
        //       writes every datum outside column 0 as zero, which would erase
        //       columns 1..ht-1 of a column-packed tile. `reduce_uninit()` (mask
        //       clear) must therefore be issued between `tile_regs_commit` and the
        //       pack. Measured: with the mask on, only column 0 survived; with it
        //       cleared, all packed columns did. No helper exposes that seam.
        // The precision contract is untouched — this moves the same sums through
        // the same FPU datapath at the same fidelity, in fewer ops.
        if constexpr (COLPACK) {
            MaybeDeviceZoneScope("cmp_colpack");
            cb_wait_front(cb_partials, ht);
            cb_wait_front(cb_packsel, HT_BLOCK);
            cb_reserve_back(cb_partial_out, 1);
            // REDUCE_ROW SUM maps scaler -> SrcA, data -> SrcB.
            reconfig_data_format(cb_packsel, cb_partials);
            pack_reconfig_data_format(cb_partial_out);
            reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_partials, cb_packsel, cb_partial_out);
            tile_regs_acquire();
            for (uint32_t h = 0; h < ht; ++h) {
                reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_partials, cb_packsel, /*itile=*/h, /*itile_scaler=*/h, /*idst=*/0);
            }
            tile_regs_commit();
            reduce_uninit();  // MUST precede the pack — see (b) above.
            tile_regs_wait();
            pack_tile(0, cb_partial_out);
            tile_regs_release();
            cb_push_back(cb_partial_out, 1);
            cb_pop_front(cb_partials, ht);
        }

        // ========== phase 3b: cross-core combine (W-split only) ============
        // The combine folds the raw slice-accumulators the writers gathered into
        // ONE mean(x^2) per tile-row. That fold is EXACTLY the local chunk
        // accumulate, done across cores instead of across chunks: AccumulateViaAdd
        // elementwise-adds the gathered tiles into DEST, folds the result within
        // the tile ONCE, and applies 1/n_reduced with n_reduced = W, the GRAND
        // total (§4.2 "Finalize"). Gathered tiles are laid out h-major
        // (tile h*fan_in + slot), so of(ht, fan_in) reads them contiguously.
        //
        // With CW2 == 1 there is one fold, on the root, over all CW tiles.
        // With CW2 > 1 it is staged: every LEADER first folds its row's CW1
        // tiles WITHOUT finalizing — Accumulate::at (never at_last) keeps the raw
        // elementwise accumulator, the same object a worker's chunk loop
        // produces, so the second fold cannot double-count the surviving x^2
        // lanes — and republishes it through cb_partial_out for its own writer.
        // The root then finalizes over just the CW2 row-sums.
        if constexpr (COLPACK) {
            // ===== Perf 2: the COLUMN-PACKED combine ==========================
            //
            // The gathered payload is CW1 column-packed tiles, one per core, in
            // which column h is that core's row-sum for tile-row h. Two steps,
            // CW1 + ht FPU ops total, against the pre-Perf-2 root's ht * CW1:
            //
            //  (1) ELEMENTWISE sum the CW1 packed tiles. Column h of the sum is
            //      the sum of column h, so this is exactly the raw accumulate the
            //      leader stage already uses (`AccumulateViaAdd` +
            //      `Accumulate::at`, never `at_last` — it must NOT fold within the
            //      tile, which would collapse the ht columns into one and
            //      double-count them. That is the Refinement-2 trap, and here it
            //      would be a silent per-row RESCALE that PCC scores >= 0.9998;
            //      only an ABSOLUTE all-ones check catches it).
            //  (2) COLUMN-SELECT, in PHASE 4 rather than here: `cb_colsel[h]` is a
            //      one-hot 1.0 at reduce-axis position h, so ONE reduce_tile pulls
            //      column h into column 0. The packer edge mask is deliberately
            //      LEFT ON for it — its zeroing of everything outside column 0 is
            //      exactly the shape phase 5 wants — and cleared once afterwards,
            //      before phases 5/6 pack full tiles again.
            //
            // The root stops HERE: it publishes the ONE packed tile as the
            // broadcast payload and does NOT extract. Extraction is deferred to
            // every receiving core, where it fuses with phase 4 (see the
            // RsqrtMeanColPacked element and phase 4 below). That deferral is what
            // makes phase 4 ONE SFPU pass instead of `ht`, and it shrinks the
            // multicast payload from `ht` tiles to 1 at the same time — the root
            // was extracting `ht` tiles only for every core to re-pack them. The
            // 1/N moves with it: cb_colsel carries a plain 1.0 now and the mean is
            // applied inside the rsqrt body, which is where the packed sum first
            // meets an fp32 SFPU datapath.
            if (is_root) {
                MaybeDeviceZoneScope("cmp_combine");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_group_partials,
                    cb_ones,
                    cb_rms_mean,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    ckl::ReduceInputBlockShape::of(1, CW1, 1),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_rms_mean, 0),
                    ckl::NoOp{},
                    ckl::ReducePartialScaler::none());
            }
        } else if constexpr (W_SPLIT) {
            MaybeDeviceZoneScope("cmp_combine");
            if constexpr (TWO_STAGE) {
                if (is_leader) {
                    // One accumulate call, never reloaded -> pack the row sum
                    // straight back into cb_partial_out for this core's own
                    // writer to ship on to the root (same CB the slice partial
                    // rode: one producer, one consumer, two sequential pushes).
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_group_partials,
                        cb_ones,
                        cb_partial_out,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, CW1, 1),
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::Accumulate::at(cb_partial_out, 0),
                        ckl::NoOp{},
                        ckl::ReducePartialScaler::none());
                    if (ht < HT_BLOCK) {
                        cb_pop_front(cb_group_partials, (HT_BLOCK - ht) * CW1);
                    }
                }
                if (is_root) {
                    ckl::reduce_mean<
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_group_partials2,
                        cb_ones,
                        cb_rms_mean,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(ht, CW2, 1),
                        N_REDUCED,
                        ckl::ReduceInputMemoryLayout::contiguous(),
                        ckl::NoAccumulation{},
                        ckl::ReducePartialScaler::none());
                    if (ht < HT_BLOCK) {
                        cb_pop_front(cb_group_partials2, (HT_BLOCK - ht) * CW2);
                    }
                }
            } else if (is_root) {
                ckl::reduce_mean<
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_group_partials,
                    cb_ones,
                    cb_rms_mean,
                    ckl::ReduceInputPolicy::BulkWaitBulkPop,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(
                    ckl::ReduceInputBlockShape::of(ht, CW1, 1),
                    N_REDUCED,
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::NoAccumulation{},
                    ckl::ReducePartialScaler::none());
                // The reader publishes a fixed HT_BLOCK*CW1 block so the gather
                // slots stay at a constant L1 offset; drop the unused tail.
                if (ht < HT_BLOCK) {
                    cb_pop_front(cb_group_partials, (HT_BLOCK - ht) * CW1);
                }
            }
        }

        // ================= phase 4: 1/sqrt(mean + eps) =====================
        // One dst-sync window for both SFPU ops; the FPU consumer in phase 5
        // reads it back from L1 (DEST reuse measures slower for an FPU consumer).
        // Under a W-split cb_rms_sum is produced by the READER (the root's
        // broadcast), not by phase 3 — every core then finalizes identically.
        {
            MaybeDeviceZoneScope("cmp_rsqrt");
#if RMS_NORM_COL0_RSQRT
            if constexpr (COLPACK) {
                // Perf 2 fast path — ONE SFPU pass for the whole row-block.
                //
                // cb_rms_sum holds the broadcast COLUMN-PACKED raw sum: column h is
                // tile-row h's cross-core Sum(x^2). So `rsqrt(x/N + eps)` for all
                // `ht` tile-rows is one pass over one tile (the element folds the
                // 1/N in), and the per-tile-row col-0 tiles phase 5 consumes are
                // then produced by `ht` cheap FPU COLUMN-SELECTS instead of `ht`
                // SFPU passes. Measured on the stage: 8_679 -> 4_497 ns, 1.930x,
                // against a per-tile scaffolding floor of 81 ns/tile that the old
                // spelling paid `ht` times over for 32 live datums each.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(1),
                    ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                    RsqrtMeanColPacked<ckl::Dst::D0>{eps_bits, inv_n_bits},
                    ckl::PackTile<cb_rootsum, ckl::OutputLifecycle::Streaming>{});

                // EXTRACT: cb_colsel[h] is a one-hot 1.0 at reduce-axis position h,
                // so one reduce_tile moves column h of the packed 1/rms into column
                // 0 of its own tile. The packer edge mask stays ON (column-0-only
                // output is exactly what BroadcastDim::Col reads) and is cleared
                // once afterwards, before phases 5/6 pack full tiles again. Same
                // raw-LLK justification as the pack: no reduce helper takes a
                // per-output-column scaler INDEX, and none exposes the mask seam.
                cb_wait_front(cb_rootsum, 1);
                cb_wait_front(cb_colsel, HT_BLOCK);
                cb_reserve_back(cb_rms_recip, ht);
                reconfig_data_format(cb_colsel, cb_rootsum);
                pack_reconfig_data_format(cb_rms_recip);
                reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_rootsum, cb_colsel, cb_rms_recip);
                for (uint32_t h = 0; h < ht; ++h) {
                    tile_regs_acquire();
                    reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                        cb_rootsum, cb_colsel, /*itile=*/0, /*itile_scaler=*/h, /*idst=*/0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_rms_recip);
                    tile_regs_release();
                }
                reduce_uninit();
                cb_push_back(cb_rms_recip, ht);
                cb_pop_front(cb_rootsum, 1);
            } else {
                // Perf 1 fast path: one SFPU pass over the 8 vectors that hold
                // column 0 (see the justification above the element). 3.53x
                // measured.
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ht),
                    ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                    RsqrtAddUnaryColZero<ckl::Dst::D0>{eps_bits},
                    ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
            }
#else
            // Fallback (Quasar): byte-identical to the pre-Perf-1 spelling.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(ht),
                ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                ckl::AddUnary<ckl::Dst::D0>{eps_bits},
                ckl::Rsqrt<>{},
                ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
#endif
        }
        if constexpr (W_SPLIT && !COLPACK) {
            // Same fixed-block contract as the gather: the reader publishes
            // HT_BLOCK pages so the multicast lands at a constant L1 offset.
            // Under COLPACK the payload is exactly ONE page per row-block and the
            // CB is declared with one page, so the offset is constant already and
            // there is no tail to drop.
            if (ht < HT_BLOCK) {
                cb_pop_front(cb_rms_sum, HT_BLOCK - ht);
            }
        }

        // ================= pass B: scale (and gamma), then write ===========
        for (uint32_t wc = 0; wc < NW; ++wc) {
            if constexpr (IS_RM && !X_RESIDENT) {
                MaybeDeviceZoneScope("cmp_tilize_b");
                ckl::tilize<WT_CHUNK, cb_input_rm, cb_input_tiles>(ht, ht * 32u);
            }
            const uint32_t x_base = wc * WT_CHUNK;

            // ---- phase 5: x * (1/rms), broadcast across columns ----
            {
                MaybeDeviceZoneScope("cmp_scale");
                if constexpr (X_RESIDENT) {
                    ckl::eltwise_chain(
                        blk,
                        ckl::BinaryFpu<
                            cb_input_tiles,
                            cb_rms_recip,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Col,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            rms_kind,
                            ckl::TileOffset::Set,
                            ckl::TileOffset::Unset>{x_base, 0},
                        ckl::PackTile<cb_scale_out, ckl::OutputLifecycle::Chunked>{});
                } else {
                    ckl::mul<
                        cb_input_tiles,
                        cb_rms_recip,
                        cb_scale_out,
                        ckl::BroadcastDim::Col,
                        x_life,
                        ckl::InputLifecycle::HeldBulk,
                        ckl::OutputLifecycle::Chunked,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        rms_kind>(blk);
                }
            }

            // ---- phase 6: * gamma, broadcast down the rows ----
            if constexpr (HAS_GAMMA) {
                MaybeDeviceZoneScope("cmp_gamma_mul");
                if constexpr (IS_RM_GAMMA && !GAMMA_RESIDENT) {
                    ckl::tilize<WT_CHUNK, cb_gamma_rm, cb_gamma>(/*num_blocks=*/1, /*total_input_pages=*/1);
                }
                if constexpr (GAMMA_RESIDENT) {
                    ckl::eltwise_chain(
                        blk,
                        ckl::BinaryFpu<
                            cb_scaled,
                            cb_gamma,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Bulk,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Block,
                            gamma_kind,
                            ckl::TileOffset::Unset,
                            ckl::TileOffset::Set>{0, x_base},
                        ckl::PackTile<cb_output_tiles, ckl::OutputLifecycle::Chunked>{});
                } else {
                    ckl::mul<
                        cb_scaled,
                        cb_gamma,
                        cb_output_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::InputLifecycle::Bulk,
                        ckl::InputLifecycle::Bulk,
                        ckl::OutputLifecycle::Chunked,
                        ckl::BinaryDataFormatReconfig::Input,
                        ckl::PackTileReconfig::Output,
                        ckl::OperandKind::Block,
                        gamma_kind>(blk);
                }
            }

            // ---- phase 7: back to row-major sticks ----
            if constexpr (IS_RM) {
                MaybeDeviceZoneScope("cmp_untilize");
                ckl::untilize<WT_CHUNK, cb_output_tiles, cb_output_rm>(ht);
            }
        }

        // ================= phase 8: release the held CBs ===================
        // R2: cb_rms_recip is HeldBulk across all NW chunks of pass B.
        cb_pop_front(cb_rms_recip, ht);
        if constexpr (X_RESIDENT) {
            cb_pop_front(cb_input_tiles, ht * WT);
        }
    }
}
