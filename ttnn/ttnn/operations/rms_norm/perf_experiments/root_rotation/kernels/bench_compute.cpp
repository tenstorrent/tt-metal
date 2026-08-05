// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for rms_norm (UNPACK / MATH / PACK).
//
//   out = x * rsqrt( (1/W) * sum_w x^2 + eps ) * gamma
//
// One loop nest covers ALL THREE regimes (op_design.md section 7).  Per row-block:
//
//   pass A   [RM] tilize            cb_input_sticks -> cb_input_tiles
//            square                 cb_input_tiles  -> cb_x_squared
//            accumulate_reduce_block cb_x_squared   -> cb_row_stat   (sum x^2)
//   finalize transform_in_place     cb_row_stat     -> cb_row_stat   (1/rms)
//   pass B   mul<Col>               cb_input_tiles, cb_row_stat -> NORM_OUT
//            mul<Row>               cb_normalized, cb_gamma_tiles -> cb_output_tiles
//            [RM] untilize          cb_output_tiles -> cb_output_sticks
//
// The regimes differ ONLY in whether cb_input_tiles / cb_gamma_tiles are held
// across both passes, and how wide they are (X_RESIDENT / X_HOLD_WT, from the
// descriptor -- deviation D14):
//   RESIDENT      X_RESIDENT, NUM_W_CHUNKS == 1.  The whole row is one chunk and
//                 is held, so x is read from DRAM once.
//   ROW_RESIDENT  X_RESIDENT, NUM_W_CHUNKS >  1.  One whole tile-row of x and the
//                 whole row of gamma are held while the DERIVED CBs stay chunked,
//                 so x is STILL read once: each helper call indexes the held CBs
//                 at a TILE OFFSET (TileOffset::Set, base = c * WT_CHUNK) and they
//                 are popped once per row-block rather than per chunk.
//   STREAM        !X_RESIDENT.  Each pass pops its chunk and the reader re-reads x
//                 (and gamma) for pass B -- the L1 fallback, ~2x the DRAM bytes.
//
// Every phase is a kernel_lib helper.  The only raw LLK is inside the
// transform_in_place lambda (x1/W, +eps, rsqrt) — that helper's documented
// calling convention, and the family explicitly routes multi-instruction
// finalizers like rsqrt-with-eps here rather than to a chain
// (streaming_reduce_helpers.hpp:75-78).  Refinement 4 adds ONE raw-LLK function
// inside that same lambda, `rsqrt_tile_col`; its justification is at its
// definition below (the rsqrt API exposes no VectorMode seam).
//
// Pass A's square and pass B's two multiplies are spelled as `eltwise_chain`
// rather than the `square` / `mul` convenience one-liners for one reason: the
// convenience wrappers take no element constructor arguments, and the
// ROW_RESIDENT regime needs to pass a runtime TILE OFFSET to the chain element.
// Each is exactly what the corresponding convenience call expands to.
//
// Explicit cb_pop_front calls on cb_input_tiles / cb_row_stat / cb_gamma_tiles /
// cb_scaler are the sanctioned pattern for operands whose lifetime spans more
// calls than any single PopPolicy can express (op_design.md section 6.1).

#include <cstdint>

// ---- TEMPORARY ABLATION SWITCHES (/perf-measure cumulative peel) ------------
// Uncomment to strip a stage's PAYLOAD while keeping every CB handshake and trip
// count.  Perf measurement only -- the op is wrong with any of these on.  These
// stay commented in the committed tree.
// #define RMS_ABLATE_ROOT_SUM
// #define RMS_ABLATE_ROOT_FINALIZE

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
// add_tiles / add_tiles_init with acc_to_dest -- the root's fused pairwise DEST fold
// (Perf 2 / D22; justification at the fused chain in the COMBINE branch).
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
// PERMANENT per-stage device-profiler instrumentation (never remove; free when
// the profiler is off -- see the header's durability contract).
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

// Lamp L6b+ (Perf 1, descriptor D17): the WHOLE finalize chain scoped to the lanes
// pass B actually reads, in TWO passes over DEST instead of three.
//
// RAW-LLK SUBSTITUTION -- one comment, four functions, one reason.  The finalize's
// SFPU ops each hard-code `VectorMode::RC` and expose NO VectorMode seam -- neither a
// template parameter nor a runtime argument:
//     mul_unary_tile / add_unary_tile   api/compute/eltwise_unary/binop_with_scalar.h
//     rsqrt_tile                        api/compute/eltwise_unary/rsqrt.h:38
// and the SFPU walks a face as [rg0-even, rg0-odd, rg1-even, ...], so COLUMN PARITY is
// the INNER walk axis -- unreachable through `ITERATIONS`, which truncates contiguously.
// cb_row_stat is a REDUCE_ROW result whose ONLY consumer is pass B's
// mul<BroadcastDim::Col>, i.e. COLUMN 0.  Column 0 lives in faces 0 and 2
// (== VectorMode::C, llk_math_eltwise_sfpu_common.h) and is EVEN, so an even-parity
// walk over DEST offsets 0,2,4,6 reaches it with 4 vector ops per face instead of 8.
// The NET dst_reg advance is +8 == the stock ITERATIONS=8, so VectorMode::C's
// face-0 -> face-2 stepping (_llk_math_eltwise_sfpu_apply_vector_mode_) composes
// unchanged.
//
// `rms_stat_scale_body` additionally folds *(1/W) and +eps into ONE pass over DEST.  At
// fp32_dest_acc_en == false a DEST word is 16 bit, so the stock 3-call chain rounded the
// `*(1/W)` result to bf16 on its way through DEST; keeping it in an fp32 LREG removes
// that rounding.  Accuracy is therefore >= the chain this replaces, not a trade -- the
// user's precision contract (math_fidelity / fp32_dest_acc_en / math_approx_mode /
// dtypes) is untouched, and these use the same APPROX / DST_ACCUM_MODE / DST_SYNC_MODE
// macros the stock calls use.
//
// MEASURED AUTHORISATION (blackhole p150b, 1350 MHz, at the op's pinned config --
// bf16 / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False, UNCHANGED; isolated
// bench perf_experiments/root_finalize_scope, copy+pack+inits outside the timed zone for
// the isolated column and inside it for the stage column):
//     isolated MATH-thread ns per finalize call   600.7 (RC mul+add, C rsqrt) -> 244.5
//     stage ns/tile, copy+pack+CB handshake incl. 762.1                       -> 372.8
//   i.e. 2.04x on the finalize stage; rsqrt costs ~23.1 ns per 32-lane vector op and
//   mul_unary/add_unary ~3.6 ns each, so 38% of the previous stage was scaling the 32
//   vectors nobody reads.  Do NOT "restore" this to helper calls without re-measuring.
//
// SAFETY IS MEASURED, NOT ASSUMED.  An isolated bench ran pass B's exact consumer
// (BinaryFpu<x, stat, Mul, BroadcastDim::Col>, OperandKind::Col) on a stat tile whose
// columns 1..31 were seeded five orders of magnitude wrong, and got pcc 0.999992 /
// rel-RMS 0.00403: the column broadcast reads COLUMN 0 ONLY.  The skipped lanes hold the
// raw, finite reduce result -- the same kind of defined-but-meaningless datum the
// gather's faces 1/3 already carry -- and nothing zeroes them, so this is not
// Refinement 4's zeroing race.  If a future consumer ever reads the stat tile whole (a
// debug dump, stat-as-output, an SFPU or reduce pass over it), the scope must widen.
//
// INVARIANT: STRIDE/ITERS must be IDENTICAL in the two bodies (both <2,4>), and their
// product must stay 8.  The rsqrt must never run on a lane the scale body skipped, or an
// all-zero row would be rsqrt(0) = inf -- the +eps guard only exists on the lanes the
// scale body visited.
//
// Precedent for the substitution: sdpa/.../compute_common.hpp:251-256
// `recip_tile_first_column`.  It stays inside the finalize, which is this kernel's one
// sanctioned raw-LLK site.
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"              // ckernel::sfpu::_calculate_sqrt_body_
#include "ckernel_sfpu_binop_with_unary.h"  // ckernel::sfpu::Converter::as_float

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

// *(1/W) and +eps in ONE pass, 4 even-parity vectors per face -> columns 0,2,..,14.
ALWI void stat_scale_col_skip(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2, 4>, idst, VectorMode::C, inv_w_bits, eps_bits);
}
// rsqrt over exactly the same lane set.
ALWI void rsqrt_tile_col_skip(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<2, 4>, idst, VectorMode::C);
}
#endif  // TRISC_MATH

// The SFPU payload only, no init -- so the eltwise_chain element can hoist the init out
// of the per-tile loop while the transform_in_place lambda keeps it inside.
template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
ALWI void stat_finalize_payload(uint32_t dst) {
    MATH((stat_scale_col_skip(dst, RMS_INV_W, RMS_EPS)));
    MATH((rsqrt_tile_col_skip(dst)));
}

// User-defined eltwise_chain element on the documented UnaryOp<Derived, Slot> CRTP
// surface (eltwise_chain.inl:644-660).  It exists because NO stock element exposes a
// VectorMode seam, so the element has to carry the scoped chain itself rather than
// composing MulUnary + AddUnary + Rsqrt.  `init()` runs ONCE per chain call -- that init
// hoist is worth a measured 25 ns/tile over transform_in_place, which re-emits it per
// tile.
template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
struct StatFinalize : compute_kernel_lib::UnaryOp<StatFinalize<RMS_INV_W, RMS_EPS>, compute_kernel_lib::Dst::D0> {
    static ALWI void init() { rsqrt_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) { stat_finalize_payload<RMS_INV_W, RMS_EPS>(slot_offset); }
};

// Largest divisor of `wt` that is <= `cap` -- pass B's DEST-lane block size (Perf 2,
// descriptor D21; the full measured justification is at PASS_B_BLK's use below).
constexpr uint32_t pass_b_blk(uint32_t wt, uint32_t cap) {
    uint32_t b = (cap < wt) ? cap : wt;
    while (b > 1 && (wt % b) != 0) {
        --b;
    }
    return b;
}

// ---- the bake-off switch (see kernels/bench_writer.cpp for the full legend) ----
// Only bit 0 (ROTATE) has any meaning on the compute side; the other bits are writer /
// host levers and are accepted here so all three kernels take the same define.
#ifndef RMS_ROT_VARIANT
#define RMS_ROT_VARIANT 0
#endif
namespace {
constexpr bool ROTATE = (static_cast<uint32_t>(RMS_ROT_VARIANT) & 1u) != 0u;
// bit 16 = the ATTRIBUTION variant: rotating machinery, placement held STILL at slot 0.
constexpr bool STILL = (static_cast<uint32_t>(RMS_ROT_VARIANT) & 16u) != 0u;
}  // namespace

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_x_squared = 2;
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_row_stat = 4;
constexpr uint32_t cb_gamma_sticks = 5;
constexpr uint32_t cb_gamma_tiles = 6;
constexpr uint32_t cb_normalized = 7;
constexpr uint32_t cb_output_tiles = 8;
constexpr uint32_t cb_output_sticks = 9;
// Cross-core width combine (op_design.md section 3.4) -- allocated only when the
// plan says COMBINE.
constexpr uint32_t cb_sum_handoff = 10;
constexpr uint32_t cb_partials_gathered = 11;
constexpr uint32_t cb_stat_handoff = 12;
constexpr uint32_t cb_row_final = 13;
}  // namespace

void kernel_main() {
    // ---- compile-time knobs (all from rms_norm_program_descriptor.py) -----
    constexpr uint32_t IS_TILE = get_compile_time_arg_val(0);
    constexpr uint32_t WT_CHUNK = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_W_CHUNKS = get_compile_time_arg_val(2);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(3);
    constexpr uint32_t PARTIAL_W = get_compile_time_arg_val(4);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(5);
    constexpr uint32_t GAMMA_IS_RM = get_compile_time_arg_val(6);
    constexpr uint32_t INV_W_BITS = get_compile_time_arg_val(7);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(8);
    constexpr uint32_t REDUCE_BULK = get_compile_time_arg_val(9);
    constexpr uint32_t REDUCE_ACC_VIA_ADD = get_compile_time_arg_val(10);
    constexpr uint32_t SCALER_TILES = get_compile_time_arg_val(11);
    // Refinement 2: the cross-core width combine.  COMBINE == 1 means this core
    // owns only a width SLICE of its rows, so pass A yields a PARTIAL sum(x^2)
    // that the group root sums, finalizes and multicasts back (the writer owns
    // the dataflow; see op_design.md section 3.4).
    constexpr uint32_t COMBINE = get_compile_time_arg_val(12);
    constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(13);
    // Refinement 4 / Lamp L6d (descriptor D12): cb_x_squared's width tiles per
    // tile-row, and hence the reduce's per-call reduce-dim width.  1 means pass A's
    // `square` folds the chunk's width tiles straight into DEST
    // (DestAccumulation::PerRow) rather than packing WT_CHUNK x^2 tiles out to L1
    // for the reduce to read back; WT_CHUNK is the unfolded (Phase-0) path.
    constexpr uint32_t X_SQUARED_WT = get_compile_time_arg_val(14);
    // Refinement 4 / Lamp L5 (descriptor D14).  X_RESIDENT is now an EXPLICIT flag
    // rather than `NUM_W_CHUNKS == 1`, which is what decouples "x is held across
    // both passes" from "the width is one chunk" and gives the op its third regime:
    //   RESIDENT      X_RESIDENT=1, NUM_W_CHUNKS==1   whole row in one chunk
    //   ROW_RESIDENT  X_RESIDENT=1, NUM_W_CHUNKS>1    x + gamma held for the whole
    //                                                 tile-row, only the DERIVED CBs
    //                                                 chunked -> x read ONCE
    //   STREAM        X_RESIDENT=0, NUM_W_CHUNKS>1    x re-read in pass B
    constexpr uint32_t X_RES = get_compile_time_arg_val(15);
    // Perf 2 (descriptor D25): is cb_input_tiles the ZERO-COPY resident shard?  That is the
    // precondition for the combine pipeline below -- see its justification at PIPE_A.
    constexpr uint32_t NATIVE_IN = get_compile_time_arg_val(16);

    const uint32_t num_rows = get_arg_val<uint32_t>(0);  // tile-rows owned by this core
    // Only the core holding the row's LAST width tile applies the partial-W
    // scaler/mask; 1 on the whole-row schemes.
    const uint32_t owns_last_w = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);  // group root: sums + finalizes
    // BENCH (perf_experiments/root_rotation): this core's slot in its width group and its
    // group's rotation phase.  Under ROTATE the root duty moves PER ROW-BLOCK, so the
    // `is_root` runtime arg above is replaced by a per-block predicate the writer computes
    // IDENTICALLY -- if the two ever disagreed, one side would fold a block the other never
    // gathered.  Ignored (and 0-emitted) in the baseline variant.
    const uint32_t my_slot = get_arg_val<uint32_t>(3);
    const uint32_t rot_phase = get_arg_val<uint32_t>(4);

    // An INACTIVE core (see the reader): no shard, no work, and its reader pushed
    // nothing -- return before any CB or LLK state is touched.
    if (num_rows == 0) {
        return;
    }

    constexpr bool RM = (IS_TILE == 0);
    constexpr bool HAS_G = (HAS_GAMMA != 0);
    constexpr bool G_RM = (GAMMA_IS_RM != 0);
    // X_RESIDENT == GAMMA_RESIDENT, from the descriptor's regime decision (D14).
    constexpr bool X_RESIDENT = (X_RES != 0);
    static_assert(NUM_W_CHUNKS > 1 || X_RESIDENT, "rms_norm: a one-chunk width is resident by definition");
    // Lamp L5's regime: resident x/gamma, chunked derived CBs.  The two held CBs
    // then span the WHOLE tile-row while every helper call still works on one
    // WT_CHUNK, so each call indexes them at a TILE OFFSET (TileOffset::Set) and
    // neither is popped until the row-block is done.
    constexpr bool ROW_RESIDENT = X_RESIDENT && (NUM_W_CHUNKS > 1);
    static_assert(!ROW_RESIDENT || BLOCK_ROWS == 1, "rms_norm: ROW_RESIDENT holds ONE tile-row of x");
    // Width tiles the HELD CBs (cb_input_tiles, cb_gamma_tiles) span.  Equals
    // WT_CHUNK in both Phase-0 regimes, so this is byte-identical off the L5 path.
    constexpr uint32_t X_HOLD_WT = X_RESIDENT ? (WT_CHUNK * NUM_W_CHUNKS) : WT_CHUNK;
    constexpr auto XOFF = ROW_RESIDENT ? ckl::TileOffset::Set : ckl::TileOffset::Unset;

    // Perf 2 (descriptor D25) -- THE COMBINE PIPELINE.  Run block blk+1's pass A BEFORE
    // block blk's cross-core combine, so the root's gather wait and its whole fold +
    // multicast overlap independent square+reduce work instead of idling.
    //
    // WHAT IT HARVESTS, measured.  Ablating the root chain's payload while keeping every CB
    // handshake left the root's own `cb_wait_front(cb_partials_gathered)` at 13610 ns over 4
    // rounds -- ~3400 ns per round of the root sitting idle, 21% of the pre-Perf-2 wall.
    // That residue is LATENCY, not payload: no amount of making the fold cheaper removes it.
    // And it IS hideable rather than being the slowest member finishing: across a group the
    // `compute_reduce` END spread is 112-127 ns (members finish pass A together) while the
    // `writer_gather_ship` END spread is 448-1704 ns, strictly monotone in hop distance from
    // the root -- i.e. ~250 ns per sender of SERIALIZED NoC ingress at the root's L1.
    //
    // MEASURED (blackhole p150b 1350 MHz, at the op's pinned config; whole-op ns from
    // perf_experiments/combine_pipeline_depth, whose serial baseline reproduces the real op
    // to 0.1% -- 64707 vs 64801 ns -- because it patches the op's own descriptor):
    //     serial 64707 -> pipe 57000 (1.135x) -> + handoff depth 2 55740 (1.161x)
    //            -> + the writer's early stat publish (D24) 53740 (1.204x)
    // Mechanism on the root core: `compute_root_sum`'s per-round idle 3246 -> 0 ns.
    // `torch.equal`-IDENTICAL to serial -- this changes WHEN work is issued, never what.
    //
    // EARNED CARVE-OUT, and it is a CORRECTNESS one, not a perf one: this requires
    // cb_input_tiles to hold the core's WHOLE assignment, i.e. the zero-copy resident shard
    // (`NATIVE_IN`).  Pass A for blk+1 addresses x at a TILE OFFSET past a front that pass B
    // has not popped yet, and a tile offset cannot cross a CB ring WRAP.  A shard-backed CB
    // is the whole assignment so its front never wraps; a reader-fed `CB_X_DEPTH == 2` ring
    // straddles once every two rounds.  Measured on the interleaved width split: the
    // pipeline is WRONG there (pcc 0.980150, not bit-exact), and sizing that ring to
    // num_blocks+1 blocks to make it right costs +196608 B/core and is STILL 0.894x
    // ((1,1,8192,1024) INTERLEAVED GRID_W=8: 116712 -> 130164 ns) because that regime is
    // reader/DRAM-bound.  So the carve-out is doubly earned -- incorrect AND slower -- and
    // it is written as the narrow exception: everything shard-backed gets the pipeline.
    //
    // `num_blocks > 1` is not a guard, it is the mechanism: with one block there is no
    // blk+1 to hoist.  It is checked at runtime below, not here.
    constexpr bool PIPE_A = (NATIVE_IN != 0) && (COMBINE != 0);
    // Pass A's x operand needs a RUNTIME tile base once it can run ahead of the front.
    // Compile-time-elided (hence byte-identical to Refinement 4) when PIPE_A is off.
    constexpr auto AOFF = PIPE_A ? ckl::TileOffset::Set : XOFF;

    // srcA at boot is whichever CB the first helper unpacks from.
    constexpr uint32_t CB_A = RM ? cb_input_sticks : cb_input_tiles;
    compute_kernel_hw_startup(CB_A, cb_scaler, cb_output_tiles);

    // ---- policy / shape knobs --------------------------------------------
    constexpr auto REDUCE_POLICY =
        (REDUCE_BULK != 0) ? ckl::ReduceInputPolicy::BulkWaitBulkPop : ckl::ReduceInputPolicy::WaitAndPopPerTile;

    // Reduce datapath, chosen host-side from WT_CHUNK (the reduce-dim tiles per
    // reduce() call) -- see REDUCE_ACC_VIA_ADD_MIN_WT in the descriptor.
    //
    //   AccumulateViaAdd sums the width tiles ELEMENTWISE into DST with pairwise
    //   add_tiles and finishes the within-tile 32-column sum on the SFPU (fp32
    //   LREGs).  ReduceTile (the FPU matmul-with-ones) instead accumulates all
    //   WT_CHUNK*32 all-positive addends of sum(x^2) into a single DEST word --
    //   which at fp32_dest_acc_en=False is 16-bit, and is exactly the wide-W
    //   error Refinement 1 diagnosed.  AccumulateViaAdd cuts the DEST-resident
    //   accumulation depth by 32x (WT_CHUNK/2 pairwise adds instead of
    //   WT_CHUNK*32 serial ones) and is also the faster path once the reduce dim
    //   spans >= 4 tiles (examples/reduce_block/report_reduced_sweep.md).
    constexpr bool ACC_VIA_ADD = (REDUCE_ACC_VIA_ADD != 0);
    constexpr auto REDUCE_ALGO = ACC_VIA_ADD ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::Auto;
    // AccumulateViaAdd's cross-chunk Accumulate indexes a resident block, so it
    // is BulkWaitBulkPop-only (reduce_helpers_compute.inl static_assert). The
    // descriptor already couples the two knobs; assert it so a future flip of
    // REDUCE_BULK fails here instead of deep inside the library.
    static_assert(
        !ACC_VIA_ADD || REDUCE_BULK != 0,
        "rms_norm: ReduceAlgorithm::AccumulateViaAdd + Accumulate requires BulkWaitBulkPop (REDUCE_BULK == 1)");

    // Non-tile-aligned W: the two datapaths take DIFFERENT partial mechanisms.
    //   ReduceTile       : reader emitted [full scaler, partial scaler]; route the
    //                      partial one to the last width tile (pad lanes * 0).
    //   AccumulateViaAdd : reader emitted a single 0/1 MASK tile at index 0; the
    //                      last width tile folds in through a masked accumulating
    //                      broadcast-mul, PARTIAL_W valid lanes.
    // Both zero the pad lanes by multiplying them with an exact 0, so the reader's
    // pad-lane invariant (no inf/NaN in padding) is what it always was.
    //
    // Under a cross-core width split only ONE core in the group holds the row's
    // last width tile, so the choice is per-core (runtime `owns_last_w`).  A
    // non-owning core takes none(), which is exactly right on BOTH datapaths:
    // ReduceTile then uses cb_scaler tile 0 (the FULL 1.0 scaler) everywhere, and
    // AccumulateViaAdd ignores cb_scaler entirely when there is no partial.
    const auto PARTIAL_SCALER = (PARTIAL_W == 0 || owns_last_w == 0)
                                    ? ckl::ReducePartialScaler::none()
                                    : (ACC_VIA_ADD ? ckl::ReducePartialScaler::partial_mask(PARTIAL_W, /*mask_idx=*/0)
                                                   : ckl::ReducePartialScaler::last_tile_at(1));
    // RESIDENT holds x across both passes -> pass A must not pop it.
    constexpr auto PASS_A_POP = X_RESIDENT ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd;

    // Lamp L6d / D12: fold pass A's square straight into DEST.  The chunk's width
    // tiles are multiplied and ACCUMULATED in one DEST slot, so cb_x_squared takes
    // one tile per tile-row (X_SQUARED_WT == 1) and the reduce's per-call width is 1
    // -- deleting WT_CHUNK-1 packs and the matching unpacks per tile-row.  The
    // cross-chunk carry still runs through the fp32 cb_row_stat, so the accumulation
    // DEST sees is bounded by WT_CHUNK, which is exactly what the descriptor's
    // DEST_ACC_SQUARE_MAX_WT ceiling bounds.
    constexpr bool SQ_FOLD = (X_SQUARED_WT == 1) && (WT_CHUNK > 1);
    static_assert(
        X_SQUARED_WT == 1 || X_SQUARED_WT == WT_CHUNK,
        "rms_norm: X_SQUARED_WT must be 1 (the DEST fold) or WT_CHUNK (the packed path)");
    static_assert(
        !SQ_FOLD || PARTIAL_W == 0,
        "rms_norm: the DEST fold folds the last width tile's pad lanes in BEFORE the "
        "reduce, so the reduce's partial scaler / mask can no longer reach them");
    // The fold's pack is per-OUTER (one tile per tile-row of the grid), which is the
    // policy pair DestAccumulation::PerRow requires.
    constexpr auto SQ_OUT_FOLDED = ckl::output(
        cb_x_squared,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        ckl::DataFormatReconfig::Enabled,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::Disabled,
        ckl::DestAccumulation::PerRow);
    constexpr auto SQ_OUT = SQ_FOLD ? SQ_OUT_FOLDED : ckl::output(cb_x_squared);

    // Perf 1's D16 `ROOT_FOLD_OUT` -- the root fold's packer-L1-accumulation output spec
    // (`cb_row_stat`, OneUpfront/OneAtEnd, `L1Accumulation::SeedFirst`) -- is DELETED.
    // Perf 2 / D22 accumulates the group sum in DEST and fuses the finalize into that same
    // window, so nothing packs into cb_row_stat on the COMBINE path at all any more.  Its
    // measured justification (2.18x, and MORE accurate than the packer fold) is at the
    // fused chain below.

    // ---- the operand specs, ONE definition each ---------------------------
    // Every one carries XOFF, so the L5 regime differs from Phase 0 only in the
    // (compile-time-elided) `+ base` on the tile index -- there is no second
    // code path.  base is 0 whenever XOFF is Unset, and `tile_base_value<Unset>`
    // folds the whole term away.
    constexpr auto X_IN_A = ckl::input(
        cb_input_tiles,
        ckl::WaitPolicy::Upfront,
        PASS_A_POP,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        AOFF);
    // Pass B's x: held CBs are popped ONCE per row-block below (an explicit pop is
    // the sanctioned pattern for a lifetime no single PopPolicy can express), so
    // that a chunk's `AtEnd` cannot pop the base tiles the next chunk still needs.
    constexpr auto PASS_B_X_POP = ROW_RESIDENT ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd;
    constexpr auto X_IN_B = ckl::input(
        cb_input_tiles,
        ckl::WaitPolicy::Upfront,
        PASS_B_X_POP,
        ckl::OperandKind::Block,
        ckl::DataFormatReconfig::Enabled,
        XOFF);
    constexpr auto G_IN = ckl::input(
        cb_gamma_tiles,
        ckl::WaitPolicy::Upfront,
        ckl::PopPolicy::None,
        ckl::OperandKind::Row,
        ckl::DataFormatReconfig::Enabled,
        XOFF);
    constexpr uint32_t NORM_OUT = HAS_G ? cb_normalized : cb_output_tiles;

    // Perf 2 (descriptor D21): pass B's DEST-LANE BLOCK SIZE.
    //
    // Pass B's two chains walk (rows x WT_CHUNK) tiles.  At block_size 1 each tile pays
    // its own CB reserve/push and its own MATH<->PACK `tile_regs` handshake; at
    // block_size B one outer iter drives B DEST lanes and the per-element init, the
    // format reconfig and the CB flow control all amortize over B tiles.  The chain walks
    // element-major inside a block (eltwise_chain.inl `elem_apply_compute`), which is what
    // makes the amortization real rather than nominal.
    //
    // BITWISE IDENTICAL to block_size 1 -- this changes WHEN work is issued, never what.
    // Verified `torch.equal` against the previous spelling at every geometry swept.
    //
    // MEASURED (blackhole p150b 1350 MHz, at the op's pinned config -- bf16 / HiFi2 /
    // fp32_dest_acc_en=False; isolated bench perf_experiments/pass_b_fuse_scale_gamma, a
    // kernel containing pass B and nothing else, one fresh-cache profiled run per variant):
    //     rows=8, WT_CHUNK=4 (the focus shape)   14050 -> 8860 ns   1.59x
    //   decomposed: -3.3 us from one reserve/push per CHUNK instead of per tile, then
    //   -1.9 us from 4 DEST lanes per outer iter.  The block_size curve at 128 tiles is
    //   monotonic and diminishing: 13266 / 9229 / 8804 / 8209 ns at B = 1 / 2 / 4 / 8.
    //   Wins across the whole (rows x WT_CHUNK) space: 1.28x at rows=1/wt=4, 1.49x at
    //   rows=1/wt=32, 1.62x at rows=8/wt=16, 1.65x at rows=32/wt=4, and 1.66x with
    //   HAS_GAMMA=0 -- i.e. the lever is not gamma-specific, the scale pass alone gains.
    //
    // NEVER a literal 8: DEST_AUTO_LIMIT is 8 lanes at fp32_dest_acc_en=False but 4 at
    // True, and it is the build-flag-derived cap (dest_helpers.hpp).  A DIVISOR of
    // WT_CHUNK keeps every outer iter full, so the Chunked pack lifecycle below always
    // covers exactly `PASS_B_BLK` pages.
    //
    // The block size and the PerChunk pack lifecycle are ONE change, not two: at
    // block_size > 1 the chain emits the pack lifecycle once per OUTER iter (outside the
    // lane loop), so a `PerTile` reserve would reserve 1 page and then pack
    // PASS_B_BLK -- it corrupts the CB ring and HANGS (observed in the bench before the
    // fix).  Do not change one without the other.
    constexpr uint32_t PASS_B_BLK = pass_b_blk(WT_CHUNK, ckl::DEST_AUTO_LIMIT);
    // Reserve/push once per DEST-lane block. `PerChunk` (not `Upfront`) deliberately:
    // it keeps the per-block page handover the ROW_MAJOR path's `untilize` consumer
    // needs, and it measured within noise of `Upfront` (8860 vs 8901 ns) wherever both
    // are legal -- one path, no untested regime. `Upfront`/`AtEnd` is 1.22-1.23x at
    // WT_CHUNK == 1 (where PASS_B_BLK clamps to 1 and this is inert), which is the one
    // geometry that pays ~0.4% for the single path; see the changelog for that trade.
    constexpr auto PASS_B_OUT_NORM = ckl::output(NORM_OUT, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk);
    constexpr auto PASS_B_OUT_GAMMA =
        ckl::output(cb_output_tiles, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk);
    constexpr bool CROSS_CORE = (COMBINE != 0);
    // Perf 2 (descriptor D22): the gather's slots per tile-row -- GROUP_SIZE rounded UP TO
    // EVEN -- and the half-stride the root's fused pairwise fold walks.  DERIVED, never
    // passed: a pure function of GROUP_SIZE that the writer derives identically, so the
    // landing layout has one definition per kernel and no CT arg can drift between them.
    // Equal to GROUP_SIZE at every even group (8 / 28 / 32, including the focus shape's 8);
    // the one extra slot at odd GROUP_SIZE is boot-zeroed by the writer and pairs against
    // the odd contributor as an exact +0.0.  See the fused chain below.
    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;
    constexpr uint32_t GATHER_HALF = GATHER_SLOTS / 2;
    // Pass B's Col operand: the multicast landing CB when the stat was combined
    // across cores, the local accumulator otherwise.
    constexpr uint32_t CB_STAT_B = CROSS_CORE ? cb_row_final : cb_row_stat;
    // Perf 1 (descriptor D18): on the COMBINE path pass A's reduce packs its partial
    // STRAIGHT into cb_sum_handoff, deleting the fp32 tile copy that used to move
    // cb_row_stat -> cb_sum_handoff on EVERY core of EVERY group.
    //
    // Legal because the combine path takes its width slice in ONE chunk -- the writer
    // already `static_assert`s `!CROSS_CORE || NUM_W_CHUNKS == 1`.  With num_blocks == 1
    // the reduce never re-reads its accumulator (reduce_helpers_compute.inl only enters
    // the reload branch for a later chunk), so the accumulator CB is WRITE-ONLY here and
    // does not have to be a re-readable accumulator at all.
    //
    // This SATISFIES the CB-ownership rule rather than bending it: cb_sum_handoff now has
    // compute's pack as its single producer and the writer as its single consumer, and
    // cb_row_stat becomes strictly compute-private AND root-only.  Page counts are
    // unchanged -- the reduce pushes exactly the `rows` pages the copy used to push.
    //
    // MEASURED (isolated bench perf_experiments/reduce_pack_to_handoff, blackhole p150b
    // 1350 MHz, one fresh-cache profiled run per variant): the modelled pass-A tail goes
    // 11933 -> 9377 ns at rows=10/width=1 (1.27x), and wins at all 11 rows x width
    // geometries swept (1.13x-1.46x); the output is `torch.equal`-IDENTICAL to
    // reduce-then-copy at every gated point.  The win GROWS as rows-per-block shrinks
    // (-158 ns/tile-row at rows=1 vs -60 at rows=32) because the deleted copy paid a
    // per-CALL cost regardless of tile count -- so the decode / width-sharded profiles
    // gain most.
    constexpr uint32_t CB_REDUCE_ACC = CROSS_CORE ? cb_sum_handoff : cb_row_stat;

    // 1/rms = rsqrt(sum/W + eps).  ONE definition, used by the local path and by the
    // root's post-combine finalize -- INV_W is the LOGICAL width either way.
    // rsqrt_tile_init() is MANDATORY, not decorative: rms_stat_rsqrt_body reads
    // sfpi::vConstIntPrgm0 / vConstFloatPrgm1..2, which sfpu::rsqrt_init programs
    // (ckernel_sfpu_sqrt.h) -- persistent SFPU PROGRAM registers, which is what makes
    // hoisting it out of a per-tile loop legal at all.
    auto finalize = [](uint32_t dst) {
        rsqrt_tile_init();
        stat_finalize_payload<INV_W_BITS, EPS_BITS>(dst);
    };

    // ---- gamma: resident for the whole core's assignment (RESIDENT) -------
    // ROW_RESIDENT holds the whole tile-row of gamma too, so it tilizes every
    // chunk the reader staged; NUM_W_CHUNKS == 1 makes this the Phase-0 single call.
    if constexpr (HAS_G && X_RESIDENT && G_RM) {
        MaybeDeviceZoneScope("compute_gamma_tilize");
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            ckl::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1);
        }
    }

    const uint32_t num_blocks = (num_rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    auto rows_of = [&](uint32_t blk) {
        const uint32_t r0 = blk * BLOCK_ROWS;
        return (num_rows - r0 < BLOCK_ROWS) ? (num_rows - r0) : BLOCK_ROWS;
    };

    // ================= pass A: sum(x^2) over the whole width ===============
    // Hoisted into a lambda so D25's pipeline can issue it for block blk+1 before block
    // blk's combine.  `pipe_base` is that not-yet-fronted block's tile offset inside
    // cb_input_tiles, and is a compile-time 0 (fully elided) whenever PIPE_A is off.
    auto pass_a = [&](uint32_t rows, uint32_t pipe_base) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            // Tile offset of this chunk inside the HELD CBs.  0 (and elided) unless
            // ROW_RESIDENT, where cb_input_tiles / cb_gamma_tiles span the whole row.
            const uint32_t hold_base = (ROW_RESIDENT ? (c * WT_CHUNK) : 0) + pipe_base;
            if constexpr (RM) {
                MaybeDeviceZoneScope("compute_tilize_x");
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            // x^2, either packed to cb_x_squared per width tile or folded into DEST
            // (D12).  `square` cannot carry the tile base, so the chain is spelled
            // out; it is exactly what square<> expands to.
            {
                MaybeDeviceZoneScope("compute_square");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows, WT_CHUNK),
                    ckl::BinaryFpu<
                        X_IN_A,
                        X_IN_A,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None,
                        ckl::Dst::D0,
                        SQ_OUT.dest_accumulation>{hold_base, hold_base},
                    ckl::PackTile<SQ_OUT>{});
            }

            MaybeDeviceZoneScope("compute_reduce");
            ckl::accumulate_reduce_block<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_x_squared,
                cb_scaler,
                CB_REDUCE_ACC,
                REDUCE_POLICY,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Fast,
                REDUCE_ALGO>(ckl::ReduceInputBlockShape::of(rows, X_SQUARED_WT), c, NUM_W_CHUNKS, PARTIAL_SCALER);
        }
    };

    // D25's PROLOGUE: block 0's pass A runs before the loop, so from here on the loop body
    // issues block blk+1's pass A first and the root's arrival wait + fold + multicast for
    // block blk overlap it.  Elided entirely when PIPE_A is off.
    if constexpr (PIPE_A) {
        pass_a(rows_of(0), 0);
    }

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t rows = rows_of(blk);

        if constexpr (PIPE_A) {
            // cb_input_tiles' front is still block blk (pass B has not popped it), so block
            // blk+1 begins `rows * X_HOLD_WT` tiles further in.  Widen the wait to cover
            // both blocks -- on the shard-backed CB this is already satisfied (the whole
            // assignment is published up front), and PIPE_A is gated on exactly that.
            if (blk + 1 < num_blocks) {
                const uint32_t next_rows = rows_of(blk + 1);
                cb_wait_front(cb_input_tiles, (rows + next_rows) * X_HOLD_WT);
                pass_a(next_rows, rows * X_HOLD_WT);
            }
        } else {
            pass_a(rows, 0);
        }

        // ================= finalize: 1/rms = rsqrt(sum/W + eps) ============
        // Pops before reserving, so the `rows`-page accumulator CB suffices.
        if constexpr (!CROSS_CORE) {
            MaybeDeviceZoneScope("compute_finalize");
            for (uint32_t i = 0; i < rows; ++i) {
                ckl::transform_in_place(cb_row_stat, finalize);
            }
        } else {
            // Pass A's reduce has ALREADY packed this core's raw partial into
            // cb_sum_handoff (D18), so there is nothing to hand off here -- the writer
            // ships it to the group root straight out of the reduce's own output pages.
            // The `compute_partial_handoff` zone that used to sit here is retired with the
            // copy it measured.
            // BENCH: under ROTATE the fold lands on the round's rotating root -- the SAME
            // predicate the writer uses, spelled once here.  The chain itself, its DEST
            // walk, its raw-sfpi finalize and its mandatory reconfig pair are UNTOUCHED, so
            // a rotated fold is bit-identical to the fixed one; only the core it runs on
            // moves.  (A core now alternates rounds with and without the fold, but both
            // transitions are already exercised by the baseline -- a fixed root does
            // fold->passB every round and a member does passB with no fold every round.)
            const bool is_root_blk =
                ROTATE ? (STILL ? (my_slot == 0) : (my_slot == (blk + rot_phase) % GROUP_SIZE)) : (is_root != 0);
            if (is_root_blk) {
                // Sum the group's GROUP_SIZE partials ELEMENTWISE and finalize the row
                // total, in ONE DEST WINDOW per tile-row.  Each partial is a column-shaped
                // REDUCE_ROW result, so the row total is their elementwise sum regardless
                // of which lanes the datapath filled.
                //
                // Perf 2 (descriptor D22) -- THE FUSED ROOT CHAIN.  This replaces the two
                // stages Perf 1 left here:
                //   (D16) a per-row chain that copied each partial into DEST and let the
                //         PACKER fold it onto the resident fp32 cb_row_stat
                //         (L1Accumulation::SeedFirst), then
                //   (D19) a second chain that UNPACKED cb_row_stat, ran StatFinalize, and
                //         packed cb_stat_handoff.
                // Now the row's partials are accumulated PAIRWISE IN DEST
                // (`add_tiles(..., acc_to_dest=true)` over the two halves of the row's
                // GATHER_SLOTS window), the finalize runs on that same DEST slot, and ONE
                // `pack_tile` writes cb_stat_handoff.  Deleted: GROUP_SIZE packs, one fp32
                // pack and one fp32 unpack per tile-row -- cb_row_stat's whole L1 round
                // trip on the root, and with it every remaining use of cb_row_stat on the
                // COMBINE path.
                //
                // MEASURED (blackhole p150b 1350 MHz, at the op's pinned config -- bf16 /
                // HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False, UNCHANGED;
                // isolated bench perf_experiments/root_chain_dest_fuse, ns for the stage
                // PAIR per combine round, one fresh-cache profiled run per variant):
                //     baseline (D16 fold + D19 finalize)   5874 ns
                //     DEST fold only, finalize separate    3048 ns   1.93x
                //     THIS (fused, one DEST window)        2698 ns   2.18x
                //   Sweep 1.73x-3.86x over GROUP_SIZE {4,8,9,16,28,32} x rows {1,8,32};
                //   the win GROWS as the group widens (3.04x at GROUP_SIZE=32, rows=8) and
                //   is largest at rows=1, i.e. the decode / width-shard profiles.
                //   Bench calibration: baseline x 4 rounds = 23496 ns against the op's own
                //   cumulative-peel value of 23230 ns for these two stages -- 1.1%.
                //
                // MORE ACCURATE THAN THE CHAIN IT REPLACES, measured, not argued:
                // rel-RMS 2.42e-03 vs 3.38e-03 at the focus geometry (3.36e-03 vs 5.09e-03
                // at GROUP_SIZE=28), pcc_out 0.999998.  This REFUTES D16's recorded
                // reasoning that an fp32-L1 accumulator is "at least as accurate": the
                // packer fold rounds EVERY contributor into a 16-bit DEST word before its
                // exact fp32 L1 add, so it pays GROUP_SIZE roundings in a LINEAR chain,
                // while the pairwise DEST walk pays the same per-addend rounding but sums
                // as a TREE, shortening the error chain to log2(GROUP_SIZE)+1.  A
                // precision HEDGE was measured too (pair in DEST, accumulate in fp32 L1)
                // and was both slower (4142 ns) AND less accurate (2.91e-03) -- there was
                // nothing to hedge.  The user's precision contract is untouched.
                //
                // RAW-LLK SUBSTITUTION -- the fusion is INEXPRESSIBLE through eltwise_chain.
                // `DestAccumulation::PerRow` gives exactly the DEST window this needs, but
                // EVERY chain element's apply runs on EVERY inner iteration of that row
                // (eltwise_chain.inl `elem_apply_compute`; a DestOnly/UnaryOp element's
                // exec is called `inner_count` times unconditionally).  So a StatFinalize
                // element placed after the accumulating BinaryFpu would rsqrt a PARTIAL sum
                // GROUP_SIZE/2 times instead of once on the completed one.  There is no
                // apply-after-the-accumulation element kind and no per-row tail hook -- the
                // chain's only per-row tail is the pack itself.  The helper-expressible
                // split form is measured at 1.93x against this 2.18x, and the bench keeps
                // both so the gap is re-checkable.  Do NOT "restore" this to chain calls
                // without re-measuring.
                //
                // The finalize is the op's existing sanctioned raw-sfpi StatFinalize (D17),
                // called through the SAME `stat_finalize_payload` the chain element wraps,
                // so the SFPU spelling is bit-identical to before.  Its lane invariant
                // survives unchanged: both bodies stay <STRIDE=2, ITERS=4> at
                // VectorMode::C, so `*(1/W)+eps` and `rsqrt` visit exactly the same lanes
                // and the +eps guard still covers every lane the rsqrt touches (no
                // rsqrt(0)=inf on an all-zero row).  The fusion cannot disturb that: the
                // FPU has no lane scope, so the DEST accumulation fills the WHOLE tile and
                // the finalize sees the completed sum on every lane it visits -- exactly as
                // it did when it unpacked cb_row_stat.  The lanes it skips hold the raw
                // finite group sum, the same defined-but-meaningless datum as before.
                //
                // GATHER_SLOTS (== GROUP_SIZE rounded up to even) is what makes the
                // pairwise walk universal: at odd GROUP_SIZE the writer boot-zeroes one pad
                // slot per tile-row, which pairs against the odd contributor and adds an
                // exact +0.0.  So there is no odd/even code path and no GROUP_SIZE guard.
                MaybeDeviceZoneScope("compute_root_fused");
#if defined(RMS_ABLATE_ROOT_SUM) || defined(RMS_ABLATE_ROOT_FINALIZE)
                // ABLATION (temporary, /perf-measure): payload removed, every CB handshake
                // and trip count preserved.
                cb_wait_front(cb_partials_gathered, GATHER_SLOTS * rows);
                for (uint32_t r = 0; r < rows; ++r) {
                    cb_reserve_back(cb_stat_handoff, 1);
                    cb_push_back(cb_stat_handoff, 1);
                }
                cb_pop_front(cb_partials_gathered, GATHER_SLOTS * rows);
#else
                // The whole block's gather window is waited/popped ONCE: the pairwise walk
                // addresses two tiles of the same CB at a stride, which a per-tile wait
                // cannot express.  Legal exactly as it stands -- the writer already
                // publishes the block atomically (`cb_push_back(cb_partials_gathered,
                // GATHER_SLOTS * rows)`) and the CB is sized to that same window.
                cb_wait_front(cb_partials_gathered, GATHER_SLOTS * rows);
                // The DATA-FORMAT RECONFIG the eltwise_chain helpers emit for us has to be
                // written out by hand here, and it is NOT optional: pass A left the
                // unpacker configured for cb_x_squared (the input dtype, bf16 in the
                // measured config) and the packer for cb_sum_handoff, while the gather and
                // the handoff are fp32.  Without these two calls the fold unpacks fp32 L1
                // through a bf16 srcA/srcB and the accumulated sum reads as ~0, which the
                // finalize then turns into rsqrt(eps) -- a uniform ~1/sqrt(eps) scale error
                // that keeps pcc at 0.9997 and shows up only in rel-RMS (measured 994
                // against a 0.04 bound during integration).  That is exactly why the
                // op's regression nets bound rms and not just pcc.
                reconfig_data_format(cb_partials_gathered, cb_partials_gathered);
                pack_reconfig_data_format(cb_stat_handoff);
                add_tiles_init(cb_partials_gathered, cb_partials_gathered, /*acc_to_dest=*/true);
                rsqrt_tile_init();
                for (uint32_t r = 0; r < rows; ++r) {
                    const uint32_t base = r * GATHER_SLOTS;
                    tile_regs_acquire();
                    for (uint32_t p = 0; p < GATHER_HALF; ++p) {
                        add_tiles(cb_partials_gathered, cb_partials_gathered, base + p, base + GATHER_HALF + p, 0);
                    }
                    stat_finalize_payload<INV_W_BITS, EPS_BITS>(0);
                    tile_regs_commit();
                    // Reserve/push PER TILE-ROW, not per block: the writer's stat multicast
                    // starts on the first finalized row, and a block-granular publish would
                    // make it wait for the block's LAST row.  Measured cost of keeping that
                    // overlap: zero (2698 vs 2701 ns).
                    cb_reserve_back(cb_stat_handoff, 1);
                    tile_regs_wait();
                    pack_tile(0, cb_stat_handoff);
                    tile_regs_release();
                    cb_push_back(cb_stat_handoff, 1);
                }
                cb_pop_front(cb_partials_gathered, GATHER_SLOTS * rows);
#endif
            }
        }

        // ================= pass B: scale ===================================
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            const uint32_t hold_base = ROW_RESIDENT ? (c * WT_CHUNK) : 0;
            // ROW_RESIDENT never re-stages either held operand: pass A already put
            // the whole tile-row of x tiles (and of gamma) in L1.  This is the
            // pass-B re-read that Lamp L5 exists to delete.
            if constexpr (RM && !X_RESIDENT) {
                MaybeDeviceZoneScope("compute_tilize_x_b");
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            if constexpr (HAS_G && !X_RESIDENT && G_RM) {
                MaybeDeviceZoneScope("compute_gamma_tilize_b");
                ckl::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1);
            }

            // x * (1/rms). The stat is a REDUCE_ROW result: column-shaped, so it
            // broadcasts back ACROSS columns (BroadcastDim::Col) and must be
            // operand B. OperandKind::Col indexes it by row only, and it is not
            // popped -- every width chunk of this block re-reads it.
            {
                MaybeDeviceZoneScope("compute_scale");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows, WT_CHUNK, PASS_B_BLK),
                    ckl::BinaryFpu<
                        X_IN_B,
                        ckl::input(CB_STAT_B, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Col),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Col>{hold_base},
                    ckl::PackTile<PASS_B_OUT_NORM>{});
            }

            if constexpr (HAS_G) {
                // gamma is row-shaped (1 x W, valid in row 0) -> broadcasts DOWN
                // rows (BroadcastDim::Row), indexed by column (OperandKind::Row).
                MaybeDeviceZoneScope("compute_gamma_mul");
                ckl::eltwise_chain(
                    ckl::EltwiseShape::grid(rows, WT_CHUNK, PASS_B_BLK),
                    ckl::BinaryFpu<
                        ckl::input(
                            cb_normalized, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                        G_IN,
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::Row>{0u, hold_base},
                    ckl::PackTile<PASS_B_OUT_GAMMA>{});
            }

            if constexpr (RM) {
                MaybeDeviceZoneScope("compute_untilize");
                ckl::untilize<WT_CHUNK, cb_output_tiles, cb_output_sticks>(rows);
            }

            if constexpr (HAS_G && !X_RESIDENT) {
                cb_pop_front(cb_gamma_tiles, WT_CHUNK);
            }
        }

        // The held CBs' lifetime is the whole row-block, which no PopPolicy on a
        // per-chunk call can express (an `AtEnd` would drop the base tiles the next
        // chunk still indexes), so ROW_RESIDENT pops x here -- the same sanctioned
        // pattern as cb_row_stat / cb_gamma_tiles / cb_scaler.
        if constexpr (ROW_RESIDENT) {
            cb_pop_front(cb_input_tiles, rows * X_HOLD_WT);
        }
        cb_pop_front(CB_STAT_B, rows);
    }

    // SCALER_TILES is the descriptor's single source of truth for how many tiles
    // the reader pushed into cb_scaler (datapath- and PARTIAL_W-dependent).
    cb_pop_front(cb_scaler, SCALER_TILES);
    if constexpr (HAS_G && X_RESIDENT) {
        cb_pop_front(cb_gamma_tiles, X_HOLD_WT);
    }
}
