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
// Under the cross-core width COMBINE the finalize step is replaced by three stages
// (Perf 3 / D27 -- the compact partial transpose; full justification at the
// `member_pack` lambda and the fold below):
//   member_pack  matmul-permute  cb_sum_handoff (rows tiles) -> cb_compact_handoff
//                                (ONE tile, columns 0..rows-1 = the rows' partials)
//   root fused   ONE DEST window over the group's GATHER_SLOTS compact pages +
//                the finalize + one pack             -> cb_stat_handoff
//   recv_unpack  matmul-un-permute cb_mcast_in (1 tile) -> cb_row_final (rows tiles)
// Pass B is untouched by all of it: it still reads a column-shaped stat.
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
// matmul_tiles / matmul_init -- the COMBINE's compact partial transpose (Perf 3 / D27;
// justification at the permute call sites).  The FPU's only horizontal-mixing primitive.
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/rsqrt.h"
// PERMANENT per-stage device-profiler instrumentation (never remove; free when
// the profiler is off -- see the header's durability contract).
#include "perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

// `accumulate_reduce_block` lived in kernel_lib/streaming_reduce_helpers.hpp, which upstream
// retired ("kernel_lib: drop the streaming-reduce wrappers"). It was a thin router over
// reduce() + Accumulate; reinstated here verbatim so the call sites below are unchanged.
namespace rms_norm_local {
template <
    ckernel::PoolType pool,
    ckernel::ReduceDim rdim,
    uint32_t cb_in,
    uint32_t cb_scaler,
    uint32_t cb_acc,
    ckl::ReduceInputPolicy in_policy = ckl::ReduceInputPolicy::WaitAndPopPerTile,
    ckl::ReduceDataFormatReconfigMode reconfig_mode = ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    ReduceFp32Mode fp32_mode = ReduceFp32Mode::Fast,
    ckl::ReduceAlgorithm algorithm = ckl::ReduceAlgorithm::Auto,
    typename PostOp = ckl::NoOp>
ALWI void accumulate_reduce_block(
    ckl::ReduceInputBlockShape block_shape,
    uint32_t b,
    uint32_t num_blocks,
    ckl::ReducePartialScaler partial,
    PostOp post_op_final = PostOp{}) {
    const bool is_last = (b + 1 == num_blocks);
    if (is_last) {
        ckl::reduce<pool, rdim, cb_in, cb_scaler, cb_acc, in_policy, reconfig_mode, fp32_mode, algorithm>(
            block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at_last(cb_acc, b),
            post_op_final,
            partial);
    } else {
        ckl::reduce<pool, rdim, cb_in, cb_scaler, cb_acc, in_policy, reconfig_mode, fp32_mode, algorithm>(
            block_shape,
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::Accumulate::at(cb_acc, b),
            ckl::NoOp{},
            ckl::ReducePartialScaler::none());
    }
}

// `transform_in_place` also lived in the retired streaming_reduce_helpers.hpp. Verbatim
// reinstatement: pop BEFORE reserve_back so a 1-page CB suffices.
template <typename Transform>
ALWI void transform_in_place(uint32_t cb, Transform t) {
    constexpr uint32_t onetile = 1;
    cb_wait_front(cb, onetile);
    tile_regs_acquire();
    reconfig_data_format_srca(cb);
    pack_reconfig_data_format(cb);
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, 0, 0);
    t(0);
    tile_regs_commit();
    cb_pop_front(cb, onetile);
    cb_reserve_back(cb, onetile);
    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();
    cb_push_back(cb, onetile);
}
}  // namespace rms_norm_local

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

// ---- THE COMPACT FINALIZE'S TWO WIDER SCOPES (Perf 3 / D27) ------------------------
// THESE ARE A CORRECTNESS REQUIREMENT, NOT A PERF CHOICE, and the pair above must NOT be
// reused on a compact stat tile.  D27's combine finalizes ONE tile whose columns
// 0..BLOCK_ROWS-1 each hold a different tile-row's group sum, so the finalize has to
// visit EVERY one of those columns.  The <STRIDE=2, ITERS=4> pair above walks even
// parity only -- columns 0,2,..,14 -- which is exactly right for a stat that lives in
// column 0 and SILENTLY WRONG from BLOCK_ROWS = 2 up: the ODD rows' sums are never
// scaled by 1/W and never rsqrt-ed.
//
// MEASURED, twice (perf_experiments/compact_partial_transpose_r2 and _r3's bench A, at
// the op's pinned config): the narrow scope on a compact tile gives pcc 0.9972987 with
// rel-RMS 1036 against this op's 0.04 bound -- i.e. a bug pcc ALONE would have waved
// through, the third of that kind in this op.  It is also 1.39x FASTER (553 vs 770
// ns/round), so it is exactly the sort of "win" that has to be refused.
// _r3's test_combine_bench.py keeps `test_finalize_scope_hazard` as a LIVE assertion
// that fails if the narrow scope ever starts passing on a compact tile.
//
// <STRIDE=1, ITERS=8> keeps the product at 8 (the invariant above), so the net dst_reg
// advance is unchanged and VectorMode's face stepping composes exactly as before:
//   VectorMode::C   faces 0 and 2  -> columns 0..15   (BLOCK_ROWS <= 16)
//   VectorMode::RC  all four faces -> columns 0..31   (BLOCK_ROWS > 16)
// Widening C to RC is a measured flat +452 ns/round, which is why it is taken only where
// BLOCK_ROWS actually needs columns 16..31.  Both bodies stay at the SAME <1,8> in the
// scale and the rsqrt, so the +eps guard still covers every lane the rsqrt touches (no
// rsqrt(0) = inf on an all-zero row) -- the same invariant the narrow pair carries.
ALWI void stat_scale_col_full(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<1, 8>, idst, VectorMode::C, inv_w_bits, eps_bits);
}
ALWI void rsqrt_tile_col_full(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<1, 8>, idst, VectorMode::C);
}
ALWI void stat_scale_all(uint32_t idst, uint32_t inv_w_bits, uint32_t eps_bits) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<1, 8>, idst, VectorMode::RC, inv_w_bits, eps_bits);
}
ALWI void rsqrt_tile_all(uint32_t idst) {
    _llk_math_eltwise_unary_sfpu_params_(rms_stat_rsqrt_body<1, 8>, idst, VectorMode::RC);
}
#endif  // TRISC_MATH

// The SFPU payload only, no init -- so the eltwise_chain element can hoist the init out
// of the per-tile loop while the transform_in_place lambda keeps it inside.
template <uint32_t RMS_INV_W, uint32_t RMS_EPS>
ALWI void stat_finalize_payload(uint32_t dst) {
    MATH((stat_scale_col_skip(dst, RMS_INV_W, RMS_EPS)));
    MATH((rsqrt_tile_col_skip(dst)));
}

// The COMPACT (D27) finalize's payload.  `RMS_WIDE` selects VectorMode::RC over C and is
// a pure function of BLOCK_ROWS at the one call site -- see the scope note above.
template <uint32_t RMS_INV_W, uint32_t RMS_EPS, bool RMS_WIDE>
ALWI void compact_finalize_payload(uint32_t dst) {
    if constexpr (RMS_WIDE) {
        MATH((stat_scale_all(dst, RMS_INV_W, RMS_EPS)));
        MATH((rsqrt_tile_all(dst)));
    } else {
        MATH((stat_scale_col_full(dst, RMS_INV_W, RMS_EPS)));
        MATH((rsqrt_tile_col_full(dst)));
    }
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
// Perf 3 / D27 -- the compact partial transpose.  cb_bank is the one-hot permutation
// bank the READER synthesizes (never popped); cb_compact_handoff carries this core's
// permuted partial out to the writer; cb_mcast_in is where the root's compact stat
// lands.  cb_sum_handoff and cb_row_final are now compute-private.
constexpr uint32_t cb_bank = 14;
constexpr uint32_t cb_compact_handoff = 15;
constexpr uint32_t cb_mcast_in = 16;
// Perf 3 / D28 -- the SLOT TREE's two extra CBs (allocated only when it is taken).
// cb_partials_gathered above becomes the LEVEL-0 ring there.
constexpr uint32_t cb_gather_l1 = 17;
constexpr uint32_t cb_node_out = 18;
}  // namespace

// =======================================================================================
// ONE FOLD, the D22 fused chain, with WHICH ring / WHETHER TO FINALIZE lifted to template
// parameters (Perf 3 / D28).
// =======================================================================================
// This is the code that used to sit inline in the root's combine branch, moved out
// UNCHANGED so the slot tree's three call sites are one implementation:
//     flat root            <GATHER_SLOTS, FINALIZE=true >  gather ring -> cb_stat_handoff
//     tree level 0         <TREE_SL0,     FINALIZE=false>  level-0 ring -> cb_node_out
//     tree level 1 (root)  <TREE_SL1,     FINALIZE=true >  level-1 ring -> cb_stat_handoff
// Everything stays a template parameter (rather than a runtime argument) so the pairwise
// walk keeps its compile-time trip count and the FLAT instantiation is the same code it was
// before D28.
//
// THE ONE THING AN INTERIOR NODE MUST NOT DO IS FINALIZE.  It packs the RAW sum and forwards
// it; only the LAST level (slot 0, the multicast root -- unique because f0 * f1 >=
// GROUP_SIZE) applies `*(1/W) + eps` and the rsqrt.  A finalize at an interior node would
// rsqrt a partial sum, and it would do it to a value the next level then adds to.
//
// AND THE TWO CALLS THAT ARE NOT OPTIONAL AT ANY FOLD SITE, new ones included:
// `reconfig_data_format` + `pack_reconfig_data_format`.  The preceding stage leaves the
// unpacker on the permute's (bf16 bank, fp32 handoff) pair or on pass A's cb_x_squared /
// cb_scaler (bf16), while every gather ring and every handoff here is fp32.  Without them
// the fold unpacks fp32 L1 through a bf16 srcA/srcB, the accumulated sum reads as ~0, and
// the finalize turns that into rsqrt(eps) -- a uniform ~1/sqrt(eps) SCALE error that HOLDS
// pcc at 0.9997 and shows up only in rel-RMS (measured 994 against a 0.04 bound during
// integration).  That is exactly why this op's regression nets bound rms and not just pcc.
template <
    uint32_t CB_IN,
    uint32_t SLOTS,
    uint32_t CB_OUT,
    bool FINALIZE,
    bool COMPACT_SCOPE,
    bool WIDE_SCOPE,
    uint32_t IW_BITS,
    uint32_t EP_BITS>
ALWI void combine_fold() {
    // AN EVEN WINDOW, AND IT IS CHEAPER THAN THE ALTERNATIVE -- MEASURED.  The pairwise walk
    // halves the window, so D22 rounds every ring up to an even slot count and the writer
    // boot-zeroes the one slot no sender writes (an exact +0.0).  The identity operand is not
    // strictly necessary: seeding DEST with a `copy_tile` (a ONE-operand accumulate) consumes
    // an ODD window with no pad at all, which deletes the whole `writer_gather_zero` stage.
    // That was BUILT AND MEASURED during D28's integration and it LOST, on both sides:
    //     (1,1,32,2304) WIDTH 9c  (odd GROUP_SIZE, flat root)  4442 -> 4610 ns   0.964x
    //     (1,1,32,7168) WIDTH 28c (odd f1, slot tree)          5873 -> 5841 ns   1.005x
    // i.e. the `copy_tile_init` + `add_tiles_init` pair inside the DEST window costs about
    // what the 314 ns boot zero it replaces costs, and MORE at the geometry where the pad was
    // the only thing being deleted.  So the even pad stays, and the reason is a number.
    constexpr uint32_t HALF = SLOTS / 2;
    static_assert(SLOTS % 2 == 0 && HALF >= 1, "rms_norm: the pairwise DEST walk needs an even, non-empty window");
    // The round's window is waited/popped ONCE: the pairwise walk addresses two tiles of
    // the same CB at a stride, which a per-tile wait cannot express.  Legal exactly as it
    // stands -- the writer publishes the round atomically (`cb_push_back(CB_IN, SLOTS)`)
    // and the CB is sized to that same window, which is also what keeps a remote sender's
    // locally-computed landing address equal to the gatherer's.
    cb_wait_front(CB_IN, SLOTS);
    reconfig_data_format(CB_IN, CB_IN);
    pack_reconfig_data_format(CB_OUT);
    add_tiles_init(CB_IN, CB_IN, /*acc_to_dest=*/true);
    if constexpr (FINALIZE) {
        // MANDATORY, not decorative: rms_stat_rsqrt_body reads sfpi::vConstIntPrgm0 /
        // vConstFloatPrgm1..2, which sfpu::rsqrt_init programs -- persistent SFPU PROGRAM
        // registers, which is what makes hoisting it out of a per-tile loop legal.
        rsqrt_tile_init();
    }
    tile_regs_acquire();
    for (uint32_t p = 0; p < HALF; ++p) {
        add_tiles(CB_IN, CB_IN, p, HALF + p, 0);
    }
    if constexpr (FINALIZE) {
        // The scope FOLLOWS the layout (see the note at stat_scale_col_full): at
        // BLOCK_ROWS == 1 the stat is a column-0 vector and D17's narrow <2,4> C walk is
        // right (and 1.39x cheaper); above it the stats span columns and the wide scope is
        // a CORRECTNESS requirement.  Two spellings, one predicate -- and it is the SAME
        // predicate on the tree path, because the tree changes WHO folds, never the layout
        // of what is folded.
        if constexpr (COMPACT_SCOPE) {
            compact_finalize_payload<IW_BITS, EP_BITS, WIDE_SCOPE>(0);
        } else {
            stat_finalize_payload<IW_BITS, EP_BITS>(0);
        }
    }
    tile_regs_commit();
    cb_reserve_back(CB_OUT, 1);
    tile_regs_wait();
    pack_tile(0, CB_OUT);
    tile_regs_release();
    cb_push_back(CB_OUT, 1);
    cb_pop_front(CB_IN, SLOTS);
}

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
    // Perf 3 (descriptor D28): the SLOT TREE's arity.  TREE_F0 == 0 means "keep the flat
    // root" and every tree body below is `if constexpr`-ed away, so a build the descriptor
    // did not select the tree for emits the same kernel it did before D28.
    constexpr uint32_t TREE_F0 = get_compile_time_arg_val(17);
    constexpr uint32_t TREE_F1 = get_compile_time_arg_val(18);

    const uint32_t num_rows = get_arg_val<uint32_t>(0);  // tile-rows owned by this core
    // Only the core holding the row's LAST width tile applies the partial-W
    // scaler/mask; 1 on the whole-row schemes.
    const uint32_t owns_last_w = get_arg_val<uint32_t>(1);
    const uint32_t is_root = get_arg_val<uint32_t>(2);  // group root: sums + finalizes
    // D28: this core's slot within its width group -- the ONLY thing that decides which
    // tree levels it folds at (level 0 iff my_slot % f0 == 0; level 1 iff my_slot == 0,
    // which is `is_root`).  Unread off the tree path.
    const uint32_t my_slot = get_arg_val<uint32_t>(3);

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
    const auto PARTIAL_SCALER =
        (PARTIAL_W == 0 || owns_last_w == 0)
            ? ckl::ReducePartialScaler::none()
            : (ACC_VIA_ADD ? ckl::ReducePartialScaler::only_partial() : ckl::ReducePartialScaler::with_partial());
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
    constexpr auto PASS_B_OUT_NORM =
        ckl::output(NORM_OUT, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    constexpr auto PASS_B_OUT_GAMMA =
        ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    constexpr bool CROSS_CORE = (COMBINE != 0);
    // Perf 2 (descriptor D22): the gather's slots -- GROUP_SIZE rounded UP TO EVEN, so the
    // root's fused pairwise fold always has a partner to halve against.  DERIVED, never
    // passed: a pure function of GROUP_SIZE that the writer derives identically, so the
    // landing layout has one definition per kernel and no CT arg can drift between them.
    // Equal to GROUP_SIZE at every even group (8 / 28 / 32, including the focus shape's 8);
    // the one extra slot at odd GROUP_SIZE is boot-zeroed by the writer and pairs against
    // the odd contributor as an exact +0.0.  The pad-free alternative (a `copy_tile` DEST
    // seed) was built and MEASURED SLOWER -- see the note at combine_fold.
    // (The half-stride itself now lives inside `combine_fold`, which derives it from the
    // window it is folding -- one definition for the flat root and both tree levels.)
    constexpr uint32_t GATHER_SLOTS = GROUP_SIZE + GROUP_SIZE % 2;
    // Perf 3 (descriptor D27) -- the COMPACT partial's two derived knobs.
    //
    // COMPACT_FIN_WIDE: which VectorMode the compact finalize needs.  A compact stat tile
    // carries BLOCK_ROWS stats in columns 0..BLOCK_ROWS-1; VectorMode::C reaches faces 0
    // and 2 == columns 0..15, RC reaches all 32.  A CORRECTNESS threshold, not a tuning
    // one -- see the scope note at the definitions of stat_scale_col_full / stat_scale_all.
    // A compact tile holds ONE tile-row's stat per COLUMN, and a tile has 32 columns, so
    // this is the structural ceiling on a combine row-block.  The descriptor caps its
    // BLOCK_ROWS solve at the same 32; asserted here so a future budget change that lifts
    // the cap fails at compile time instead of silently dropping rows past column 31 (which
    // it did, at pcc 0.949109 / rel-RMS 0.31, on (1,1,3232,96) WIDTH-sharded).
    static_assert(!CROSS_CORE || BLOCK_ROWS <= 32, "rms_norm: a compact combine block is at most 32 tile-rows");
    // THE ONE CARVE-OUT, and it is an IDENTITY, not a benchmark boundary.  At BLOCK_ROWS
    // == 1 a block has exactly one tile-row, so "permute the block's stats into columns
    // 0..BLOCK_ROWS-1" is `partial_0 x E_0`, which is the tile it started as: BOTH matmuls
    // are the identity map, and the compact tile IS the column-shaped partial.  So the
    // compact layout DEGENERATES into the flat one there -- same GATHER_SLOTS ring, same
    // one-whole-tile ship, same ONE DEST window in the fold, same one-page multicast -- and
    // all the permute pair can add is an extra L1 round trip (cb_sum_handoff ->
    // cb_compact_handoff on the way out, cb_mcast_in -> cb_row_final on the way back) that
    // no round has any latency left to hide, plus the reader's bank boot.
    //
    // MEASURED, whole op, one fresh-cache profiled run each, on the four pinned WIDTH-shard
    // geometries (all of which solve to BLOCK_ROWS == 1, num_blocks == 1):
    //   (1,1,32,1024) 8c   3724 -> 4880 ns   0.76x
    //   (1,1,32,2304) 9c   4527 -> 5644 ns   0.80x
    //   (1,1,32,5120) 32c  5406 -> 7119 ns   0.76x
    //   (1,1,32,7168) 28c  5724 -> 7509 ns   0.76x
    // i.e. a MATERIAL REGRESSION, not noise, and it is earned by the identity above rather
    // than by the shapes: the isolated bench DID measure BLOCK_ROWS == 1 as a win, but its
    // baseline still paid the gather's boot-zeroing, which D26 has since deleted from the
    // op -- so that credit was already banked and there was nothing left for the permute to
    // buy.  Everything from BLOCK_ROWS >= 2 up is on the compact path with no further
    // qualification (measured 1.11x-14.2x on the fold across BLOCK_ROWS 2..32 x GROUP_SIZE
    // 4..32, and flat is inside the domain).
    //
    // What this carve-out does NOT re-introduce: the D13 face-run gather.  The BLOCK_ROWS
    // == 1 path ships the partial as ONE WHOLE TILE too -- one transaction instead of two
    // face writes, and every landing byte defined.  The two paths differ ONLY by the elided
    // permute pair and by the finalize's lane scope.
    constexpr bool COMPACT = CROSS_CORE && (BLOCK_ROWS > 1);
    constexpr bool COMPACT_FIN_WIDE = (BLOCK_ROWS > 16);
    // ---- THE SLOT TREE's derived geometry (Perf 3 / D28) -----------------------------
    // TWO LEVELS of contiguous slot runs: level 0 folds runs of TREE_F0 slots on TREE_F1 =
    // ceil(GROUP_SIZE / F0) cores IN PARALLEL and forwards the RAW sums; level 1 folds those
    // TREE_F1 sums on slot 0 (the multicast root, unique because F0 * F1 >= GROUP_SIZE) and
    // finalizes.  So the root's fold drops from GROUP_SIZE tiles to TREE_F1, its L1 ingress
    // fan-in from GROUP_SIZE - 1 remote writes to TREE_F1 - 1, and every other core's fold
    // goes from nothing to TREE_F0 -- the work leaves the one core it was serialised on.
    //
    // ORTHOGONAL TO D27's COMPACT/IDENTITY split, and that is not an accident: what the tree
    // changes is WHICH CORE folds WHICH slots, never the LAYOUT of a page.  An interior
    // node's raw sum has exactly the shape of the compact (or column-shaped) partials it
    // summed, so the finalize's lane scope is decided by BLOCK_ROWS exactly as before and
    // the un-permute below is untouched.
    //
    // Every ring is rounded UP TO EVEN (D22's own trick) so every fold is a pairwise DEST
    // walk; a ragged run's missing slots and the evenness slot are boot-zeroed WHOLE by the
    // writer and pair against a real contributor as an exact +0.0 -- which is what makes one
    // code path cover odd, ragged and non-factorising group sizes with no guard.
    constexpr bool TREE = CROSS_CORE && (TREE_F0 != 0);
    constexpr uint32_t TREE_SL0 = TREE_F0 + TREE_F0 % 2;
    constexpr uint32_t TREE_SL1 = TREE_F1 + TREE_F1 % 2;
    static_assert(!TREE || TREE_F0 * TREE_F1 >= GROUP_SIZE, "rms_norm: the slot tree must cover GROUP_SIZE");
    static_assert(!TREE || TREE_F1 >= 2, "rms_norm: a slot-tree level that gathers one member is a hop, not a fold");
    // COMBINE_DEST_BATCH: DEST lanes the un-permute drives per window.  MEASURED optimum
    // (isolated bench perf_experiments/compact_partial_transpose_r3, `compute_recv_unpack`
    // ns at the op's pinned config): at BLOCK_ROWS 8, batch 1/2/4/8 = 1130/682/539/599 ns
    // (4 wins); at BLOCK_ROWS 32, 4143/2368/1548/1380 (8 wins).  Clamped to
    // DEST_AUTO_LIMIT and never a literal, because that cap is 8 lanes at
    // fp32_dest_acc_en=False but 4 at True -- the user's precision config, which this op
    // never touches, decides how many lanes exist.
    constexpr uint32_t COMBINE_DEST_BATCH_WANT = (BLOCK_ROWS <= 8) ? 4u : 8u;
    constexpr uint32_t COMBINE_DEST_BATCH = (COMBINE_DEST_BATCH_WANT < ckl::DEST_AUTO_LIMIT)
                                                ? COMBINE_DEST_BATCH_WANT
                                                : static_cast<uint32_t>(ckl::DEST_AUTO_LIMIT);
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
                    ckl::IterationShape::grid(rows, WT_CHUNK),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_A, X_IN_A, ckl::Dst::D0, SQ_OUT.dest_accumulation>{
                        hold_base, hold_base},
                    ckl::PackTile<SQ_OUT>{});
            }

            MaybeDeviceZoneScope("compute_reduce");
            rms_norm_local::accumulate_reduce_block<
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

    // ============ the COMBINE's compact partial transpose (Perf 3 / D27) ============
    //
    // WHAT CHANGES.  Pass A leaves this core's block as `rows` COLUMN-SHAPED partial tiles
    // (a REDUCE_ROW result lives in column 0).  This permutes them into `rows` COLUMNS of
    // ONE tile, so the whole block travels the combine as a SINGLE tile:
    //     C = partial_r x E_r,  E_r[0][r] = 1  ->  C[i][r] = partial_r[i][0]
    // and `compute_recv_unpack` below undoes it with the SAME bank read transposed.  The
    // bank is the reader's `reader_bank_boot` one-hot CB.
    //
    // WHAT IT BUYS, and it is four things at once, all MEASURED on the 64-core BLOCK shard
    // geometry (isolated bench perf_experiments/compact_partial_transpose_r3, blackhole
    // p150b 1350 MHz, at the op's pinned config -- bf16 / HiFi2 / fp32_dest_acc_en=False /
    // math_approx_mode=False, UNCHANGED; whole-combine device ns, GROUP_SIZE 8 /
    // BLOCK_ROWS 8 / 32 tile-rows per core, 34772 -> 10994 ns = 3.16x):
    //   fold       the root's D22 chain runs ONE DEST window per ROUND instead of one per
    //              TILE-ROW: 3024 -> 770 ns/round, and it is now FLAT in BLOCK_ROWS
    //              (777 ns at 16) where the flat fold was O(BLOCK_ROWS x GROUP_SIZE).
    //   gather     a member issues ONE whole-tile NoC write instead of BLOCK_ROWS
    //              face-runs (16 writes / 16 kB -> 1 write / 4 kB per round at the focus
    //              geometry): `writer_gather_ship` 1891 -> 1087 ns/round on a member.
    //   multicast  the root broadcasts ONE tile instead of BLOCK_ROWS: `writer_mcast_send`
    //              4133 -> 1147, `writer_mcast_recv` 6577 -> 1395 ns/round.
    //   L1         the landing ring loses its BLOCK_ROWS factor -- the combine's own CBs
    //              go 288 -> 88 kB/core at the focus geometry and 1056 -> 184 kB at
    //              GROUP_SIZE 32, which is what lets the descriptor's L1-bound BLOCK_ROWS
    //              solve take a coarse block at all (the flat ring is 1152 kB at
    //              BLOCK_ROWS 32 -- a measured L1 OOM).
    // The cost is this pack plus the un-pack, and it is paid IN PARALLEL ON EVERY CORE
    // (+219 / +940 ns/round on the root's timeline) against work that used to be the
    // root's alone -- the group ends up balanced, root 32-34 us -> ~10 us with the members
    // going from idle to ~10 us.  Zero cells below 1.00x across BLOCK_ROWS {1,2,4,8,16,32}
    // x GROUP_SIZE {4,8,9,28,32}, including two RAGGED configs.  BLOCK_ROWS == 1 is a
    // literal no-op in the fold (one packed column IS column 0) and measured FLAT there,
    // which is why there is no BLOCK_ROWS guard: flat is inside the domain.
    //
    // PRECISION, stated plainly and not hidden: the two permutation matmuls round each
    // value through a 16-bit DEST word twice more at fp32_dest_acc_en=False, so the
    // combine is slightly LESS accurate -- rel-RMS 0.00383 vs 0.00227 at the focus, pcc
    // 0.9999931 vs 0.9999978.  That is four orders inside this op's 0.04 rel-RMS bound and
    // four nines inside the 0.9995 pcc gate, so the user's precision CONTRACT is untouched
    // (nothing here reads or changes fp32_dest_acc_en / math_fidelity / math_approx_mode /
    // a dtype).  There is no fp32 path through DEST at that config, and changing the
    // config to get one would be exactly the forbidden move.
    //
    // RAW-LLK / RAW-API JUSTIFICATION.  A COLUMN PERMUTATION has no kernel_lib expression:
    // the eltwise / bcast / reduce families all preserve or collapse the column axis, and
    // `transpose_wh` transposes the WHOLE tile, which is a different map.  The FPU's only
    // horizontal-mixing primitive is the matmul, so `matmul_tiles` against a one-hot bank
    // IS the operation -- with `matmul_init`'s srcB `transpose` flag reading E_r as E_r^T
    // so ONE bank serves both directions.  DEST accumulation is free here: `matmul_tiles`
    // is DST += A*B and `tile_regs_release` clears DST (the packer's ZEROACC), so `rows`
    // matmuls into one DEST slot cost ONE pack and need no explicit zero seed -- measured
    // 1.1x faster on the pack and 2.2-2.5x on the un-pack than seeding with a zero-tile
    // copy, for a BIT-IDENTICAL result.
    //
    // SAFETY INVARIANT any later change must preserve: a matmul sums 32 products, so EVERY
    // column of BOTH operands must be FINITE -- an inf/NaN in an unused column becomes
    // inf*0 = NaN and poisons column 0.  Two things guarantee that here, and both are
    // load-bearing.  (1) A compact page is shipped WHOLE, out of a fully-defined
    // `pack_tile`, so no landing column is ever un-written L1 -- which is also why D26's
    // face-zeroing deletion has nothing left to delete on this path and why the gather can
    // never go back to shipping a face subset.  (2) The finalized compact stat's UNUSED
    // columns (rows..31) hold rsqrt(0 * 1/W + eps) = 1/sqrt(eps), which is finite for
    // every eps > 0.  At eps == 0 exactly they would be +inf and the un-permute would
    // return NaN everywhere; the flat path degraded only on an all-zero row there.  eps is
    // a user argument with no axis in the op's SUPPORTED rectangle and a 1e-6 default; the
    // whole test suite runs 1e-12 .. 1e-2.  If eps == 0 ever needs supporting, the fix is
    // to clamp the finalize's additive term, not to widen the scope.
    auto member_pack = [&](uint32_t rows) {
        // Compile the body only where the permute is not the identity: cb_bank /
        // cb_compact_handoff are not allocated otherwise, and an uncalled-but-emitted zone
        // would report a phantom stage.  This is also what makes the BLOCK_ROWS == 1
        // carve-out a single predicate rather than a condition at every call site.
        if constexpr (!COMPACT) {
            (void)rows;
            return;
        } else {
            MaybeDeviceZoneScope("compute_member_pack");
            cb_wait_front(cb_sum_handoff, rows);
            cb_reserve_back(cb_compact_handoff, 1);
            // NOT optional, for the same reason D22's fold spells its reconfigs out (a missing
            // one there was a uniform ~1000x scale error that HELD pcc at 0.9997 and showed
            // only in rel-RMS): pass A leaves the unpacker on cb_x_squared / cb_scaler (bf16)
            // and the packer on cb_sum_handoff, and `matmul_init` does NOT reconfigure formats
            // (its `state_configure` is the debug sentinel, not a reconfig).  SrcOrder::Reverse
            // because matmul maps in0 -> SrcB and in1 -> SrcA, so the operands are passed in
            // the same natural order as to `matmul_tiles` and the helper does the swap.
            reconfig_data_format<ckernel::SrcOrder::Reverse>(cb_sum_handoff, cb_bank);
            pack_reconfig_data_format(cb_compact_handoff);
            matmul_init(cb_sum_handoff, cb_bank, /*transpose=*/0);
            tile_regs_acquire();
            for (uint32_t r = 0; r < rows; ++r) {
                matmul_tiles(cb_sum_handoff, cb_bank, r, r, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_compact_handoff);
            tile_regs_release();
            cb_push_back(cb_compact_handoff, 1);
            cb_pop_front(cb_sum_handoff, rows);
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
            // D27 x D25 ORDERING, and it is the whole point of the pipeline: block blk's
            // PERMUTE is issued BEFORE block blk+1's hoisted pass A.  The writer's ship for
            // block blk waits on cb_compact_handoff, so permuting first puts every member's
            // partial on the wire while pass A for blk+1 runs -- which is what leaves the
            // root's gather wait overlapping independent work.  Permuting after the hoist
            // would delay every member's ship by a whole pass A and re-serialize exactly
            // the latency D25 exists to hide.  Legal in this order because pass A for block
            // blk already ran (the prologue for blk == 0, the previous iteration after).
            member_pack(rows);
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
            // Off the pipeline (D25's carved-out reader-fed combine) pass A for THIS block
            // has only just produced the partials, so the permute has to follow it.
            if constexpr (CROSS_CORE) {
                member_pack(rows);
            }
        }

        // ================= finalize: 1/rms = rsqrt(sum/W + eps) ============
        // Pops before reserving, so the `rows`-page accumulator CB suffices.
        if constexpr (!CROSS_CORE) {
            MaybeDeviceZoneScope("compute_finalize");
            for (uint32_t i = 0; i < rows; ++i) {
                rms_norm_local::transform_in_place(cb_row_stat, finalize);
            }
        } else {
            // Pass A's reduce packs this core's raw per-row partials into cb_sum_handoff
            // (D18) and `member_pack` above has already permuted them into ONE compact tile
            // in cb_compact_handoff, which is what the writer ships to the group root.  The
            // `compute_partial_handoff` zone that used to sit here is retired with the copy
            // it measured.
            // ======== THE SLOT TREE's interior fold (Perf 3, descriptor D28) ==========
            // A core folds the level-0 run it gathers -- a run of TREE_F0 slots, one of
            // TREE_F1 runs, all folded IN PARALLEL on TREE_F1 different cores -- and packs
            // the RAW sum, WITHOUT finalizing, for the writer to forward to the root.  The
            // root then folds only TREE_F1 pages instead of GROUP_SIZE (below).
            //
            // WHY THIS IS WORTH A NoC HOP, and where it stops being worth one: the flat root
            // is the ONE core that pays both per-GROUP_SIZE terms -- GROUP_SIZE - 1 remote
            // writes serialising into its L1 ingress, and a GROUP_SIZE-tile fold -- while
            // every other core in the group idles.  The tree caps both at max(f0, f1) and
            // spends the idle cores.  MEASURED (isolated bench
            // perf_experiments/slot_tree_gather, blackhole p150b 1350 MHz, whole-combine
            // device ns, one fresh-cache profiled run per variant, at the op's pinned
            // config; f0 = 4 is itself measured -- see COMBINE_TREE_F0):
            //     GROUP_SIZE 32, 1 page/sender/round   flat 5424 -> 3744   1.45x
            //     GROUP_SIZE 28, 1 page/sender/round   flat 5007 -> 3576   1.40x
            //     GROUP_SIZE 32, 4 rounds              flat 13788 -> 11036 1.25x
            //     GROUP_SIZE 16, 4 rounds              flat 9741 -> 10410  0.94x  REGRESSION
            //     GROUP_SIZE  8, 4 rounds              flat 7007 -> 9174   0.76x  REGRESSION
            // The descriptor's ONE predicate (`_combine_tree_arity`) is what keeps the op off
            // the last two, and it is a threshold on the deleted fold-tiles, not on a shape.
            //
            // MORE ACCURATE, not less -- the same mechanism D22 recorded against D16: a
            // deeper pairwise DEST tree shortens the error chain.  Measured rel-RMS
            // 0.00213 (tree) vs 0.00250 (flat) at GROUP_SIZE 32 and 0.00292 vs 0.00336 at
            // GROUP_SIZE 28, at IDENTICAL pcc-or-better.  Nothing here reads or changes
            // fp32_dest_acc_en / math_fidelity / math_approx_mode / a dtype.
            if constexpr (TREE) {
                if (my_slot % TREE_F0 == 0) {
                    MaybeDeviceZoneScope("compute_tree_fold_l0");
                    // FINALIZE=false is the load-bearing half: an interior node must forward
                    // the RAW sum.  A finalize here would rsqrt a partial sum -- and then the
                    // root would ADD rsqrt'd values together.
                    combine_fold<
                        cb_partials_gathered,
                        TREE_SL0,
                        cb_node_out,
                        /*FINALIZE=*/false,
                        COMPACT,
                        COMPACT_FIN_WIDE,
                        INV_W_BITS,
                        EPS_BITS>();
                }
            }

            if (is_root != 0) {
                // Sum the group's GROUP_SIZE partials ELEMENTWISE and finalize the whole
                // BLOCK, in ONE DEST WINDOW.  Each landing page is a COMPACT partial whose
                // columns 0..rows-1 are that sender's per-tile-row sums (D27), so the
                // block's row totals are the elementwise sum of the group's pages -- one
                // pairwise walk, one finalize, one pack, INDEPENDENT of BLOCK_ROWS.
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
                // The finalize is the op's raw-sfpi StatFinalize body (D17) at the SCOPE the
                // compact layout requires -- <STRIDE=1, ITERS=8>, VectorMode::C up to
                // BLOCK_ROWS 16 and RC above (COMPACT_FIN_WIDE).  THAT SCOPE IS A
                // CORRECTNESS REQUIREMENT, not a tuning choice, and it is the ONE thing D27
                // had to change in this chain: the stats no longer live in column 0 alone,
                // so D17's even-parity <2,4> walk (columns 0,2,..,14) would leave every ODD
                // tile-row's sum unscaled and un-rsqrt-ed -- measured pcc 0.9972987 with
                // rel-RMS 1036 against a 0.04 bound, and 1.39x FASTER, i.e. precisely the
                // kind of "win" that has to be refused.  The full note (and why D17's narrow
                // scope stays right for the LOCAL finalize above, which really does own only
                // column 0) is at stat_scale_col_full's definition.  Both bodies stay at the
                // same <1,8>, so the +eps guard still covers every lane the rsqrt touches --
                // no rsqrt(0) = inf on an all-zero row.  The FPU has no lane scope, so the
                // DEST accumulation fills the whole tile and the finalize sees completed
                // sums on every lane it visits.
                //
                // GATHER_SLOTS (== GROUP_SIZE rounded up to even) is what makes the
                // pairwise walk universal: at odd GROUP_SIZE the writer boot-zeroes the ONE
                // pad slot (D27 shrank it from GATHER_SLOTS * BLOCK_ROWS pages to one),
                // which pairs against the odd contributor and adds an exact +0.0.  So there
                // is no odd/even code path and no GROUP_SIZE guard.
                //
                // D28 lifts the whole chain into `combine_fold` (definition above, with the
                // two mandatory reconfigs and the finalize-scope predicate) so the tree's
                // interior fold and this one are ONE implementation.  What the tree changes
                // here is only WHICH RING the root reads and HOW MANY pages are in it: the
                // level-1 ring of TREE_SL1 forwarded sums instead of GATHER_SLOTS partials.
                MaybeDeviceZoneScope("compute_root_fused");
#if defined(RMS_ABLATE_ROOT_SUM) || defined(RMS_ABLATE_ROOT_FINALIZE)
                // ABLATION (temporary, /perf-measure): payload removed, every CB handshake
                // and trip count preserved.  Under D27 the round's handshake is ONE window
                // in and ONE page out, whatever BLOCK_ROWS is -- peel recipe: uncomment
                // RMS_ABLATE_ROOT_SUM at the head of this file (and, on the writer,
                // RMS_ABLATE_GATHER_ZERO) and diff the profiled zones against the
                // unablated run; the difference is that stage's WALL contribution, which is
                // what makes the cumulative peel additive.  D28: the ROOT's window is the
                // level-1 ring when the tree is built, and `compute_tree_fold_l0` above
                // peels with the same pair of switches.
                cb_wait_front(TREE ? cb_gather_l1 : cb_partials_gathered, TREE ? TREE_SL1 : GATHER_SLOTS);
                cb_reserve_back(cb_stat_handoff, 1);
                cb_push_back(cb_stat_handoff, 1);
                cb_pop_front(TREE ? cb_gather_l1 : cb_partials_gathered, TREE ? TREE_SL1 : GATHER_SLOTS);
#else
                if constexpr (TREE) {
                    combine_fold<
                        cb_gather_l1,
                        TREE_SL1,
                        cb_stat_handoff,
                        /*FINALIZE=*/true,
                        COMPACT,
                        COMPACT_FIN_WIDE,
                        INV_W_BITS,
                        EPS_BITS>();
                } else {
                    combine_fold<
                        cb_partials_gathered,
                        GATHER_SLOTS,
                        cb_stat_handoff,
                        /*FINALIZE=*/true,
                        COMPACT,
                        COMPACT_FIN_WIDE,
                        INV_W_BITS,
                        EPS_BITS>();
                }
#endif
            }

            // ---- every core: UN-PERMUTE the multicast compact stat (D27) ------------
            // C = compact x E_r^T, read straight out of the SAME one-hot bank via
            // matmul_init's srcB `transpose` flag (E_r^T[r][0] = 1), so page r of the bank
            // both WROTE column r on the way out and READS it on the way back and there is
            // only one constant in L1.  Output: `rows` column-shaped 1/rms tiles in
            // cb_row_final -- byte-for-byte the operand shape pass B already consumed
            // (BroadcastDim::Col reads column 0), so pass B is UNCHANGED by D27.
            //
            // cb_row_final therefore becomes COMPUTE-PRIVATE (this pack is its only
            // producer, pass B its only consumer); the multicast now lands in cb_mcast_in.
            //
            // DEST-batched at COMBINE_DEST_BATCH lanes so one MATH<->PACK handshake and one
            // format reconfig amortize over several tiles -- measured 1130 -> 539 ns at
            // BLOCK_ROWS 8 going from 1 lane to 4.  See COMBINE_DEST_BATCH's definition.
            // Elided at BLOCK_ROWS == 1: the multicast then lands straight in cb_row_final
            // (an identity un-permute is nothing to do), exactly as it did before D27.
            if constexpr (COMPACT) {
                MaybeDeviceZoneScope("compute_recv_unpack");
                cb_wait_front(cb_mcast_in, 1);
                cb_reserve_back(cb_row_final, rows);
                reconfig_data_format<ckernel::SrcOrder::Reverse>(cb_mcast_in, cb_bank);
                pack_reconfig_data_format(cb_row_final);
                matmul_init(cb_mcast_in, cb_bank, /*transpose=*/1);
                for (uint32_t b = 0; b < rows; b += COMBINE_DEST_BATCH) {
                    const uint32_t n = (rows - b < COMBINE_DEST_BATCH) ? (rows - b) : COMBINE_DEST_BATCH;
                    tile_regs_acquire();
                    for (uint32_t d = 0; d < n; ++d) {
                        matmul_tiles(cb_mcast_in, cb_bank, 0, b + d, d);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t d = 0; d < n; ++d) {
                        pack_tile(d, cb_row_final, b + d);
                    }
                    tile_regs_release();
                }
                cb_push_back(cb_row_final, rows);
                cb_pop_front(cb_mcast_in, 1);
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
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        X_IN_B,
                        ckl::input(
                            CB_STAT_B,
                            ckl::BroadcastDim::Col,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::None,
                            ckl::OperandKind::Col)>{hold_base},
                    ckl::PackTile<PASS_B_OUT_NORM>{});
            }

            if constexpr (HAS_G) {
                // gamma is row-shaped (1 x W, valid in row 0) -> broadcasts DOWN
                // rows (BroadcastDim::Row), indexed by column (OperandKind::Row).
                MaybeDeviceZoneScope("compute_gamma_mul");
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            cb_normalized, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                        ckl::input(G_IN, ckl::BroadcastDim::Row)>{0u, hold_base},
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
