// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for rms_norm_ttnn (UNPACK / MATH / PACK).
//
//   t   = x + r                                     (A1, optional)
//   out = t * rsqrt( (1/W) * sum_w t^2 + eps ) * w + b   (w, b optional)
//
// One loop nest covers ALL THREE regimes and every operand combination.  Per
// row-block (bracketed stages are compile-time-elided when absent, so an
// operand-free build is the seed's kernel):
//
//   pass A   [RM]      tilize             cb_input_sticks    -> cb_input_tiles
//            [RM,R]    tilize             cb_residual_sticks -> cb_residual_tiles
//            [R]       residual_add_block cb_input_tiles + cb_residual_tiles
//                                                            -> cb_x_sum
//                      square             CB_T               -> cb_x_squared
//                      accumulate_reduce  cb_x_squared       -> CB_REDUCE_ACC
//   finalize           transform_in_place cb_row_stat        -> cb_row_stat
//   pass B             mul<Col>           CB_T, CB_STAT_B    -> NORM_OUT
//            [G]       mul<Row>           cb_normalized, cb_gamma_tiles
//                                            -> cb_normalized (in place, if [B])
//                                            -> cb_output_tiles (otherwise)
//            [B]       add<Row>           cb_normalized, cb_bias_tiles
//                                                            -> cb_output_tiles
//            [RM]      untilize           cb_output_tiles    -> cb_output_sticks
//
// CB_T is the tensor the statistics and pass B operate on: cb_x_sum when a
// residual is present (which is why cb_x_sum, not cb_input_tiles, is the HELD
// buffer there), cb_input_tiles otherwise.  Pass-B routing, stated once: with
// S = [scale if G] + [bias if B], `normalize` writes cb_output_tiles when S is
// empty, else cb_normalized; every stage in S but the LAST writes cb_normalized
// in place, and the last writes cb_output_tiles.  At S = [scale] or S = [] that
// collapses to exactly the seed's routing.
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
// The regimes differ ONLY in whether CB_T / cb_gamma_tiles / cb_bias_tiles are
// held across both passes, and how wide they are (X_RESIDENT / X_HOLD_WT, from
// the descriptor -- deviation D14):
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
// Explicit cb_pop_front calls on CB_T / cb_row_stat / cb_gamma_tiles /
// cb_bias_tiles / cb_scaler are the sanctioned pattern for operands whose
// lifetime spans more calls than any single PopPolicy can express
// (op_design.md section 6.1).
//
// The ONE in-place site in this kernel is `scale_block` when a bias follows it
// (D29).  Its lifecycle pair is not a free choice: an in-place chain needs an
// incrementally POPPING input and an incrementally RESERVING output, so both
// sides are PerBlockSize -- the device-verified case 1 of
// kernel_lib/tests/eltwise/chain/lifecycle/inplace_chain.cpp.  An upfront-reserve
// output on an aliased CB DEADLOCKS rather than returning a wrong answer, and a
// Row/Col operand may never be the aliased CB (chain.inl:82-85) -- which is why
// cb_x_sum, the HELD Upfront/None srcA, is never aliased even though the
// design's CB table suggests it.

#include <cstdint>

// ---- ABLATION SWITCHES (/perf-measure cumulative peel) ---------------------
// `RMS_ABLATE_<STAGE>` strips that stage's PAYLOAD while keeping every CB
// handshake, barrier, loop trip count and zone.  Perf measurement only -- the op
// is WRONG with any of them on.
//
// They are DEFINES SUPPLIED BY THE HOST, from the `RMS_ABLATE` env var:
//
//     RMS_ABLATE=READ_X,WRITE scripts/tt-probe.sh rms_norm_ttnn < bench.py
//
// and NOT `#define`s to uncomment here.  Perf 3 measured why: the JIT kernel
// cache key does not include the source's CONTENT, so an in-place edit is a
// CACHE HIT on the previously compiled binary -- a "clean baseline" reproduced
// twice at 56,090 ns with pcc=nan when the truth was 84,510 ns, because it was
// still the all-stubbed build.  A define is part of the key.  See
// `_kernel_defines()` in the program descriptor for the one source of truth.

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
// ckl::Square -- the DEST-only SFPU square Lamp L-RES-FUSE's fused pass-A chain
// applies to the residual sum without unpacking it back out of cb_x_sum.
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"
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

// The SMALLEST divisor of `wt` that is >= 2 and <= `cap` -- pass B's DEST-lane block size
// when the core owns exactly ONE tile-row (Perf 1).  See PASS_B_AUTO below for why the
// two rules exist and which one a block gets.
constexpr uint32_t pass_b_blk_small(uint32_t wt, uint32_t cap) {
    for (uint32_t b = 2; b <= cap && b <= wt; ++b) {
        if ((wt % b) == 0) {
            return b;
        }
    }
    return pass_b_blk(wt, cap);
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
// A1 / A2 -- allocated only under their HAS_* flag, at NEW slots so an
// operand-free build is byte-identical to the seed's.
constexpr uint32_t cb_residual_sticks = 19;
constexpr uint32_t cb_residual_tiles = 20;
constexpr uint32_t cb_x_sum = 21;  // t = x + r; takes over cb_input_tiles' HELD role
constexpr uint32_t cb_bias_sticks = 22;
constexpr uint32_t cb_bias_tiles = 23;
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
    static_assert(SLOTS % 2 == 0 && HALF >= 1, "rms_norm_ttnn: the pairwise DEST walk needs an even, non-empty window");
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
    // ---- compile-time knobs (all from rms_norm_ttnn_program_descriptor.py) -----
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
    // ---- A1 / A2 / A5: appended, so an operand-free build is the seed's -------
    // A2: the per-channel SHIFT, applied AFTER the scale.  Its own CB pair and its
    // own data format (the two per-channel operands share a layout, not a dtype).
    constexpr uint32_t HAS_BIAS = get_compile_time_arg_val(19);
    // A1: the residual add runs BEFORE the statistics, so the tensor this kernel
    // squares and normalizes is `t = x + r`, materialized once into cb_x_sum.
    constexpr uint32_t HAS_RESIDUAL = get_compile_time_arg_val(20);
    // A5: the caller's `program_config.subblock_w`, or 0 for "the op's own
    // choice" -- which is exactly the seed's pass_b_blk(WT_CHUNK, DEST_AUTO_LIMIT).
    constexpr uint32_t PASS_B_BLK_CT = get_compile_time_arg_val(21);
    // D30: the reader staged each ROW_MAJOR per-channel operand one tile COLUMN per
    // page instead of one WT_CHUNK-wide block, so the tilize walks WT_CHUNK
    // one-tile blocks.  Same tiles bit-for-bit; the descriptor takes it only when
    // the L1 budget asks (a per-channel staging ring is WT_CHUNK whole tiles of L1
    // to carry ONE stick, which is the biggest single term on a wide band).
    constexpr uint32_t NARROW_PC_STAGE = get_compile_time_arg_val(22);
    // Refinement 2 / LAMP L-FIN: WHERE THE FINALIZE RUNS.
    //   0  ROOT   -- the last-level fold FUSES `*(1/W) + eps` and the rsqrt into its own
    //                DEST window (D22) and the multicast carries the FINALIZED stat.
    //   1  SPREAD -- the root packs the RAW group sum, the multicast carries THAT, and
    //                every core finalizes its own copy in `spread_finalize` below.
    // Correct either way; the descriptor's `_combine_fin_spread` owns the choice and
    // carries the measurement.  Off the combine path this is dead (there is no root).
    constexpr uint32_t FIN_SPREAD_CT = get_compile_time_arg_val(23);

    // ---- Refinement 3: pass A's two knobs -----------------------------------
    // PASS_A_SQ_BLOCK_CT: run pass A's `square` at pass B's DEST-LANE BLOCK SIZE
    // (`PASS_B_BLK`) instead of the per-tile spelling, so one per-element init, one
    // format reconfig and one CB reserve/push cover PASS_B_BLK tiles.  That is D21's
    // measured 1.28-1.66x lever applied to the one chain in the kernel that never got
    // it.  0 == the seed's per-tile chain, byte-identical.
    constexpr uint32_t PASS_A_SQ_BLOCK_CT = get_compile_time_arg_val(24);
    // RES_FUSE_CT: Lamp L-RES-FUSE.  1 == pass A runs `t = x + r` and its square as ONE
    // chain -- Add -> PackTile(cb_x_sum) -> Square(DEST) -> PackTile(cb_x_squared) --
    // so the square takes the sum out of DEST instead of unpacking cb_x_sum back in.
    // cb_x_sum is STILL materialized (every RESIDENT regime's pass B reads it, and
    // STREAM's pass B rebuilds it with the unfused chain), so there is no L1 change.
    constexpr uint32_t RES_FUSE_CT = get_compile_time_arg_val(25);
    // D39: the per-channel TILE CB is a CHUNKED window (WT_CHUNK pages,
    // popped after every chunk) rather than a whole row held for the core's life.
    // This WAS `!X_RESIDENT`; the reader's compact per-channel cache decouples the
    // two, so a ROW_RESIDENT build can hold x while chunking gamma/bias.
    constexpr uint32_t PC_CHUNK_CT = get_compile_time_arg_val(26);

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
    // A2 / A1.  The two per-channel operands SHARE a layout by contract, so one
    // flag drives both readers (PC_RM keeps the seed's `G_RM` meaning at its own
    // CT index; a bias-only call is a per-channel call too).
    constexpr bool HAS_B = (HAS_BIAS != 0);
    constexpr bool HAS_R = (HAS_RESIDUAL != 0);
    constexpr bool G_RM = (GAMMA_IS_RM != 0);
    constexpr bool PC_RM = G_RM;
    constexpr bool PC_NARROW = (NARROW_PC_STAGE != 0);
    // ---- A1: WHICH CB CARRIES `t` ------------------------------------------
    // The statistics are taken over `t = x + r`, never over x alone, so with a
    // residual present the tensor pass A squares and pass B normalizes is
    // cb_x_sum -- and cb_x_sum therefore takes over cb_input_tiles' HELD role
    // (the descriptor sizes it at `x_hold_wt` for exactly that reason) while
    // cb_input_tiles and cb_residual_tiles become STREAMS that
    // `residual_add_block` pops.  Without a residual CB_T *is* cb_input_tiles and
    // every operand spec below is the seed's, character for character.
    //
    // Materializing `t` rather than fusing the add into pass B's chain is not a
    // convenience: a second BinaryFpu reads CBs (not DEST) and DestReuseBinary
    // carries no broadcast parameter, so `(x + r) * stat<Col>` cannot share one
    // DEST window (chain.hpp:526).  Materializing it once additionally makes it
    // available to BOTH passes, which is what keeps the resident regimes from
    // re-reading two activations in pass B instead of one.
    constexpr uint32_t CB_T = HAS_R ? cb_x_sum : cb_input_tiles;
    // X_RESIDENT == GAMMA_RESIDENT, from the descriptor's regime decision (D14).
    constexpr bool X_RESIDENT = (X_RES != 0);
    static_assert(NUM_W_CHUNKS > 1 || X_RESIDENT, "rms_norm_ttnn: a one-chunk width is resident by definition");
    // Lamp L5's regime: resident x/gamma, chunked derived CBs.  The two held CBs
    // then span the WHOLE tile-row while every helper call still works on one
    // WT_CHUNK, so each call indexes them at a TILE OFFSET (TileOffset::Set) and
    // neither is popped until the row-block is done.
    constexpr bool ROW_RESIDENT = X_RESIDENT && (NUM_W_CHUNKS > 1);
    constexpr bool PC_CHUNKED = (PC_CHUNK_CT != 0);
    static_assert(PC_CHUNKED || X_RESIDENT, "rms_norm_ttnn: a streamed per-channel ring is chunked by definition");
    // The compact cache is a TILE-operand mechanism; a ROW_MAJOR operand is staged
    // through cb_*_sticks and tilized, which the resident boot does once per core.
    static_assert(
        !PC_CHUNKED || !X_RESIDENT || !PC_RM, "rms_norm_ttnn: a chunked resident per-channel ring is TILE-only");
    static_assert(!ROW_RESIDENT || BLOCK_ROWS == 1, "rms_norm_ttnn: ROW_RESIDENT holds ONE tile-row of x");
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
    // A1 CARVE-OUT, and it is a CORRECTNESS one.  D25's pipeline issues block
    // blk+1's pass A at a TILE OFFSET past a front block blk's pass B has not
    // popped yet, which is legal only because a shard-backed cb_input_tiles holds
    // the WHOLE assignment so its front never wraps.  With a residual the hoisted
    // pass A also writes cb_x_sum, whose ring is ONE block deep and whose front
    // block blk's pass B still owns -- so the hoist would either overwrite live
    // data or self-deadlock on the reserve.  Making it legal needs cb_x_sum at
    // depth 2 AND a runtime output base on the pack, and `output(...)` carries no
    // tile base, so it is not expressible without a new chain seam.
    //
    // Gated OFF rather than reworked: it costs the residual configurations the
    // pipeline's overlap on the combine path (recorded as a follow-up -- the
    // measurement to take is cb_x_sum at depth 2 with a pack-side base against
    // the serial order), and it leaves every operand-free build byte-identical.
    constexpr bool PIPE_A = (NATIVE_IN != 0) && (COMBINE != 0) && !HAS_R;
    // Pass A's x operand needs a RUNTIME tile base once it can run ahead of the front.
    // Compile-time-elided (hence byte-identical to Refinement 4) when PIPE_A is off.
    constexpr auto AOFF = PIPE_A ? ckl::TileOffset::Set : XOFF;

    // srcA at boot is whichever CB the first helper unpacks from.
    constexpr uint32_t CB_A = RM ? cb_input_sticks : cb_input_tiles;
    compute_kernel_hw_startup(CB_A, cb_scaler, cb_output_tiles);

    // ==== Refinement 3 / lever 1: DATA-FORMAT RECONFIG, one named constant per
    // ==== boundary ==========================================================
    //
    // Every chain in this kernel emits its dtype reconfig ONCE per CALL (the
    // eltwise_chain fold is boot-hoisted -- chain.inl `emit_pre_element_transitions`
    // runs in the one-time setup, not per tile), so the cost is
    // `stages x num_blocks x NUM_W_CHUNKS` per core, not `stages x tiles`.  The
    // constants below exist so that (a) the boundaries are NAMED rather than a
    // repeated literal, and (b) the ablation switch above can strip all of them in
    // one place to bound what the lever can ever be worth.  RMS_ABLATE_RECONFIG is a
    // MEASUREMENT-ONLY build: with the formats not programmed the op is numerically
    // wrong wherever any two of the CBs differ, which is exactly the set of builds
    // the elision predicate would have to exclude.
    //
    // The shipped default is Enabled on every boundary.  See the changelog for the
    // measured ablation number that justifies leaving it there.
#ifdef RMS_ABLATE_RECONFIG
    constexpr auto DFR = ckl::DataFormatReconfig::Disabled;
    constexpr auto REDUCE_RECONFIG = ckl::ReduceDataFormatReconfigMode::NONE;
#else
    constexpr auto DFR = ckl::DataFormatReconfig::Enabled;
    constexpr auto REDUCE_RECONFIG = ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT;
#endif

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
        "rms_norm_ttnn: ReduceAlgorithm::AccumulateViaAdd + Accumulate requires BulkWaitBulkPop (REDUCE_BULK == 1)");

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
    // D43 (Perf 3) -- THE GROUPED FOLD.  `X_SQUARED_WT` is now any DIVISOR of
    // WT_CHUNK, and `SQ_GROUP = WT_CHUNK / X_SQUARED_WT` is how many width tiles
    // accumulate into ONE DEST slot before a pack.  SQ_GROUP is therefore BOTH the
    // pack/unpack saving (SQ_GROUP : 1) AND the serial 16-bit accumulation depth that
    // the descriptor's DEST_ACC_SQUARE_MAX_WT ceiling exists to bound -- and D43's
    // whole point is that the two are now decoupled from WT_CHUNK.  A chunk of 32 can
    // fold in groups of 16 and still delete 15 of every 16 packs.
    //   X_SQUARED_WT == 1           the FLAT fold (SQ_GROUP == WT_CHUNK), pre-D43
    //   1 < X_SQUARED_WT < WT_CHUNK the GROUPED fold
    //   X_SQUARED_WT == WT_CHUNK    the PACKED path (SQ_GROUP == 1, no fold)
    constexpr uint32_t SQ_GROUP = WT_CHUNK / X_SQUARED_WT;
    constexpr bool SQ_FOLD = (SQ_GROUP > 1);
    static_assert(
        X_SQUARED_WT >= 1 && WT_CHUNK % X_SQUARED_WT == 0,
        "rms_norm_ttnn: X_SQUARED_WT must be a divisor of WT_CHUNK (1 == the flat DEST fold, D43)");
    static_assert(
        !SQ_FOLD || PARTIAL_W == 0,
        "rms_norm_ttnn: the DEST fold folds the last width tile's pad lanes in BEFORE the "
        "reduce, so the reduce's partial scaler / mask can no longer reach them");
    // Lamp L-RES-FUSE (R3).  Three gates, and the THIRD is the lamp's real boundary:
    //   * HAS_R          -- there is nothing to fuse without a residual;
    //   * !SQ_FOLD       -- the D12 fold's DestAccumulation::PerRow keeps D0 live across
    //                       the whole tile-row, so there is no per-tile DEST value left
    //                       to square;
    //   * !X_RESIDENT    -- MEASURED, not argued.  The lamp's premise is "pass A only
    //                       squares t -- it does not need t to survive", and that is true
    //                       ONLY in STREAM, where pass B rebuilds `t` from the re-read x
    //                       and r.  In both RESIDENT regimes cb_x_sum IS the tensor pass B
    //                       normalizes, so `t` has to be packed as well as squared -- and
    //                       an eltwise_chain CANNOT publish an intermediate DEST value:
    //                       Pack is its OWN COHORT, disjoint from math-MOP/SFPU
    //                       (chain.inl `elem_pack_init`), so every pack in a chain runs
    //                       AFTER every compute element.  A
    //                       `Add -> PackTile(cb_x_sum) -> Square -> PackTile(cb_x_squared)`
    //                       chain therefore writes the SQUARE into cb_x_sum too, and pass B
    //                       normalizes t^2 instead of t.  It is not a subtle error: built
    //                       and measured at pcc 0.260 on (1,1,8192,5120) ROW_RESIDENT
    //                       gamma_bias_residual (and 0.947x, so it was not even faster).
    //                       Publishing `t` and squaring it in one DEST window needs a
    //                       DEST->DEST copy element the chain does not expose; that is a
    //                       helper gap, recorded here rather than worked around with raw
    //                       LLK.
    constexpr bool RES_FUSE = HAS_R && (RES_FUSE_CT != 0) && !SQ_FOLD && !X_RESIDENT;
    // The fold's pack is per-OUTER (one tile per tile-row of the grid), which is the
    // policy pair DestAccumulation::PerRow requires.
    constexpr auto SQ_OUT_FOLDED = ckl::output(
        cb_x_squared,
        ckl::ReservePolicy::PerOuter,
        ckl::PushPolicy::PerOuter,
        DFR,
        ckl::PackRelu::Disabled,
        ckl::L1Accumulation::Disabled,
        ckl::DestAccumulation::PerRow);
    // A5: the caller's `subblock_w` when they supplied one -- HONOURED, never
    // clamped and never absorbed.  Every way it could be illegal (< 1, not a
    // divisor of block_w, above the DEST capacity their own fp32_dest_acc_en
    // bought) was REFUSED host-side in resolve_program_config, which is the only
    // place that decision lives; 0 means "the op's own choice" and is the seed's
    // expression exactly.
    // PERF 1: A ONE-TILE-ROW BLOCK WANTS THE *SMALL* DEST BLOCK, NOT THE LARGE ONE.
    // D21's rule -- the LARGEST divisor of WT_CHUNK that fits DEST -- amortizes pass B's
    // per-element init, format reconfig and CB reserve/push over as many tiles as the
    // register file holds.  That is the right trade when a core has many tile-rows to get
    // through and pass B is THROUGHPUT-bound.  At BLOCK_ROWS == 1 it is not: pass B is the
    // TAIL after the combine's multicast, the block is a handful of tiles, and a smaller
    // DEST block lets the packer overlap the math instead of waiting out one long window.
    // So the small rule is the op's choice, and the LARGE rule is the exception -- earned
    // by a MEASURED regression, not by a hunch (blackhole p150b 1350 MHz, pinned config
    // unchanged, output BIT-IDENTICAL in every cell below; a smaller block changes only
    // the DEST windowing, never an operand, a format or a rounding):
    //     BLOCK_ROWS > 1, small block LOSES  (1,1,8192,1024) BLOCK 64c   0.958x
    //                                        (1,1,7168,1024) BLOCK gbr   0.976x
    //     BLOCK_ROWS == 1, small block WINS  (1,1,32,7168) WIDTH 28c   3891 -> 3724  1.045x
    //                                        (1,1,32,2304) WIDTH  9c   3156 -> 2989  1.056x
    //                                        (1,1,32,1024) WIDTH  8c   2688 -> 2593  1.037x
    //                                        (1,1,2048,256) HEIGHT     3564 -> 3433  1.038x
    //                                        (1,1,32,7168) INTERLEAVED 8530 -> 8356  1.021x
    // Geometries whose WT_CHUNK has no divisor between 2 and the auto value (WT_CHUNK 5 on
    // the (1,1,32,5120) WIDTH shards) get the SAME number from both rules and are
    // byte-identical, which is why they read flat.  `>= 2` rather than a flat 1: block 1
    // measured 0.994x / 0.991x / 0.987x -- the per-tile lifecycle costs more than the
    // overlap buys -- so 1 is a fallback for WT_CHUNK == 1, never a choice.
    constexpr uint32_t PASS_B_AUTO = (BLOCK_ROWS > 1) ? pass_b_blk(WT_CHUNK, ckl::DEST_AUTO_LIMIT)
                                                      : pass_b_blk_small(WT_CHUNK, ckl::DEST_AUTO_LIMIT);
    constexpr uint32_t PASS_B_BLK = (PASS_B_BLK_CT != 0) ? PASS_B_BLK_CT : PASS_B_AUTO;
    static_assert(PASS_B_BLK >= 1 && WT_CHUNK % PASS_B_BLK == 0, "rms_norm_ttnn: PASS_B_BLK must divide WT_CHUNK");
    static_assert(PASS_B_BLK <= ckl::DEST_AUTO_LIMIT, "rms_norm_ttnn: PASS_B_BLK exceeds the DEST lane capacity");

    // R3: pass A's DEST-lane block size.  The D12 fold already owns D0 across the whole
    // tile-row (DestAccumulation::PerRow), so it stays at 1 there; everywhere else the
    // block size is PASS_B_BLK -- the SAME expression pass B uses, never a second
    // literal.  The block size and the pack lifecycle are ONE change: at block_size > 1
    // the chain emits the pack lifecycle once per OUTER iter, so a PerTile reserve would
    // reserve 1 page and pack PASS_A_SQ_BLK of them (a corrupted ring, i.e. a hang).
    constexpr bool SQ_BLOCKED = (PASS_A_SQ_BLOCK_CT != 0) && !SQ_FOLD && (PASS_B_BLK > 1);
    constexpr uint32_t SQ_BLK = SQ_BLOCKED ? PASS_B_BLK : 1u;
    constexpr auto SQ_OUT_BLOCKED =
        ckl::output(cb_x_squared, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize);
    constexpr auto SQ_OUT = SQ_FOLD ? SQ_OUT_FOLDED : (SQ_BLOCKED ? SQ_OUT_BLOCKED : ckl::output(cb_x_squared));

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
    constexpr auto X_IN_A = ckl::input(CB_T, ckl::WaitPolicy::Upfront, PASS_A_POP, ckl::OperandKind::Block, DFR, AOFF);
    // ---- A1: `residual_add_block`'s two operands, and its output ------------
    // BOTH activation streams are consumed and POPPED here, whatever the regime:
    // the held role has moved to cb_x_sum, so nothing downstream indexes back
    // into either of them.  Under NATIVE_IN they are the zero-copy shard CBs --
    // already resident, published once by the reader -- and this pop is what
    // advances their front from one row-block to the next.
    constexpr auto R_X_IN =
        ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block, DFR);
    constexpr auto R_R_IN =
        ckl::input(cb_residual_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block, DFR);
    // ONE reserve and ONE push for the whole chunk -- `Upfront`/`AtEnd`, NOT the
    // `PerBlockSize` pair pass B's stages use.
    //
    // WHY IT DIFFERS FROM PASS B, and it is a MEASURED difference, not a
    // stylistic one.  D21 chose `PerBlockSize` for pass B deliberately: those
    // chains' output feeds the ROW_MAJOR `untilize` consumer, which needs the
    // per-block page handover, and there `PerBlockSize` measured within noise of
    // `Upfront` (8860 vs 8901 ns).  cb_x_sum has NO such consumer -- it is
    // compute-private and both of its readers (`square_block` in pass A,
    // `normalize_block` in pass B) wait `Upfront` for the whole window -- so the
    // incremental handover buys nothing and the flow-control steps are pure
    // overhead: at WT_CHUNK 32 and PASS_B_BLK 8 that is 4 reserve/push pairs per
    // (block, chunk) where 1 will do.
    //
    // MEASURED (blackhole p150b, bf16 / HiFi2 / fp32_dest_acc_en=False, THREE
    // fresh-cache profiled runs per variant, median; the two distributions are
    // DISJOINT on the first row, which is why 2.5% is reported at all on a shape
    // whose run-to-run spread is ~2%):
    //     (1,1,8192,1024) INTERLEAVED gamma_bias_residual  139758 -> 136361  1.025x
    //     (1,1,32,5120)   WIDTH 32c   gamma_bias_residual    6824 ->   6730  1.014x
    //     (1,1,8192,1024) INTERLEAVED residual only        125154 -> 124890  1.002x
    //     (1,1,7168,1024) BLOCK 64c   gamma_bias_residual   33822 ->  33876  0.998x
    // i.e. it wins where a bias makes pass B long enough that pass A's own
    // flow-control is visible, and is flat where the shape is DRAM-bound.
    //
    // LEGAL AT EVERY REGIME, and the reason is the ring arithmetic rather than a
    // sweep: `Upfront` reserves the chunk's whole window at the chain's start and
    // `AtEnd` publishes it once, and cb_x_sum's write pointer sits at
    // `c * WT_CHUNK` from the ring base in every regime -- RESIDENT (one chunk,
    // front 0), ROW_RESIDENT (chunks accumulate to exactly the ring's
    // `wt_core` pages, so free space is never less than WT_CHUNK), STREAM
    // (`square_block` pops each chunk, so the front returns to 0).  No window
    // ever straddles the wrap.
    constexpr auto X_SUM_OUT = ckl::output(cb_x_sum, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);
    // Pass B's x: held CBs are popped ONCE per row-block below (an explicit pop is
    // the sanctioned pattern for a lifetime no single PopPolicy can express), so
    // that a chunk's `AtEnd` cannot pop the base tiles the next chunk still needs.
    constexpr auto PASS_B_X_POP = ROW_RESIDENT ? ckl::PopPolicy::None : ckl::PopPolicy::AtEnd;
    constexpr auto X_IN_B =
        ckl::input(CB_T, ckl::WaitPolicy::Upfront, PASS_B_X_POP, ckl::OperandKind::Block, DFR, XOFF);
    constexpr auto G_IN =
        ckl::input(cb_gamma_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row, DFR, XOFF);
    // A2: the bias CB mirrors gamma's spec exactly -- same Row broadcast, same
    // held lifetime -- at its OWN data format.  The chain's reconfig fold must NOT
    // elide the second format switch: `weight` and `bias` may be at different
    // dtypes, so the packer/unpacker really does change twice per chunk.
    constexpr auto B_IN =
        ckl::input(cb_bias_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row, DFR, XOFF);
    // Pass-B output routing, stated ONCE.  Let S = [scale] if HAS_G + [bias] if
    // HAS_B.  `normalize_block` writes cb_output_tiles when S is empty, else
    // cb_normalized; each stage in S except the LAST writes cb_normalized IN
    // PLACE, and the last writes cb_output_tiles.  With S = [scale] or S = [] this
    // collapses to exactly the seed's routing.
    constexpr uint32_t NORM_OUT = (HAS_G || HAS_B) ? cb_normalized : cb_output_tiles;

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
    static_assert(!CROSS_CORE || BLOCK_ROWS <= 32, "rms_norm_ttnn: a compact combine block is at most 32 tile-rows");
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
    // LAMP L-FIN (Refinement 2).  ORTHOGONAL to COMPACT, exactly as the tree is: what it
    // moves is WHICH CORE applies the rsqrt, never the LAYOUT of the tile it is applied
    // to -- so the lane scope below is still decided by BLOCK_ROWS alone and BOTH branches
    // (identity and compact) are covered by the one predicate the verifier's note asks for.
    //
    //   ROOT   fold(FINALIZE=true)  -> cb_stat_handoff (finalized) -> mcast -> [un-permute]
    //   SPREAD fold(FINALIZE=false) -> cb_stat_handoff (RAW)       -> mcast ->
    //          spread_finalize -> cb_row_stat -> [un-permute]
    //
    // cb_row_stat is the landing CB because it is the ONE fp32 per-row CB that is dead on
    // the combine path (D22 deleted its last use there), so the spread needs no new buffer
    // index and, at the default, allocates nothing at all.
    constexpr bool FIN_SPREAD = CROSS_CORE && (FIN_SPREAD_CT != 0);
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
    static_assert(!TREE || TREE_F0 * TREE_F1 >= GROUP_SIZE, "rms_norm_ttnn: the slot tree must cover GROUP_SIZE");
    static_assert(
        !TREE || TREE_F1 >= 2, "rms_norm_ttnn: a slot-tree level that gathers one member is a hop, not a fold");
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
    // Under FIN_SPREAD on the IDENTITY path the multicast lands RAW in cb_row_final and the
    // finalized tile pass B reads is `spread_finalize`'s output; on the COMPACT path the
    // un-permute still writes cb_row_final, so pass B's operand is unchanged there.
    constexpr uint32_t CB_STAT_B = CROSS_CORE ? ((FIN_SPREAD && !COMPACT) ? cb_row_stat : cb_row_final) : cb_row_stat;
    // The un-permute's SOURCE: the multicast landing CB at ROOT, the spread finalize's
    // output under SPREAD.  One name so the matmul, its reconfig and its pop cannot drift.
    constexpr uint32_t CB_UNPERM_SRC = FIN_SPREAD ? cb_row_stat : cb_mcast_in;
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
    // ONE definition of "tilize one chunk of a per-channel operand", read by the
    // resident boot below and by the STREAM re-stage in pass B.  D30's two staging
    // widths differ ONLY in how the same tiles are walked:
    //   wide   tilize<WT_CHUNK>(1)          one WT_CHUNK-wide block, one LLK call
    //   narrow tilize<1>(WT_CHUNK)          WT_CHUNK one-tile blocks, WT_CHUNK calls
    // Both leave exactly WT_CHUNK tiles in the destination CB, in the same order.
    auto tilize_per_channel_chunk = [&]() {
        if constexpr (HAS_G) {
            if constexpr (PC_NARROW) {
                ckl::tilize<1, cb_gamma_sticks, cb_gamma_tiles>(WT_CHUNK);
            } else {
                ckl::tilize<WT_CHUNK, cb_gamma_sticks, cb_gamma_tiles>(1);
            }
        }
        if constexpr (HAS_B) {
            if constexpr (PC_NARROW) {
                ckl::tilize<1, cb_bias_sticks, cb_bias_tiles>(WT_CHUNK);
            } else {
                ckl::tilize<WT_CHUNK, cb_bias_sticks, cb_bias_tiles>(1);
            }
        }
    };

    if constexpr ((HAS_G || HAS_B) && X_RESIDENT && PC_RM) {
        MaybeDeviceZoneScope("compute_gamma_tilize");
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            tilize_per_channel_chunk();
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
    // ================= A1: residual_add_block -- t = x + r ==================
    // The FIRST stage of the pipeline, upstream of the statistics, so `sum(t^2)`
    // is taken over x + r and never over x alone.  One `eltwise_chain` element:
    // an elementwise BinaryFpu Add of two full-shape Block operands (no
    // broadcast, both All-valid) packing into cb_x_sum.
    //
    // Spelled as `eltwise_chain` rather than the `add<>` one-liner for the same
    // reason pass A's square and pass B's multiplies are: the convenience
    // wrappers default-construct their elements (convenience.inl:8-71) and cannot
    // carry a per-operand policy pair, and this one needs BOTH inputs at
    // Upfront/AtEnd (they are streams, including the zero-copy shard CBs whose
    // front this pop advances) with a PerBlockSize pack (cb_x_sum accumulates
    // chunk by chunk under the ROW_RESIDENT hold).  It is exactly what
    // `add<cb_input_tiles, cb_residual_tiles, cb_x_sum>` expands to, plus those
    // policies.
    //
    // DEST-blocked at PASS_B_BLK, like pass B's stages, so one per-element init,
    // one format reconfig and one CB flow-control step amortize over PASS_B_BLK
    // tiles instead of being paid per tile (D21's measured 1.28-1.66x, on the
    // same (rows x WT_CHUNK) walk).
    auto residual_add_block = [&](uint32_t rows) {
        if constexpr (!HAS_R) {
            (void)rows;
            return;
        } else {
            MaybeDeviceZoneScope("compute_residual_add");
            ckl::eltwise_chain(
                ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                ckl::BinaryFpu<ckl::BinaryFpuOp::Add, R_X_IN, R_R_IN>{},
                ckl::PackTile<X_SUM_OUT>{});
        }
    };

    // PERF 1: the boot constants are waited ONCE, at first use, via these one-shot flags.
    // `cb_scaler` and `cb_bank` are pushed once by the reader and never popped until the
    // end, so the front only has to be established the first time -- and a flag keeps the
    // poll off the per-chunk path of the streaming regimes, where pass A runs it hundreds
    // of times per core.
    bool scaler_ready = false;
    bool bank_ready = false;

    auto pass_a = [&](uint32_t rows, uint32_t pipe_base) {
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            // Tile offset of this chunk inside the HELD CB.  0 (and elided) unless
            // ROW_RESIDENT, where CB_T / cb_gamma_tiles / cb_bias_tiles span the
            // whole row.
            const uint32_t hold_base = (ROW_RESIDENT ? (c * WT_CHUNK) : 0) + pipe_base;
            if constexpr (RM) {
                MaybeDeviceZoneScope("compute_tilize_x");
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            if constexpr (RM && HAS_R) {
                MaybeDeviceZoneScope("compute_tilize_r");
                ckl::tilize<WT_CHUNK, cb_residual_sticks, cb_residual_tiles>(rows);
            }
            if constexpr (RES_FUSE) {
                // Lamp L-RES-FUSE: ONE chain for `t = x + r` AND `t^2`.  The FPU add
                // leaves the sum in D0 and the SFPU `Square` squares that DEST slot in
                // place, so the square costs NO unpack and NO pack of `t` at all -- two
                // chain setups, one pack and two unpacks per tile become one setup, one
                // pack and two unpacks.  cb_x_sum is not written in pass A here and is
                // not read there either: this branch is STREAM-only (see RES_FUSE above),
                // and STREAM's pass B calls `residual_add_block` to rebuild `t` itself.
                // The two activation CBs are still popped by the chain, so the push/pop
                // balance is exactly the unfused pair's minus cb_x_sum's matched pair.
                MaybeDeviceZoneScope("compute_res_square");
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(SQ_BLK),
                    ckl::BinaryFpu<ckl::BinaryFpuOp::Add, R_X_IN, R_R_IN>{},
                    ckl::Square<>{},
                    ckl::PackTile<SQ_OUT>{});
            } else {
                // t = x + r, materialized into the CB the rest of the kernel reads.
                residual_add_block(rows);
                // x^2, either packed to cb_x_squared per width tile or folded into DEST
                // (D12).  `square` cannot carry the tile base, so the chain is spelled
                // out; it is exactly what square<> expands to.
                MaybeDeviceZoneScope("compute_square");
                // D43: the fold's grid is (rows * X_SQUARED_WT) rows of SQ_GROUP tiles,
                // so `DestAccumulation::PerRow` acquires / packs / clears DEST once per
                // GROUP instead of once per chunk.  `OperandKind::Block` indexes
                // `base + r * Wt + c`, so the walk over the (rows x WT_CHUNK) block is
                // the flat shape's walk, tile for tile -- the reshape is the WHOLE
                // mechanism and no helper is bypassed.  The PACKED branch keeps
                // grid(rows, WT_CHUNK) AND its SQ_BLK DEST blocking, because
                // `block_size` applies to the INNER extent: folding it into the reshaped
                // grid would silently drop D21's blocking on the un-folded path.
                const auto sq_shape = SQ_FOLD ? ckl::IterationShape::grid(rows * X_SQUARED_WT, SQ_GROUP)
                                              : ckl::IterationShape::grid(rows, WT_CHUNK).block_size(SQ_BLK);
                // ABLATION LIMIT, stated rather than left to fail at compile time:
                // the payload-free stub is ONE unpack + ONE pack with the identical CB
                // lifecycle and trip count, and `CopyTile` carries no DEST-accumulation
                // parameter -- while a FOLDING `SQ_OUT` requires accumulation on both the
                // math element and the output (`chain.inl:2906` static_asserts it).  So
                // where the fold is on, `RMS_ABLATE=COMPUTE` leaves the square's payload
                // IN.  That is a real limit of the switch and not a new one (it predates
                // D43; D43 only widens the set of folding geometries), and it does not
                // reach the peel this round used, which ran on a non-folding chunk of 32.
                // Closing it needs a `CopyTile` with a DestAccumulation parameter.
                constexpr bool SQ_STUB =
#ifdef RMS_ABLATE_COMPUTE
                    !SQ_FOLD;
#else
                    false;
#endif
                if constexpr (SQ_STUB) {
                    ckl::eltwise_chain(sq_shape, ckl::CopyTile<X_IN_A>{hold_base}, ckl::PackTile<SQ_OUT>{});
                } else {
                    ckl::eltwise_chain(
                        sq_shape,
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_A, X_IN_A, ckl::Dst::D0, SQ_OUT.dest_accumulation>{
                            hold_base, hold_base},
                        ckl::PackTile<SQ_OUT>{});
                }
            }

            // PERF 1: the reduce helper does NOT wait on its scaler CB -- its own contract
            // hands that to the caller (reduce_helpers_compute.hpp:37) -- and since the
            // reader now publishes the input shard BEFORE synthesizing the scaler, this is
            // the wait that used to be implied by the reader's ordering.
            if (!scaler_ready) {
                cb_wait_front(cb_scaler, SCALER_TILES);
                scaler_ready = true;
            }
            MaybeDeviceZoneScope("compute_reduce");
            rms_norm_local::accumulate_reduce_block<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_x_squared,
                cb_scaler,
                CB_REDUCE_ACC,
                REDUCE_POLICY,
                REDUCE_RECONFIG,
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
            // PERF 1: `matmul_tiles` has no CB lifecycle, so the one-hot bank the reader
            // synthesizes needs an explicit front here -- see the note above the row-block
            // loop for the corruption this prevents.  BLOCK_ROWS pages, not `rows`: a
            // RAGGED last block has fewer rows but the bank is always full-size, and this
            // wait must cover the whole bank the first time whichever block runs first.
            if (!bank_ready) {
                cb_wait_front(cb_bank, BLOCK_ROWS);
                bank_ready = true;
            }
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

    // ---- PERF 1: THE BOOT-CONSTANT CBs ARE WAITED FOR *EXPLICITLY*, ONCE --------------
    // `cb_scaler` and `cb_bank` are reader-synthesized constants that this kernel reads
    // WITHOUT any per-use wait: the reduce helper's own contract puts the burden on the
    // caller ("the scaler CB must contain the scaling factor tile BEFORE calling reduce()",
    // reduce_helpers_compute.hpp:37), and `member_pack` / `compute_recv_unpack` hand
    // `cb_bank` straight to `matmul_tiles`, which has no CB lifecycle at all.  Until Perf 1
    // both were ordered only IMPLICITLY, by the reader publishing the input shard LAST --
    // so pass A could not start until every boot constant was already in L1.
    //
    // Perf 1 hoists that publish to the TOP of the reader (a 1.29x win; see the note there),
    // which DELETES that implicit ordering.  Without these two waits the compute kernel
    // races the reader's boot: MEASURED as a non-deterministic corruption on the COMPACT
    // combine path -- `1x1x2048x256` and `4x1x512x512` BLOCK_SHARDED came back at
    // pcc 0.08-0.12 / rel-RMS 1e5 with a DIFFERENT set of cells failing on every run,
    // because the permutation matmul was multiplying against a bank the reader had not
    // finished zeroing.  The identity path (BLOCK_ROWS == 1) has no bank and happened to
    // survive on the scaler, which is precisely the kind of luck a wait replaces.
    //
    // EACH WAIT SITS AT ITS OWN FIRST USE, not in a joint prologue: the scaler's is in
    // `pass_a` immediately before the reduce and the bank's at the top of `member_pack` and
    // `compute_recv_unpack`.  That matters, and it is measured -- a joint prologue makes
    // `compute_square` (which needs NEITHER constant) block on both, and on the COMPACT
    // BLOCK-shard geometries that cost 7-8% of the whole op (22001 vs 20419 ns on
    // `(1,1,8192,1024)` BLOCK 64c) by re-serialising pass A behind the reader's boot.  A
    // repeated `cb_wait_front` on an already-satisfied, never-popped CB is an L1 poll that
    // returns immediately, so paying it per block is free where the constant is ready.
    // These are the matching FRONTS for the `cb_pop_front(cb_scaler, SCALER_TILES)` at the
    // end of this kernel; the bank is never popped at all.

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
            // config; f0 is itself measured -- see COMBINE_TREE_F0_MIN/_MAX):
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
                //
                // PERF 1: THE FOLD PACKS STRAIGHT INTO THE MULTICAST LANDING PAGE.
                // The last level's pack target used to be `cb_stat_handoff`, which the
                // WRITER then copied 3 kB L1->L1 into the landing CB (with an acked
                // barrier) purely to satisfy D24's "publish the root's own copy before the
                // broadcast".  Packing into the landing CB directly makes that publish the
                // pack that already existed: D24's requirement is unchanged and now free,
                // and `cb_stat_handoff` degenerates to the one-page READY TOKEN pushed
                // below.  This is the ROOT's branch only -- an interior tree node still
                // packs `cb_node_out`, and a member never runs this code.
                // MEASURED bit-identical, 1.013x-1.025x on the focus WIDTH shard and
                // 1.006x-1.024x across every other combine geometry; see the writer's
                // `writer_mcast_send` note for the table.
                constexpr uint32_t CB_FOLD_OUT = COMPACT ? cb_mcast_in : cb_row_final;
                // Under FIN_SPREAD the last level forwards a RAW sum, so the fold's pack
                // would NOT be the stat and the landing page must not receive it.
                // COMBINE_FIN_SPREAD is measured off (Refinement 2), so this guards a dead
                // path rather than gating a live one.
                static_assert(!FIN_SPREAD, "rms_norm_ttnn: the direct-to-landing fold assumes the ROOT finalizes");
                MaybeDeviceZoneScope("compute_root_fused");
#if defined(RMS_ABLATE_ROOT_SUM) || defined(RMS_ABLATE_ROOT_FINALIZE)
                // ABLATION (temporary, /perf-measure): payload removed, every CB handshake
                // and trip count preserved.  Under D27 the round's handshake is ONE window
                // in and ONE page out, whatever BLOCK_ROWS is -- peel recipe:
                // `RMS_ABLATE=ROOT_SUM` (add `,GATHER_ZERO` for the writer's half of the
                // same stage) and diff the profiled zones against the
                // unablated run; the difference is that stage's WALL contribution, which is
                // what makes the cumulative peel additive.  D28: the ROOT's window is the
                // level-1 ring when the tree is built, and `compute_tree_fold_l0` above
                // peels with the same pair of switches.
                // PERF 1: the round's OUTPUT handshake is the CB_FOLD_OUT page plus the
                // ready token below (which is emitted unconditionally, ablated or not), so
                // the stub keeps the landing page's reserve/push and nothing else.
                cb_wait_front(TREE ? cb_gather_l1 : cb_partials_gathered, TREE ? TREE_SL1 : GATHER_SLOTS);
                cb_reserve_back(CB_FOLD_OUT, 1);
                cb_push_back(CB_FOLD_OUT, 1);
                cb_pop_front(TREE ? cb_gather_l1 : cb_partials_gathered, TREE ? TREE_SL1 : GATHER_SLOTS);
#else
                // FINALIZE = !FIN_SPREAD is the ONE line Lamp L-FIN changes on the root:
                // under SPREAD the last level forwards the RAW group sum and every core
                // applies the rsqrt to its own copy after the multicast.  Note this is the
                // LAST level either way -- an INTERIOR node still must never finalize (it
                // would rsqrt a partial sum, and the next level would then add rsqrt'd
                // values), which is why `compute_tree_fold_l0` above stays FINALIZE=false
                // unconditionally.
                if constexpr (TREE) {
                    combine_fold<
                        cb_gather_l1,
                        TREE_SL1,
                        CB_FOLD_OUT,
                        /*FINALIZE=*/!FIN_SPREAD,
                        COMPACT,
                        COMPACT_FIN_WIDE,
                        INV_W_BITS,
                        EPS_BITS>();
                } else {
                    combine_fold<
                        cb_partials_gathered,
                        GATHER_SLOTS,
                        CB_FOLD_OUT,
                        /*FINALIZE=*/!FIN_SPREAD,
                        COMPACT,
                        COMPACT_FIN_WIDE,
                        INV_W_BITS,
                        EPS_BITS>();
                }
#endif
                // The one-page READY TOKEN.  Its BYTES ARE NEVER READ: the writer waits it
                // only to learn that CB_FOLD_OUT now holds this round's finalized stat, and
                // then multicasts out of that page.  It is what gives the writer the
                // happens-before it used to get from the copy it no longer makes.
                cb_reserve_back(cb_stat_handoff, 1);
                cb_push_back(cb_stat_handoff, 1);
            }

            // ---- LAMP L-FIN: the SPREAD finalize (Refinement 2) ---------------------
            // Under FIN_SPREAD the tile that just arrived (or, on the root, that it
            // published for itself before broadcasting -- D24) is the RAW group sum, so
            // `1/rms = rsqrt(sum/W + eps)` is applied HERE, on every core of the group,
            // in parallel, instead of once on the root ahead of the multicast.
            //
            // The SCOPE is the same predicate the root's fused finalize uses and for the
            // same reason: a compact tile carries one tile-row's sum per COLUMN, so it
            // needs the <1,8> walk (VectorMode::C to BLOCK_ROWS 16, RC above), while the
            // identity path's stat really does live in column 0 alone and D17's narrow
            // <2,4> C walk is both right and 1.39x cheaper.  Getting this wrong is silent:
            // the narrow scope on a compact tile measured pcc 0.997 with rel-RMS 1036.
            //
            // WHY IT IS A COPY + PACK AND NOT AN IN-PLACE TRANSFORM: the raw tile's CB is
            // the WRITER's (cb_mcast_in on the compact path, cb_row_final on the identity
            // one -- both are where the multicast lands), so packing back into it would
            // give that CB two producers.  cb_row_stat is the compute-private landing CB;
            // it is dead on the combine path otherwise, and it is allocated only when this
            // knob is on.
            //
            // THE STRUCTURAL COST, stated because it is what the measurement found: at
            // ROOT the rsqrt is FUSED into the fold's DEST window (D22) and costs no pack
            // at all; spread, it needs its own copy_tile + pack + unpack on every core.
            // And the finalize is a REPLICATED term -- every core needs the same value --
            // so relocating it moves it along the identical serial chain rather than
            // dividing it.  See `_combine_fin_spread` in the descriptor for the numbers.
            if constexpr (FIN_SPREAD) {
                constexpr uint32_t CB_RAW = COMPACT ? cb_mcast_in : cb_row_final;
                MaybeDeviceZoneScope("compute_spread_fin");
                cb_wait_front(CB_RAW, 1);
                cb_reserve_back(cb_row_stat, 1);
                reconfig_data_format_srca(CB_RAW);
                pack_reconfig_data_format(cb_row_stat);
                copy_tile_to_dst_init_short(CB_RAW);
                // MANDATORY here for the same reason it is inside `combine_fold`:
                // rms_stat_rsqrt_body reads the persistent SFPU PROGRAM registers
                // sfpu::rsqrt_init programs.
                rsqrt_tile_init();
                tile_regs_acquire();
                copy_tile(CB_RAW, 0, 0);
                if constexpr (COMPACT) {
                    compact_finalize_payload<INV_W_BITS, EPS_BITS, COMPACT_FIN_WIDE>(0);
                } else {
                    stat_finalize_payload<INV_W_BITS, EPS_BITS>(0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_row_stat);
                tile_regs_release();
                cb_push_back(cb_row_stat, 1);
                cb_pop_front(CB_RAW, 1);
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
                if (!bank_ready) {  // PERF 1: matmul has no CB lifecycle
                    cb_wait_front(cb_bank, BLOCK_ROWS);
                    bank_ready = true;
                }
                cb_wait_front(CB_UNPERM_SRC, 1);
                cb_reserve_back(cb_row_final, rows);
                reconfig_data_format<ckernel::SrcOrder::Reverse>(CB_UNPERM_SRC, cb_bank);
                pack_reconfig_data_format(cb_row_final);
                matmul_init(CB_UNPERM_SRC, cb_bank, /*transpose=*/1);
                for (uint32_t b = 0; b < rows; b += COMBINE_DEST_BATCH) {
                    const uint32_t n = (rows - b < COMBINE_DEST_BATCH) ? (rows - b) : COMBINE_DEST_BATCH;
                    tile_regs_acquire();
                    for (uint32_t d = 0; d < n; ++d) {
                        matmul_tiles(CB_UNPERM_SRC, cb_bank, 0, b + d, d);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t d = 0; d < n; ++d) {
                        pack_tile(d, cb_row_final, b + d);
                    }
                    tile_regs_release();
                }
                cb_push_back(cb_row_final, rows);
                cb_pop_front(CB_UNPERM_SRC, 1);
            }
        }

        // ================= pass B: scale ===================================
        for (uint32_t c = 0; c < NUM_W_CHUNKS; ++c) {
            const uint32_t hold_base = ROW_RESIDENT ? (c * WT_CHUNK) : 0;
            // D39: gamma/bias index from 0 when their ring IS the chunk.
            const uint32_t pc_base = PC_CHUNKED ? 0u : hold_base;
            // ROW_RESIDENT never re-stages either held operand: pass A already put
            // the whole tile-row of x tiles (and of gamma) in L1.  This is the
            // pass-B re-read that Lamp L5 exists to delete.
            if constexpr (RM && !X_RESIDENT) {
                MaybeDeviceZoneScope("compute_tilize_x_b");
                ckl::tilize<WT_CHUNK, cb_input_sticks, cb_input_tiles>(rows);
            }
            if constexpr (RM && HAS_R && !X_RESIDENT) {
                MaybeDeviceZoneScope("compute_tilize_r_b");
                ckl::tilize<WT_CHUNK, cb_residual_sticks, cb_residual_tiles>(rows);
            }
            if constexpr ((HAS_G || HAS_B) && !X_RESIDENT && PC_RM) {
                MaybeDeviceZoneScope("compute_gamma_tilize_b");
                tilize_per_channel_chunk();
            }
            // A1 in STREAM: neither activation survived pass A, so `t` has to be
            // rebuilt from the re-read x and r.  This is the whole cost of the
            // STREAM regime doubling with a residual (two activations crossing
            // DRAM twice instead of one), and it is exactly why ROW_RESIDENT --
            // which holds cb_x_sum and re-reads NOTHING -- matters more here than
            // it did in the seed.
            if constexpr (!X_RESIDENT) {
                residual_add_block(rows);
            }

            // ================= D44 (Perf 3): PASS B'S ORDER ====================
            //
            // Pass B's FIRST op is the one that needs the finalized stat.  On a
            // cross-core plan the stat arrives by gather -> root fold -> multicast, and
            // the shipped order (scale, then gamma) stalls pass B on that arrival with
            // the whole gamma traversal still to do.  The gamma mul depends on x and
            // gamma ONLY -- never on the stat -- so doing it FIRST fills the wait with
            // the traversal instead of idling through it.
            //
            // WHY THE TRAVERSAL AND NOT THE MATH.  An ablation settles it: replacing the
            // gamma broadcast-mul with a bare CopyTile over the same tiles -- traversal,
            // unpack, pack and CB lifecycle all kept, ONLY the multiply deleted -- is
            // 0.982 / 1.002 / 0.982 / 1.004 on perf cases 8 / 11 / 12 / 17, while
            // deleting the whole TRAVERSAL is 1.096 / 1.113 / 1.244 / 1.168.  Pass B's
            // second pass costs its traversal; the multiply is free.  That is also why
            // Perf 1's DEST-reuse fusion of these two stages LOST (13.5 vs 8.5 us/core):
            // it deleted packs, which were never the cost, and added per-face MOP
            // restarts.  So the only thing left to do with these two stages was to
            // reorder them, and on a combine plan `swapx` measures AS FAST AS DELETING
            // the traversal outright (1.095 vs 1.096 on case 8; 1.129 vs 1.113 on
            // case 11; 1.036 vs 1.034 on case 13; 1.074 vs 1.075 on case 18) -- which
            // only latency-hiding explains.
            //
            // MEASURED 1.010x-1.129x on all 7 combine=True plans of the perf group and
            // BIT-EXACT with the shipped order on all 6 combine=False plans.
            //
            // THE COST, stated because it is real: the reordered intermediate is
            // `x * gamma`, which is UN-normalized, so it saturates the intermediate CB's
            // dtype where the shipped intermediate (approximately 1) cannot.  The
            // boundary is exactly |x * gamma| > dtype_max; measured, x=1e10 with
            // gamma=1e29 gives 9.965e28 shipped and 3.373e28 reordered.  Nothing in the
            // op's tested universe reaches that band (all 19 perf cases and 31
            // structural cases match the shipped order to 6 decimal places of pcc), and
            // the op's own sum(x^2) already saturates by |x| ~ 1.8e19 -- but this
            // narrows the dynamic range and that is the honest price of the win.
            //
            // TWO EXCEPTIONS, each earned, each written as the NARROW carve-out so it
            // shrinks rather than has to be widened:
            //   * !HAS_G -- INFEASIBLE.  With no gamma there is no second mul to move.
            //   * !CROSS_CORE -- MEASURED REGRESSION.  With the stat computed locally
            //     there is no arrival to hide behind, and the reorder cost a reproducible
            //     0.983x and 0.989x in two independent sessions on perf case 05
            //     (1,1,8192,2304).  Every other combine=False case was flat, so the
            //     carve-out is the regime, not that one shape.
            if constexpr (!HAS_G || !CROSS_CORE) {
                // ---- the shipped order: scale, then gamma ----
                // x * (1/rms). The stat is a REDUCE_ROW result: column-shaped, so it
                // broadcasts back ACROSS columns (BroadcastDim::Col) and must be
                // operand B. OperandKind::Col indexes it by row only, and it is not
                // popped -- every width chunk of this block re-reads it.
                {
                    MaybeDeviceZoneScope("compute_scale");
#ifdef RMS_ABLATE_COMPUTE
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                        ckl::CopyTile<X_IN_B>{hold_base},
                        ckl::PackTile<PASS_B_OUT_NORM>{});
#else
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
#endif
                }

                if constexpr (HAS_G) {
                    // gamma is row-shaped (1 x W, valid in row 0) -> broadcasts DOWN
                    // rows (BroadcastDim::Row), indexed by column (OperandKind::Row).
                    MaybeDeviceZoneScope("compute_gamma_mul");
                    if constexpr (HAS_B) {
                        // A2: a bias FOLLOWS the scale, so the scale is not the last
                        // stage and must not write cb_output_tiles.  It transforms
                        // cb_normalized IN PLACE instead of taking a third block CB.
                        //
                        // In-place is legal ONLY with an incrementally POPPING input
                        // and an incrementally RESERVING output -- the packer's
                        // reserve cannot succeed while the reader's tiles still occupy
                        // the CB -- so both sides are PerBlockSize: the
                        // device-verified case 1 of
                        // kernel_lib/tests/eltwise/chain/lifecycle/inplace_chain.cpp.
                        // (An Upfront-reserve output on an aliased CB DEADLOCKS rather
                        // than returning a wrong answer, which is why the pair is not
                        // a free choice.)  gamma stays Row/Upfront/None and is NEVER
                        // the aliased CB -- chain.inl:82-85 forbids a Row/Col operand
                        // as an in-place target.
                        //
                        // The rotation this costs is why cb_normalized is TWO blocks
                        // deep when both stages are present: the front advances by
                        // `rows * WT_CHUNK` per block, which is a whole revolution for
                        // a FULL block but not for the partial final one, and the bias
                        // stage's bulk wait + linear indexing would then straddle the
                        // ring wrap.  See _norm_cb_depth in the descriptor (it is D6's
                        // hazard on a different CB, and it takes D6's fix).
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                ckl::input(
                                    cb_normalized,
                                    ckl::WaitPolicy::PerBlockSize,
                                    ckl::PopPolicy::PerBlockSize,
                                    ckl::OperandKind::Block),
                                ckl::input(G_IN, ckl::BroadcastDim::Row)>{0u, pc_base},
                            ckl::PackTile<ckl::output(
                                cb_normalized, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
                    } else {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                ckl::input(
                                    cb_normalized,
                                    ckl::WaitPolicy::Upfront,
                                    ckl::PopPolicy::AtEnd,
                                    ckl::OperandKind::Block),
                                ckl::input(G_IN, ckl::BroadcastDim::Row)>{0u, pc_base},
                            ckl::PackTile<PASS_B_OUT_GAMMA>{});
                    }
                }
            } else {
                // ---- D44: the reordered order -- gamma FIRST, then the scale ----
                {
                    MaybeDeviceZoneScope("compute_gamma_mul");
                    ckl::eltwise_chain(
                        ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                        ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, X_IN_B, ckl::input(G_IN, ckl::BroadcastDim::Row)>{
                            hold_base, pc_base},
                        ckl::PackTile<PASS_B_OUT_NORM>{});
                }
                {
                    // The scale is now the SECOND stage, so it inherits the lifecycle the
                    // gamma stage used to own: in place on cb_normalized when a bias
                    // follows (PerBlockSize both sides -- an Upfront-reserve output on an
                    // aliased CB DEADLOCKS), and Upfront/AtEnd packing cb_output_tiles
                    // when it is last.  The stat operand stays Col/Upfront/None and is
                    // never the aliased CB (chain.inl:82-85 forbids a Row/Col operand as
                    // an in-place target).
                    MaybeDeviceZoneScope("compute_scale");
                    if constexpr (HAS_B) {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
#ifdef RMS_ABLATE_COMPUTE
                            ckl::CopyTile<ckl::input(
                                cb_normalized,
                                ckl::WaitPolicy::PerBlockSize,
                                ckl::PopPolicy::PerBlockSize,
                                ckl::OperandKind::Block)>{0u},
#else
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                ckl::input(
                                    cb_normalized,
                                    ckl::WaitPolicy::PerBlockSize,
                                    ckl::PopPolicy::PerBlockSize,
                                    ckl::OperandKind::Block),
                                ckl::input(
                                    CB_STAT_B,
                                    ckl::BroadcastDim::Col,
                                    ckl::WaitPolicy::Upfront,
                                    ckl::PopPolicy::None,
                                    ckl::OperandKind::Col)>{0u, 0u},
#endif
                            ckl::PackTile<ckl::output(
                                cb_normalized, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
                    } else {
                        ckl::eltwise_chain(
                            ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
#ifdef RMS_ABLATE_COMPUTE
                            ckl::CopyTile<ckl::input(
                                cb_normalized,
                                ckl::WaitPolicy::Upfront,
                                ckl::PopPolicy::AtEnd,
                                ckl::OperandKind::Block)>{0u},
#else
                            ckl::BinaryFpu<
                                ckl::BinaryFpuOp::Mul,
                                ckl::input(
                                    cb_normalized,
                                    ckl::WaitPolicy::Upfront,
                                    ckl::PopPolicy::AtEnd,
                                    ckl::OperandKind::Block),
                                ckl::input(
                                    CB_STAT_B,
                                    ckl::BroadcastDim::Col,
                                    ckl::WaitPolicy::Upfront,
                                    ckl::PopPolicy::None,
                                    ckl::OperandKind::Col)>{0u, 0u},
#endif
                            ckl::PackTile<PASS_B_OUT_GAMMA>{});
                    }
                }
            }

            if constexpr (HAS_B) {
                // A2: the per-channel SHIFT, and it is the LAST stage -- it packs
                // cb_output_tiles.  Same operand shape and the same Row broadcast
                // as the scale (a bias is a 1 x W vector valid in row 0), so it is
                // the same chain with BinaryFpuOp::Add; the device-tested Add+Row
                // pairing is kernel_lib/tests/eltwise/chain/axes/bcast_binary_add.cpp.
                //
                // NOT fused with the scale into one DEST window: a second
                // BinaryFpu reads CBs rather than DEST, and DestReuseBinary
                // carries no broadcast parameter (chain.hpp:526), so
                // `(y * w) + b` with two Row broadcasts is inexpressible as one
                // chain.  It is also why bias cannot be folded into the scale.
                MaybeDeviceZoneScope("compute_bias_add");
                ckl::eltwise_chain(
                    ckl::IterationShape::grid(rows, WT_CHUNK).block_size(PASS_B_BLK),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            cb_normalized, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                        ckl::input(B_IN, ckl::BroadcastDim::Row)>{0u, pc_base},
                    ckl::PackTile<PASS_B_OUT_GAMMA>{});
            }

            if constexpr (RM) {
                MaybeDeviceZoneScope("compute_untilize");
                ckl::untilize<WT_CHUNK, cb_output_tiles, cb_output_sticks>(rows);
            }

            if constexpr (HAS_G && PC_CHUNKED) {
                cb_pop_front(cb_gamma_tiles, WT_CHUNK);
            }
            if constexpr (HAS_B && PC_CHUNKED) {
                cb_pop_front(cb_bias_tiles, WT_CHUNK);
            }
        }

        // The held CBs' lifetime is the whole row-block, which no PopPolicy on a
        // per-chunk call can express (an `AtEnd` would drop the base tiles the next
        // chunk still indexes), so ROW_RESIDENT pops x here -- the same sanctioned
        // pattern as cb_row_stat / cb_gamma_tiles / cb_scaler.
        if constexpr (ROW_RESIDENT) {
            cb_pop_front(CB_T, rows * X_HOLD_WT);
        }
        cb_pop_front(CB_STAT_B, rows);
    }

    // SCALER_TILES is the descriptor's single source of truth for how many tiles
    // the reader pushed into cb_scaler (datapath- and PARTIAL_W-dependent).
    cb_pop_front(cb_scaler, SCALER_TILES);
    if constexpr (HAS_G && !PC_CHUNKED) {
        cb_pop_front(cb_gamma_tiles, X_HOLD_WT);
    }
    if constexpr (HAS_B && !PC_CHUNKED) {
        cb_pop_front(cb_bias_tiles, X_HOLD_WT);
    }
}
