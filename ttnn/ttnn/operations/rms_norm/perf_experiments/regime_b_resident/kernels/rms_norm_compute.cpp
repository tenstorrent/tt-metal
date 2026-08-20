// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// W-SPLIT combine, step 3 (only on a group ROOT, only when W_SPLIT=1): the
// GROUP_SIZE gathered PARTIAL sum-of-squares tiles are summed element-wise and
// collapsed along W in ONE reduce<SUM, REDUCE_ROW, ..., AccumulateViaAdd> call.
//
// WHY the tile that crosses the NoC is the PRE-collapse accumulator (32 live
// columns) and not a col0-collapsed one: `AccumulateViaAdd` IS "pairwise FPU
// add_tiles into one DEST register, then ONE SFPU within-tile finalize".  Feeding
// it the GROUP_SIZE accumulator tiles therefore performs the cross-core
// element-wise sum AND the within-tile collapse in a single call of the SAME
// helper this op already uses for its Regime B reduce - no new datapath.
// Collapsing per core first would need a separate element-wise add over
// GROUP_SIZE tiles plus the collapse.  It is also why the split requires Regime A
// per core: only Regime A produces that pre-collapse tile.
//
// rms_norm compute (TRISC x3).  Every phase is kernel_lib-helper covered; there
// is no raw-LLK compute anywhere in this file.  `compute_kernel_hw_startup` is
// the chain's documented caller-init contract, not a bypass.
//
// Regime A (RESIDENT-FUSED, single DRAM read):
//     [RM] tilize                              cb_rm_in       -> cb_input_tiles
//     sum_of_squares  (fused x*x + per-row DEST accumulate, NO intermediate CB;
//                      PopPolicy::None keeps x resident for the scale pass)
//                                              cb_input_tiles -> cb_sumsq_acc
//     reduce<SUM, REDUCE_ROW>  (the within-tile finalize of that accumulator)
//                                              cb_sumsq_acc   -> cb_sumsq
//     eltwise_chain   (*1/W, +eps, rsqrt in ONE dst-sync window, fp32 scalars)
//                                              cb_sumsq       -> cb_rms_recip
//     mul <Col bcast>                          x, 1/rms       -> cb_normed | cb_output_tiles
//     mul <Row bcast>                          normed, gamma  -> cb_output_tiles
//     [RM] untilize                            cb_output_tiles-> cb_rm_out
//
// Regime B (STREAMING-MASKED, two DRAM reads) chunks the dependent W axis at
// WT_REDUCE_BLOCK / WT_SCALE_BLOCK and runs THE SAME fused sum-of-squares per
// chunk:
//     sum_of_squares  cb_input_tiles -> cb_sumsq_acc   (ONE tile per row)
//     reduce<SUM, REDUCE_ROW, Accumulate>  cb_sumsq_acc -> cb_sumsq
// i.e. Regime A's shape, chunked.  x*x is folded into ONE DEST row accumulator
// per chunk, so the reduce collapses a 1-tile window instead of a
// WT_REDUCE_BLOCK-tile one and there is no full-block x^2 intermediate CB at all.
// The partial scaler (partial_mask / last_tile_at at scaler index 1) zeroes the
// pad columns of the last W-tile on the LAST chunk only, so implicit tile padding
// never enters the sum (risk R1).  The divisor is always 1/W_true, applied in
// fp32 by MulUnary - never folded into the mandatory-bfloat16 reduce scaler
// (risk R2).
//
// CB-WRAP INVARIANT: the W-chunk divides Wt_core and every row-block is exactly
// BLOCK_HT tile-rows, so every multi-page CB access here is a fixed size that
// divides the CB's page count.  See the reader for the full rationale.

// ===========================================================================
// LAB FORK (perf_experiments/regime_b_resident) - see rbr_plan.py for the idea.
// Adds, on top of the shipped kernel:
//   * REGIME C - x resident (ONE DRAM read), scale pass chunked at WT_SCALE_BLOCK.
//   * THE MASKED RESIDENT SUM-OF-SQUARES - the last W-tile gets its own
//     accumulator, so a non-tile-aligned W can use a resident plan at all.
// ===========================================================================
#include <cstdint>

// Lab-only zone gate: with -DRMSN_NO_ZONES every MaybeDeviceZoneScope becomes a
// no-op, so a chunk/depth sweep is not paying a marker cost that DIFFERS between
// arms (a resident plan runs a different NUMBER of zone executions than a
// streaming one, which would leak straight into the measured delta).  Defined
// BEFORE the include, which guards its own definition with #ifndef.
#ifdef RMSN_NO_ZONES
#define MaybeDeviceZoneScope(name) ((void)0)
#endif

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_sumsq = 4;
constexpr uint32_t cb_rms_recip = 5;
constexpr uint32_t cb_normed = 6;
constexpr uint32_t cb_output_tiles = 7;
constexpr uint32_t cb_rm_in = 8;
constexpr uint32_t cb_rm_out = 9;
constexpr uint32_t cb_sumsq_acc = 10;
constexpr uint32_t cb_gamma_rm = 11;
constexpr uint32_t cb_partial_gather = 12;
constexpr uint32_t cb_sumsq_bcast = 13;

constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
// REGIME (CT 1) - three plans, not two:
//   0 = B  STREAMING   two DRAM reads of x (reduce pass + scale pass).
//   1 = A  RESIDENT    one DRAM read; the WHOLE per-core width of x AND gamma AND
//                      cb_normed AND the output CB are resident.
//   2 = C  RESIDENT-X  one DRAM read; only x (and optionally gamma) is resident,
//                      and the scale pass walks W in WT_SCALE_BLOCK chunks.
constexpr uint32_t REGIME = get_compile_time_arg_val(1);
constexpr bool REGIME_A = (REGIME == 1);
constexpr bool REGIME_C = (REGIME == 2);
// x is read from DRAM exactly ONCE per row-block and stays in cb_input_tiles for
// the scale pass.  This is the property the experiment is about.
constexpr bool RESIDENT_X = REGIME_A || REGIME_C;
constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(2);
constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(3);
constexpr uint32_t Wt_core = get_compile_time_arg_val(4);
constexpr uint32_t W_PARTIAL = get_compile_time_arg_val(5);
constexpr uint32_t BLOCK_HT = get_compile_time_arg_val(6);
constexpr uint32_t WT_REDUCE_BLOCK = get_compile_time_arg_val(7);
constexpr uint32_t WT_SCALE_BLOCK = get_compile_time_arg_val(8);
constexpr uint32_t Rt = get_compile_time_arg_val(9);
constexpr uint32_t DEST_BLOCK_CT = get_compile_time_arg_val(15);
constexpr uint32_t GAMMA_INGEST_BLOCK = get_compile_time_arg_val(18);
// Regime B reduce datapath (host-chosen, see blocking_plan.reduce_via_add):
//   1 = AccumulateViaAdd - pairwise FPU add_tiles into ONE DST register, then a
//       single SFPU within-tile finalize on the LAST chunk.
//   0 = ReduceTile       - the Phase-0 datapath, one FPU matmul-with-ones per
//       input tile accumulating straight into DEST.
// WHY this is a knob and why 1 is the default: at fp32_dest_acc_en=False the
// DEST datum is 16-bit, and ReduceTile's long per-tile DEST accumulation carries
// a systematic sum-of-squares OVERESTIMATE that grows with the reduced width
// (measured: +0.84% at Wt=32, +1.9% at 64, +5.6% at 128, +10.4% at 224, which
// shows up as a uniform ~5% low output scale at W=7168).  It is invariant to the
// W-chunk size, the accumulator CB format and DEST_BLOCK, so it is the datapath
// and not the blocking.  Regime A's element-wise accumulate never had it, and
// AccumulateViaAdd is that same pairwise-accumulate shape - the helper documents
// it as "more accurate ... wins for wide reduces (many tiles per output)".
constexpr uint32_t REDUCE_VIA_ADD = get_compile_time_arg_val(22);
// --- W-split work distribution (blocking_plan._choose_group_size) ------------
// W_SPLIT == 0 is the row-parallel plan and every branch below compiles out.
constexpr uint32_t W_SPLIT = get_compile_time_arg_val(23);
constexpr uint32_t GROUP_SIZE = get_compile_time_arg_val(24);
// LAB (CT 30): gamma is resident (read once per core, never popped).  Always 1 in
// Regime A; in Regime C the solver picks it, and when it is 0 the scale pass
// consumes a freshly pushed gamma slice per chunk (Regime B's protocol).
constexpr uint32_t RESIDENT_GAMMA_CT = get_compile_time_arg_val(30);
constexpr bool GAMMA_RESIDENT = (RESIDENT_GAMMA_CT != 0);
// LAB (CT 31) POSITIVE CONTROL, never shippable: drop the mask from the masked
// resident fold.  Must produce a WRONG answer on a pad-poisoned case - that is
// what proves the poison landed and the passing arms are not vacuous.
constexpr uint32_t RESIDENT_NO_MASK = get_compile_time_arg_val(31);
// The masked resident fold: the aligned head, then the last W-tile on its own.
// The mask is a TILE-path property: on ROW_MAJOR input the reader zero-fills each
// stick's pad tail, so the pad is exactly 0 and the resident fold needs no mask.
constexpr bool RESIDENT_MASKED = RESIDENT_X && (W_PARTIAL > 0) && !IS_ROW_MAJOR;
// Which plans consume the PARTIAL scaler tile the reader emits at index 1.
constexpr bool USES_PARTIAL_SCALER = (W_PARTIAL > 0) && (!RESIDENT_X || RESIDENT_MASKED);

constexpr uint32_t NUM_REDUCE_CHUNKS = Wt_core / WT_REDUCE_BLOCK;
constexpr uint32_t NUM_SCALE_CHUNKS = Wt_core / WT_SCALE_BLOCK;

// The host-chosen DEST block knob, clamped against the REAL hardware constant.
// Never a literal 4 or 8.
constexpr uint32_t DEST_BLOCK = (DEST_BLOCK_CT < ckl::DEST_AUTO_LIMIT) ? DEST_BLOCK_CT : ckl::DEST_AUTO_LIMIT;

// The two datapaths differ in ONE more thing than the algorithm enum: cross-call
// Accumulate on AccumulateViaAdd is BulkWaitBulkPop-only (helper contract).  Both
// aliases are derived from the single REDUCE_VIA_ADD arg so they cannot drift.
constexpr auto REDUCE_ALGORITHM =
    REDUCE_VIA_ADD ? ckl::ReduceAlgorithm::AccumulateViaAdd : ckl::ReduceAlgorithm::ReduceTile;
constexpr auto REDUCE_POLICY =
    REDUCE_VIA_ADD ? ckl::ReduceInputPolicy::BulkWaitBulkPop : ckl::ReduceInputPolicy::WaitAndPopPerTile;

// The no-gamma path writes the 1/rms scale straight into the output CB, so it
// pays zero extra copies.
constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_normed : cb_output_tiles;

// ROW_MAJOR gamma: tilize N staged stick-tiles into cb_gamma_tiles.  N is a
// compile-time multiple of GAMMA_INGEST_BLOCK by construction of the host plan
// (Wt_core in Regime A, WT_SCALE_BLOCK in Regime B), so this never
// over-produces gamma tiles.  Only tile row 0 of each staged block carries
// data; BroadcastDim::Row reads nothing else.
//
// NOTE: each chunk gets a FULL tilize init/uninit.  The helper's back-to-back
// InitOnly/Neither/UninitOnly form was tried here and is NOT correct for this
// loop: the chunks are separate cb_reserve/push groups on cb_gamma_tiles, and
// skipping the per-call init silently corrupts every chunk after the first
// (measured: PCC 0.0035-0.24 on (1,1,32,4096) and (1,1,32,16384)).  Keep
// InitAndUninit.
template <uint32_t N>
ALWI void ingest_gamma() {
    if constexpr (HAS_GAMMA && GAMMA_IS_ROW_MAJOR) {
        MaybeDeviceZoneScope("cp_gamma_tilize");
        for (uint32_t o = 0; o < N; o += GAMMA_INGEST_BLOCK) {
            ckl::tilize<GAMMA_INGEST_BLOCK, cb_gamma_rm, cb_gamma_tiles>(1);
        }
    }
}

// The partial form is datapath-specific (the reader emits the matching tile at
// scaler index 1; see its scaler block).  `last_tile_at` carries only a scaler
// INDEX, which AccumulateViaAdd reads as "tile-aligned" - it needs
// `partial_mask`, which also carries the valid-element COUNT.
constexpr auto partial_scaler = (W_PARTIAL == 0)
                                    ? ckl::ReducePartialScaler::none()
                                    : (REDUCE_VIA_ADD ? ckl::ReducePartialScaler::partial_mask(W_PARTIAL, 1)
                                                      : ckl::ReducePartialScaler::last_tile_at(1));

// ===========================================================================
// Regime B's sum-of-squares phase: ONE fused accumulate per W-chunk
// ===========================================================================
// HELPER SUBSTITUTION (kernel_lib CAPABILITY gap - do NOT "fix" this back).
// `fs_out` below is byte-for-byte what `ckl::row_output(cb_sumsq_acc)` expands to
// inside `ckl::sum_of_squares` (eltwise/api/convenience.inl: PerOuter reserve +
// PerOuter push + DestAccumulation::PerRow), and `sumsq_strided` is byte-for-byte
// what `sum_of_squares` expands to (`square` -> `eltwise_chain(BinaryFpu<Mul, In,
// In, D0, Output.dest_accumulation>, PackTile<Output>)`).  It is spelled out
// because the NON-TILE-ALIGNED last chunk must walk a GAPPED column window
// ("every column but the last", then "the last column"), which needs a
// caller-supplied `StridedTileRange` per operand - and the `sum_of_squares`
// wrapper takes no runtime element arguments, so there is no overload to pass the
// stride to.  This is the composable surface UNDER the same helper family, not
// raw LLK; the tile-aligned chunk still calls `sum_of_squares` itself.
//
// .INL-ONLY CONTRACT worth recording: `PackTile` static_asserts "L1 and DEST
// accumulation cannot be combined" (eltwise/core/chain.inl), so the CROSS-chunk
// accumulation can NOT be an L1-accumulating pack into cb_sumsq - it must go
// through the reduce's `Accumulate`.  Hence the two-level shape: DEST accumulate
// WITHIN a chunk, reduce `Accumulate` ACROSS chunks.
constexpr auto fs_out = ckl::output(
    cb_sumsq_acc,
    ckl::ReservePolicy::PerOuter,
    ckl::PushPolicy::PerOuter,
    ckl::DataFormatReconfig::Enabled,
    ckl::PackRelu::Disabled,
    ckl::L1Accumulation::Disabled,
    ckl::DestAccumulation::PerRow);

// How the cross-chunk fold reloads cb_sumsq's running raw partial sum.
// FoldViaAdd hands it to add_tiles as the SrcB operand instead of paying the
// default CopySeedPairs' copy_tile + DEST-reuse add per chunk (measured 5-9% of
// the WHOLE op at Wt_core <= 4, ~1.3% of the compute payload at Wt_core = 224).
// Legal because nothing in this op tags cb_sumsq UnpackToDestFp32, which is
// exactly the helper's stated contract (reduce_helpers_compute.hpp).  Ignored
// outright on the ReduceTile datapath, which always reloads via copy_tile.
constexpr auto FOLD_RELOAD = ckl::AccumulateReloadMode::FoldViaAdd;

// x*x accumulated into ONE DEST tile per row over `NCOL` columns starting at
// column `base` of a resident (BLOCK_HT x WT_REDUCE_BLOCK) window.  Caller owns
// the cb_input_tiles wait/pop - TileOffset::Strided is (None, None)-only, since a
// gapped window has no single wait/pop count.
template <uint32_t NCOL, uint32_t ROW_STRIDE_T = WT_REDUCE_BLOCK>
ALWI void sumsq_strided(uint32_t base) {
    ckl::eltwise_chain(
        ckl::IterationShape::grid(BLOCK_HT, NCOL).block_size(DEST_BLOCK),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Mul,
            ckl::input(
                cb_input_tiles,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileOffset::Strided),
            ckl::input(
                cb_input_tiles,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::OperandKind::Block,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileOffset::Strided),
            ckl::Dst::D0,
            ckl::DestAccumulation::PerRow>{
            ckl::StridedTileRange{base, ROW_STRIDE_T}, ckl::StridedTileRange{base, ROW_STRIDE_T}},
        ckl::PackTile<fs_out>{});
}

// Fold ONE raw partial tile per row out of cb_sumsq_acc into the running cb_sumsq
// accumulator.  `k` is the cross-call Accumulate iteration (0 = seed); `last`
// runs the single within-tile REDUCE_ROW finalize; `partial` masks the pad
// columns - only ever on the accumulator tile that owns the LAST W-tile.
ALWI void fold_partial_sum(uint32_t k, bool last, bool partial) {
    MaybeDeviceZoneScope("cp_reduce");
    ckl::reduce<
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        cb_sumsq_acc,
        cb_reduce_scaler,
        cb_sumsq,
        REDUCE_POLICY,
        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast,  // default (global scope, not in ckl)
        REDUCE_ALGORITHM>(
        ckl::ReduceInputBlockShape::of(BLOCK_HT, 1, 1),
        ckl::ReduceInputMemoryLayout::contiguous(),
        (last ? ckl::Accumulate::at_last(cb_sumsq, k) : ckl::Accumulate::at(cb_sumsq, k)).with_reload(FOLD_RELOAD),
        ckl::NoOp{},
        partial ? partial_scaler : ckl::ReducePartialScaler::none());
}

// One W-chunk of the sum-of-squares phase; `k` is the running cross-call
// Accumulate index into cb_sumsq and the return value is the next one.
ALWI uint32_t sumsq_chunk(uint32_t k, bool is_last_chunk) {
    // NARROW CARVE-OUT, and the only one: a ONE-TILE chunk has nothing to fuse.
    // The per-row DEST accumulate spans a single column, so `sum_of_squares`
    // reduces to `square` while still paying its PerOuter reserve/push (BLOCK_HT
    // CB round-trips instead of one) and the cross-chunk fold machinery - a
    // MEASURED regression on the smallest supported cell ((32, 17): 3,365 ->
    // 3,460 ns).  The degenerate fused form IS `square`, so this branch is the
    // pre-fusion datapath exactly, just writing the shared cb_sumsq_acc (there is
    // no separate x^2 CB on any path any more).  It is a carve-out around the
    // case that CANNOT benefit, not an allow-list of measured widths.
    if constexpr (WT_REDUCE_BLOCK == 1) {
        {
            MaybeDeviceZoneScope("cp_sumsq");
            ckl::square<
                ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::output(cb_sumsq_acc, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::grid(BLOCK_HT, 1).block_size(DEST_BLOCK));
        }
        // The single tile of the chunk IS the last W-tile, so the mask lands on it.
        fold_partial_sum(k, is_last_chunk, W_PARTIAL > 0 && is_last_chunk);
        return k + 1;
    } else if (!(W_PARTIAL > 0 && is_last_chunk)) {
        // TILE-ALIGNED chunk: the whole chunk folds into ONE DEST row accumulator
        // and the helper owns the input lifecycle.  No mask is needed - the
        // accumulator's 32 columns are all meaningful.
        {
            MaybeDeviceZoneScope("cp_sumsq");
            ckl::sum_of_squares<
                ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::row_output(cb_sumsq_acc)>(
                ckl::IterationShape::grid(BLOCK_HT, WT_REDUCE_BLOCK).block_size(DEST_BLOCK));
        }
        fold_partial_sum(k, is_last_chunk, false);
        return k + 1;
    } else {
        // LAST chunk of a NON-TILE-ALIGNED W.  The pad columns live ONLY in the
        // very last W-tile, so that tile gets its OWN accumulator: its 32 columns
        // then map 1:1 onto the last W-tile's columns, which is what makes the
        // reduce's existing masked-last-tile fold - partial_mask(W_PARTIAL, 1) on
        // AccumulateViaAdd, last_tile_at(1) on ReduceTile - zero PRECISELY the pad
        // columns (risk R1: the pad must never enter the sum).  Everything before
        // it is a plain aligned accumulate.  The strided walk is what lets the
        // split work for BLOCK_HT > 1, where "all columns but the last" is not
        // contiguous.
        constexpr uint32_t HEAD = (WT_REDUCE_BLOCK > 1) ? (WT_REDUCE_BLOCK - 1) : 1;
        cb_wait_front(cb_input_tiles, BLOCK_HT * WT_REDUCE_BLOCK);
        {
            MaybeDeviceZoneScope("cp_sumsq");
            sumsq_strided<HEAD>(0);
        }
        fold_partial_sum(k, false, false);
        {
            MaybeDeviceZoneScope("cp_sumsq");
            sumsq_strided<1>(WT_REDUCE_BLOCK - 1);
        }
        fold_partial_sum(k + 1, true, true);
        cb_pop_front(cb_input_tiles, BLOCK_HT * WT_REDUCE_BLOCK);
        return k + 2;
    }
}

// ===========================================================================
// LAB: THE MASKED RESIDENT SUM-OF-SQUARES
// ===========================================================================
// This is the ONE thing that was missing to let a non-tile-aligned W use a
// resident (single-DRAM-read) plan at all.
//
// WHY THE SHIPPED REGIME A CANNOT MASK.  Its fused `sum_of_squares` folds the
// whole tile-row of x*x into ONE accumulator tile ELEMENT-WISE: column j of the
// accumulator ends up holding sum over tiles t of x[t][j]^2.  Only the LAST
// W-tile has pad columns, so after the fold columns j >= W_PARTIAL carry a MIX of
// valid contributions (from tiles 0..Wt-2) and pad ones (from tile Wt-1).  There
// is no column position left that is pure pad, so no mask can be applied - the
// pad is already smeared across live data (risk R1).
//
// THE FIX is the shape Regime B's `sumsq_chunk` third branch already uses: give
// the LAST W-TILE ITS OWN ACCUMULATOR.  Two strided passes over the resident
// window - "every column but the last" then "the last column" - produce two
// accumulator tiles, and the 32 columns of the second map 1:1 onto the last
// W-tile's columns.  The op's existing masked fold (`partial_mask(W_PARTIAL, 1)`
// on AccumulateViaAdd, `last_tile_at(1)` on ReduceTile) then zeroes PRECISELY the
// pad columns.  The only thing that is new here is the ROW STRIDE: the resident
// window is Wt_core wide, not WT_REDUCE_BLOCK.
//
// The strided walk is also what makes it work for BLOCK_HT > 1, where "all
// columns but the last" is not contiguous in the CB.
//
// Cost: exactly ONE extra reduce fold per row-block versus the unmasked resident
// path (two 1-tile folds instead of one), against a whole second DRAM read of x.
ALWI void sumsq_resident_masked() {
    constexpr uint32_t HEAD = (Wt_core > 1) ? (Wt_core - 1) : 1;
    // TileOffset::Strided is caller-managed (WaitPolicy::None, PopPolicy::None) -
    // a gapped window has no single wait/pop count - so the wait is here and the
    // pop is in kernel_main, AFTER the scale pass has read the same resident x.
    cb_wait_front(cb_input_tiles, BLOCK_HT * Wt_core);
    if constexpr (Wt_core > 1) {
        {
            MaybeDeviceZoneScope("cp_sumsq");
            sumsq_strided<HEAD, Wt_core>(0);
        }
        fold_partial_sum(0, false, false);
    }
    {
        MaybeDeviceZoneScope("cp_sumsq");
        sumsq_strided<1, Wt_core>(Wt_core - 1);
    }
    // The masked fold, and the within-tile finalize, in one call.
    fold_partial_sum(Wt_core > 1 ? 1 : 0, true, RESIDENT_NO_MASK == 0);
}

// x * (1/rms) [* gamma] over one (BLOCK_HT x cw) chunk.
//   RmsPop   : AtEnd in Regime A (one call per row-block); None in Regime B,
//              where cb_rms_recip is reused across every W-chunk and popped by
//              the caller after the last one.
//   GammaPop : None in Regime A (cb_gamma_tiles is resident and never popped);
//              AtEnd in Regime B, where the reader re-pushes the chunk's slice.
template <ckl::PopPolicy RmsPop, ckl::PopPolicy GammaPop>
ALWI void scale_chunk(uint32_t cw) {
    {
        MaybeDeviceZoneScope("cp_scale_mul");
        ckl::mul<
            ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(cb_rms_recip, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, RmsPop, ckl::OperandKind::Col),
            ckl::output(cb_scale_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::grid(BLOCK_HT, cw).block_size(DEST_BLOCK));
    }

    if constexpr (HAS_GAMMA) {
        MaybeDeviceZoneScope("cp_gamma_mul");
        ckl::mul<
            ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(
                cb_gamma_tiles, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, GammaPop, ckl::OperandKind::Row),
            ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::grid(BLOCK_HT, cw).block_size(DEST_BLOCK));
    }
}

// ---------------------------------------------------------------------------
// LAB / REGIME C: addressing a W-CHUNK of the RESIDENT x
// ---------------------------------------------------------------------------
// cb_input_tiles holds the whole (BLOCK_HT x Wt_core) row-block, written by the
// reader in row-major tile order, and is NOT popped between chunks.  Chunk `c` is
// therefore not at the CB front - it is the sub-block
//     tile_id = (c * WT_SCALE_BLOCK) + r * Wt_core + w
// which is exactly `TileOffset::Strided` with StridedTileRange{base, Wt_core}.
// Strided is contracted to caller-managed (None, None) policies, so the enclosing
// kernel owns the lifecycle: the sum-of-squares above already did the ONE wait for
// BLOCK_HT * Wt_core tiles, and kernel_main pops the whole window once, after the
// last chunk.  CB-WRAP: every access is a multiple of that fixed window measured
// from an aligned fifo pointer, so no chunk can run off the end.
//
// HELPER NOTE (same substitution the shipped `sumsq_strided` records): this is
// NOT raw LLK.  `ckl::mul` is the two-argument convenience forwarder, which has
// no place to pass an operand's StridedTileRange, so the call drops ONE level to
// the `eltwise_chain` + `BinaryFpu` + `PackTile` form the forwarder itself
// expands to (convenience.inl).  Same elements, same policies, same helper family.
template <ckl::PopPolicy RmsPop, ckl::PopPolicy GammaPop>
ALWI void scale_chunk_resident(uint32_t cw, uint32_t base) {
    {
        MaybeDeviceZoneScope("cp_scale_mul");
        ckl::eltwise_chain(
            ckl::IterationShape::grid(BLOCK_HT, cw).block_size(DEST_BLOCK),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    cb_input_tiles,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::OperandKind::Block,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Strided),
                ckl::input(
                    cb_rms_recip, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, RmsPop, ckl::OperandKind::Col),
                ckl::Dst::D0>{ckl::StridedTileRange{base, Wt_core}},
            ckl::PackTile<ckl::output(
                cb_scale_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
    }

    if constexpr (HAS_GAMMA) {
        MaybeDeviceZoneScope("cp_gamma_mul");
        if constexpr (GAMMA_RESIDENT) {
            // gamma is resident and never popped: this chunk's columns are
            // [base, base + cw) of the CB, i.e. an ordinary Row operand plus a
            // TileOffset base.  Caller-managed, waited once before the loop.
            ckl::eltwise_chain(
                ckl::IterationShape::grid(BLOCK_HT, cw).block_size(DEST_BLOCK),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::input(
                        cb_gamma_tiles,
                        ckl::BroadcastDim::Row,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        ckl::OperandKind::Row,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileOffset::Set),
                    ckl::Dst::D0>{0u, base},
                ckl::PackTile<ckl::output(
                    cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
        } else {
            ckl::mul<
                ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                ckl::input(
                    cb_gamma_tiles, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, GammaPop, ckl::OperandKind::Row),
                ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                ckl::IterationShape::grid(BLOCK_HT, cw).block_size(DEST_BLOCK));
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);
    const uint32_t num_row_blocks_here = get_arg_val<uint32_t>(3);
    const uint32_t IS_ROOT = get_arg_val<uint32_t>(4);  // W-split: this core is its group's combine root

    if constexpr (IS_ROW_MAJOR) {
        compute_kernel_hw_startup(cb_rm_in, cb_rm_in, cb_input_tiles);
    } else {
        compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_output_tiles);
    }

    // Regime A holds the whole per-core gamma width resident for the whole
    // kernel: ingested exactly once, never popped.  (Regime C's resident-gamma
    // ingest happens on its first row-block instead - see the scale block - so it
    // matches the reader's push order, where gamma lands after the first x.)
    if constexpr (GAMMA_RESIDENT) {
        ingest_gamma<Wt_core>();
        if constexpr (REGIME_C && HAS_GAMMA) {
            // Regime A's scale chain waits on cb_gamma_tiles itself
            // (WaitPolicy::Upfront); Regime C's resident-gamma chain is
            // caller-managed (None, None) because it addresses this chunk's
            // columns through a TileOffset base, so the ONE wait is here.
            cb_wait_front(cb_gamma_tiles, Wt_core);
        }
    }

    for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
        // ---------------- sum of squares over the reduced axis ---------------
        if constexpr (RESIDENT_X) {
            if constexpr (IS_ROW_MAJOR) {
                MaybeDeviceZoneScope("cp_tilize");
                ckl::tilize<Wt_core, cb_rm_in, cb_input_tiles>(BLOCK_HT);
            }
            // accumulate-then-finalize (catalog `row_reduce_accumulate`):
            // sum_of_squares folds the whole tile-row of x*x into ONE tile per
            // row, element-wise; the within-tile collapse along W is the single
            // reduce<SUM, REDUCE_ROW> below.  No mask is needed here - the
            // accumulator's 32 columns are all meaningful, since the only padded
            // columns live in the last W-tile and the RM reader zero-fills them.
            if constexpr (RESIDENT_MASKED) {
                // LAB: non-tile-aligned W on a resident plan.  This call also does
                // the folds AND the within-tile finalize, so the reduce below is
                // skipped.  Never coincides with W_SPLIT (the split's P2 keeps a
                // masked shape at G == 1, asserted host-side).
                sumsq_resident_masked();
            } else {
                {
                    MaybeDeviceZoneScope("cp_sumsq");
                    ckl::sum_of_squares<
                        ckl::input(
                            cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block),
                        ckl::row_output(cb_sumsq_acc)>(
                        ckl::IterationShape::grid(BLOCK_HT, Wt_core).block_size(DEST_BLOCK));
                }

                if constexpr (!W_SPLIT) {
                    MaybeDeviceZoneScope("cp_reduce");
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_sumsq_acc,
                        cb_reduce_scaler,
                        cb_sumsq>(ckl::ReduceInputBlockShape::of(BLOCK_HT, 1, 1));
                } else if (IS_ROOT) {
                    // The reader has landed slot i of every group member at tile
                    // (r * GROUP_SIZE + i) - a contiguous (BLOCK_HT x GROUP_SIZE) block -
                    // and pushed it.  cb_sumsq itself is filled by the reader's broadcast
                    // on EVERY core of the group, root included, so the rms chain below
                    // is identical on all of them.
                    MaybeDeviceZoneScope("cp_combine");
                    ckl::reduce<
                        ckernel::PoolType::SUM,
                        ckernel::ReduceDim::REDUCE_ROW,
                        cb_partial_gather,
                        cb_reduce_scaler,
                        cb_sumsq_bcast,
                        ckl::ReduceInputPolicy::BulkWaitBulkPop,
                        ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                        ReduceFp32Mode::Fast,
                        ckl::ReduceAlgorithm::AccumulateViaAdd>(
                        ckl::ReduceInputBlockShape::of(BLOCK_HT, GROUP_SIZE, 1));
                }
            }
        } else {
            // `at_last` (inside fold_partial_sum) marks the finalizing fold: on the
            // AccumulateViaAdd datapath cb_sumsq holds the RAW partial sum between
            // chunks and the within-tile collapse runs exactly once, on the last
            // one.  ReduceTile ignores the flag (it finalizes every chunk), so the
            // same call site is correct for both datapaths.  `k` is a running
            // counter, not `c`: the non-tile-aligned last chunk contributes TWO
            // folds (its head, then the masked last W-tile).
            uint32_t k = 0;
            for (uint32_t c = 0; c < NUM_REDUCE_CHUNKS; ++c) {
                if constexpr (IS_ROW_MAJOR) {
                    MaybeDeviceZoneScope("cp_tilize");
                    ckl::tilize<WT_REDUCE_BLOCK, cb_rm_in, cb_input_tiles>(BLOCK_HT);
                }
                k = sumsq_chunk(k, c + 1 == NUM_REDUCE_CHUNKS);
            }
        }

        // ---------------- rms chain: *1/W, +eps, rsqrt -----------------------
        // One helper call, one dst-sync window, no constant CBs.  MulUnary /
        // AddUnary take fp32 bit patterns, so 1/W and epsilon are applied at
        // full fp32 precision.
        {
            MaybeDeviceZoneScope("cp_rms_chain");
            ckl::eltwise_chain(
                ckl::IterationShape::tiles(BLOCK_HT),
                ckl::CopyTile<ckl::input(cb_sumsq)>{},
                ckl::MulUnary<>{inv_w_bits},
                ckl::AddUnary<>{eps_bits},
                ckl::Rsqrt<>{},
                ckl::PackTile<ckl::output(cb_rms_recip)>{});
        }

        // ---------------- scale (and gamma) ----------------------------------
        if constexpr (REGIME_A) {
            scale_chunk<ckl::PopPolicy::AtEnd, ckl::PopPolicy::None>(Wt_core);
            if constexpr (IS_ROW_MAJOR) {
                MaybeDeviceZoneScope("cp_untilize");
                ckl::untilize<Wt_core, cb_output_tiles, cb_rm_out>(BLOCK_HT);
            }
        } else if constexpr (REGIME_C) {
            // LAB: the scale pass walks W in chunks over the RESIDENT x.  Nothing
            // is re-read from DRAM here - that is the whole point.
            for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                if constexpr (!GAMMA_RESIDENT) {
                    ingest_gamma<WT_SCALE_BLOCK>();
                }
                scale_chunk_resident<ckl::PopPolicy::None, ckl::PopPolicy::AtEnd>(WT_SCALE_BLOCK, c * WT_SCALE_BLOCK);
                if constexpr (IS_ROW_MAJOR) {
                    MaybeDeviceZoneScope("cp_untilize");
                    ckl::untilize<WT_SCALE_BLOCK, cb_output_tiles, cb_rm_out>(BLOCK_HT);
                }
            }
            // cb_rms_recip was held across every chunk (PopPolicy::None), and the
            // resident window is popped ONCE, after the last chunk read it.
            cb_pop_front(cb_rms_recip, BLOCK_HT);
            cb_pop_front(cb_input_tiles, BLOCK_HT * Wt_core);
        } else {
            for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                // Ordered to match the reader's push order (gamma, then input);
                // reversing it deadlocks on the depth-1 staging CB.
                ingest_gamma<WT_SCALE_BLOCK>();
                if constexpr (IS_ROW_MAJOR) {
                    MaybeDeviceZoneScope("cp_tilize");
                    ckl::tilize<WT_SCALE_BLOCK, cb_rm_in, cb_input_tiles>(BLOCK_HT);
                }
                scale_chunk<ckl::PopPolicy::None, ckl::PopPolicy::AtEnd>(WT_SCALE_BLOCK);
                if constexpr (IS_ROW_MAJOR) {
                    MaybeDeviceZoneScope("cp_untilize");
                    ckl::untilize<WT_SCALE_BLOCK, cb_output_tiles, cb_rm_out>(BLOCK_HT);
                }
            }
            // cb_rms_recip was held across every W-chunk (PopPolicy::None).
            cb_pop_front(cb_rms_recip, BLOCK_HT);
        }
    }

    // reduce() waits on the scaler CB but never pops it - release the pages
    // exactly once, at the very end (risk R9).  Only Regime B ever emits the
    // second (partial) tile.
    // LAB: the partial tile now exists on EVERY plan with a non-tile-aligned W,
    // not only the streaming one - that is exactly what the masked resident fold
    // consumes.  In the baseline arm W_PARTIAL > 0 implies Regime B, so this is
    // the same count the shipped kernel pops.
    cb_pop_front(cb_reduce_scaler, USES_PARTIAL_SCALER ? 2 : 1);
}
