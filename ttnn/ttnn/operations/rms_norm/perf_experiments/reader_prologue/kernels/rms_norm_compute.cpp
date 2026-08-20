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

#include <cstdint>

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
constexpr uint32_t REGIME_A = get_compile_time_arg_val(1);
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
// reader_prologue arm 3 ("defer"): the reader pushes the resident Regime A gamma
// slice AFTER the W-split combine.  The compute-side tilize has to move with it -
// waiting on cb_gamma_rm at the top of the kernel would deadlock against a reader
// that is itself waiting on cb_sumsq_acc.  Inert on every non-ROW_MAJOR gamma
// (ingest_gamma compiles to nothing there).
constexpr uint32_t GAMMA_LATE = (get_compile_time_arg_val(30) == 3 || get_compile_time_arg_val(30) == 5) ? 1 : 0;

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
template <uint32_t NCOL>
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
            ckl::StridedTileRange{base, WT_REDUCE_BLOCK}, ckl::StridedTileRange{base, WT_REDUCE_BLOCK}},
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
    // kernel: ingested exactly once, never popped.
    if constexpr (REGIME_A && !GAMMA_LATE) {
        ingest_gamma<Wt_core>();
    }

    for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
        // ---------------- sum of squares over the reduced axis ---------------
        if constexpr (REGIME_A) {
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
            {
                MaybeDeviceZoneScope("cp_sumsq");
                ckl::sum_of_squares<
                    ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block),
                    ckl::row_output(cb_sumsq_acc)>(ckl::IterationShape::grid(BLOCK_HT, Wt_core).block_size(DEST_BLOCK));
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
                    ckl::ReduceAlgorithm::AccumulateViaAdd>(ckl::ReduceInputBlockShape::of(BLOCK_HT, GROUP_SIZE, 1));
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

        // Arm 3's late gamma: after the (combine) reduce, once per core.
        if constexpr (REGIME_A && GAMMA_LATE) {
            if (b == 0) {
                ingest_gamma<Wt_core>();
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
    cb_pop_front(cb_reduce_scaler, (!REGIME_A && W_PARTIAL > 0) ? 2 : 1);
}
