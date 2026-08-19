// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
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
// WT_REDUCE_BLOCK / WT_SCALE_BLOCK and replaces the fused sum-of-squares with
// square -> accumulating reduce<SUM, REDUCE_ROW>.  The partial scaler
// (last_tile_at(1)) zeroes the pad columns of the last W-tile on the LAST chunk
// only, so implicit tile padding never enters the sum (risk R1).  The divisor is
// always 1/W_true, applied in fp32 by MulUnary - never folded into the
// mandatory-bfloat16 reduce scaler (risk R2).
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

// ---------------------------------------------------------------------------
// gamma_row0 bake-off: the per-stage zones are compiled OUT in this copy.
// The bake-off metric is DEVICE KERNEL DURATION [ns], which comes from the
// always-on *-KERNEL firmware markers, not from these optional user zones.  The
// device profiler hashes each zone source location into 16 bits and THROWS on a
// collision; with several parallel experiment copies of these kernels in one
// build log the table gets dense enough to collide.  Dropping the optional zones
// also keeps their marker cost out of every arm's measurement.
// ---------------------------------------------------------------------------
#undef MaybeDeviceZoneScope
#define MaybeDeviceZoneScope(name) ((void)0)

namespace ckl = compute_kernel_lib;

namespace {

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_squared = 3;
constexpr uint32_t cb_sumsq = 4;
constexpr uint32_t cb_rms_recip = 5;
constexpr uint32_t cb_normed = 6;
constexpr uint32_t cb_output_tiles = 7;
constexpr uint32_t cb_rm_in = 8;
constexpr uint32_t cb_rm_out = 9;
constexpr uint32_t cb_sumsq_acc = 10;
constexpr uint32_t cb_gamma_rm = 11;

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

    if constexpr (IS_ROW_MAJOR) {
        compute_kernel_hw_startup(cb_rm_in, cb_rm_in, cb_input_tiles);
    } else {
        compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_output_tiles);
    }

    // Regime A holds the whole per-core gamma width resident for the whole
    // kernel: ingested exactly once, never popped.
    if constexpr (REGIME_A) {
        ingest_gamma<Wt_core>();
    }

    // The partial form is datapath-specific (the reader emits the matching tile at
    // scaler index 1; see its scaler block).  `last_tile_at` carries only a scaler
    // INDEX, which AccumulateViaAdd reads as "tile-aligned" - it needs
    // `partial_mask`, which also carries the valid-element COUNT.
    constexpr auto partial_scaler = (W_PARTIAL == 0)
                                        ? ckl::ReducePartialScaler::none()
                                        : (REDUCE_VIA_ADD ? ckl::ReducePartialScaler::partial_mask(W_PARTIAL, 1)
                                                          : ckl::ReducePartialScaler::last_tile_at(1));

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

            {
                MaybeDeviceZoneScope("cp_reduce");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_sumsq_acc,
                    cb_reduce_scaler,
                    cb_sumsq>(ckl::ReduceInputBlockShape::of(BLOCK_HT, 1, 1));
            }
        } else {
            for (uint32_t c = 0; c < NUM_REDUCE_CHUNKS; ++c) {
                const bool is_last = (c + 1 == NUM_REDUCE_CHUNKS);
                if constexpr (IS_ROW_MAJOR) {
                    MaybeDeviceZoneScope("cp_tilize");
                    ckl::tilize<WT_REDUCE_BLOCK, cb_rm_in, cb_input_tiles>(BLOCK_HT);
                }
                {
                    MaybeDeviceZoneScope("cp_square");
                    ckl::square<
                        ckl::input(
                            cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                        ckl::output(cb_squared, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                        ckl::IterationShape::grid(BLOCK_HT, WT_REDUCE_BLOCK).block_size(DEST_BLOCK));
                }

                // `at_last` marks the finalizing chunk: on the AccumulateViaAdd
                // datapath cb_sumsq holds the RAW partial sum between chunks and
                // the within-tile collapse runs exactly once, on the last one.
                // ReduceTile ignores the flag (it finalizes every chunk), so the
                // same call site is correct for both datapaths.
                MaybeDeviceZoneScope("cp_reduce");
                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_squared,
                    cb_reduce_scaler,
                    cb_sumsq,
                    REDUCE_POLICY,
                    ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                    ReduceFp32Mode::Fast,  // default (global scope, not in ckl)
                    REDUCE_ALGORITHM>(
                    ckl::ReduceInputBlockShape::of(BLOCK_HT, WT_REDUCE_BLOCK, 1),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    is_last ? ckl::Accumulate::at_last(cb_sumsq, c) : ckl::Accumulate::at(cb_sumsq, c),
                    ckl::NoOp{},
                    is_last ? partial_scaler : ckl::ReducePartialScaler::none());
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
