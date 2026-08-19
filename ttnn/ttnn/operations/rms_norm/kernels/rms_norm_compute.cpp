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
//                                              cb_input_tiles -> cb_sumsq
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

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

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

constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(0);
constexpr uint32_t REGIME_A = get_compile_time_arg_val(1);
constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(2);
constexpr uint32_t GAMMA_IS_ROW_MAJOR = get_compile_time_arg_val(3);
constexpr uint32_t Wt_core = get_compile_time_arg_val(4);
constexpr uint32_t W_PARTIAL = get_compile_time_arg_val(5);
constexpr uint32_t BLOCK_HT = get_compile_time_arg_val(6);
constexpr uint32_t WT_REDUCE_BLOCK = get_compile_time_arg_val(7);
constexpr uint32_t WT_REDUCE_TAIL = get_compile_time_arg_val(8);
constexpr uint32_t WT_SCALE_BLOCK = get_compile_time_arg_val(9);
constexpr uint32_t WT_SCALE_TAIL = get_compile_time_arg_val(10);
constexpr uint32_t Rt = get_compile_time_arg_val(11);
constexpr uint32_t NUM_ROWS = get_compile_time_arg_val(12);
constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(13);
constexpr uint32_t ELEM_SIZE = get_compile_time_arg_val(14);
constexpr uint32_t GAMMA_ELEM_SIZE = get_compile_time_arg_val(15);
constexpr uint32_t GAMMA_ROW_BYTES = get_compile_time_arg_val(16);
constexpr uint32_t DEST_BLOCK_CT = get_compile_time_arg_val(17);

constexpr uint32_t NUM_REDUCE_CHUNKS = (Wt_core + WT_REDUCE_BLOCK - 1) / WT_REDUCE_BLOCK;
constexpr uint32_t NUM_SCALE_CHUNKS = (Wt_core + WT_SCALE_BLOCK - 1) / WT_SCALE_BLOCK;

// The host-chosen DEST block knob, clamped against the REAL hardware constant.
// Never a literal 4 or 8.
constexpr uint32_t DEST_BLOCK = (DEST_BLOCK_CT < ckl::DEST_AUTO_LIMIT) ? DEST_BLOCK_CT : ckl::DEST_AUTO_LIMIT;

// The no-gamma path writes the 1/rms scale straight into the output CB, so it
// pays zero extra copies.
constexpr uint32_t cb_scale_out = HAS_GAMMA ? cb_normed : cb_output_tiles;

FORCE_INLINE uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

template <uint32_t W>
ALWI void tilize_w(uint32_t num_blocks) {
    ckl::tilize<W, cb_rm_in, cb_input_tiles>(num_blocks);
}

template <uint32_t W>
ALWI void untilize_w(uint32_t num_blocks) {
    ckl::untilize<W, cb_output_tiles, cb_rm_out>(num_blocks);
}

// tilize/untilize take their block width as a TEMPLATE parameter, so a W-chunk
// loop whose last chunk is narrower needs both widths instantiated.  When the
// tail equals the block the second instantiation folds away at compile time.
template <uint32_t BLK, uint32_t TAIL>
ALWI void tilize_chunk(uint32_t cw, uint32_t num_blocks) {
    if constexpr (BLK == TAIL) {
        tilize_w<BLK>(num_blocks);
    } else {
        if (cw == BLK) {
            tilize_w<BLK>(num_blocks);
        } else {
            tilize_w<TAIL>(num_blocks);
        }
    }
}

template <uint32_t BLK, uint32_t TAIL>
ALWI void untilize_chunk(uint32_t cw, uint32_t num_blocks) {
    if constexpr (BLK == TAIL) {
        untilize_w<BLK>(num_blocks);
    } else {
        if (cw == BLK) {
            untilize_w<BLK>(num_blocks);
        } else {
            untilize_w<TAIL>(num_blocks);
        }
    }
}

// x * (1/rms) [* gamma] over one (ht x cw) chunk.
//   RmsPop   : AtEnd in Regime A (one call per row-block); None in Regime B,
//              where cb_rms_recip is reused across every W-chunk and popped by
//              the caller after the last one.
//   GammaPop : None in Regime A (cb_gamma_tiles is resident and never popped);
//              AtEnd in Regime B, where the reader re-pushes the chunk's slice.
template <ckl::PopPolicy RmsPop, ckl::PopPolicy GammaPop>
ALWI void scale_chunk(uint32_t ht, uint32_t cw) {
    ckl::mul<
        ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
        ckl::input(cb_rms_recip, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, RmsPop, ckl::OperandKind::Col),
        ckl::output(cb_scale_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
        ckl::IterationShape::grid(ht, cw).block_size(DEST_BLOCK));

    if constexpr (HAS_GAMMA) {
        ckl::mul<
            ckl::input(cb_normed, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(
                cb_gamma_tiles, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, GammaPop, ckl::OperandKind::Row),
            ckl::output(cb_output_tiles, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
            ckl::IterationShape::grid(ht, cw).block_size(DEST_BLOCK));
    }
}

}  // namespace

void kernel_main() {
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);
    const uint32_t start_row_block = get_arg_val<uint32_t>(2);
    const uint32_t num_row_blocks_here = get_arg_val<uint32_t>(3);

    if constexpr (IS_ROW_MAJOR) {
        compute_kernel_hw_startup(cb_rm_in, cb_rm_in, cb_input_tiles);
    } else {
        compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_output_tiles);
    }

    constexpr auto partial_scaler =
        (W_PARTIAL > 0) ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    for (uint32_t b = 0; b < num_row_blocks_here; ++b) {
        const uint32_t rt0 = (start_row_block + b) * BLOCK_HT;
        const uint32_t ht = umin(BLOCK_HT, Rt - rt0);

        // ---------------- sum of squares over the reduced axis ---------------
        if constexpr (REGIME_A) {
            if constexpr (IS_ROW_MAJOR) {
                tilize_w<Wt_core>(ht);
            }
            // accumulate-then-finalize (catalog `row_reduce_accumulate`):
            // sum_of_squares folds the whole tile-row of x*x into ONE tile per
            // row, element-wise; the within-tile collapse along W is the single
            // reduce<SUM, REDUCE_ROW> below.  No mask is needed here - the
            // accumulator's 32 columns are all meaningful, since the only padded
            // columns live in the last W-tile and the RM reader zero-fills them.
            ckl::sum_of_squares<
                ckl::input(cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Block),
                ckl::row_output(cb_sumsq_acc)>(ckl::IterationShape::grid(ht, Wt_core).block_size(DEST_BLOCK));

            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_sumsq_acc,
                cb_reduce_scaler,
                cb_sumsq>(ckl::ReduceInputBlockShape::of(ht, 1, 1));
        } else {
            for (uint32_t c = 0; c < NUM_REDUCE_CHUNKS; ++c) {
                const bool is_last = (c + 1 == NUM_REDUCE_CHUNKS);
                const uint32_t cw = is_last ? WT_REDUCE_TAIL : WT_REDUCE_BLOCK;
                if constexpr (IS_ROW_MAJOR) {
                    tilize_chunk<WT_REDUCE_BLOCK, WT_REDUCE_TAIL>(cw, ht);
                }
                ckl::square<
                    ckl::input(
                        cb_input_tiles, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
                    ckl::output(cb_squared, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(
                    ckl::IterationShape::grid(ht, cw).block_size(DEST_BLOCK));

                ckl::reduce<
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW,
                    cb_squared,
                    cb_reduce_scaler,
                    cb_sumsq>(
                    ckl::ReduceInputBlockShape::of(ht, cw, 1),
                    ckl::ReduceInputMemoryLayout::contiguous(),
                    ckl::Accumulate::at(cb_sumsq, c),
                    ckl::NoOp{},
                    is_last ? partial_scaler : ckl::ReducePartialScaler::none());
            }
        }

        // ---------------- rms chain: *1/W, +eps, rsqrt -----------------------
        // One helper call, one dst-sync window, no constant CBs.  MulUnary /
        // AddUnary take fp32 bit patterns, so 1/W and epsilon are applied at
        // full fp32 precision.
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(ht),
            ckl::CopyTile<ckl::input(cb_sumsq)>{},
            ckl::MulUnary<>{inv_w_bits},
            ckl::AddUnary<>{eps_bits},
            ckl::Rsqrt<>{},
            ckl::PackTile<ckl::output(cb_rms_recip)>{});

        // ---------------- scale (and gamma) ----------------------------------
        if constexpr (REGIME_A) {
            scale_chunk<ckl::PopPolicy::AtEnd, ckl::PopPolicy::None>(ht, Wt_core);
            if constexpr (IS_ROW_MAJOR) {
                untilize_w<Wt_core>(ht);
            }
        } else {
            for (uint32_t c = 0; c < NUM_SCALE_CHUNKS; ++c) {
                const bool is_last = (c + 1 == NUM_SCALE_CHUNKS);
                const uint32_t cw = is_last ? WT_SCALE_TAIL : WT_SCALE_BLOCK;
                if constexpr (IS_ROW_MAJOR) {
                    tilize_chunk<WT_SCALE_BLOCK, WT_SCALE_TAIL>(cw, ht);
                }
                scale_chunk<ckl::PopPolicy::None, ckl::PopPolicy::AtEnd>(ht, cw);
                if constexpr (IS_ROW_MAJOR) {
                    untilize_chunk<WT_SCALE_BLOCK, WT_SCALE_TAIL>(cw, ht);
                }
            }
            // cb_rms_recip was held across every W-chunk (PopPolicy::None).
            cb_pop_front(cb_rms_recip, ht);
        }
    }

    // reduce() waits on the scaler CB but never pops it - release the pages
    // exactly once, at the very end (risk R9).  Only Regime B ever emits the
    // second (partial) tile.
    cb_pop_front(cb_reduce_scaler, (!REGIME_A && W_PARTIAL > 0) ? 2 : 1);
}
