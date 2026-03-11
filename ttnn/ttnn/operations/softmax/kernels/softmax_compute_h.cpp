// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Compute Kernel (dim=-2, height reduction)
// 4-phase per chunk: max(REDUCE_COL), sub+exp(ROW broadcast), sum(REDUCE_COL)+recip, mul(ROW broadcast)
// Input tiles are in chunked column order with row_stride = current_chunk

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_max = 24;
constexpr uint32_t cb_exp = 25;
constexpr uint32_t cb_recip = 26;

namespace NAMESPACE {

void MAIN {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_size = get_compile_time_arg_val(1);  // max chunk width
    constexpr uint32_t NC = get_compile_time_arg_val(2);          // always 1 (batch folded)
    constexpr uint32_t numeric_stable = get_compile_time_arg_val(3);

    // Runtime args
    const uint32_t num_cols = get_arg_val<uint32_t>(0);  // total columns assigned to this core

    // Hardware startup
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_out);

    // Process columns in chunks of chunk_size
    uint32_t cols_remaining = num_cols;

    while (cols_remaining > 0) {
        const uint32_t current_chunk = (cols_remaining < chunk_size) ? cols_remaining : chunk_size;
        const uint32_t chunk_tiles = Ht * current_chunk;

        // ==============================================================
        // Phase 1: reduce_max(input, REDUCE_COL) -> cb_max
        // Explicit wait for NoWaitNoPop; input tiles persist for Phase 2
        // ==============================================================
        cb_wait_front(cb_input, chunk_tiles);
        compute_kernel_lib::reduce<
            PoolType::MAX,
            ReduceDim::REDUCE_COL,
            compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
            cb_input,
            cb_scaler,
            cb_max,
            compute_kernel_lib::ReduceInputBlockShape::of(Ht, current_chunk, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(current_chunk));

        // ==============================================================
        // Phase 2: sub(input - max) + exp -> cb_exp
        // input: Ht*chunk tiles, already waited from Phase 1, pop at end
        // max: current_chunk tiles (ROW-shaped from REDUCE_COL), wait upfront + pop at end
        // Output: Ht*chunk tiles, bulk
        // ROW broadcast: REDUCE_COL output has Row0 valid, replicate down across Ht rows
        // ==============================================================
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_input,
            cb_max,
            cb_exp,
            compute_kernel_lib::BinaryInputBlockShape::of(Ht, current_chunk),
            [](uint32_t dst_idx) {
                exp_tile_init();
                exp_tile(dst_idx);
            });

        // ==============================================================
        // Phase 3: reduce_sum(exp, REDUCE_COL) + recip -> cb_recip
        // exp: Ht*chunk tiles (persist for Phase 4)
        // Output: current_chunk tiles
        // ==============================================================
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_exp,
            cb_scaler,
            cb_recip,
            compute_kernel_lib::ReduceInputBlockShape::of(Ht, current_chunk, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::with_row_stride(current_chunk),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });

        // ==============================================================
        // Phase 4: mul(exp, recip_sum, ROW broadcast) -> cb_out
        // exp: Ht*chunk tiles, already waited, pop at end
        // recip: current_chunk tiles (ROW-shaped), wait+pop per tile
        // Output: per-tile for writer overlap
        // ROW broadcast: REDUCE_COL output has Row0 valid, replicate down
        // ==============================================================
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_exp, cb_recip, cb_out, compute_kernel_lib::BinaryInputBlockShape::of(Ht, current_chunk));

        cols_remaining -= current_chunk;
    }
}

}  // namespace NAMESPACE
