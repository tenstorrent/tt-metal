// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

// CB indices
constexpr uint32_t c_0 = 0;    // input
constexpr uint32_t c_1 = 1;    // scaler
constexpr uint32_t c_16 = 16;  // output
constexpr uint32_t c_24 = 24;  // max
constexpr uint32_t c_25 = 25;  // exp intermediate
constexpr uint32_t c_26 = 26;  // recip (1/sum)

// Compile-time args
constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t num_rows_or_cols = get_compile_time_arg_val(2);  // per-core work units
constexpr uint32_t dim = get_compile_time_arg_val(3);               // 0 = width, 1 = height
constexpr uint32_t numeric_stable = get_compile_time_arg_val(4);

void kernel_main() {
    // Hardware init
    compute_kernel_hw_startup(c_0, c_1, c_16);

    if constexpr (dim == 0) {
        // dim=-1 (width softmax)
        for (uint32_t wu = 0; wu < num_rows_or_cols; wu++) {
            // ============================================================
            // Phase 1: Find max along row (REDUCE_ROW on Wt tiles -> 1 tile in c_24)
            // ============================================================
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                c_0, c_1, c_24, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));

            // ============================================================
            // Phase 2: exp(x - max), accumulate sum, compute recip
            // ============================================================
            // Step 2a: sub(input, max) with exp post_op -> c_25
            // c_24 (max) uses WaitUpfrontNoPop -- persists for Phase 3
            compute_kernel_lib::sub<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                c_0, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt), [](uint32_t dst) {
                    exp_tile_init();
                    exp_tile(dst);
                });

            // Step 2b: reduce SUM on c_25 -> c_26, with recip post_reduce_op
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                c_25,
                c_1,
                c_26,
                compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });

            // ============================================================
            // Phase 3: Normalize = exp(x - max) * recip
            // ============================================================
            // Step 3a: sub(input, max) with exp post_op -> c_25
            // c_24 still persists (WaitUpfrontNoPop -- already waited, not popped)
            compute_kernel_lib::sub<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                c_0, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt), [](uint32_t dst) {
                    exp_tile_init();
                    exp_tile(dst);
                });

            // Step 3b: mul(exp_tiles, recip) -> c_16 (output)
            // c_26 (recip) uses WaitUpfrontNoPop -- popped manually after
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                c_25, c_26, c_16, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

            // Manual cleanup: pop persistent CBs
            cb_pop_front(c_24, 1);  // max
            cb_pop_front(c_26, 1);  // recip
        }
    } else {
        // dim=-2 (height softmax)
        for (uint32_t wu = 0; wu < num_rows_or_cols; wu++) {
            // ============================================================
            // Phase 1: Find max along column (REDUCE_COL on Ht tiles -> 1 tile in c_24)
            // ============================================================
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_COL,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                c_0, c_1, c_24, compute_kernel_lib::ReduceInputBlockShape::of(Ht, 1, 1));

            // ============================================================
            // Phase 2: exp(x - max), accumulate sum, compute recip
            // ============================================================
            // Step 2a: sub(input, max) with exp post_op -> c_25
            // c_24 (max) uses WaitUpfrontNoPop -- persists for Phase 3
            compute_kernel_lib::sub<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                c_0, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::of(Ht, 1), [](uint32_t dst) {
                    exp_tile_init();
                    exp_tile(dst);
                });

            // Step 2b: reduce SUM on c_25 -> c_26, with recip post_reduce_op
            compute_kernel_lib::reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_COL,
                compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                c_25,
                c_1,
                c_26,
                compute_kernel_lib::ReduceInputBlockShape::of(Ht, 1, 1),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                [](uint32_t dst_idx) {
                    recip_tile_init();
                    recip_tile(dst_idx);
                });

            // ============================================================
            // Phase 3: Normalize = exp(x - max) * recip
            // ============================================================
            // Step 3a: sub(input, max) with exp post_op -> c_25
            // c_24 still persists (already waited, not popped)
            compute_kernel_lib::sub<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                c_0, c_24, c_25, compute_kernel_lib::BinaryInputBlockShape::of(Ht, 1), [](uint32_t dst) {
                    exp_tile_init();
                    exp_tile(dst);
                });

            // Step 3b: mul(exp_tiles, recip) -> c_16 (output)
            // c_26 (recip) uses WaitUpfrontNoPop -- popped manually after
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                c_25, c_26, c_16, compute_kernel_lib::BinaryInputBlockShape::of(Ht, 1));

            // Manual cleanup: pop persistent CBs
            cb_pop_front(c_24, 1);  // max
            cb_pop_front(c_26, 1);  // recip
        }
    }
}
