// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

namespace NAMESPACE {

void MAIN {
    // --- Compile-time args ---
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);      // 0
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);     // 8
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);        // 16
    constexpr uint32_t cb_max = get_compile_time_arg_val(3);        // 24
    constexpr uint32_t cb_exp = get_compile_time_arg_val(4);        // 25
    constexpr uint32_t cb_recip_sum = get_compile_time_arg_val(5);  // 26
    constexpr uint32_t R = get_compile_time_arg_val(6);             // tiles per work unit
    constexpr uint32_t numeric_stable = get_compile_time_arg_val(7);
    constexpr uint32_t num_work_units = get_compile_time_arg_val(8);
    constexpr uint32_t is_dim_h = get_compile_time_arg_val(9);  // 1 if dim=-2, 0 if dim=-1

    // Dimension-dependent constants
    // dim=-1: REDUCE_ROW, block row(R), broadcast COL, binary shape (1, R)
    // dim=-2: REDUCE_COL, block col(R), broadcast ROW, binary shape (R, 1)
    constexpr auto reduce_dim = is_dim_h ? ReduceDim::REDUCE_COL : ReduceDim::REDUCE_ROW;
    constexpr auto bcast_dim = is_dim_h ? compute_kernel_lib::BroadcastDim::ROW : compute_kernel_lib::BroadcastDim::COL;
    constexpr auto reduce_block = is_dim_h ? compute_kernel_lib::ReduceInputBlockShape::col(R)
                                           : compute_kernel_lib::ReduceInputBlockShape::row(R);
    constexpr auto binary_block = is_dim_h ? compute_kernel_lib::BinaryInputBlockShape::of(R, 1)
                                           : compute_kernel_lib::BinaryInputBlockShape::of(1, R);

    // Post-ops
    auto exp_post_op = [](uint32_t dst_idx) {
        exp_tile_init();
        exp_tile(dst_idx);
    };

    auto recip_post_op = [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    };

    // Init: binary_op_init_common is needed for stages that use reduce + binary ops
    binary_op_init_common(cb_input, cb_scaler, cb_out);

    // Wait for scaler tile (persistent, produced once by reader)
    cb_wait_front(cb_scaler, 1);

    for (uint32_t wu = 0; wu < num_work_units; ++wu) {
        if constexpr (numeric_stable == 1) {
            // ===== STABLE MODE: 4-phase softmax =====

            // Phase 1: max = reduce(input) along reduction dim
            compute_kernel_lib::reduce<
                PoolType::MAX,
                reduce_dim,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(cb_input, cb_scaler, cb_max, reduce_block);

            // Phase 2: exp(input - max) with broadcast
            compute_kernel_lib::sub<
                bcast_dim,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryOutputPolicy::PerTile,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_input, cb_max, cb_exp, binary_block, exp_post_op);
            cb_pop_front(cb_input, R);  // manual pop -- NoWaitNoPop on A

            // Phase 3: recip_sum = 1 / sum(exp) along reduction dim
            compute_kernel_lib::reduce<
                PoolType::SUM,
                reduce_dim,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                cb_exp,
                cb_scaler,
                cb_recip_sum,
                reduce_block,
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                recip_post_op);

            // Phase 4: output = exp * (1/sum) with broadcast
            compute_kernel_lib::mul<
                bcast_dim,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryOutputPolicy::PerTile,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_exp, cb_recip_sum, cb_out, binary_block);
            cb_pop_front(cb_exp, R);  // manual pop -- NoWaitNoPop on A

        } else {
            // ===== UNSTABLE MODE: 3-phase softmax (skip max subtraction) =====

            // Phase 1: exp(input) via copy_tiles with exp post-op
            compute_kernel_lib::copy_tiles<
                compute_kernel_lib::CopyInputPolicy::WaitAndPop,
                compute_kernel_lib::CopyDataFormatReconfig::NONE>(cb_input, cb_exp, R, exp_post_op);

            // Phase 3: recip_sum = 1 / sum(exp) along reduction dim
            compute_kernel_lib::reduce<
                PoolType::SUM,
                reduce_dim,
                compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
                compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
                cb_exp,
                cb_scaler,
                cb_recip_sum,
                reduce_block,
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                recip_post_op);

            // Phase 4: output = exp * (1/sum) with broadcast
            compute_kernel_lib::mul<
                bcast_dim,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryOutputPolicy::PerTile,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_exp, cb_recip_sum, cb_out, binary_block);
            cb_pop_front(cb_exp, R);  // manual pop -- NoWaitNoPop on A
        }
    }
}

}  // namespace NAMESPACE
