// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Compute Kernel (dim=-1, width reduction)
// Unstable: exp -> reduce_sum+recip -> mul
// Stable: reduce_max -> sub+exp -> reduce_sum+recip -> mul

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
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
    constexpr uint32_t Ht = get_compile_time_arg_val(0);  // 1 for dim=-1
    constexpr uint32_t Wt = get_compile_time_arg_val(1);  // tile-cols per row
    constexpr uint32_t NC = get_compile_time_arg_val(2);  // always 1 (batch folded)
    constexpr uint32_t numeric_stable = get_compile_time_arg_val(3);

    // Runtime args
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    // Hardware startup
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_out);

    for (uint32_t unit = 0; unit < num_units; ++unit) {
        if constexpr (numeric_stable) {
            // ==============================================================
            // STABLE SOFTMAX: max -> sub+exp -> sum+recip -> mul
            // ==============================================================

            // Phase 1: reduce_max(input, REDUCE_ROW) -> c_max
            // Explicit wait for NoWaitNoPop; c_0 tiles persist for Phase 2
            cb_wait_front(cb_input, Wt);
            compute_kernel_lib::reduce<
                PoolType::MAX,
                ReduceDim::REDUCE_ROW,
                compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop,
                compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
                cb_input, cb_scaler, cb_max, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1));

            // Phase 2: sub(input - max) + exp -> c_exp
            // c_0: Wt tiles, already waited from Phase 1, pop at end
            // c_max: 1 tile (COL vector from REDUCE_ROW), wait+pop
            compute_kernel_lib::sub<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryOutputPolicy::Bulk,
                compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
                cb_input, cb_max, cb_exp, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt), [](uint32_t dst_idx) {
                    exp_tile_init();
                    exp_tile(dst_idx);
                });
        } else {
            // ==============================================================
            // UNSTABLE SOFTMAX: exp only -> c_exp
            // ==============================================================
            copy_tile_to_dst_init_short(cb_input);
            exp_tile_init();

            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(cb_input, 1);

                tile_regs_acquire();
                copy_tile(cb_input, 0, 0);
                exp_tile(0);
                tile_regs_commit();

                tile_regs_wait();
                cb_reserve_back(cb_exp, 1);
                pack_tile(0, cb_exp);
                cb_push_back(cb_exp, 1);
                tile_regs_release();

                cb_pop_front(cb_input, 1);
            }
        }

        // ==============================================================
        // Phase 3: reduce_sum(exp, REDUCE_ROW) + recip -> c_recip
        // c_exp has Wt tiles (persist for Phase 4)
        // ==============================================================
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_exp,
            cb_scaler,
            cb_recip,
            compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });

        // ==============================================================
        // Phase 4: mul(exp, recip_sum, COL broadcast) -> c_out
        // REDUCE_ROW output is COL-shaped (col0 valid, per-row values)
        // c_exp: Wt tiles, already waited, pop at end
        // c_recip: 1 tile (COL vector), wait+pop per tile
        // ==============================================================
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_exp, cb_recip, cb_out, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
    }
}

}  // namespace NAMESPACE
