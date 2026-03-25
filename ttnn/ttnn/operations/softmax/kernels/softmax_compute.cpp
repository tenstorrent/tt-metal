// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Compute Kernel
// 4-phase softmax: max_reduce, sub+exp, sum_reduce+recip, multiply
// Branches on is_dim_w at compile time for REDUCE_ROW vs REDUCE_COL.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"

constexpr uint32_t cb_input = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_max = 24;
constexpr uint32_t cb_exps = 25;
constexpr uint32_t cb_recip_sum = 26;

namespace {
using namespace compute_kernel_lib;
}  // namespace

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t is_dim_w = get_compile_time_arg_val(3);
    constexpr uint32_t numeric_stable = get_compile_time_arg_val(4);

    // Runtime args
    const uint32_t num_slices = get_arg_val<uint32_t>(0);

    // Slice size depends on reduction dimension
    // dim=-1: slice = one tile-row (Wt tiles), dim=-2: slice = one tile-column (Ht tiles)
    constexpr uint32_t slice_tiles = is_dim_w ? Wt : Ht;

    // Compile-time constants for dimension-dependent parameters
    constexpr ReduceDim reduce_dim = is_dim_w ? ReduceDim::REDUCE_ROW : ReduceDim::REDUCE_COL;
    constexpr BroadcastDim bcast_dim = is_dim_w ? BroadcastDim::COL : BroadcastDim::ROW;
    constexpr auto reduce_block = is_dim_w ? ReduceInputBlockShape::row(Wt) : ReduceInputBlockShape::col(Ht);
    constexpr auto binary_block = is_dim_w ? BinaryInputBlockShape::of(1, Wt) : BinaryInputBlockShape::of(Ht, 1);

    // Hardware init
    compute_kernel_hw_startup(cb_input, cb_scaler, cb_out);

    for (uint32_t slice = 0; slice < num_slices; ++slice) {
        if constexpr (numeric_stable) {
            // ====== Phase 1: Max reduction ======
            // WaitUpfrontNoPop: waits for all slice_tiles in cb_input, does NOT pop
            compute_kernel_lib::reduce<PoolType::MAX, reduce_dim, ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_input, cb_scaler, cb_max, reduce_block);

            // ====== Phase 2: Subtract max + exp ======
            // cb_input: already waited from Phase 1, NoWaitNoPop
            // cb_max: WaitUpfrontPopAtEnd -- waited and popped after all tiles processed
            compute_kernel_lib::sub<bcast_dim, BinaryInputPolicy::NoWaitNoPop, BinaryInputPolicy::WaitUpfrontPopAtEnd>(
                cb_input, cb_max, cb_exps, binary_block, [](uint32_t dst_idx) {
                    exp_tile_init();
                    exp_tile(dst_idx);
                });

            // Pop cb_input manually (NoWaitNoPop didn't pop it)
            cb_pop_front(cb_input, slice_tiles);
        } else {
            // ====== Phase 2 (unstable): Just exp, no max subtraction ======
            compute_kernel_lib::copy_tiles(cb_input, cb_exps, slice_tiles, [](uint32_t dst_idx) {
                exp_tile_init();
                exp_tile(dst_idx);
            });
        }
        // cb_exps has slice_tiles tiles
        // Must wait on cb_exps since NoWaitNoPop policies below expect caller to manage wait
        cb_wait_front(cb_exps, slice_tiles);

        // ====== Phase 3: Sum reduction + reciprocal ======
        compute_kernel_lib::reduce<PoolType::SUM, reduce_dim, ReduceInputPolicy::NoWaitNoPop>(
            cb_exps,
            cb_scaler,
            cb_recip_sum,
            reduce_block,
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            [](uint32_t dst_idx) {
                recip_tile_init();
                recip_tile(dst_idx);
            });

        // ====== Phase 4: Multiply exp * recip_sum ======
        compute_kernel_lib::mul<bcast_dim, BinaryInputPolicy::NoWaitPopAtEnd, BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_exps, cb_recip_sum, cb_out, binary_block);
    }
}
