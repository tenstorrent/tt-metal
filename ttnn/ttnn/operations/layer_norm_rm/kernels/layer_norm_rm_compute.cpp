// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 4 (variance_rsqrt): full layer norm without affine transform.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

namespace NAMESPACE {

void MAIN {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // ========== Runtime args ==========
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    // ========== CB IDs ==========
    constexpr uint32_t cb_input_rm = 0;       // c_0: RM sticks from reader
    constexpr uint32_t cb_tilized = 1;        // c_1: tilized input (2*Wt for reuse)
    constexpr uint32_t cb_reduce_scaler = 8;  // c_8: reduce scaler (1/W)
    constexpr uint32_t cb_eps = 9;            // c_9: epsilon constant tile
    constexpr uint32_t cb_out_rm = 16;        // c_16: untilized output for writer
    constexpr uint32_t cb_mean = 24;          // c_24: mean tile (1 tile)
    constexpr uint32_t cb_centered = 25;      // c_25: centered values (2*Wt for reuse)
    constexpr uint32_t cb_sq = 26;            // c_26: squared centered
    constexpr uint32_t cb_var = 27;           // c_27: variance tile
    constexpr uint32_t cb_rstd = 28;          // c_28: rsqrt(var + eps) tile
    constexpr uint32_t cb_norm = 29;          // c_29: normalized output

    // ========== Hardware init ==========
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_out_rm);

    // ========== Main loop ==========
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Phase 1: Tilize c_0 -> c_1 (Wt tiles)
        compute_kernel_lib::tilize<Wt, cb_input_rm, cb_tilized>(1);

        // Phase 2: Reduce row for mean: c_1 -> c_24
        // WaitUpfrontNoPop: c_1 tiles persist for Phase 3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 3: sub(tilized_input, mean) with COL broadcast -> centered
        // A: c_1 (already waited by reduce, NoWaitNoPop)
        // B: c_24 (1 tile mean, COL broadcast, WaitAndPopPerTile)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop c_1 after sub
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square centered values: c_25 -> c_26
        // WaitUpfrontNoPop on c_25: tiles persist for Phase 7
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce row for variance: c_26 -> c_27
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_sq, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 6: Add epsilon + rsqrt: c_27 + c_9 -> c_28 with rsqrt post-op
        // A: c_27 (variance, WaitAndPopPerTile)
        // B: c_9 (epsilon, WaitUpfrontNoPop -- persistent, wait is no-op after first iter)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var, cb_eps, cb_rstd, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply centered by rstd: c_25 * c_28 with COL broadcast -> c_29
        // A: c_25 (already waited from Phase 4, NoWaitNoPop)
        // B: c_28 (1 tile rstd, COL broadcast, WaitAndPopPerTile)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_rstd, cb_norm, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop c_25 after mul
        cb_pop_front(cb_centered, Wt);

        // Phase 10: Untilize normalized -> c_16
        compute_kernel_lib::untilize<Wt, cb_norm, cb_out_rm>(1);
    }
}

}  // namespace NAMESPACE
