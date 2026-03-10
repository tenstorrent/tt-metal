// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Compute Kernel
//
// Stage 1 (data_pipeline): tilize + untilize (identity passthrough)
// Stage 2 (reduce_mean):   + reduce SUM_ROW -> mean, sub COL bcast -> centered
// Stage 3 (variance_normalize): + square, reduce var, add_eps+rsqrt, mul inv_std
//
// Compile-time args:
//   [0] Wt               — tiles per row (W / 32)
//   [1] nblocks_per_core — (legacy, now read from runtime args)
//   [2] has_gamma        — 1 if gamma present, 0 otherwise
//   [3] has_beta         — 1 if beta present, 0 otherwise
//   [4] epsilon_packed   — epsilon as uint32 IEEE-754 bits

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

namespace {
constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_mean = 2;
constexpr uint32_t cb_centered = 3;
constexpr uint32_t cb_var_input = 4;
constexpr uint32_t cb_var = 5;
constexpr uint32_t cb_eps = 8;
constexpr uint32_t cb_scaler = 9;
constexpr uint32_t cb_normed = 16;
constexpr uint32_t cb_out_rm = 17;
constexpr uint32_t cb_inv_std = 24;
}  // namespace

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    // CT args [1]-[4] used in later stages

    // ========== Runtime args ==========
    const uint32_t nblocks_per_core = get_arg_val<uint32_t>(0);

    // ========== Hardware startup ==========
    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm);

    // Wait for cb_eps once (program-lifetime, filled by reader)
    cb_wait_front(cb_eps, 1);

    for (uint32_t block = 0; block < nblocks_per_core; ++block) {
        // Phase 1: Tilize (cb_in_rm -> cb_tilized)
        compute_kernel_lib::tilize<
            cb_in_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 2: Reduce mean (SUM row with 1/W scaler)
        cb_wait_front(cb_tilized, Wt);
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::NoWaitNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean (x - mean, COL broadcast)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square centered values
        cb_wait_front(cb_centered, Wt);
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_centered, cb_var_input, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5a: Reduce variance (SUM row with 1/W scaler)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_var_input, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 5b: Add epsilon + rsqrt post_op (var + eps -> inv_std)
        // cb_eps: NoWaitNoPop (program-lifetime, already waited before loop)
        // cb_var: WaitAndPopPerTile
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_var, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 6: Multiply by inv_std (COL broadcast)
        // cb_centered: already waited from Phase 4, NoWaitNoPop
        // cb_inv_std: WaitAndPopPerTile
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_inv_std, cb_normed, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_centered, Wt);

        // Phase 8: Untilize (cb_normed -> cb_out_rm)
        compute_kernel_lib::untilize<
            Wt,
            cb_normed,
            cb_out_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }

    // Release cb_eps (program-lifetime)
    cb_pop_front(cb_eps, 1);
}
