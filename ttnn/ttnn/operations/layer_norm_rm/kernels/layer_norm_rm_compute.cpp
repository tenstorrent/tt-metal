// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Compute Kernel
//
// Full pipeline:
//   Phase 1: tilize(cb_in_rm -> cb_tilized)
//   Phase 2: reduce SUM_ROW -> cb_mean
//   Phase 3: sub COL(cb_tilized, cb_mean) -> cb_centered
//   Phase 4: square(cb_centered) -> cb_var_input
//   Phase 5: reduce SUM_ROW -> cb_var, add_eps+rsqrt -> cb_inv_std
//   Phase 6: mul COL(cb_centered, cb_inv_std) -> cb_normed
//   Phase 7: [optional] mul NONE(normed, gamma), add NONE(scaled, beta)
//   Phase 8: untilize -> cb_out_rm
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
constexpr uint32_t cb_gamma = 6;
constexpr uint32_t cb_beta = 7;
constexpr uint32_t cb_eps = 8;
constexpr uint32_t cb_scaler = 9;
constexpr uint32_t cb_normed = 16;
constexpr uint32_t cb_out_rm = 17;
constexpr uint32_t cb_inv_std = 24;
}  // namespace

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

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
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_var, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 6: Multiply by inv_std (COL broadcast)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_inv_std, cb_normed, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_centered, Wt);

        // Phase 7: Optional affine transform (gamma * normed + beta)
        // After Phase 6: cb_tilized is free, cb_centered is free.
        // Strategy: reuse cb_tilized for tilized gamma/beta, cb_centered as intermediate.

        // The CB that will be untilized (final output before untilize)
        // Default: cb_normed (no affine). Updated below if affine is applied.
        constexpr uint32_t cb_pre_untilize = []() -> uint32_t {
            if constexpr (has_gamma && has_beta) {
                return cb_normed;  // gamma*normed -> cb_centered, then centered+beta -> cb_normed
            } else if constexpr (has_gamma) {
                return cb_centered;  // gamma*normed -> cb_centered
            } else if constexpr (has_beta) {
                return cb_centered;  // normed+beta -> cb_centered
            } else {
                return cb_normed;
            }
        }();

        if constexpr (has_gamma) {
            // Tilize gamma RM data: cb_gamma -> cb_tilized
            compute_kernel_lib::tilize<
                cb_gamma,
                cb_tilized,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

            if constexpr (has_beta) {
                // gamma*normed -> cb_centered (intermediate)
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                    cb_normed, cb_tilized, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            } else {
                // gamma*normed -> cb_centered (final)
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                    cb_normed, cb_tilized, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            }
        }

        if constexpr (has_beta) {
            // Tilize beta RM data: cb_beta -> cb_tilized
            compute_kernel_lib::tilize<
                cb_beta,
                cb_tilized,
                compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

            if constexpr (has_gamma) {
                // centered+beta -> cb_normed (final, reuse cb_normed)
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                    cb_centered, cb_tilized, cb_normed, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            } else {
                // normed+beta -> cb_centered (final)
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                    cb_normed, cb_tilized, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            }
        }

        // Phase 8: Untilize (cb_pre_untilize -> cb_out_rm)
        compute_kernel_lib::untilize<
            Wt,
            cb_pre_untilize,
            cb_out_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }

    // Release cb_eps (program-lifetime)
    cb_pop_front(cb_eps, 1);
}
