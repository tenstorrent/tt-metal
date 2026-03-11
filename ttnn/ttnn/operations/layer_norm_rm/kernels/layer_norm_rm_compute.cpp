// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 4: tilize, reduce_row(mean), sub(COL), square, reduce_row(var),
//          add_eps+rsqrt, mul_inv_std(COL), mul_gamma(ROW), add_beta(ROW), untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB indices
constexpr uint32_t cb_rm_input = 0;    // RM input sticks
constexpr uint32_t cb_gamma = 1;       // Tilized gamma (Wt tiles, persistent)
constexpr uint32_t cb_beta = 2;        // Tilized beta (Wt tiles, persistent)
constexpr uint32_t cb_scaler = 8;      // Reduce scaler (1/W)
constexpr uint32_t cb_eps = 9;         // Epsilon constant tile
constexpr uint32_t cb_rm_output = 16;  // RM output sticks
constexpr uint32_t cb_tilized = 24;    // Tilized input / reused intermediate
constexpr uint32_t cb_mean = 25;       // Row mean / reused for variance (1 tile)
constexpr uint32_t cb_centered = 26;   // x - mean (Wt tiles)
constexpr uint32_t cb_squared = 27;    // (x - mean)^2 / reused for affine intermediate (Wt tiles)
constexpr uint32_t cb_inv_std = 28;    // rsqrt(var + eps) (1 tile)

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t max_blocks = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

// Determine which CB to untilize from based on affine config:
// no gamma, no beta: c_24 (normalized output)
// gamma + beta: c_24 (gamma c_24->c_27, beta c_27->c_24)
// gamma only: c_27 (gamma c_24->c_27)
// beta only: c_27 (beta c_24->c_27)
constexpr uint32_t cb_pre_untilize = (!has_gamma && !has_beta) ? cb_tilized
                                     : (has_gamma && has_beta) ? cb_tilized
                                                               : cb_squared;

void kernel_main() {
    // Hardware startup
    compute_kernel_hw_startup(cb_tilized, cb_scaler, cb_rm_output);

    // Get actual num_blocks for this core from runtime args
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Wait for persistent CBs (scaler and eps persist entire program)
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    // ========== Pre-loop: Tilize gamma/beta from c_0 to c_1/c_2 ==========
    if constexpr (has_gamma) {
        // Reader pushed gamma sticks into c_0 (Wt pages). Tilize c_0 -> c_1.
        compute_kernel_lib::
            tilize<cb_rm_input, cb_gamma, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);
        // c_1 now has Wt tiles of tilized gamma (Row0 valid for ROW broadcast)
        // Wait for gamma to be available (persistent)
        cb_wait_front(cb_gamma, Wt);
    }

    if constexpr (has_beta) {
        // Reader pushed beta sticks into c_0 (Wt pages). Tilize c_0 -> c_2.
        compute_kernel_lib::
            tilize<cb_rm_input, cb_beta, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);
        // c_2 now has Wt tiles of tilized beta (Row0 valid for ROW broadcast)
        // Wait for beta to be available (persistent)
        cb_wait_front(cb_beta, Wt);
    }

    // Main loop: per block
    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize (c_0 -> c_24)
        compute_kernel_lib::
            tilize<cb_rm_input, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);

        // Phase 2: Reduce Row for mean (c_24, c_8 -> c_25)
        // WaitUpfrontNoPop: c_24 tiles persist for Phase 3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 3: Sub mean with COL broadcast (c_24 - c_25 -> c_26)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square centered (c_26 -> c_27)
        // WaitUpfrontNoPop: c_26 tiles persist for Phase 7
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_squared, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce Row for variance (c_27, c_8 -> c_25)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_squared, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));

        // Phase 6: Add epsilon + rsqrt (c_25 + c_9 -> c_28)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_mean, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::of(1, 1), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply centered by inv_std with COL broadcast (c_26 * c_28 -> c_24)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_centered, cb_inv_std, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        cb_pop_front(cb_centered, Wt);
        // c_24 now has normalized output (Wt tiles)

        // Phase 8 (optional): Multiply by gamma with ROW broadcast (c_24 * c_1 -> c_27)
        if constexpr (has_gamma) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_tilized, cb_gamma, cb_squared, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
            // c_24 consumed, c_27 has gamma-scaled output
        }

        // Phase 9 (optional): Add beta with ROW broadcast
        if constexpr (has_beta) {
            // Input comes from Phase 8 output (c_27) if gamma, else from Phase 7 output (c_24)
            constexpr uint32_t affine_in = has_gamma ? cb_squared : cb_tilized;
            // Output goes to: c_24 if gamma+beta, c_27 if beta only
            constexpr uint32_t affine_out = has_gamma ? cb_tilized : cb_squared;
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                affine_in, cb_beta, affine_out, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        }

        // Phase 10: Untilize output (cb_pre_untilize -> c_16)
        compute_kernel_lib::untilize<
            Wt,
            cb_pre_untilize,
            cb_rm_output,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit>(1);
    }

    // Pop persistent CBs
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
    if constexpr (has_gamma) {
        cb_pop_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
