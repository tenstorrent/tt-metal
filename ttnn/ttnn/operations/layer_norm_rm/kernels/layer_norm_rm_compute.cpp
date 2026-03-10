// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
// Stage 4 (affine_transform): Full layer norm with optional gamma/beta
// tilize -> reduce_mean -> sub_mean -> square -> reduce_var
// -> eps+rsqrt -> mul_rstd -> [mul_gamma] -> [add_beta] -> untilize

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // CB indices
    constexpr uint32_t cb_input_rm = 0;     // c_0: RM sticks from reader
    constexpr uint32_t cb_gamma = 5;        // c_5: tilized gamma (persistent)
    constexpr uint32_t cb_beta = 6;         // c_6: tilized beta (persistent)
    constexpr uint32_t cb_scaler = 8;       // c_8: reduce scaler (1/W)
    constexpr uint32_t cb_eps = 9;          // c_9: epsilon constant
    constexpr uint32_t cb_tilized = 16;     // c_16: tilized data (multi-use)
    constexpr uint32_t cb_output_rm = 17;   // c_17: untilized output for writer
    constexpr uint32_t cb_reduce_out = 24;  // c_24: reduce output (mean/variance)
    constexpr uint32_t cb_centered = 25;    // c_25: centered values / affine intermediates
    constexpr uint32_t cb_rstd = 27;        // c_27: rsqrt(var+eps)

    // Hardware init - must come first
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_tilized);

    // --- Tilize gamma/beta at program start (before main loop) ---
    // Reader has already pushed replicated gamma/beta sticks into c_0.
    // We tilize them into persistent CBs c_5 and c_6.
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<cb_input_rm, cb_gamma>(Wt, 1);
    }
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<cb_input_rm, cb_beta>(Wt, 1);
    }

    for (uint32_t block = 0; block < num_blocks_per_core; block++) {
        // Phase 1: Tilize (c_0 -> c_16)
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Wt, 1);

        // Phase 2: Reduce Mean (c_16 -> c_24)
        // WaitUpfrontNoPop: waits for Wt tiles in c_16, does NOT pop them (needed for Phase 3)
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_tilized, cb_scaler, cb_reduce_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract Mean (c_16, c_24 -> c_25)
        // A: c_16 already waited from Phase 2 (NoWaitPopAtEnd - pops at end)
        // B: c_24 freshly pushed (WaitUpfrontPopAtEnd - waits then pops at end)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_tilized, cb_reduce_out, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 4: Square Centered Values (c_25 -> c_16)
        // WaitUpfrontNoPop: waits for Wt tiles in c_25, holds them for Phase 7
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Reduce Variance (c_16 -> c_24)
        // BulkWaitBulkPop: waits for Wt tiles, processes, pops Wt tiles
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_tilized, cb_scaler, cb_reduce_out, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Epsilon + Rsqrt (c_24, c_9 -> c_27)
        // A: c_24 variance (WaitAndPopPerTile)
        // B: c_9 epsilon (WaitUpfrontNoPop - persistent across blocks)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_reduce_out, cb_eps, cb_rstd, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply by Rstd (c_25, c_27 -> c_16)
        // A: c_25 already waited from Phase 4 (NoWaitPopAtEnd - pops at end)
        // B: c_27 freshly pushed (WaitUpfrontPopAtEnd - waits then pops at end)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_centered, cb_rstd, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // After Phase 7: final_cb = cb_tilized (c_16)

        // Phase 8 (optional): Multiply Gamma (c_16 * c_5 -> c_25)
        if constexpr (has_gamma) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_gamma, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            // After Phase 8: final_cb = cb_centered (c_25)
        }

        // Phase 9 (optional): Add Beta (c_25 + c_6 -> c_16)
        if constexpr (has_beta) {
            // Determine input CB based on whether gamma was applied
            if constexpr (has_gamma) {
                // Input is c_25 from Phase 8
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_centered, cb_beta, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
                // After Phase 9: final_cb = cb_tilized (c_16)
            } else {
                // No gamma, input is c_16 from Phase 7
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_tilized, cb_beta, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
                // After Phase 9 (beta only): final_cb = cb_centered (c_25)
            }
        }

        // Phase 10: Untilize (final_cb -> c_17)
        // Determine which CB has the final result
        if constexpr (has_gamma && has_beta) {
            // gamma+beta: Phase 9 wrote to c_16
            compute_kernel_lib::untilize<Wt, cb_tilized, cb_output_rm>(1);
        } else if constexpr (has_gamma) {
            // gamma only: Phase 8 wrote to c_25
            compute_kernel_lib::untilize<Wt, cb_centered, cb_output_rm>(1);
        } else if constexpr (has_beta) {
            // beta only: Phase 9 wrote to c_25
            compute_kernel_lib::untilize<Wt, cb_centered, cb_output_rm>(1);
        } else {
            // no affine: Phase 7 wrote to c_16
            compute_kernel_lib::untilize<Wt, cb_tilized, cb_output_rm>(1);
        }
    }
}
