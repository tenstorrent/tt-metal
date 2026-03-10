// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
//
// Per-block pipeline:
//   Phase 1:  Tilize (cb_rm_in -> cb_tilized)
//   Phase 2:  Reduce row for mean (cb_tilized, cb_reduce_scaler -> cb_mean)
//   Phase 3:  Subtract mean, broadcast COL (cb_tilized, cb_mean -> cb_centered)
//             Manual pop cb_tilized after Phase 3.
//   -- Stage 1 bypass: route cb_centered -> cb_out_pre_untilize (skip phases 4-7) --
//   Phase 10: Untilize (cb_out_pre_untilize -> cb_rm_out)

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// CB indices
constexpr uint32_t cb_rm_in = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_reduce_scaler = 2;
constexpr uint32_t cb_eps = 3;
constexpr uint32_t cb_mean = 4;
constexpr uint32_t cb_centered = 5;
constexpr uint32_t cb_centered_sq = 6;
constexpr uint32_t cb_var = 7;
constexpr uint32_t cb_out_pre_untilize = 16;
constexpr uint32_t cb_rm_out = 17;
constexpr uint32_t cb_inv_std = 24;
constexpr uint32_t cb_gamma = 25;
constexpr uint32_t cb_beta = 26;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
    constexpr uint32_t has_beta = get_compile_time_arg_val(2);

    // Runtime args
    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    // Hardware init
    compute_kernel_hw_startup(cb_rm_in, cb_reduce_scaler, cb_out_pre_untilize);
    binary_op_init_common(cb_tilized, cb_mean, cb_out_pre_untilize);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Phase 1: Tilize (cb_rm_in -> cb_tilized)
        compute_kernel_lib::
            tilize<cb_rm_in, cb_tilized, compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit>(Wt, 1);

        // Phase 2: Reduce row for mean (cb_tilized -> cb_mean)
        // WaitUpfrontNoPop: tiles persist in cb_tilized for Phase 3
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean with COL broadcast (cb_tilized - cb_mean -> cb_out_pre_untilize)
        // Stage 1 bypass: output goes directly to cb_out_pre_untilize instead of cb_centered
        // A: cb_tilized [Wt tiles, already waited from Phase 2, NoWaitNoPop]
        // B: cb_mean [1 tile, WaitAndPopPerTile]
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_mean, cb_out_pre_untilize, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop cb_tilized after Phase 3
        cb_pop_front(cb_tilized, Wt);

        // Phase 10: Untilize (cb_out_pre_untilize -> cb_rm_out)
        compute_kernel_lib::untilize<
            Wt,
            cb_out_pre_untilize,
            cb_rm_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
    }
}
