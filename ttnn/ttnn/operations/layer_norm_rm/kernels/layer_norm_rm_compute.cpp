// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
//
// Per-block pipeline:
//   Phase 1:  Tilize (cb_rm_in -> cb_tilized)
//   Phase 2:  Reduce row for mean (cb_tilized, cb_reduce_scaler -> cb_mean)
//   Phase 3:  Subtract mean, broadcast COL (cb_tilized, cb_mean -> cb_centered)
//             Manual pop cb_tilized after Phase 3.
//   Phase 4:  Square centered (cb_centered -> cb_centered_sq)
//   Phase 5:  Reduce row for variance (cb_centered_sq, cb_reduce_scaler -> cb_var)
//   Phase 6:  Add eps + rsqrt post_op (cb_var, cb_eps -> cb_inv_std)
//   Phase 7:  Multiply inv_std, broadcast SCALAR (cb_centered, cb_inv_std -> cb_out_pre_untilize)
//             Manual pop cb_centered after Phase 7.
//   Phase 8:  Optional: multiply gamma, broadcast ROW (stage 3)
//   Phase 9:  Optional: add beta, broadcast ROW (stage 3)
//   Phase 10: Untilize (cb_out_pre_untilize -> cb_rm_out)

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

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

    // Wait for program-lifetime CBs once at kernel start
    // cb_eps: needed by Phase 6 (add eps), NoWaitNoPop policy there
    cb_wait_front(cb_eps, 1);

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

        // Phase 3: Subtract mean with COL broadcast (cb_tilized - cb_mean -> cb_centered)
        // A: cb_tilized [Wt tiles, already waited from Phase 2, NoWaitNoPop]
        // B: cb_mean [1 tile, WaitAndPopPerTile]
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop cb_tilized after Phase 3
        cb_pop_front(cb_tilized, Wt);

        // Phase 4: Square centered values (cb_centered -> cb_centered_sq)
        // WaitUpfrontNoPop: cb_centered persists for Phase 7
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_centered, cb_centered_sq, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce row for variance (cb_centered_sq -> cb_var)
        // WaitAndPopPerTile: cb_centered_sq is consumed and freed
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
            cb_centered_sq, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Add epsilon + rsqrt (cb_var + cb_eps -> cb_inv_std)
        // A: cb_var [1 tile, WaitAndPopPerTile - consumed]
        // B: cb_eps [1 tile, program lifetime, NoWaitNoPop - already waited at kernel start]
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_var, cb_eps, cb_inv_std, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 7: Multiply by inv_std with COL broadcast (cb_centered * cb_inv_std -> cb_out_pre_untilize)
        // inv_std is a REDUCE_ROW output with per-row values in col0 -- needs COL broadcast
        // A: cb_centered [Wt tiles, already waited from Phase 4, NoWaitNoPop]
        // B: cb_inv_std [1 tile, WaitAndPopPerTile - consumed]
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryOutputPolicy::Bulk,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_centered, cb_inv_std, cb_out_pre_untilize, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Manual pop cb_centered after Phase 7
        cb_pop_front(cb_centered, Wt);

        // Phase 10: Untilize (cb_out_pre_untilize -> cb_rm_out)
        compute_kernel_lib::untilize<
            Wt,
            cb_out_pre_untilize,
            cb_rm_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(1);
    }

    // Pop program-lifetime CBs at kernel end
    cb_pop_front(cb_eps, 1);
}
