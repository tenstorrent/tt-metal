// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Compute Kernel
//
// Stage 2: Full standardization pipeline
//   Phase 1: Tilize (cb_in_rm -> cb_tilized)
//   Phase 2: Reduce SUM for mean (cb_tilized -> cb_mean, WaitUpfrontNoPop)
//   Phase 3: Raw LLK sub_bcast_cols (cb_tilized - cb_mean -> cb_centered)
//   Phase 4: Square (cb_centered -> cb_centered_sq, WaitUpfrontNoPop on centered)
//   Phase 5a: Reduce SUM for variance (cb_centered_sq -> cb_mean reused)
//   Phase 5b: Raw LLK add_bcast_scalar (cb_mean + cb_eps -> cb_inv_std_tmp)
//   Phase 5c: Manual copy_tile + rsqrt_tile (cb_inv_std_tmp -> cb_mean reused as inv_std)
//   Phase 6: mul_bcast_cols (cb_centered * cb_mean -> cb_normed)
//   Phase 7: Untilize (cb_normed -> cb_out_rm)
//
// Compile-time args:
//   [0] Wt - tiles per row (W / 32)
//
// Runtime args:
//   [0] num_blocks - number of tile-rows for this core

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_tilized = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_eps = 3;
constexpr uint32_t cb_out_rm = 16;
constexpr uint32_t cb_mean = 24;  // Also reused as inv_std after Phase 5c
constexpr uint32_t cb_centered = 25;
constexpr uint32_t cb_centered_sq = 26;
constexpr uint32_t cb_inv_std_tmp = 27;  // var + eps intermediate
constexpr uint32_t cb_normed = 28;

constexpr uint32_t Wt = get_compile_time_arg_val(0);

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm);

    if (num_blocks == 0) {
        return;
    }

    constexpr uint32_t ndst = compute_kernel_lib::DEST_AUTO_LIMIT;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // ================================================================
        // Phase 1: Tilize (cb_in_rm -> cb_tilized)
        // ================================================================
        compute_kernel_lib::tilize<
            cb_in_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // ================================================================
        // Phase 2: Reduce SUM for mean (tiles persist in cb_tilized)
        // Output: cb_mean = 1 tile (row mean)
        // ================================================================
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ================================================================
        // Phase 3: Subtract mean using raw LLK (softmax pattern)
        // cb_tilized: Wt tiles (persistent from WaitUpfrontNoPop)
        // cb_mean: 1 tile from reduce
        // Output: cb_centered = Wt tiles (x - mean)
        // ================================================================
        reconfig_data_format_srcb(cb_mean);
        cb_wait_front(cb_mean, 1);
        sub_bcast_cols_init_short(cb_tilized, cb_mean);

        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            uint32_t chunk = (wt + ndst <= Wt) ? ndst : (Wt - wt);
            tile_regs_acquire();
            for (uint32_t i = 0; i < chunk; ++i) {
                sub_tiles_bcast_cols(cb_tilized, cb_mean, wt + i, 0, i);
            }
            cb_reserve_back(cb_centered, chunk);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < chunk; ++i) {
                pack_tile(i, cb_centered);
            }
            tile_regs_release();
            cb_push_back(cb_centered, chunk);
        }
        cb_pop_front(cb_tilized, Wt);
        cb_pop_front(cb_mean, 1);

        // ================================================================
        // Phase 4: Square (cb_centered -> cb_centered_sq)
        // WaitUpfrontNoPop: cb_centered tiles persist for Phase 6
        // ================================================================
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile,
            compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
            cb_centered, cb_centered_sq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // ================================================================
        // Phase 5a: Reduce SUM for variance (cb_centered_sq -> cb_mean)
        // WaitAndPopPerTile: cb_centered_sq consumed
        // Scaler = 1/W, so SUM * scaler = mean of squares = variance
        // Reusing cb_mean (c_24) for variance output
        // ================================================================
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_centered_sq, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ================================================================
        // Phase 5b: Add epsilon using raw LLK (after reduce, avoid pack_reconfig)
        // cb_mean (c_24): 1 tile (variance)
        // cb_eps (c_3): 1 tile (epsilon scalar, persistent)
        // Output: cb_inv_std_tmp (c_27) = var + eps
        // ================================================================
        reconfig_data_format_srcb(cb_eps);
        cb_wait_front(cb_mean, 1);
        cb_wait_front(cb_eps, 1);
        add_bcast_scalar_init_short(cb_mean, cb_eps);

        tile_regs_acquire();
        add_tiles_bcast_scalar(cb_mean, cb_eps, 0, 0, 0);
        cb_reserve_back(cb_inv_std_tmp, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_inv_std_tmp);
        tile_regs_release();
        cb_push_back(cb_inv_std_tmp, 1);
        cb_pop_front(cb_mean, 1);
        // cb_eps: persistent, do NOT pop

        // ================================================================
        // Phase 5c: Manual rsqrt (cb_inv_std_tmp -> cb_mean reused as inv_std)
        // ================================================================
        cb_wait_front(cb_inv_std_tmp, 1);
        cb_reserve_back(cb_mean, 1);
        tile_regs_acquire();
        copy_tile_init(cb_inv_std_tmp);
        copy_tile(cb_inv_std_tmp, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mean);
        tile_regs_release();
        cb_pop_front(cb_inv_std_tmp, 1);
        cb_push_back(cb_mean, 1);

        // ================================================================
        // Phase 6: Multiply centered by inv_std (cb_centered * cb_mean -> cb_normed)
        // cb_centered: Wt tiles (persistent from Phase 4's WaitUpfrontNoPop)
        // cb_mean: 1 tile (inv_std, col vector)
        // Using raw LLK mul_bcast_cols for safety after manual rsqrt
        // ================================================================
        reconfig_data_format_srcb(cb_mean);
        cb_wait_front(cb_mean, 1);
        mul_bcast_cols_init_short(cb_centered, cb_mean);

        for (uint32_t wt = 0; wt < Wt; wt += ndst) {
            uint32_t chunk = (wt + ndst <= Wt) ? ndst : (Wt - wt);
            tile_regs_acquire();
            for (uint32_t i = 0; i < chunk; ++i) {
                mul_tiles_bcast_cols(cb_centered, cb_mean, wt + i, 0, i);
            }
            cb_reserve_back(cb_normed, chunk);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < chunk; ++i) {
                pack_tile(i, cb_normed);
            }
            tile_regs_release();
            cb_push_back(cb_normed, chunk);
        }
        cb_pop_front(cb_centered, Wt);
        cb_pop_front(cb_mean, 1);

        // ================================================================
        // Phase 7: Untilize (cb_normed -> cb_out_rm)
        // ================================================================
        compute_kernel_lib::untilize<
            Wt,
            cb_normed,
            cb_out_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
    }
}
