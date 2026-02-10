// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"

#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t num_tile_rows = get_compile_time_arg_val(1);
constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
constexpr uint32_t has_beta = get_compile_time_arg_val(3);

constexpr uint32_t cb_input_rm = tt::CBIndex::c_0;
constexpr uint32_t cb_tilized_input = tt::CBIndex::c_1;
constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma_tilized = tt::CBIndex::c_3;
constexpr uint32_t cb_beta_rm = tt::CBIndex::c_4;
constexpr uint32_t cb_beta_tilized = tt::CBIndex::c_5;
constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
constexpr uint32_t cb_eps_scalar = tt::CBIndex::c_7;
constexpr uint32_t cb_output_tiles = tt::CBIndex::c_8;
constexpr uint32_t cb_output_rm = tt::CBIndex::c_16;
constexpr uint32_t cb_mean = tt::CBIndex::c_24;
constexpr uint32_t cb_centered = tt::CBIndex::c_25;
constexpr uint32_t cb_centered_sq = tt::CBIndex::c_26;
constexpr uint32_t cb_var = tt::CBIndex::c_27;
constexpr uint32_t cb_rstd = tt::CBIndex::c_28;
constexpr uint32_t cb_normed = tt::CBIndex::c_29;
constexpr uint32_t cb_gamma_applied = tt::CBIndex::c_30;

void kernel_main() {
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm);

    // Pre-tilize gamma and beta (persistent, done once before main loop)
    if constexpr (has_gamma) {
        compute_kernel_lib::tilize<cb_gamma_rm, cb_gamma_tilized>(Wt, 1);
    }
    if constexpr (has_beta) {
        compute_kernel_lib::tilize<cb_beta_rm, cb_beta_tilized>(Wt, 1);
    }

    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Phase 1: Tilize input (c_0 -> c_1)
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized_input>(Wt, 1);

        // Phase 2: Row-wise mean via SUM reduce with 1/W scaler (c_1 -> c_24)
        // WaitUpfrontNoPop: tiles remain in c_1 for Phase 3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized_input, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Center x - mean (c_1, c_24 -> c_25)
        // A=c_1 (NoWaitNoPop - already waited by Phase 2), B=c_24 (WaitAndPopPerTile)
        // COL broadcast: mean is 1 tile broadcast across Wt tiles
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized_input, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        // Pop tilized input (NoWaitNoPop policy means we must pop manually)
        cb_pop_front(cb_tilized_input, Wt);

        // Phase 4: Square centered values (c_25 -> c_26)
        // NoWaitNoPop: centered tiles already in c_25, keep them for Phase 7
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_centered_sq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Reduce variance (c_26 -> c_27)
        // SUM reduce of squared centered values with 1/W scaler
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_centered_sq, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6a: Add epsilon (c_27 + c_7 -> c_24, reusing mean CB)
        // SCALAR broadcast: eps is 1 tile broadcast to var
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var, cb_eps_scalar, cb_mean, compute_kernel_lib::BinaryInputBlockShape::single());

        // Phase 6b: rsqrt(var+eps) -> c_28
        // Manual: copy from c_24 to DST, apply rsqrt, pack to c_28
        {
            cb_wait_front(cb_mean, 1);
            cb_reserve_back(cb_rstd, 1);

            copy_tile_to_dst_init_short(cb_mean);
            tile_regs_acquire();
            copy_tile(cb_mean, 0, 0);

            rsqrt_tile_init();
            rsqrt_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_rstd);
            pack_tile(0, cb_rstd);
            cb_push_back(cb_rstd, 1);
            tile_regs_release();
            cb_pop_front(cb_mean, 1);
        }

        // Phase 7: Multiply centered * rstd (c_25, c_28 -> c_8 or c_29)
        // COL broadcast: rstd is 1 tile broadcast across Wt centered tiles
        if constexpr (has_gamma || has_beta) {
            // Output to c_29 (normed), gamma/beta still to come
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                cb_centered, cb_rstd, cb_normed, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        } else {
            // No gamma/beta: output directly to c_8
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
                cb_centered, cb_rstd, cb_output_tiles, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        if constexpr (has_gamma && has_beta) {
            // Phase 8: gamma * normed (c_29, c_3 -> c_30)
            {
                cb_wait_front(cb_normed, Wt);
                reconfig_data_format(cb_normed, cb_gamma_tilized);
                pack_reconfig_data_format(cb_gamma_applied);
                mul_tiles_init(cb_normed, cb_gamma_tilized, false);
                for (uint32_t i = 0; i < Wt; i++) {
                    tile_regs_acquire();
                    mul_tiles(cb_normed, cb_gamma_tilized, i, i, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_gamma_applied, 1);
                    pack_tile(0, cb_gamma_applied);
                    cb_push_back(cb_gamma_applied, 1);
                    tile_regs_release();
                }
                cb_pop_front(cb_normed, Wt);
            }

            // Phase 9: gamma_applied + beta (c_30, c_5 -> c_8)
            {
                cb_wait_front(cb_gamma_applied, Wt);
                reconfig_data_format(cb_gamma_applied, cb_beta_tilized);
                pack_reconfig_data_format(cb_output_tiles);
                add_tiles_init(cb_gamma_applied, cb_beta_tilized, false);
                for (uint32_t i = 0; i < Wt; i++) {
                    tile_regs_acquire();
                    add_tiles(cb_gamma_applied, cb_beta_tilized, i, i, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_output_tiles, 1);
                    pack_tile(0, cb_output_tiles);
                    cb_push_back(cb_output_tiles, 1);
                    tile_regs_release();
                }
                cb_pop_front(cb_gamma_applied, Wt);
            }
        } else if constexpr (has_gamma) {
            // Phase 8: gamma * normed (c_29, c_3 -> c_8)
            {
                cb_wait_front(cb_normed, Wt);
                reconfig_data_format(cb_normed, cb_gamma_tilized);
                pack_reconfig_data_format(cb_output_tiles);
                mul_tiles_init(cb_normed, cb_gamma_tilized, false);
                for (uint32_t i = 0; i < Wt; i++) {
                    tile_regs_acquire();
                    mul_tiles(cb_normed, cb_gamma_tilized, i, i, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_output_tiles, 1);
                    pack_tile(0, cb_output_tiles);
                    cb_push_back(cb_output_tiles, 1);
                    tile_regs_release();
                }
                cb_pop_front(cb_normed, Wt);
            }
        } else if constexpr (has_beta) {
            // Phase 9: normed + beta (c_29, c_5 -> c_8)
            {
                cb_wait_front(cb_normed, Wt);
                reconfig_data_format(cb_normed, cb_beta_tilized);
                pack_reconfig_data_format(cb_output_tiles);
                add_tiles_init(cb_normed, cb_beta_tilized, false);
                for (uint32_t i = 0; i < Wt; i++) {
                    tile_regs_acquire();
                    add_tiles(cb_normed, cb_beta_tilized, i, i, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_output_tiles, 1);
                    pack_tile(0, cb_output_tiles);
                    cb_push_back(cb_output_tiles, 1);
                    tile_regs_release();
                }
                cb_pop_front(cb_normed, Wt);
            }
        }

        // Phase 10: Untilize (c_8 -> c_16)
        compute_kernel_lib::untilize<Wt, cb_output_tiles, cb_output_rm>(1);
    }
}
