// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm - Compute Kernel
// Stage 4: affine_transform - full normalization with optional gamma/beta.
// Per row: mean -> center -> square -> variance -> eps+rsqrt -> normalize -> gamma*x+beta

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);
    constexpr uint32_t gamma_has_value = get_compile_time_arg_val(2);
    constexpr uint32_t beta_has_value = get_compile_time_arg_val(3);

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_scaler = 1;
    constexpr uint32_t cb_eps = 2;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t cb_mean = 24;
    constexpr uint32_t cb_centered = 25;
    constexpr uint32_t cb_squared = 26;
    constexpr uint32_t cb_var = 27;
    constexpr uint32_t cb_rstd = 28;
    constexpr uint32_t cb_normalized = 29;
    constexpr uint32_t cb_gamma_out = 30;
    constexpr uint32_t cb_gamma = 3;
    constexpr uint32_t cb_beta = 4;

    // Determine the output CB for Phase 6 (normalize):
    // If affine transform is enabled, normalize outputs to cb_normalized;
    // otherwise it outputs directly to cb_out.
    constexpr bool has_affine = (gamma_has_value || beta_has_value);
    constexpr uint32_t cb_norm_out = has_affine ? cb_normalized : cb_out;

    // CRITICAL: Must call binary_op_init_common before any reduce/bcast operations.
    binary_op_init_common(cb_input, cb_scaler, cb_out);

    // Wait for scaler tile (filled once by reader, used every row)
    cb_wait_front(cb_scaler, 1);
    // Wait for epsilon tile (filled once by reader, used every row)
    cb_wait_front(cb_eps, 1);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Wait for reader to push Wt input tiles for this row
        cb_wait_front(cb_input, Wt);

        // ====== Phase 1: Reduce input row to mean (REDUCE_ROW) ======
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_input, cb_scaler, cb_mean);

        cb_reserve_back(cb_mean, 1);
        tile_regs_acquire();
        for (uint32_t w = 0; w < Wt; ++w) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_input, cb_scaler, w, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_mean);
        tile_regs_release();
        cb_push_back(cb_mean, 1);

        reduce_uninit();

        // ====== Phase 2: Subtract mean from each input tile (COL broadcast) ======
        cb_wait_front(cb_mean, 1);

        init_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::COL>(cb_input, cb_mean, cb_centered);

        for (uint32_t w = 0; w < Wt; ++w) {
            cb_reserve_back(cb_centered, 1);
            tile_regs_acquire();
            sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_mean, w, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_centered);
            tile_regs_release();
            cb_push_back(cb_centered, 1);
        }

        // Pop input and mean
        cb_pop_front(cb_input, Wt);
        cb_pop_front(cb_mean, 1);

        // ====== Phase 3: Square centered tiles (element-wise mul) ======
        cb_wait_front(cb_centered, Wt);

        // Full init for non-broadcast multiply, including packer for cb_squared
        binary_op_init_common(cb_centered, cb_centered, cb_squared);
        mul_tiles_init(cb_centered, cb_centered);

        for (uint32_t w = 0; w < Wt; ++w) {
            cb_reserve_back(cb_squared, 1);
            tile_regs_acquire();
            mul_tiles(cb_centered, cb_centered, w, w, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_squared);
            tile_regs_release();
            cb_push_back(cb_squared, 1);
        }

        // ====== Phase 4: Reduce squared to variance (REDUCE_ROW) ======
        cb_wait_front(cb_squared, Wt);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_squared, cb_scaler, cb_var);

        cb_reserve_back(cb_var, 1);
        tile_regs_acquire();
        for (uint32_t w = 0; w < Wt; ++w) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_squared, cb_scaler, w, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_var);
        tile_regs_release();
        cb_push_back(cb_var, 1);

        cb_pop_front(cb_squared, Wt);

        reduce_uninit();

        // ====== Phase 5: Add epsilon to variance, then rsqrt -> rstd ======
        cb_wait_front(cb_var, 1);

        // Full init for add with scalar broadcast, configures packer for cb_rstd
        init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR>(cb_var, cb_eps, cb_rstd);

        cb_reserve_back(cb_rstd, 1);
        tile_regs_acquire();

        add_tiles_bcast_scalar(cb_var, cb_eps, 0, 0, 0);

        // Wait for add to complete before SFPU rsqrt operates on DST
        tile_regs_wait();

        // rsqrt in-place in DST: 1/sqrt(var + eps)
        rsqrt_tile_init<true>();
        rsqrt_tile<true>(0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_rstd);
        tile_regs_release();
        cb_push_back(cb_rstd, 1);

        cb_pop_front(cb_var, 1);

        // ====== Phase 6: Normalize: centered * rstd (COL broadcast) ======
        // Output goes to cb_normalized if affine transform follows, else cb_out.
        cb_wait_front(cb_rstd, 1);

        init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_centered, cb_rstd, cb_norm_out);

        for (uint32_t w = 0; w < Wt; ++w) {
            cb_reserve_back(cb_norm_out, 1);
            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::COL>(cb_centered, cb_rstd, w, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_norm_out);
            tile_regs_release();
            cb_push_back(cb_norm_out, 1);
        }

        // Pop persistent CBs for this row
        cb_pop_front(cb_centered, Wt);
        cb_pop_front(cb_rstd, 1);

        // ====== Phase 7 (if gamma): Multiply normalized by gamma (ROW broadcast) ======
        if constexpr (gamma_has_value) {
            // Determine output CB: if beta follows, write to cb_gamma_out; else write to cb_out
            constexpr uint32_t cb_phase7_out = beta_has_value ? cb_gamma_out : cb_out;

            cb_wait_front(cb_normalized, Wt);
            cb_wait_front(cb_gamma, Wt);

            init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(cb_normalized, cb_gamma, cb_phase7_out);

            for (uint32_t w = 0; w < Wt; ++w) {
                cb_reserve_back(cb_phase7_out, 1);
                tile_regs_acquire();
                mul_tiles_bcast<BroadcastType::ROW>(cb_normalized, cb_gamma, w, w, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_phase7_out);
                tile_regs_release();
                cb_push_back(cb_phase7_out, 1);
            }

            cb_pop_front(cb_normalized, Wt);
            cb_pop_front(cb_gamma, Wt);
        }

        // ====== Phase 8 (if beta): Add beta (ROW broadcast) ======
        if constexpr (beta_has_value) {
            // Input CB depends on whether gamma was applied
            constexpr uint32_t cb_phase8_in = gamma_has_value ? cb_gamma_out : cb_normalized;

            cb_wait_front(cb_phase8_in, Wt);
            cb_wait_front(cb_beta, Wt);

            init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(cb_phase8_in, cb_beta, cb_out);

            for (uint32_t w = 0; w < Wt; ++w) {
                cb_reserve_back(cb_out, 1);
                tile_regs_acquire();
                add_tiles_bcast<BroadcastType::ROW>(cb_phase8_in, cb_beta, w, w, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
                cb_push_back(cb_out, 1);
            }

            cb_pop_front(cb_phase8_in, Wt);
            cb_pop_front(cb_beta, Wt);
        }
    }

    // Pop scaler and eps (filled once, used for all rows)
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
}
