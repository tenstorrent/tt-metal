// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Compute Kernel (Stage 4: affine)
// Phase 0: Tilize (cb_input_rm -> cb_tilized, persistent)
// Phase 1: Compute per-group mean via mul_tiles(x, binary_mask) + reduce_tile
// Phase 2: Compute per-group E[x^2] via mul_tiles(x^2, binary_mask) + reduce_tile
// Phase 3: Compute den_g = rsqrt(E[x^2] - mean_g^2 + eps) per group
// Phase 4: Build mean_tile, subtract mean, build den_tile, multiply by den
// Phase 4b: Affine transform: gamma * normalized + beta
// Phase 5: Untilize (cb_normalized -> cb_output_rm)

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t num_samples = get_compile_time_arg_val(3);

    // ========== CB INDICES ==========
    constexpr uint32_t cb_input_rm = 0;
    constexpr uint32_t cb_tilized = 1;
    constexpr uint32_t cb_gamma = 2;
    constexpr uint32_t cb_beta = 3;
    constexpr uint32_t cb_eps = 4;
    constexpr uint32_t cb_scaler = 5;
    constexpr uint32_t cb_mean = 6;
    constexpr uint32_t cb_den = 7;
    constexpr uint32_t cb_normalized = 16;
    constexpr uint32_t cb_output_rm = 17;
    constexpr uint32_t cb_scratch = 8;
    constexpr uint32_t cb_sq_sum = 24;
    constexpr uint32_t cb_tmp = 25;
    constexpr uint32_t cb_group_scaler = 26;

    // ========== HW STARTUP ==========
    compute_kernel_hw_startup(cb_input_rm, cb_tilized);

    for (uint32_t n = 0; n < num_samples; ++n) {
        // ========== PHASE 0: TILIZE ==========
        compute_kernel_lib::tilize<cb_input_rm, cb_tilized>(Ct, Ht);

        // Wait for all tilized data (persistent CB)
        cb_wait_front(cb_tilized, Ht * Ct);

        // Wait for persistent CBs needed by compute
        cb_wait_front(cb_scaler, 1);
        cb_wait_front(cb_eps, 1);
        cb_wait_front(cb_group_scaler, G * Ct);
        cb_wait_front(cb_gamma, Ct);
        cb_wait_front(cb_beta, Ct);

        // ========== PHASE 1: COMPUTE PER-GROUP MEAN ==========
        // mean_g = (1/K) * sum(x * mask_g) for each group
        for (uint32_t g = 0; g < G; ++g) {
            bool first_tile = true;

            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t ct = 0; ct < Ct; ++ct) {
                    uint32_t tilized_idx = ht * Ct + ct;
                    uint32_t mask_idx = g * Ct + ct;

                    // mul_tiles: x * mask -> cb_tmp
                    mul_tiles_init(cb_tilized, cb_group_scaler);
                    tile_regs_acquire();
                    mul_tiles(cb_tilized, cb_group_scaler, tilized_idx, mask_idx, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_tmp, 1);
                    pack_tile(0, cb_tmp);
                    tile_regs_release();
                    cb_push_back(cb_tmp, 1);

                    // reduce_tile: accumulate with 1/K scaler
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, cb_sq_sum);
                    tile_regs_acquire();

                    if (!first_tile) {
                        copy_tile_to_dst_init_short_with_dt(cb_sq_sum, cb_tmp);
                        cb_wait_front(cb_sq_sum, 1);
                        copy_tile(cb_sq_sum, 0, 0);
                        cb_pop_front(cb_sq_sum, 1);
                        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, cb_sq_sum);
                    }

                    cb_wait_front(cb_tmp, 1);
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, 0, 0, 0);
                    cb_pop_front(cb_tmp, 1);

                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_sq_sum, 1);
                    pack_tile(0, cb_sq_sum);
                    tile_regs_release();
                    cb_push_back(cb_sq_sum, 1);

                    reduce_uninit(cb_tmp);
                    first_tile = false;
                }
            }

            // Move final mean from cb_sq_sum to cb_mean
            copy_tile_to_dst_init_short(cb_sq_sum);
            tile_regs_acquire();
            cb_wait_front(cb_sq_sum, 1);
            copy_tile(cb_sq_sum, 0, 0);
            cb_pop_front(cb_sq_sum, 1);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_mean, 1);
            pack_tile(0, cb_mean);
            tile_regs_release();
            cb_push_back(cb_mean, 1);
        }

        // cb_mean now has G tiles
        cb_wait_front(cb_mean, G);

        // ========== PHASE 2: COMPUTE PER-GROUP E[x^2] ==========
        // E[x^2]_g = (1/K) * sum(x^2 * mask_g) for each group
        // Same pattern as Phase 1 but with x^2
        for (uint32_t g = 0; g < G; ++g) {
            bool first_tile = true;

            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t ct = 0; ct < Ct; ++ct) {
                    uint32_t tilized_idx = ht * Ct + ct;
                    uint32_t mask_idx = g * Ct + ct;

                    // Step 1: x^2 = mul_tiles(x, x) -> cb_tmp
                    mul_tiles_init(cb_tilized, cb_tilized);
                    tile_regs_acquire();
                    mul_tiles(cb_tilized, cb_tilized, tilized_idx, tilized_idx, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_tmp, 1);
                    pack_tile(0, cb_tmp);
                    tile_regs_release();
                    cb_push_back(cb_tmp, 1);

                    // Step 2: x^2 * mask = mul_tiles(cb_tmp, mask) -> cb_tmp
                    mul_tiles_init(cb_tmp, cb_group_scaler);
                    tile_regs_acquire();
                    cb_wait_front(cb_tmp, 1);
                    mul_tiles(cb_tmp, cb_group_scaler, 0, mask_idx, 0);
                    cb_pop_front(cb_tmp, 1);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_tmp, 1);
                    pack_tile(0, cb_tmp);
                    tile_regs_release();
                    cb_push_back(cb_tmp, 1);

                    // Step 3: reduce-accumulate with 1/K scaler
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, cb_sq_sum);
                    tile_regs_acquire();

                    if (!first_tile) {
                        copy_tile_to_dst_init_short_with_dt(cb_sq_sum, cb_tmp);
                        cb_wait_front(cb_sq_sum, 1);
                        copy_tile(cb_sq_sum, 0, 0);
                        cb_pop_front(cb_sq_sum, 1);
                        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, cb_sq_sum);
                    }

                    cb_wait_front(cb_tmp, 1);
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, 0, 0, 0);
                    cb_pop_front(cb_tmp, 1);

                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_sq_sum, 1);
                    pack_tile(0, cb_sq_sum);
                    tile_regs_release();
                    cb_push_back(cb_sq_sum, 1);

                    reduce_uninit(cb_tmp);
                    first_tile = false;
                }
            }

            // ========== PHASE 3: COMPUTE DEN = rsqrt(E[x^2] - mean^2 + eps) ==========
            // cb_sq_sum has E[x^2] for group g (1 tile)
            // cb_mean[g] has mean_g

            // Step 1: mean^2 = mul_tiles_bcast_scalar(cb_mean, cb_mean) -- mean * mean
            // Use scalar self-multiply: mean_g * mean_g
            init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_mean, cb_mean, cb_tmp);
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_mean, cb_mean, g, g, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_tmp, 1);
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            // Step 2: var = E[x^2] - mean^2
            sub_tiles_init(cb_sq_sum, cb_tmp);
            tile_regs_acquire();
            cb_wait_front(cb_sq_sum, 1);
            cb_wait_front(cb_tmp, 1);
            sub_tiles(cb_sq_sum, cb_tmp, 0, 0, 0);
            cb_pop_front(cb_sq_sum, 1);
            cb_pop_front(cb_tmp, 1);
            tile_regs_commit();
            tile_regs_wait();
            // Pack var to cb_tmp
            cb_reserve_back(cb_tmp, 1);
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            // Step 3: var + eps
            init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR>(cb_tmp, cb_eps, cb_den);
            tile_regs_acquire();
            cb_wait_front(cb_tmp, 1);
            add_tiles_bcast_scalar(cb_tmp, cb_eps, 0, 0, 0);
            cb_pop_front(cb_tmp, 1);

            // Step 4: rsqrt(var + eps) -- in-place on DST[0]
            rsqrt_tile_init();
            rsqrt_tile(0);

            // Pack den to cb_den
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_den, 1);
            pack_tile(0, cb_den);
            tile_regs_release();
            cb_push_back(cb_den, 1);
        }

        // cb_den now has G tiles (one per group)
        cb_wait_front(cb_den, G);

        // ========== PHASE 4: NORMALIZE = (x - mean) * den ==========
        // For each tile: build mean_tile, subtract from x, build den_tile, multiply
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_reserve_back(cb_normalized, Ct);

            for (uint32_t ct = 0; ct < Ct; ++ct) {
                uint32_t tilized_idx = ht * Ct + ct;

                // --- Build mean_tile for this tile column ---
                // mean_tile = sum_g(mean_g * binary_mask[g*Ct+ct])
                init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_group_scaler, cb_mean, cb_tmp);
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_group_scaler, cb_mean, 0 * Ct + ct, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_tmp, 1);
                pack_tile(0, cb_tmp);
                tile_regs_release();
                cb_push_back(cb_tmp, 1);

                for (uint32_t g = 1; g < G; ++g) {
                    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_group_scaler, cb_mean, cb_sq_sum);
                    tile_regs_acquire();
                    mul_tiles_bcast_scalar(cb_group_scaler, cb_mean, g * Ct + ct, g, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_sq_sum, 1);
                    pack_tile(0, cb_sq_sum);
                    tile_regs_release();
                    cb_push_back(cb_sq_sum, 1);

                    add_tiles_init(cb_tmp, cb_sq_sum);
                    tile_regs_acquire();
                    cb_wait_front(cb_tmp, 1);
                    cb_wait_front(cb_sq_sum, 1);
                    add_tiles(cb_tmp, cb_sq_sum, 0, 0, 0);
                    cb_pop_front(cb_tmp, 1);
                    cb_pop_front(cb_sq_sum, 1);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_tmp, 1);
                    pack_tile(0, cb_tmp);
                    tile_regs_release();
                    cb_push_back(cb_tmp, 1);
                }

                // --- Subtract mean: centered = x - mean_tile ---
                sub_tiles_init(cb_tilized, cb_tmp);
                tile_regs_acquire();
                cb_wait_front(cb_tmp, 1);
                sub_tiles(cb_tilized, cb_tmp, tilized_idx, 0, 0);
                cb_pop_front(cb_tmp, 1);
                // DST[0] = centered tile
                // Pack centered to cb_tmp for the multiply step
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_tmp, 1);
                pack_tile(0, cb_tmp);
                tile_regs_release();
                cb_push_back(cb_tmp, 1);

                // --- Build den_tile for this tile column ---
                // den_tile = sum_g(den_g * binary_mask[g*Ct+ct])
                init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_group_scaler, cb_den, cb_sq_sum);
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_group_scaler, cb_den, 0 * Ct + ct, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_sq_sum, 1);
                pack_tile(0, cb_sq_sum);
                tile_regs_release();
                cb_push_back(cb_sq_sum, 1);

                for (uint32_t g = 1; g < G; ++g) {
                    // den_g * mask[g*Ct+ct] -> cb_scratch (dedicated scratch)
                    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_group_scaler, cb_den, cb_scratch);
                    tile_regs_acquire();
                    mul_tiles_bcast_scalar(cb_group_scaler, cb_den, g * Ct + ct, g, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_scratch, 1);
                    pack_tile(0, cb_scratch);
                    tile_regs_release();
                    cb_push_back(cb_scratch, 1);

                    add_tiles_init(cb_sq_sum, cb_scratch);
                    tile_regs_acquire();
                    cb_wait_front(cb_sq_sum, 1);
                    cb_wait_front(cb_scratch, 1);
                    add_tiles(cb_sq_sum, cb_scratch, 0, 0, 0);
                    cb_pop_front(cb_sq_sum, 1);
                    cb_pop_front(cb_scratch, 1);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_sq_sum, 1);
                    pack_tile(0, cb_sq_sum);
                    tile_regs_release();
                    cb_push_back(cb_sq_sum, 1);
                }

                // --- Multiply: normalized = centered * den_tile ---
                mul_tiles_init(cb_tmp, cb_sq_sum);
                tile_regs_acquire();
                cb_wait_front(cb_tmp, 1);
                cb_wait_front(cb_sq_sum, 1);
                mul_tiles(cb_tmp, cb_sq_sum, 0, 0, 0);
                cb_pop_front(cb_tmp, 1);
                cb_pop_front(cb_sq_sum, 1);
                tile_regs_commit();
                tile_regs_wait();
                // Pack normalized to cb_tmp for affine transform
                cb_reserve_back(cb_tmp, 1);
                pack_tile(0, cb_tmp);
                tile_regs_release();
                cb_push_back(cb_tmp, 1);

                // --- Phase 4b: Affine = gamma[ct] * normalized + beta[ct] ---
                // Step 1: gamma * normalized
                mul_tiles_init(cb_tmp, cb_gamma);
                tile_regs_acquire();
                cb_wait_front(cb_tmp, 1);
                mul_tiles(cb_tmp, cb_gamma, 0, ct, 0);
                cb_pop_front(cb_tmp, 1);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_tmp, 1);
                pack_tile(0, cb_tmp);
                tile_regs_release();
                cb_push_back(cb_tmp, 1);

                // Step 2: gamma * normalized + beta
                add_tiles_init(cb_tmp, cb_beta);
                tile_regs_acquire();
                cb_wait_front(cb_tmp, 1);
                add_tiles(cb_tmp, cb_beta, 0, ct, 0);
                cb_pop_front(cb_tmp, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_normalized);
                tile_regs_release();
            }

            cb_push_back(cb_normalized, Ct);

            // ========== PHASE 5: UNTILIZE ==========
            compute_kernel_lib::untilize<Ct, cb_normalized, cb_output_rm>(1);
        }

        // Pop persistent tiles for this sample
        cb_pop_front(cb_tilized, Ht * Ct);
        cb_pop_front(cb_mean, G);
        cb_pop_front(cb_den, G);
    }
    // Note: cb_scaler, cb_eps, and cb_group_scaler are persistent across samples, never popped
}
