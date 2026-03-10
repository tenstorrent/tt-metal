// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Compute Kernel (Stage 2: group_mean_subtract)
// Phase 0: Tilize (cb_input_rm -> cb_tilized, persistent)
// Phase 1: Compute per-group mean via mul_tiles(x, binary_mask) + reduce_tile
// Phase 2: Build per-tile mean tile via mul_tiles_bcast_scalar(mask, mean) + add_tiles, then sub_tiles
// Phase 3: Untilize (cb_normalized -> cb_output_rm)

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
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
    constexpr uint32_t cb_scaler = 5;
    constexpr uint32_t cb_mean = 6;
    constexpr uint32_t cb_normalized = 16;
    constexpr uint32_t cb_output_rm = 17;
    constexpr uint32_t cb_sq_sum = 24;  // Used as intermediate accumulator for reduce
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
        cb_wait_front(cb_group_scaler, G * Ct);

        // ========== PHASE 1: COMPUTE PER-GROUP MEAN ==========
        // For each group g:
        //   mean_g = (1/K) * sum over all tiles of (x_tile * binary_mask[g, ct])
        // Uses mul_tiles for masking -> cb_tmp, then reduce_tile with 1/K scaler.
        // Intermediate accumulator stored in cb_sq_sum (FIFO-safe: 1 tile, push/pop each iter).
        // Final result for each group pushed to cb_mean (FIFO: G tiles, one per group).

        for (uint32_t g = 0; g < G; ++g) {
            bool first_tile = true;

            for (uint32_t ht = 0; ht < Ht; ++ht) {
                for (uint32_t ct = 0; ct < Ct; ++ct) {
                    uint32_t tilized_idx = ht * Ct + ct;
                    uint32_t mask_idx = g * Ct + ct;

                    // --- mul_tiles: x_tile * binary_mask -> DST[0] -> pack to cb_tmp ---
                    mul_tiles_init(cb_tilized, cb_group_scaler);
                    tile_regs_acquire();
                    mul_tiles(cb_tilized, cb_group_scaler, tilized_idx, mask_idx, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_tmp, 1);
                    pack_tile(0, cb_tmp);
                    tile_regs_release();
                    cb_push_back(cb_tmp, 1);

                    // --- reduce_tile: accumulate masked tile with 1/K scaler ---
                    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, cb_sq_sum);
                    tile_regs_acquire();

                    if (!first_tile) {
                        // Reload previous accumulator from cb_sq_sum into DST[0]
                        copy_tile_to_dst_init_short_with_dt(cb_sq_sum, cb_tmp);
                        cb_wait_front(cb_sq_sum, 1);
                        copy_tile(cb_sq_sum, 0, 0);
                        cb_pop_front(cb_sq_sum, 1);
                        // Re-init reduce after copy_tile changed SRCA config
                        // Full reduce_init is safe (reconfigures unpack+math+pack)
                        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, cb_sq_sum);
                    }

                    cb_wait_front(cb_tmp, 1);
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_tmp, cb_scaler, 0, 0, 0);
                    cb_pop_front(cb_tmp, 1);

                    // Pack accumulated result to cb_sq_sum (intermediate accumulator)
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
            // cb_sq_sum has 1 tile (the final mean for group g)
            // Copy it to cb_mean via copy_tile
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

        // Now cb_mean has G tiles (one per group, each a scalar broadcast of mean_g)
        cb_wait_front(cb_mean, G);

        // ========== PHASE 2: SUBTRACT PER-GROUP MEAN ==========
        // For each tile-row and tile column:
        //   Construct mean_tile[ct] = sum_g(mean_g * binary_mask[g*Ct+ct])
        //   output_tile = x_tile - mean_tile
        // This correctly handles sub-tile group boundaries.

        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_reserve_back(cb_normalized, Ct);

            for (uint32_t ct = 0; ct < Ct; ++ct) {
                uint32_t tilized_idx = ht * Ct + ct;

                // Build the mean broadcast tile for this tile column
                // mean_tile = sum over g of: mean_g * binary_mask[g*Ct+ct]

                // Group 0: mean_tile = mean_0 * mask[0*Ct+ct]
                init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_group_scaler, cb_mean, cb_tmp);
                tile_regs_acquire();
                mul_tiles_bcast_scalar(cb_group_scaler, cb_mean, 0 * Ct + ct, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(cb_tmp, 1);
                pack_tile(0, cb_tmp);
                tile_regs_release();
                cb_push_back(cb_tmp, 1);

                // Groups 1..G-1: accumulate mean_g * mask contributions
                for (uint32_t g = 1; g < G; ++g) {
                    // Compute mean_g * mask[g*Ct+ct] -> pack to cb_sq_sum (temp)
                    init_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR>(cb_group_scaler, cb_mean, cb_sq_sum);
                    tile_regs_acquire();
                    mul_tiles_bcast_scalar(cb_group_scaler, cb_mean, g * Ct + ct, g, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    cb_reserve_back(cb_sq_sum, 1);
                    pack_tile(0, cb_sq_sum);
                    tile_regs_release();
                    cb_push_back(cb_sq_sum, 1);

                    // add_tiles(cb_tmp, cb_sq_sum) -> new cb_tmp
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

                // Now cb_tmp has the mean_tile for this tile column
                // Subtract: output = x_tile - mean_tile
                sub_tiles_init(cb_tilized, cb_tmp);
                tile_regs_acquire();
                cb_wait_front(cb_tmp, 1);
                sub_tiles(cb_tilized, cb_tmp, tilized_idx, 0, 0);
                cb_pop_front(cb_tmp, 1);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_normalized);
                tile_regs_release();
            }

            cb_push_back(cb_normalized, Ct);

            // ========== PHASE 3: UNTILIZE ==========
            compute_kernel_lib::untilize<Ct, cb_normalized, cb_output_rm>(1);
        }

        // Pop persistent tiles for this sample
        cb_pop_front(cb_tilized, Ht * Ct);
        cb_pop_front(cb_mean, G);
    }
    // Note: cb_scaler and cb_group_scaler are persistent across samples, never popped
}
