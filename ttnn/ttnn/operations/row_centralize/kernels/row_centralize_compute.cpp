// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row Centralize - Compute Kernel
//
// TDD Stage: add_beta (full LayerNorm with affine)
// All 9+2 phases: tilize, reduce_mean, sub, square, reduce_var, add_eps, rsqrt, mul,
//                 [if affine: mul_gamma, add_beta], untilize
//
// Startup (if has_affine):
//   Phase 0a: tilize(c_0 -> c_9)  — gamma RM stick to tilized gamma (program lifetime)
//   Phase 0b: tilize(c_0 -> c_10) — beta RM stick to tilized beta (program lifetime)
//
// Per tile-row (Ht_total iterations):
//   Phase 1: tilize(c_0 -> c_1)                   - RM sticks to tiles
//   Phase 2: reduce_row SUM (c_1 -> c_2)          - row mean with 1/W scaler
//   Phase 3: sub_col (c_1, c_2 -> c_3)            - centralize: x - mean
//   Phase 4: square (c_3 -> c_24)                  - squared centered values
//   Phase 5: reduce_row SUM (c_24 -> c_4)          - variance with 1/W scaler
//   Phase 6: add_scalar (c_4, c_7 -> c_25)         - var + epsilon
//   Phase 7: rsqrt (c_25 -> c_5)                   - 1/sqrt(var + eps)
//   Phase 8: mul_col (c_3, c_5 -> c_6)             - standardize: centered * inv_std
//   Phase 8.5 [affine]: mul_row (c_6, c_9 -> c_26) - standardized * gamma
//   Phase 8.6 [affine]: add_row (c_26, c_10 -> c_6)- (standardized * gamma) + beta
//   Phase 9: untilize(c_6 -> c_16)                 - tiles back to RM sticks
//
// Compile-time args:
//   [0]  Wt              - tiles per tile-row
//   [1]  Ht_total        - total tile-rows
//   [2]  cb_rm_in        - c_0
//   [3]  cb_tilized      - c_1
//   [4]  cb_mean         - c_2
//   [5]  cb_centered     - c_3
//   [6]  cb_squared      - c_24
//   [7]  cb_var          - c_4
//   [8]  cb_var_plus_eps - c_25
//   [9]  cb_inv_std      - c_5
//   [10] cb_result       - c_6
//   [11] cb_rm_out       - c_16
//   [12] cb_eps          - c_7
//   [13] cb_scaler       - c_8
//   [14] cb_gamma_tiled  - c_9  (0 when no affine)
//   [15] cb_beta_tiled   - c_10 (0 when no affine)
//   [16] cb_after_gamma  - c_26 (0 when no affine)
//   [17] has_affine      - 1 if gamma/beta, 0 otherwise

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t Ht_total = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rm_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_tilized = get_compile_time_arg_val(3);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(4);
    constexpr uint32_t cb_centered = get_compile_time_arg_val(5);
    constexpr uint32_t cb_squared = get_compile_time_arg_val(6);
    constexpr uint32_t cb_var = get_compile_time_arg_val(7);
    constexpr uint32_t cb_var_plus_eps = get_compile_time_arg_val(8);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(9);
    constexpr uint32_t cb_result = get_compile_time_arg_val(10);
    constexpr uint32_t cb_rm_out = get_compile_time_arg_val(11);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(12);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(13);
    constexpr uint32_t cb_gamma_tiled = get_compile_time_arg_val(14);
    constexpr uint32_t cb_beta_tiled = get_compile_time_arg_val(15);
    constexpr uint32_t cb_after_gamma = get_compile_time_arg_val(16);
    constexpr uint32_t has_affine = get_compile_time_arg_val(17);

    // ========== Hardware startup ==========
    // Span all CB IDs used: c_0 (0) through c_26 (26) when affine, c_25 otherwise
    constexpr uint32_t cb_min = cb_rm_in;  // c_0
    constexpr uint32_t cb_max = has_affine ? 26 : 25;
    compute_kernel_hw_startup(cb_min, cb_max);

    // ========== Affine startup: tilize gamma and beta ==========
    if constexpr (has_affine) {
        // Phase 0a: Tilize gamma (c_0 -> c_9)
        // Reader pushes 1 gamma stick as Wt RM pages into c_0.
        // Tilize converts RM data to tile format. Only row 0 has valid data.
        compute_kernel_lib::tilize<cb_rm_in, cb_gamma_tiled>(Wt, 1);

        // Phase 0b: Tilize beta (c_0 -> c_10)
        // Reader pushes 1 beta stick as Wt RM pages into c_0 (after Phase 0a pops c_0).
        compute_kernel_lib::tilize<cb_rm_in, cb_beta_tiled>(Wt, 1);
    }

    // ========== Main loop: per tile-row ==========
    for (uint32_t tr = 0; tr < Ht_total; ++tr) {
        // Phase 1: Tilize (c_0 -> c_1)
        compute_kernel_lib::tilize<cb_rm_in, cb_tilized>(Wt, 1);

        // Phase 2: Reduce row mean (c_1 -> c_2)
        // WaitUpfrontNoPop: waits for Wt tiles in c_1, does NOT pop them (tiles persist for Phase 3)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilized, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 3: Subtract mean (c_1, c_2 -> c_3)
        // Input A (c_1): NoWaitPopAtEnd - already waited in Phase 2, pop all Wt tiles at end
        // Input B (c_2): WaitAndPopPerTile - wait for 1 mean tile, pop after row (COL broadcast)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 4: Square centered values (c_3 -> c_24)
        // WaitUpfrontNoPop: c_3 tiles persist for Phase 8 mul
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_centered, cb_squared, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 5: Reduce row variance (c_24 -> c_4)
        // BulkWaitBulkPop: waits for Wt tiles in c_24, processes all, pops all
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_squared, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 6: Add epsilon (c_4 + c_7 -> c_25)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var, cb_eps, cb_var_plus_eps, compute_kernel_lib::BinaryInputBlockShape::single());

        // Phase 7: Rsqrt (c_25 -> c_5) - RAW implementation, no helper
        rsqrt_tile_init();
        copy_tile_to_dst_init_short(cb_var_plus_eps);
        cb_wait_front(cb_var_plus_eps, 1);
        cb_reserve_back(cb_inv_std, 1);
        tile_regs_acquire();
        copy_tile(cb_var_plus_eps, 0, 0);
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_inv_std);
        cb_push_back(cb_inv_std, 1);
        tile_regs_release();
        cb_pop_front(cb_var_plus_eps, 1);

        // Phase 8: Multiply by inv_std (c_3, c_5 -> c_6)
        // Input A (c_3): NoWaitPopAtEnd - already waited in Phase 4, pop all Wt tiles after processing
        // Input B (c_5): WaitAndPopPerTile - wait for 1 inv_std tile, pop after row (COL broadcast)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::NoWaitPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_inv_std, cb_result, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

        // Phase 8.5 & 8.6: Apply gamma and beta (affine transform)
        if constexpr (has_affine) {
            // Phase 8.5: Multiply by gamma (c_6, c_9 -> c_26)
            // Input A (c_6): WaitUpfrontPopAtEnd - wait for Wt standardized tiles, pop after processing
            // Input B (c_9 gamma): WaitUpfrontNoPop - gamma persists for all tile-rows, never popped
            // ROW broadcast: row 0 of each gamma tile broadcasts across all 32 rows
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_result, cb_gamma_tiled, cb_after_gamma, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

            // Phase 8.6: Add beta (c_26, c_10 -> c_6)
            // Input A (c_26): WaitUpfrontPopAtEnd - wait for Wt tiles from phase 8.5, pop after
            // Input B (c_10 beta): WaitUpfrontNoPop - beta persists for all tile-rows, never popped
            // ROW broadcast: row 0 of each beta tile broadcasts across all 32 rows
            // Output goes to c_6 (freed in Phase 8.5) for untilize in Phase 9
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_after_gamma, cb_beta_tiled, cb_result, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));
        }

        // Phase 9: Untilize (c_6 -> c_16)
        compute_kernel_lib::untilize<Wt, cb_result, cb_rm_out>(1);
    }
}
