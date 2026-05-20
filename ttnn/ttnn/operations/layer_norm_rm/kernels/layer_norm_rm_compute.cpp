// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for layer_norm_rm (Refinement 2 — streaming + multi-core).
//
// Per-core slice = `Ht_local` consecutive tile-rows (from RT arg).
//
// Per tile-row:
//   Pass 1: accumulate_reduce<SUM, REDUCE_ROW>(cb_input_tiles, cb_scaler,
//           cb_mean) — streaming chunked along W into NUM_BLOCKS blocks of
//           BLOCK_SIZE tiles each. For RM input, per-block tilize feeds
//           cb_input_tiles.
//   Pass 2: per block b:
//             (RM input) tilize<BLOCK_SIZE,…>(1, 32) → cb_input_tiles
//             sub<COL>(cb_input_tiles, cb_mean, cb_centered)
//             square_in_place(cb_centered)
//             accumulate_reduce_block<SUM, REDUCE_ROW>(cb_centered, cb_scaler,
//                                                      cb_inv_std, b, NUM_BLOCKS)
//           After loop: transform_in_place on cb_inv_std → rsqrt(var + eps).
//   Pass 3: per block b:
//             (RM input) tilize<BLOCK_SIZE,…>(1, 32) → cb_input_tiles
//             sub<COL>(cb_input_tiles, cb_mean, cb_centered)
//             mul_in_place<COL>(cb_centered, cb_inv_std)
//             (HAS_GAMMA) tilize<BLOCK_SIZE,…>(1, 1) → cb_gamma_tiles (1 stick→
//                          BLOCK_SIZE tiles, row-0 valid only)
//             (HAS_GAMMA) mul_in_place<ROW>(cb_centered, cb_gamma_tiles)
//             (HAS_BETA) similar for beta
//             drain → cb_output (copy_tiles for TILE / untilize for RM)
//             pop BLOCK_SIZE tiles from cb_gamma_tiles / cb_beta_tiles
//
// Persistent CBs (cb_mean, cb_inv_std) are popped at the end of each tile-row.
// cb_scaler is popped at kernel end.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_input_sticks = 0;
constexpr uint32_t cb_input_tiles = 1;
constexpr uint32_t cb_gamma_sticks = 2;
constexpr uint32_t cb_beta_sticks = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_output = 16;
constexpr uint32_t cb_gamma_tiles = 24;
constexpr uint32_t cb_beta_tiles = 25;
constexpr uint32_t cb_mean = 26;
constexpr uint32_t cb_inv_std = 27;
constexpr uint32_t cb_centered = 28;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(2);
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(3) != 0;
    constexpr bool IS_RM_INPUT = get_compile_time_arg_val(4) != 0;
    constexpr bool IS_RM_OUTPUT = get_compile_time_arg_val(5) != 0;
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(6) != 0;
    constexpr bool HAS_BETA = get_compile_time_arg_val(7) != 0;
    constexpr uint32_t eps_bits = get_compile_time_arg_val(8);

    uint32_t Ht_local = get_arg_val<uint32_t>(0);

    // Single full hardware init for the kernel — helpers will reconfig as needed.
    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output);

    // Cores past the work boundary may be assigned Ht_local=0; bail before
    // touching any CB so we don't deadlock against an empty cb_scaler.
    if (Ht_local == 0) {
        return;
    }

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(/*rows=*/1, /*cols=*/BLOCK_SIZE, /*batches=*/1);
    constexpr auto bin_block_shape = ckl::BinaryInputBlockShape::of(/*rows=*/1, /*cols=*/BLOCK_SIZE);
    // For non-tile-aligned W, the partial scaler tile (idx 1) is routed to the
    // last W-tile of the last block by accumulate_reduce / accumulate_reduce_block.
    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    // ---------------- Per-tile-row loop ----------------
    for (uint32_t tr = 0; tr < Ht_local; ++tr) {
        // ============ Pass 1: streaming mean ============
        if constexpr (IS_RM_INPUT) {
            // RM input: tilize each block from cb_input_sticks into cb_input_tiles
            // and feed accumulate_reduce_block per block.
            for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
                ckl::tilize<BLOCK_SIZE, cb_input_sticks, cb_input_tiles>(/*num_blocks=*/1, /*total_input_pages=*/32);
                ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_input_tiles, cb_scaler, cb_mean, reduce_block_shape, b, NUM_BLOCKS, partial_scaler);
            }
        } else {
            // TILE input: reader pushes tiles block by block; the streaming
            // reduce helper owns the block loop.
            ckl::accumulate_reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_input_tiles, cb_scaler, cb_mean, reduce_block_shape, NUM_BLOCKS, partial_scaler);
        }

        // ============ Pass 2: variance → inv_std ============
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            if constexpr (IS_RM_INPUT) {
                ckl::tilize<BLOCK_SIZE, cb_input_sticks, cb_input_tiles>(/*num_blocks=*/1, /*total_input_pages=*/32);
            }
            // sub<COL>: A = cb_input_tiles streaming, B = cb_mean (1 tile,
            // WaitUpfrontNoPop) — produces BLOCK_SIZE centered tiles.
            ckl::sub<
                ckl::BroadcastDim::COL,
                ckl::BinaryInputPolicy::WaitAndPopPerTile,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_input_tiles, cb_mean, cb_centered, bin_block_shape);

            ckl::square_in_place(cb_centered, bin_block_shape);

            ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_centered, cb_scaler, cb_inv_std, reduce_block_shape, b, NUM_BLOCKS, partial_scaler);
        }
        // variance → 1/sqrt(variance + eps) in place.
        ckl::transform_in_place(cb_inv_std, [](uint32_t dst) {
            binop_with_scalar_tile_init();
            add_unary_tile(dst, eps_bits);
            rsqrt_tile_init();
            rsqrt_tile(dst);
        });

        // ============ Pass 3: normalize + (gamma) + (beta) + drain ============
        for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
            if constexpr (IS_RM_INPUT) {
                ckl::tilize<BLOCK_SIZE, cb_input_sticks, cb_input_tiles>(/*num_blocks=*/1, /*total_input_pages=*/32);
            }
            ckl::sub<
                ckl::BroadcastDim::COL,
                ckl::BinaryInputPolicy::WaitAndPopPerTile,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_input_tiles, cb_mean, cb_centered, bin_block_shape);

            ckl::mul_in_place<ckl::BroadcastDim::COL, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_centered, cb_inv_std, bin_block_shape);

            if constexpr (HAS_GAMMA) {
                // Tilize this block's gamma slice: 1 partial stick → BLOCK_SIZE
                // tiles, row-0 valid only.  mul_in_place<ROW> reads BLOCK_SIZE
                // tiles from cb_gamma_tiles using indices [0, BLOCK_SIZE).
                ckl::tilize<BLOCK_SIZE, cb_gamma_sticks, cb_gamma_tiles>(/*num_blocks=*/1, /*total_input_pages=*/1);
                ckl::mul_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_centered, cb_gamma_tiles, bin_block_shape);
                cb_pop_front(cb_gamma_tiles, BLOCK_SIZE);
            }
            if constexpr (HAS_BETA) {
                ckl::tilize<BLOCK_SIZE, cb_beta_sticks, cb_beta_tiles>(/*num_blocks=*/1, /*total_input_pages=*/1);
                ckl::add_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
                    cb_centered, cb_beta_tiles, bin_block_shape);
                cb_pop_front(cb_beta_tiles, BLOCK_SIZE);
            }

            // Drain this block's BLOCK_SIZE tiles from cb_centered → cb_output.
            if constexpr (IS_RM_OUTPUT) {
                ckl::untilize<BLOCK_SIZE, cb_centered, cb_output>(/*num_blocks=*/1);
            } else {
                ckl::copy_tiles<ckl::CopyInputPolicy::WaitAndPop>(cb_centered, cb_output, BLOCK_SIZE);
            }
        }

        // Per-row persistent CBs (cb_mean, cb_inv_std were WaitUpfrontNoPop'd).
        cb_pop_front(cb_mean, 1);
        cb_pop_front(cb_inv_std, 1);
    }

    // ---------------- Teardown ----------------
    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
