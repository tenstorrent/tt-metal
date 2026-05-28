// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// layer_norm_rm compute kernel.
//
// Per-strip algorithm (one strip = 32 RM rows × Wt tiles wide,
// chunked into NUM_BLOCKS chunks of BLOCK_SIZE tiles each):
//
//   Pass A — per chunk c in [0, NUM_BLOCKS):
//     tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1)
//     accumulate_reduce_block<SUM, REDUCE_ROW>(
//         cb_tilized_x, cb_scaler, cb_mean,
//         shape=(1, BLOCK_SIZE, 1), b=c, num_blocks=NUM_BLOCKS)
//   → cb_mean holds the per-row means in column 0 (1 tile) after the last
//     chunk. The scaler 1/W (set by the reader) folds SUM into the mean.
//
//   Pass B — per chunk c:
//     tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1)
//     sub<COL, WaitAndPop, WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, shape)
//     square_in_place(cb_centered, shape)
//     accumulate_reduce_block<SUM, REDUCE_ROW>(
//         cb_centered, cb_scaler, cb_inv_std, ..., b=c, num_blocks=NUM_BLOCKS)
//   → cb_inv_std holds variance (1 tile) after the last chunk.
//
//   eps + rsqrt: transform_in_place(cb_inv_std, lambda(dst){
//       binop_with_scalar_tile_init();
//       add_unary_tile(dst, epsilon_bits);
//       rsqrt_tile_init();
//       rsqrt_tile(dst);
//   })
//   → cb_inv_std now holds 1/sqrt(var + eps).
//
//   Pass C — per chunk c:
//     tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1)
//     if HAS_GAMMA: tilize<BLOCK_SIZE, cb_gamma_rm, cb_gamma_tilized>(1, 1)
//     if HAS_BETA:  tilize<BLOCK_SIZE, cb_beta_rm,  cb_beta_tilized >(1, 1)
//     sub<COL, WaitAndPop, WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, shape)
//     mul_in_place<COL, WaitUpfrontNoPop>(cb_centered, cb_inv_std, shape)
//     if HAS_GAMMA: mul_in_place<ROW, WaitAndPop>(cb_centered, cb_gamma_tilized, shape)
//     if HAS_BETA:  add_in_place<ROW, WaitAndPop>(cb_centered, cb_beta_tilized,  shape)
//     untilize<BLOCK_SIZE, cb_centered, cb_output_rm>(1)
//
//   End-of-strip: cb_pop_front(cb_mean, 1); cb_pop_front(cb_inv_std, 1);
//   End-of-kernel: cb_pop_front(cb_scaler, 1);

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"

#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
constexpr uint32_t cb_input_rm = 0;
constexpr uint32_t cb_gamma_rm = 1;
constexpr uint32_t cb_beta_rm = 2;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_output_rm = 16;
constexpr uint32_t cb_tilized_x = 24;
constexpr uint32_t cb_centered = 25;
constexpr uint32_t cb_mean = 26;
constexpr uint32_t cb_inv_std = 27;
constexpr uint32_t cb_gamma_tilized = 28;
constexpr uint32_t cb_beta_tilized = 29;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(1);
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(2);
    constexpr uint32_t HAS_BETA = get_compile_time_arg_val(3);
    constexpr uint32_t EPSILON_BITS = get_compile_time_arg_val(4);

    const uint32_t num_strips = get_arg_val<uint32_t>(0);

    // ── Hardware init (exactly once). ──
    // srcA = cb_input_rm (first unpack source for tilize).
    // srcB = cb_scaler   (used by every accumulate_reduce_block).
    // dst  = cb_centered (the most-packed intermediate).
    compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_centered);

    constexpr auto reduce_shape = ckl::ReduceInputBlockShape::of(1, BLOCK_SIZE, 1);
    constexpr auto bin_shape = ckl::BinaryInputBlockShape::of(1, BLOCK_SIZE);

    for (uint32_t s = 0; s < num_strips; ++s) {
        // ============================================================
        // Pass A — per-chunk tilize + streaming SUM reduce → mean
        // ============================================================
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            ckl::tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(/*num_blocks=*/1);

            ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_tilized_x,
                cb_scaler,
                cb_mean,
                reduce_shape,
                /*b=*/c,
                /*num_blocks=*/NUM_BLOCKS,
                ckl::ReducePartialScaler::none());
        }

        // ============================================================
        // Pass B — per-chunk tilize, sub<COL>, square_in_place,
        //          streaming SUM reduce → variance (into cb_inv_std)
        // ============================================================
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            ckl::tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1);

            ckl::sub<
                ckl::BroadcastDim::COL,
                ckl::BinaryInputPolicy::WaitAndPopPerTile,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, bin_shape);

            ckl::square_in_place(cb_centered, bin_shape);

            ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_centered, cb_scaler, cb_inv_std, reduce_shape, c, NUM_BLOCKS, ckl::ReducePartialScaler::none());
        }

        // ============================================================
        // eps + rsqrt: cb_inv_std (variance) → cb_inv_std (1/sqrt(var+eps))
        // ============================================================
        ckl::transform_in_place(cb_inv_std, [](uint32_t dst_idx) {
            binop_with_scalar_tile_init();
            add_unary_tile(dst_idx, EPSILON_BITS);
            rsqrt_tile_init</*legacy_compat=*/false>();
            rsqrt_tile</*legacy_compat=*/false>(dst_idx);
        });

        // ============================================================
        // Pass C — per-chunk normalize, optional gamma/beta, untilize
        // ============================================================
        for (uint32_t c = 0; c < NUM_BLOCKS; ++c) {
            ckl::tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1);

            if constexpr (HAS_GAMMA != 0) {
                // Asymmetric tilize: cb_gamma_rm has 1 row-page per chunk.
                ckl::tilize<BLOCK_SIZE, cb_gamma_rm, cb_gamma_tilized>(
                    /*num_blocks=*/1, /*total_input_pages=*/1);
            }
            if constexpr (HAS_BETA != 0) {
                ckl::tilize<BLOCK_SIZE, cb_beta_rm, cb_beta_tilized>(1, 1);
            }

            // centered = (x - mean) — mean held across the whole strip.
            ckl::sub<
                ckl::BroadcastDim::COL,
                ckl::BinaryInputPolicy::WaitAndPopPerTile,
                ckl::BinaryInputPolicy::WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, bin_shape);

            // centered *= inv_std (broadcast col 0 of inv_std across width).
            ckl::mul_in_place<ckl::BroadcastDim::COL, ckl::BinaryInputPolicy::WaitUpfrontNoPop>(
                cb_centered, cb_inv_std, bin_shape);

            if constexpr (HAS_GAMMA != 0) {
                // centered *= gamma (broadcast row 0 of each gamma tile down).
                // ROW broadcast requires all B tiles upfront (binary_op_helpers.inl:576).
                // WaitUpfrontPopAtEnd: wait for BLOCK_SIZE gamma tiles, pop after consuming.
                ckl::mul_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
                    cb_centered, cb_gamma_tilized, bin_shape);
            }
            if constexpr (HAS_BETA != 0) {
                ckl::add_in_place<ckl::BroadcastDim::ROW, ckl::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
                    cb_centered, cb_beta_tilized, bin_shape);
            }

            ckl::untilize<BLOCK_SIZE, cb_centered, cb_output_rm>(1);
        }

        // End-of-strip cleanup: drain the WaitUpfrontNoPop persistent CBs.
        cb_pop_front(cb_mean, 1);
        cb_pop_front(cb_inv_std, 1);
    }

    // End-of-kernel: drain the scaler CB (pushed once at boot, never popped
    // by reduce<>).
    cb_pop_front(cb_scaler, 1);
}
