// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_variance.
//
// Computes per-row population variance via the proper two-pass algorithm:
//   variance = E[(x - E[x])^2]
//
// Pass 1: cb_mean     = mean(x)                                — streaming accumulate_reduce
// Pass 2: cb_variance = mean((x - mean)^2)                     — per block:
//                       sub<Col> (eltwise chain) →
//                       FPU square in-place (eltwise chain) →
//                       accumulate_reduce_block
//
// COMPUTE_STD_DEV: when set, sqrt is applied as the post_op_final on the
// pass-2 last-block reduce. The streaming-reduce helper routes post_op_final
// only to the last block, so sqrt runs in DST after the final accumulation,
// before pack — intermediate accumulator values stay in variance space.
//
// The reductions still go through reduce_helpers_compute / streaming_reduce_helpers
// (the reduce family is orthogonal to the eltwise helpers). The binary helpers
// (sub / square_in_place) and the final copy have been migrated to eltwise_helpers
// (eltwise_chain + eltwise_convenience).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_centered = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_mean = 3;
constexpr uint32_t cb_variance = 4;
constexpr uint32_t cb_out = 16;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t BLOCK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t NUM_BLOCKS = get_compile_time_arg_val(3);
    constexpr bool COMPUTE_STD_DEV = get_compile_time_arg_val(4) != 0;
    constexpr bool HAS_PARTIAL_W = get_compile_time_arg_val(5) != 0;

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(Ht, BLOCK_SIZE, /*NC=*/1);
    constexpr auto eltwise_block_shape = ckl::EltwiseShape::of(Ht, BLOCK_SIZE);

    // For non-tile-aligned W: select the partial scaler tile (idx 1) on the
    // last W-tile of the last block — accumulate_reduce / accumulate_reduce_block
    // route it that way internally.
    constexpr auto partial_scaler =
        HAS_PARTIAL_W ? ckl::ReducePartialScaler::last_tile_at(1) : ckl::ReducePartialScaler::none();

    // ---------- Pass 1: streaming mean ----------
    ckl::accumulate_reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
        cb_in, cb_scaler, cb_mean, reduce_block_shape, NUM_BLOCKS, partial_scaler);

    // ---------- Pass 2: streaming variance via (x - mean)^2 ----------
    // Per block:
    //   sub<Col>           : cb_in − cb_mean        → cb_centered  (eltwise chain)
    //   FPU square in-place: cb_centered²           → cb_centered  (eltwise chain, same-CB mul)
    //   accumulate_reduce_block : mean(cb_centered) → cb_variance
    //
    // cb_mean is held across all blocks of pass 2 → B-side lifecycle = HeldBulk
    // (wait Ht upfront, never pop). Caller pops cb_mean once after the block loop.
    // cb_in is per-tile streamed by the reader → A-side lifecycle = Streaming.
    for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
        // sub<Col> — cb_in (Block, Streaming) − cb_mean (Col, HeldBulk) → cb_centered.
        ckl::eltwise_chain(
            eltwise_block_shape,
            ckl::BinaryFpu<
                cb_in,
                cb_mean,
                ckl::BinaryFpuOp::Sub,
                ckl::BroadcastDim::Col,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Streaming,            // A: per-tile pop chase
                ckl::HeldBulk,             // B: wait Ht upfront, never pop
                ckl::OperandKind::Scalar,  // AIndex = FirstTile (Streaming requires)
                ckl::Dst::D0,
                ckl::OperandKind::Col  // BIndex = tile by `ht`
                >{},
            ckl::PackTile<cb_centered, ckl::Dst::D0, ckl::OutStreaming>{});

        // FPU square in-place on cb_centered (same-CB BinaryFpu mul — chain dedups
        // B-side wait/pop; pop-chase per-tile reuses freed slots on the same CB).
        ckl::eltwise_chain(
            eltwise_block_shape,
            ckl::BinaryFpu<
                cb_centered,
                cb_centered,
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::Streaming,
                ckl::Streaming,
                ckl::OperandKind::Scalar,
                ckl::Dst::D0,
                ckl::OperandKind::Scalar>{},
            ckl::PackTile<cb_centered, ckl::Dst::D0, ckl::OutStreaming>{});

        if constexpr (COMPUTE_STD_DEV) {
            ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_centered,
                cb_scaler,
                cb_variance,
                reduce_block_shape,
                b,
                NUM_BLOCKS,
                partial_scaler,
                [](uint32_t dst) {
                    sqrt_tile_init();
                    sqrt_tile(dst);
                });
        } else {
            ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_centered, cb_scaler, cb_variance, reduce_block_shape, b, NUM_BLOCKS, partial_scaler);
        }
    }

    // cb_mean was held across pass 2 with HeldBulk — release it now.
    cb_pop_front(cb_mean, Ht);

    // ---------- Drain cb_variance → cb_out ----------
    ckl::copy<cb_variance, cb_out>(Ht);

    cb_pop_front(cb_scaler, HAS_PARTIAL_W ? 2 : 1);
}
