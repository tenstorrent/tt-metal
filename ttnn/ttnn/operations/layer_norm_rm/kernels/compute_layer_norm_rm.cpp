// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Compute Kernel
// Stage 5: 3-pass full layer norm
//   Pass1: tilize + reduce_mean
//   Pass2: tilize + sub_mean + square + reduce_var + add_eps + rsqrt
//   Pass3: tilize + sub_mean + mul_rsqrt [+ gamma] [+ beta] + untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"

// CB indices
constexpr uint32_t cb_input_rm = 0;       // c_0: RM sticks from reader
constexpr uint32_t cb_tilized = 1;        // c_1: Tilized tiles
constexpr uint32_t cb_reduce_scaler = 2;  // c_2: Reduce scaler (1/W)
constexpr uint32_t cb_eps_scalar = 3;     // c_3: Epsilon tile
constexpr uint32_t cb_gamma = 4;          // c_4: Gamma tiles (Wt)
constexpr uint32_t cb_beta = 5;           // c_5: Beta tiles (Wt)
constexpr uint32_t cb_mean = 24;          // c_24: Row-reduced mean (1 tile)
constexpr uint32_t cb_centered = 25;      // c_25: Centered tiles (x - mean)
constexpr uint32_t cb_var = 26;           // c_26: Row-reduced variance (1 tile)
constexpr uint32_t cb_rsqrt_var = 27;     // c_27: rsqrt(var+eps) (1 tile)
constexpr uint32_t cb_normalized = 28;    // c_28: Normalized tiles
constexpr uint32_t cb_output_tiles = 16;  // c_16: Pre-untilize output tiles
constexpr uint32_t cb_output_rm = 17;     // c_17: Untilized RM output

// Compile-time args
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);

void kernel_main() {
    // Runtime args
    const uint32_t nblocks = get_arg_val<uint32_t>(0);

    if (nblocks == 0) {
        return;
    }

    // Hardware startup: input CB for srcA/srcB, output CB for pack
    compute_kernel_hw_startup(cb_input_rm, cb_reduce_scaler, cb_output_rm);

    for (uint32_t block = 0; block < nblocks; block++) {
        // ==================== PASS 1: Compute mean ====================

        // Phase 1: Tilize c_0 -> c_1
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 2: Reduce mean - SUM along row with 1/W scaler
        // c_24 persists for pass2 and pass3
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_tilized, cb_reduce_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // ==================== PASS 2: Variance + rsqrt ====================

        // Phase 3: Tilize (re-read from reader's second pass)
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 4: sub(x, mean) with COL broadcast -> c_25 (centered)
        // c_24 persists (NoWaitNoPop) -- needed again in pass3
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 5: Square centered -> c_1 (reuse freed c_1)
        compute_kernel_lib::square<compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_tilized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 6: Reduce variance - SUM along row with 1/W scaler
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile>(
                cb_tilized, cb_reduce_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 7: add(var, eps) with SCALAR broadcast + rsqrt post-op -> c_27
        // c_27 persists for pass3
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var,
            cb_eps_scalar,
            cb_rsqrt_var,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // ==================== PASS 3: Normalize ====================

        // Phase 8: Tilize (re-read from reader's third pass)
        compute_kernel_lib::tilize<
            cb_input_rm,
            cb_tilized,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock>(Wt, 1);

        // Phase 9: sub(x, mean) with COL broadcast -> c_25
        // c_24 still persists from pass1 (NoWaitNoPop)
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_tilized, cb_mean, cb_centered, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Pop mean after last use
        cb_pop_front(cb_mean, 1);

        // Phase 10: mul(centered, rsqrt_var) with COL broadcast
        // When no gamma and no beta: output directly to c_16
        // When gamma or beta present: output to c_28
        if constexpr (!has_gamma && !has_beta) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_centered, cb_rsqrt_var, cb_output_tiles, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        } else {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_centered, cb_rsqrt_var, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Pop rsqrt_var after last use
        cb_pop_front(cb_rsqrt_var, 1);

        // Phase 11 (conditional): Apply gamma
        if constexpr (has_gamma && !has_beta) {
            // Gamma only: c_28 * c_4 -> c_16
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_normalized, cb_gamma, cb_output_tiles, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        } else if constexpr (has_gamma && has_beta) {
            // Gamma + beta: c_28 * c_4 -> c_16
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_normalized, cb_gamma, cb_output_tiles, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 12 (conditional): Apply beta
        if constexpr (has_beta && has_gamma) {
            // c_16 + c_5 -> c_28, then need to move to c_16 for untilize
            // Actually: add c_16 + c_5 -> c_28
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_output_tiles, cb_beta, cb_normalized, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            // Untilize from c_28 instead of c_16
            compute_kernel_lib::untilize<
                Wt,
                cb_normalized,
                cb_output_rm,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
        } else if constexpr (has_beta && !has_gamma) {
            // Beta only, no gamma: normalized is in c_28
            // c_28 + c_5 -> c_16
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::ROW,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_normalized, cb_beta, cb_output_tiles, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            // Phase 13: Untilize c_16 -> c_17
            compute_kernel_lib::untilize<
                Wt,
                cb_output_tiles,
                cb_output_rm,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
        } else {
            // No beta: untilize from c_16
            // Phase 13: Untilize c_16 -> c_17
            compute_kernel_lib::untilize<
                Wt,
                cb_output_tiles,
                cb_output_rm,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock>(1);
        }
    }
}
