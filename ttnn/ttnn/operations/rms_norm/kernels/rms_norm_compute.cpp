// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Compute Kernel
// Stage 4: tilize + square + reduce_mean + add_eps + rsqrt + 2nd tilize + mul_col + [mul_row] + untilize

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_scaler = 1;
constexpr uint32_t cb_eps = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_gamma = 4;
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_x = 24;
constexpr uint32_t cb_xsq = 25;
constexpr uint32_t cb_rms = 26;
constexpr uint32_t cb_rsqrt = 27;
constexpr uint32_t cb_normed = 28;

void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t origin_w = get_arg_val<uint32_t>(2);

    if (num_rows == 0) {
        return;
    }

    // Determine compute input CB based on layout
#if IS_INPUT_RM
    constexpr uint32_t compute_input_cb = cb_x;
#else
    constexpr uint32_t compute_input_cb = cb_in;
#endif

    // Determine final output tile CB based on layout and gamma presence
    // This is the CB that the last compute phase writes to before the writer consumes it.
    // RM path: last phase before writer is untilize, so final_tile_cb is the untilize input
    // TILE path: last compute phase writes directly to cb_out
#if IS_INPUT_RM
#if HAS_GAMMA
    // RM+gamma: mul_row -> cb_x (reuse as pre-untilize staging), untilize(cb_x, cb_out)
    constexpr uint32_t final_tile_cb = cb_x;
#else
    // RM+no_gamma: mul_col -> cb_normed, untilize(cb_normed, cb_out)
    constexpr uint32_t final_tile_cb = cb_normed;
#endif
#else
    // TILE: mul_col or mul_row writes directly to cb_out
    constexpr uint32_t final_tile_cb = cb_out;
#endif

    // mul_col output CB: when gamma is present, mul_col outputs to cb_normed (streaming 1 tile)
    // which then feeds mul_row. When no gamma, mul_col outputs to final_tile_cb directly.
#if HAS_GAMMA
    constexpr uint32_t mul_col_out_cb = cb_normed;
#else
    constexpr uint32_t mul_col_out_cb = final_tile_cb;
#endif

    // Hardware startup: configure with first operation's CBs
#if IS_INPUT_RM
#if HAS_GAMMA
    // RM+gamma path: first op is tilize(cb_gamma_rm -> cb_gamma)
    compute_kernel_hw_startup(cb_gamma_rm, cb_gamma_rm, cb_gamma);
#else
    // RM path: first op is tilize(cb_in -> cb_x)
    compute_kernel_hw_startup(cb_in, cb_in, cb_x);
#endif
#else
#if HAS_GAMMA
    // TILE+gamma: first op is tilize(cb_gamma_rm -> cb_gamma)
    compute_kernel_hw_startup(cb_gamma_rm, cb_gamma_rm, cb_gamma);
#else
    // TILE path: first op is square(cb_in -> cb_xsq)
    compute_kernel_hw_startup(cb_in, cb_in, cb_xsq);
#endif
#endif

    // Wait for persistent CBs (pushed once by reader, never popped during loop)
    cb_wait_front(cb_eps, 1);

#if HAS_GAMMA
    // Tilize gamma: cb_gamma_rm -> cb_gamma (once at program start)
    compute_kernel_lib::tilize<cb_gamma_rm, cb_gamma>(Wt, 1);
    // Wait for gamma tiles to be available (persistent for entire program)
    cb_wait_front(cb_gamma, Wt);
#endif

    for (uint32_t row = 0; row < num_rows; ++row) {
        // === Pass 1: square + reduce (consumes first push of cb_in) ===

#if IS_INPUT_RM
        // Phase 1: Tilize RM input -> cb_x
        compute_kernel_lib::tilize<cb_in, cb_x>(Wt, 1);
#endif

        // Phase 2: Square - compute_input_cb -> cb_xsq
        compute_kernel_lib::square(compute_input_cb, cb_xsq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 3: Reduce SUM along row (with 1/W scaler = mean)
        compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_xsq, cb_scaler, cb_rms, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 4: Add epsilon + rsqrt
        // cb_rms + cb_eps -> cb_rsqrt, with rsqrt post-op
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_rms, cb_eps, cb_rsqrt, compute_kernel_lib::BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // === Pass 2: normalize (consumes second push of cb_in) ===

#if IS_INPUT_RM
        // Phase 5: Tilize RM input pass 2 -> cb_x
        compute_kernel_lib::tilize<cb_in, cb_x>(Wt, 1);
#endif

        // Phase 6: Multiply x * rsqrt (COL broadcast)
        // compute_input_cb * cb_rsqrt -> mul_col_out_cb
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            compute_input_cb, cb_rsqrt, mul_col_out_cb, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

#if HAS_GAMMA
        // Phase 7: Multiply by gamma (ROW broadcast)
        // cb_normed * cb_gamma -> final_tile_cb
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::ROW,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_normed, cb_gamma, final_tile_cb, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
#endif

#if IS_INPUT_RM
#if HAS_GAMMA
        // Phase 8: Untilize gamma-scaled output cb_x -> cb_out
        compute_kernel_lib::untilize<RMS_Wt, cb_x, cb_out>(1);
#else
        // Phase 8: Untilize normalized output cb_normed -> cb_out
        compute_kernel_lib::untilize<RMS_Wt, cb_normed, cb_out>(1);
#endif
#endif
    }

    // Cleanup persistent CBs
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_eps, 1);
#if HAS_GAMMA
    cb_pop_front(cb_gamma, Wt);
#endif
}
