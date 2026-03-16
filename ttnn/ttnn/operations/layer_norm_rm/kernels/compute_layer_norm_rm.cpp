// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for layer_norm_rm.
// Phases: tilize, reduce(mean), sub(center), square, reduce(var),
//         add+rsqrt, mul(normalize), mul(gamma), add(beta), untilize.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

// Include tilize/untilize utilities from layernorm (guarded by TILIZE_IN / UNTILIZE_OUT defines)
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_compute_utils.h"

namespace {

using namespace compute_kernel_lib;

// CB IDs
constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;
constexpr uint32_t cb_in = tt::CBIndex::c_1;
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
constexpr uint32_t cb_mean = tt::CBIndex::c_3;
constexpr uint32_t cb_centered = tt::CBIndex::c_4;
constexpr uint32_t cb_sq = tt::CBIndex::c_5;
constexpr uint32_t cb_var = tt::CBIndex::c_6;
constexpr uint32_t cb_eps = tt::CBIndex::c_7;
constexpr uint32_t cb_out = tt::CBIndex::c_16;
constexpr uint32_t cb_gamma = tt::CBIndex::c_17;
constexpr uint32_t cb_beta = tt::CBIndex::c_18;
constexpr uint32_t cb_gamma_rm = tt::CBIndex::c_19;
constexpr uint32_t cb_beta_rm = tt::CBIndex::c_20;
constexpr uint32_t cb_rsqrt = tt::CBIndex::c_24;
constexpr uint32_t cb_temp = tt::CBIndex::c_25;
constexpr uint32_t cb_out_rm = tt::CBIndex::c_28;

}  // namespace

void kernel_main() {
    // ================================================================
    // Compile-time args
    // ================================================================
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);  // min(Wt, 8)
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);

    // ================================================================
    // Runtime args
    // ================================================================
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);
    const uint32_t W = get_arg_val<uint32_t>(1);  // width in elements

    // Early exit for idle cores
    if (num_tile_rows == 0) {
        return;
    }

    // ================================================================
    // Dynamic CB routing for optional affine
    // ================================================================
    constexpr uint32_t cb_affine_or_out = (has_gamma || has_beta) ? cb_temp : cb_out;
    constexpr uint32_t cb_scaled_or_out = has_beta ? cb_temp : cb_out;

    // ================================================================
    // Hardware startup
    // ================================================================
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // ================================================================
    // Startup: Tilize gamma/beta (once, if present)
    // ================================================================
    if constexpr (has_gamma) {
        tilize_all_blocks_to_cb<block_size>(cb_gamma_rm, cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        tilize_all_blocks_to_cb<block_size>(cb_beta_rm, cb_beta, Wt);
    }

    // ================================================================
    // Wait for persistent CBs before main loop
    // ================================================================
    // cb_eps: waited once, never popped during loop (NoWaitNoPop in phase 6)
    cb_wait_front(cb_eps, 1);

    // cb_gamma/cb_beta: waited once, never popped
    if constexpr (has_gamma) {
        cb_wait_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_wait_front(cb_beta, Wt);
    }

    // Pre-compute 1/W as uint32 for mul_unary_tile
    const float inv_W = 1.0f / static_cast<float>(W);
    const uint32_t inv_W_bits = __builtin_bit_cast(uint32_t, inv_W);

    // ================================================================
    // Main loop: per tile-row
    // ================================================================
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // ============================================================
        // Phase 1: Tilize RM input -> tiled
        // ============================================================
        tilize_all_blocks_to_cb<block_size>(cb_in_rm, cb_in, Wt);

        // ============================================================
        // Phase 2: Reduce row to mean
        // Uses WaitUpfrontNoPop so cb_in persists for Phase 3
        // ============================================================
        reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
            cb_in,
            cb_scaler,
            cb_mean,
            ReduceInputBlockShape::row(Wt),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            [inv_W_bits](uint32_t dst_idx) {
                binop_with_scalar_tile_init();
                mul_unary_tile(dst_idx, inv_W_bits);
            });

        // ============================================================
        // Phase 3: Subtract mean (center)
        // cb_in: already waited from Phase 2, popped at end (NoWaitPopAtEnd for A)
        // cb_mean: 1 tile, Col0 broadcast across Wt tiles, consumed (WaitAndPopPerTile for B)
        // ============================================================
        sub<BroadcastDim::COL, BinaryInputPolicy::NoWaitPopAtEnd, BinaryInputPolicy::WaitAndPopPerTile>(
            cb_in, cb_mean, cb_centered, BinaryInputBlockShape::of(1, Wt));

        // ============================================================
        // Phase 4: Square centered values
        // cb_centered: WaitUpfrontNoPop so it persists for Phase 7
        // ============================================================
        square<BinaryInputPolicy::WaitUpfrontNoPop>(cb_centered, cb_sq, BinaryInputBlockShape::of(1, Wt));

        // ============================================================
        // Phase 5: Reduce row to variance
        // cb_sq: streaming, consumed
        // ============================================================
        reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(
            cb_sq,
            cb_scaler,
            cb_var,
            ReduceInputBlockShape::row(Wt),
            ReduceInputMemoryLayout::contiguous(),
            NoAccumulation{},
            [inv_W_bits](uint32_t dst_idx) {
                binop_with_scalar_tile_init();
                mul_unary_tile(dst_idx, inv_W_bits);
            });

        // ============================================================
        // Phase 6: Add epsilon + rsqrt
        // cb_var: consumed
        // cb_eps: persistent (NoWaitNoPop for B)
        // ============================================================
        add<BroadcastDim::SCALAR, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::NoWaitNoPop>(
            cb_var, cb_eps, cb_rsqrt, BinaryInputBlockShape::single(), [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // ============================================================
        // Phase 7: Multiply by rsqrt (normalize)
        // cb_centered: already waited from Phase 4, popped at end
        // cb_rsqrt: 1 tile, Col0 broadcast, consumed
        // Output goes to cb_affine_or_out (c_16 if no gamma/beta, c_25 otherwise)
        // ============================================================
        mul<BroadcastDim::COL, BinaryInputPolicy::NoWaitPopAtEnd, BinaryInputPolicy::WaitAndPopPerTile>(
            cb_centered, cb_rsqrt, cb_affine_or_out, BinaryInputBlockShape::of(1, Wt));

        // ============================================================
        // Phase 8 (optional): Multiply by gamma
        // ============================================================
        if constexpr (has_gamma) {
            mul<BroadcastDim::ROW, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::NoWaitNoPop>(
                cb_affine_or_out, cb_gamma, cb_scaled_or_out, BinaryInputBlockShape::of(1, Wt));
        }

        // ============================================================
        // Phase 9 (optional): Add beta
        // ============================================================
        if constexpr (has_beta) {
            add<BroadcastDim::ROW, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::NoWaitNoPop>(
                cb_scaled_or_out, cb_beta, cb_out, BinaryInputBlockShape::of(1, Wt));
        }

        // ============================================================
        // Phase 10: Untilize output -> RM
        // ============================================================
        untilize_all_blocks_from_cb<block_size>(cb_out, cb_out_rm, Wt);
    }

    // ================================================================
    // Cleanup: pop persistent CBs
    // ================================================================
    cb_pop_front(cb_eps, 1);
    if constexpr (has_gamma) {
        cb_pop_front(cb_gamma, Wt);
    }
    if constexpr (has_beta) {
        cb_pop_front(cb_beta, Wt);
    }
}
