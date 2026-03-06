// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// Force recompile after binary_op fidelity fix

#include <cstdint>

#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
// CB indices
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_beta = 2;
constexpr uint32_t cb_gamma_rm = 3;
constexpr uint32_t cb_beta_rm = 4;
constexpr uint32_t cb_out_tile = 16;
constexpr uint32_t cb_out_rm = 17;
constexpr uint32_t cb_tilized = 24;
constexpr uint32_t cb_scaler = 25;
constexpr uint32_t cb_eps = 26;
constexpr uint32_t cb_mean = 27;
constexpr uint32_t cb_centered = 28;
constexpr uint32_t cb_sq = 29;
constexpr uint32_t cb_var = 30;
constexpr uint32_t cb_inv_std = 31;
}  // namespace

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t num_tile_rows = get_compile_time_arg_val(1);

    using namespace compute_kernel_lib;

    // hehe

    // Hardware init - first op is tilize from cb_gamma_rm to cb_gamma
    compute_kernel_hw_startup(cb_gamma_rm, cb_gamma);

    // ==================== Startup: tilize gamma and beta ====================

    // Tilize gamma: 1 RM page -> Wt tiles (asymmetric mode)
    tilize<cb_gamma_rm, cb_gamma>(Wt, 1, 1);

    // Tilize beta: 1 RM page -> Wt tiles (asymmetric mode)
    tilize<cb_beta_rm, cb_beta>(Wt, 1, 1);

    // ==================== Per tile row loop ====================
    for (uint32_t tr = 0; tr < num_tile_rows; tr++) {
        // Step 1: Tilize input - 32 RM pages -> Wt tiles
        tilize<cb_in, cb_tilized>(Wt, 1, 32);

        // Step 2: Reduce row to get mean: sum(x)/W -> cb_mean
        // cb_tilized persists (WaitUpfrontNoPop), scaler never popped by reduce
        reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
            cb_tilized, cb_scaler, cb_mean, ReduceInputBlockShape::row(Wt));

        // Step 3: x - mean -> cb_centered
        // A=cb_tilized persists, B=cb_mean consumed
        sub<BroadcastDim::COL, BinaryInputPolicy::WaitUpfrontNoPop, BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_tilized, cb_mean, cb_centered, BinaryInputBlockShape::row(Wt));

        // Step 4: (x - mean)^2 -> cb_sq
        // cb_centered persists (needed in step 7)
        square<BinaryInputPolicy::WaitUpfrontNoPop>(cb_centered, cb_sq, BinaryInputBlockShape::row(Wt));

        // Step 5: Reduce row to get variance: sum((x-mean)^2)/W -> cb_var
        // cb_sq consumed (default WaitAndPopPerTile)
        reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            cb_sq, cb_scaler, cb_var, ReduceInputBlockShape::row(Wt));

        // Step 6: rsqrt(var + eps) -> cb_inv_std (rsqrt as post-op)
        // A=cb_var consumed, B=cb_eps persists
        add<BroadcastDim::SCALAR, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_var, cb_eps, cb_inv_std, BinaryInputBlockShape::single(), {}, {}, [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Step 7: (x-mean) * rsqrt(var+eps) -> cb_out_tile
        // A=cb_centered consumed, B=cb_inv_std consumed
        mul<BroadcastDim::COL, BinaryInputPolicy::NoWaitPopAtEnd, BinaryInputPolicy::WaitUpfrontPopAtEnd>(
            cb_centered, cb_inv_std, cb_out_tile, BinaryInputBlockShape::row(Wt));

        // Step 8: norm * gamma -> cb_sq (reuse)
        // A=cb_out_tile consumed, B=cb_gamma persists
        mul<BroadcastDim::ROW, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_out_tile, cb_gamma, cb_sq, BinaryInputBlockShape::row(Wt));

        // Step 9: gamma*norm + beta -> cb_out_tile
        // A=cb_sq consumed, B=cb_beta persists
        add<BroadcastDim::ROW, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitUpfrontNoPop>(
            cb_sq, cb_beta, cb_out_tile, BinaryInputBlockShape::row(Wt));

        // Step 10: Untilize output tiles to RM
        untilize<Wt, cb_out_tile, cb_out_rm>(1);

        // Step 11: Manual pop of persistent tilized input
        cb_pop_front(cb_tilized, Wt);
    }
}
