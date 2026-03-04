// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (Stage 3: full_normalize)
//
// Per tile-row:
//   Phase 0: Tilize cb_in -> cb_tilize_out
//   Phase 1: Reduce SUM (scaler=1/W) -> cb_mean
//   Phase 2: Sub broadcast COL (x - mean) -> cb_x_minus_mean
//   Phase 3: Square (x-mean)^2 -> cb_sq
//   Phase 4: Reduce SUM (scaler=1/W) of squares -> cb_var
//   Phase 4b: (var + eps) -> rsqrt -> cb_inv_std
//   Phase 5: Multiply (x-mean) * inv_std broadcast COL -> cb_out
//   Phase 8: Untilize cb_out -> cb_untilize_out

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

// CB indices matching op_design.md
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_eps = 1;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_weight = 3;
constexpr uint32_t cb_bias = 4;
constexpr uint32_t cb_tilize_out = 16;
constexpr uint32_t cb_out = 17;
constexpr uint32_t cb_untilize_out = 18;
constexpr uint32_t cb_mean = 24;
constexpr uint32_t cb_x_minus_mean = 25;
constexpr uint32_t cb_sq = 26;
constexpr uint32_t cb_var = 27;
constexpr uint32_t cb_inv_std = 28;
constexpr uint32_t cb_norm = 29;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_weight = get_compile_time_arg_val(2);
    constexpr uint32_t has_bias = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    for (uint32_t tile_row = 0; tile_row < Ht; ++tile_row) {
        // Phase 0: Tilize
        compute_kernel_lib::tilize<cb_in, cb_tilize_out>(Wt, 1);

        // Phase 1: Mean = reduce_SUM(x) * (1/W)
        // WaitUpfrontNoPop: cb_tilize_out tiles persist for Phase 2
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilize_out, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 2: x - mean -> cb_x_minus_mean
        // cb_tilize_out: already waited (NoPop from Phase 1), popped at end
        // cb_mean: waited and popped at end
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_tilize_out, cb_mean, cb_x_minus_mean, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 3: square (x-mean)^2 -> cb_sq
        // WaitUpfrontNoPop: cb_x_minus_mean tiles persist for Phase 5
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_x_minus_mean, cb_sq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 4: variance = reduce_SUM(sq) * (1/W)
        // BulkWaitBulkPop: cb_sq is consumed and freed
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_sq, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 4b: inv_std = rsqrt(var + eps)
        // Use add<SCALAR> helper with rsqrt post-op
        // cb_var: 1 tile, consumed and popped
        // cb_eps: 1 tile, persistent (NoPop)
        // Output: cb_inv_std (1 tile)
        compute_kernel_lib::add<
            compute_kernel_lib::BroadcastDim::SCALAR,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::PerTile>(
            cb_var,
            cb_eps,
            cb_inv_std,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            {},
            compute_kernel_lib::NoAccumulation{},
            [](uint32_t dst_idx) {
                rsqrt_tile_init();
                rsqrt_tile(dst_idx);
            });

        // Phase 5: norm = (x-mean) * inv_std (broadcast COL)
        // cb_x_minus_mean: already waited (NoPop from Phase 3), popped at end
        // cb_inv_std: 1 tile, popped at end
        // Output: cb_out (route directly to untilize for stage 3)
        compute_kernel_lib::mul<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_x_minus_mean, cb_inv_std, cb_out, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 8: Untilize cb_out -> cb_untilize_out
        compute_kernel_lib::untilize<
            Wt,
            cb_out,
            cb_untilize_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitUpfront>(1);
    }
}
