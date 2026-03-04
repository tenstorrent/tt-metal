// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Compute Kernel (Stage 4: affine_transform)
//
// Per tile-row:
//   Phase 0: Tilize cb_in -> cb_tilize_out
//   Phase 1: Reduce SUM (scaler=1/W) -> cb_mean
//   Phase 2: Sub broadcast COL (x - mean) -> cb_x_minus_mean
//   Phase 3: Square (x-mean)^2 -> cb_sq
//   Phase 4: Reduce SUM (scaler=1/W) of squares -> cb_var
//   Phase 4b: (var + eps) -> rsqrt -> cb_inv_std
//   Phase 5: Multiply (x-mean) * inv_std broadcast COL -> cb_norm
//   Phase 6: norm * weight (ROW broadcast) -> cb_out (if has_weight)
//   Phase 7: result + bias (ROW broadcast) -> alternating CB (if has_bias)
//   Phase 8: Untilize final CB -> cb_untilize_out

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

    // Determine which CB holds the final result before untilize.
    // Phase 5 -> cb_norm
    // Phase 6 (weight): cb_norm -> cb_out
    // Phase 7 (bias): alternates: if came from cb_out -> cb_norm, if from cb_norm -> cb_out
    // Logic: weight flips norm->out, bias flips again
    constexpr uint32_t cb_final = []() {
        if constexpr (has_weight && has_bias) {
            return cb_norm;  // Phase 5->norm, Phase 6->out, Phase 7->norm
        } else if constexpr (has_weight) {
            return cb_out;  // Phase 5->norm, Phase 6->out
        } else if constexpr (has_bias) {
            return cb_out;  // Phase 5->norm, Phase 7->out
        } else {
            return cb_out;  // Phase 5->cb_out directly (no affine)
        }
    }();

    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    for (uint32_t tile_row = 0; tile_row < Ht; ++tile_row) {
        // Phase 0: Tilize
        compute_kernel_lib::tilize<cb_in, cb_tilize_out>(Wt, 1);

        // Phase 1: Mean = reduce_SUM(x) * (1/W)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
                cb_tilize_out, cb_scaler, cb_mean, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 2: x - mean -> cb_x_minus_mean
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::COL,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_tilize_out, cb_mean, cb_x_minus_mean, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 3: square (x-mean)^2 -> cb_sq
        compute_kernel_lib::square<
            compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
            compute_kernel_lib::BinaryOutputPolicy::Bulk>(
            cb_x_minus_mean, cb_sq, compute_kernel_lib::BinaryInputBlockShape::row(Wt));

        // Phase 4: variance = reduce_SUM(sq) * (1/W)
        compute_kernel_lib::
            reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
                cb_sq, cb_scaler, cb_var, compute_kernel_lib::ReduceInputBlockShape::row(Wt));

        // Phase 4b: inv_std = rsqrt(var + eps)
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
        // Output destination depends on whether we have affine transform
        if constexpr (has_weight || has_bias) {
            // Write to cb_norm, affine phases will follow
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryOutputPolicy::Bulk>(
                cb_x_minus_mean, cb_inv_std, cb_norm, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        } else {
            // No affine: write directly to cb_out for untilize
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::COL,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryOutputPolicy::Bulk>(
                cb_x_minus_mean, cb_inv_std, cb_out, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 6: norm * weight (element-wise, weight expanded to full tiles)
        if constexpr (has_weight) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
                compute_kernel_lib::BinaryOutputPolicy::Bulk>(
                cb_norm, cb_weight, cb_out, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
        }

        // Phase 7: result + bias (element-wise, bias expanded to full tiles)
        if constexpr (has_bias) {
            if constexpr (has_weight) {
                // Input from Phase 6 is in cb_out, write to cb_norm
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
                    compute_kernel_lib::BinaryOutputPolicy::Bulk>(
                    cb_out, cb_bias, cb_norm, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            } else {
                // No weight: input from Phase 5 is in cb_norm, write to cb_out
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
                    compute_kernel_lib::BinaryOutputPolicy::Bulk>(
                    cb_norm, cb_bias, cb_out, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
            }
        }

        // Phase 8: Untilize final CB -> cb_untilize_out
        compute_kernel_lib::untilize<
            Wt,
            cb_final,
            cb_untilize_out,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitUpfront>(1);
    }
}
