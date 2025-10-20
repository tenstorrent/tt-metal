// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file numeric.h
 * @brief Numeric functions for compute kernels.
 */

#pragma once

#include "tt_metal/api/tt-metalium/constants.hpp"
#include "tt_metal/api/tt-metalium/math.hpp"
#include "tt_metal/include/compute_kernel_api/eltwise_binary_sfpu.h"
#include "tt_metal/include/compute_kernel_api/eltwise_unary/fill.h"
#include "tt_metal/include/compute_kernel_api/tile_move_copy.h"
#include "tt_metal/include/compute_kernel_api/eltwise_unary/recip.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/generic/bit.h"
#include "ttnn/cpp/ttnn/operations/normalization/kernel_util/generic/policies.h"

namespace norm::kernel_util::compute::numeric {

namespace sfpu {
/**
 * @brief Generate a series of tiles containing consecutive reciprocal values,
 * of consecutive integers, e.g. 1/N, 1/(N+1), 1/(N+2), ... into a CB.
 *
 * @details Computes and pushes enough tiles to @p cb_out_id to contain reciprocals
 * up to 1/(@p upper_bound) (inclusive). Uses pre-populated CB with
 * id @p cb_int_id containing consecutive integers [1, 2, ... TILE_HW] to do
 * the calculation.
 *
 * @tparam upper_bound The number of reciprocals in the output CB
 *                     (defines the number of tiles to generate)
 * @tparam T The data type of the reciprocals
 * @tparam WaitPolicy Specify whether the function should wait for the
 * pushes tiles after pushing them
 * @param cb_int_id CB containing consecutive integers [1, 2, ... TILE_HW]
 * @param cb_out_id Output CB containing the reciprocals
 *
 */
template <
    uint32_t upper_bound,
    typename T = float,
    generic::policies::CBWaitPolicy WaitPolicy = generic::policies::CBWaitPolicy::Wait>
ALWI void generate_consecutive_reciprocal_tiles(const uint32_t cb_int_id, const uint32_t cb_out_id) {
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t num_recip_tiles = tt::round_down(upper_bound, tt::constants::TILE_HW) + 1;

    reconfig_data_format_srca(cb_int_id);
    pack_reconfig_data_format(cb_out_id);
    cb_wait_front(cb_int_id, 1);
    cb_reserve_back(cb_out_id, num_recip_tiles);

    // Helper to facilitate not doing
    // multiplication on the first tile without
    // branching.
    auto copy_mul_recip = [&](auto&& mul_fn) {
        tile_regs_acquire();

        // Copy the integer tile to dst0
        copy_tile_init(cb_int_id);
        copy_tile(cb_int_id, 0, dst0);

        // Call the multiply function
        mul_fn();

        // Take the reciprocal
        recip_tile_init();
        recip_tile(dst0);

        // Pack the result to the output CB
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out_id);
        tile_regs_release();
        cb_push_back(cb_out_id, 1);
    };

    // First tile doesn't need multiplication
    copy_mul_recip([]() {});

    // Rest of the tiles multiply the base integer
    // tile by i+1 before doing the reciprocal
    for (uint32_t i = 1; i < num_recip_tiles; i++) {
        copy_mul_recip([i]() {
            binop_with_scalar_tile_init();
            mul_unary_tile(dst0, norm::kernel_util::generic::bit_cast<uint32_t>(static_cast<float>(i + 1)));
        });
    }

    // Wait for all tiles to be pushed
    if constexpr (WaitPolicy == norm::kernel_util::generic::policies::CBWaitPolicy::Wait) {
        cb_wait_front(cb_out_id, num_recip_tiles);
    }
}

}  // namespace sfpu
}  // namespace norm::kernel_util::compute::numeric
