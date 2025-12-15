// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file layernorm_utils.h
 * @brief Utility functions for the layernorm compute kernels.
 */

#pragma once

#include <tt-metalium/constants.hpp>
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"

namespace generic = norm::kernel_util::generic;

namespace norm::layernorm::device::kernels::compute {
/**
 * @brief Remove the excess contribution to the computed
 * variance due to reducing a partially-filled last tile.
 * Does nothing if tile-aligned
 *
 * @param W Width of the tensor
 * @param extra_cols Extra columns that were overaccumulated
 * @param cb_var Variance CB. Must contain a computed
 * variance, and have space for 2 total tiles
 * @param cb_mean Mean CB. Must contain a computed
 * mean tile
 *
 * @note cb_var must have space for 2 tiles
 * @note Uses destination registers dst0 and dst1
 */
template <uint32_t W, uint32_t extra_cols>
inline void adjust_variance_for_overaccumulation(uint32_t cb_var, uint32_t cb_mean) {
    if constexpr (extra_cols > 0) {
        constexpr uint32_t dst0 = 0;
        constexpr uint32_t dst1 = 1;
        cb_wait_front(cb_var, 1);
        cb_wait_front(cb_mean, 1);
        cb_reserve_back(cb_var, 1);
        reconfig_data_format_srca(cb_var);
        copy_tile_init(cb_var);
        tile_regs_acquire();

        // Copy variance tile to dst0
        copy_tile_init(cb_var);
        copy_tile(cb_var, 0, dst0);

        // Copy E[x] tile to dst1
        copy_tile_init(cb_mean);
        copy_tile(cb_mean, 0, dst1);

        // Square E[x]
        square_tile_init();
        square_tile(dst1);

        // Multiply by (#extra cols / W)
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, generic::bit_cast<uint32_t>(static_cast<float>(extra_cols) / W));

        // Subtract dst1 from dst0
        sub_binary_tile_init();
        sub_binary_tile(dst0, dst1, dst0);

        tile_regs_commit();
        tile_regs_wait();

        // Pack final value in dst0 into cb_ex2
        pack_tile(dst0, cb_var);

        tile_regs_release();

        cb_push_back(cb_var, 1);

        // Pop the old var tile
        cb_pop_front(cb_var, 1);
    }
}
}  // namespace norm::layernorm::device::kernels::compute
