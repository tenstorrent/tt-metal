// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file copy_tile_helpers.inl
 * @brief Implementation of copy_tiles helper function
 *
 * This file contains the implementation details for the copy_tiles() function.
 * It should only be included by copy_tile_helpers.hpp.
 */

namespace compute_kernel_lib {

template <
    CopyInputPolicy input_policy,
    CopyDataFormatReconfig reconfig_mode,
    typename PostOp>
ALWI void copy_tiles(uint32_t input_cb, uint32_t output_cb, uint32_t num_tiles, PostOp post_op) {
    ASSERT(num_tiles > 0);

    // Data format reconfiguration
    constexpr bool reconfig_input =
        (reconfig_mode == CopyDataFormatReconfig::INPUT) ||
        (reconfig_mode == CopyDataFormatReconfig::INPUT_AND_OUTPUT);

    constexpr bool reconfig_output =
        (reconfig_mode == CopyDataFormatReconfig::OUTPUT) ||
        (reconfig_mode == CopyDataFormatReconfig::INPUT_AND_OUTPUT);

    if constexpr (reconfig_input) {
        reconfig_data_format_srca(input_cb);
    }

    if constexpr (reconfig_output) {
        pack_reconfig_data_format(output_cb);
    }

    // Initialize unpacker for datacopy from input CB
    copy_tile_to_dst_init_short(input_cb);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        if constexpr (input_policy == CopyInputPolicy::WaitAndPop) {
            cb_wait_front(input_cb, 1);
        }

        // Tile index in the CB: 0 for streaming (WaitAndPop pops after each tile),
        // i for indexed access (NoWaitNoPop, all tiles pre-loaded).
        constexpr bool streaming = (input_policy == CopyInputPolicy::WaitAndPop);
        const uint32_t in_tile_idx = streaming ? 0 : i;

        copy_tile(input_cb, in_tile_idx, 0);

        if constexpr (input_policy == CopyInputPolicy::WaitAndPop) {
            cb_pop_front(input_cb, 1);
        }

        // Apply optional post-copy operation while tile is in DST
        post_op(0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(output_cb, 1);
        pack_tile(0, output_cb);
        cb_push_back(output_cb, 1);

        tile_regs_release();
    }
}

}  // namespace compute_kernel_lib
