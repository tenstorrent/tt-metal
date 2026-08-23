// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "experimental/kernel_args.h"
#include "ttnn/kernel_lib/untilize_helpers.hpp"

template <uint32_t tiles_per_row, uint32_t block_size>
TT_KERNEL void pack_untilize(uint32_t total_blocks) {
    constexpr uint32_t src_cb_id = dfb::input;
    constexpr uint32_t out_cb_id = dfb::untilized;

    // Initialize once before the loop. tiles_per_row is the untilize width; block_size is the
    // number of tile rows consumed by each iteration.
    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    compute_kernel_lib::untilize_init<tiles_per_row, src_cb_id, out_cb_id>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        // The legacy factory bound both alternating destinations to the same untilized CB.
        // Use Neither mode because init/uninit are handled outside the loop.
        compute_kernel_lib::untilize<
            tiles_per_row,
            src_cb_id,
            out_cb_id,
            compute_kernel_lib::untilize_config::InitUninitMode::Neither,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
    }

    // Uninitialize once after all blocks have been processed.
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_cb_id, out_cb_id>();
}
