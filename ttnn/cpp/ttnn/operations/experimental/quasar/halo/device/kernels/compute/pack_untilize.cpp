// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"
#include "experimental/kernel_args.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;

void kernel_main() {
    // Metal 2.0 named bindings:
    //   dfb::src           - borrowed input shard (compute consumes, untilizes)
    //   dfb::untilize_out0 - untilized output for reader0 (even blocks)
    //   dfb::untilize_out1 - untilized output for reader1 (odd blocks)
    constexpr uint32_t src_cb_id = dfb::src;
    constexpr uint32_t out_cb_id0 = dfb::untilize_out0;
    constexpr uint32_t out_cb_id1 = dfb::untilize_out1;
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);  // number of tiles along width of shard
    constexpr uint32_t block_size = get_arg(args::block_size);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg(args::total_blocks);

    compute_kernel_hw_startup(src_cb_id, out_cb_id0);

    // Initialize once before the loop
    compute_kernel_lib::untilize_init<tiles_per_row, src_cb_id, out_cb_id0>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        // Use unified untilize with Neither mode since we handle init/uninit outside the loop
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id0,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id1,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        }
    }

    // Uninit after loop
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_cb_id, out_cb_id0>();
}
