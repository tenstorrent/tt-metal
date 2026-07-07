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
        // QSR: the two blocks alternate output CBs (even->out0, odd->out1) so the split readers each
        // consume their own untilize_out. On Quasar the pack writes via a buffer descriptor bound at
        // pack_untilize_init time (the MOP bakes in the output CB's buf_desc_id / L1 base); it is NOT
        // re-resolved per pack_untilize_block from the runtime `ocb` the way WH/BH dynamic pack addressing
        // is. With a single up-front init (for out0) + InitUninitMode::Neither, every block therefore packs
        // to out0's descriptor -> block 1 overwrites block 0 at out0 and out1 is never written (out0 ends up
        // holding block 1's data, out1 all zeros -> the pool input is scrambled by +one tile-row). Re-init
        // (InitOnly) per block so the pack rebinds to the correct out_cb_idN. (Unpack/math re-init for the
        // same src_cb_id is idempotent; the trailing untilize_uninit still runs once after the loop.)
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id0,
                compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id1,
                compute_kernel_lib::untilize_config::InitUninitMode::InitOnly,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        }
    }

    // Uninit after loop
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_cb_id, out_cb_id0>();
}
