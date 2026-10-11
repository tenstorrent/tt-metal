// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
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

    // QUASAR: the DFB objects are constructed ONCE for the whole kernel. On Quasar ~DataflowBuffer()
    // drains the buffer (spins until the peer has consumed everything that was pushed), so going
    // through compute_kernel_lib::untilize() per block -- which constructs/destroys its own
    // DataflowBuffer objects on every call -- serialises the packer against the readers after every
    // block and, once a reader has finished its last block without popping, deadlocks the halo.
    // The untilize loop is therefore written out here with the same DEST sub-blocking as the helper.
    DataflowBuffer in_dfb(src_cb_id);
    DataflowBuffer out_dfb0(out_cb_id0);
    DataflowBuffer out_dfb1(out_cb_id1);

    // DEST sub-blocking for wide rows (same as compute_kernel_lib::UntilizeDispatchConfig).
    constexpr uint32_t dest_limit = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr bool use_sub_blocks = tiles_per_row > dest_limit;
    constexpr uint32_t num_sub_blocks =
        use_sub_blocks ? compute_kernel_lib::compute_num_blocks(tiles_per_row, dest_limit) : 1;
    constexpr uint32_t sub_block_width = use_sub_blocks ? (tiles_per_row / num_sub_blocks) : tiles_per_row;

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        const bool even = (block_idx % 2 == 0);
        const uint32_t out_cb_id = even ? out_cb_id0 : out_cb_id1;
        DataflowBuffer& out_dfb = even ? out_dfb0 : out_dfb1;

        // QSR: the Quasar pack_untilize LLK bakes the destination DFB's base L1 address at
        // pack_untilize_init time, so the packer is re-initialised per block for the block's
        // actual destination (out0 for reader0's even blocks, out1 for reader1's odd blocks).
        pack_untilize_init<sub_block_width, tiles_per_row>(src_cb_id, out_cb_id);
        for (uint32_t r = 0; r < block_size; ++r) {
            out_dfb.reserve_back(tiles_per_row);
            for (uint32_t b = 0; b < num_sub_blocks; ++b) {
                in_dfb.wait_front(sub_block_width);
                pack_untilize_block<sub_block_width, tiles_per_row>(src_cb_id, 1, out_cb_id, b);
                in_dfb.pop_front(sub_block_width);
            }
            out_dfb.push_back(tiles_per_row);
        }
        pack_untilize_uninit(out_cb_id);
    }
}
