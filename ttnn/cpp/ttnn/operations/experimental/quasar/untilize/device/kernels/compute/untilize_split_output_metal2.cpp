// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilize with two output DFBs: even blocks go to dfb::out and odd blocks to dfb::out_odd, so two
// single-thread writer kernels drain them in parallel. Each output entry is one row of the block, so
// a writer moves whole entries with implicit sync. The input DFB holds sub_block_tiles-wide
// sub-blocks, one per wait. The core's last block pushes only its last_block_rows real rows; nothing
// is packed after it, so its padding rows are simply never posted.

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    const uint32_t last_block_rows = get_arg(args::last_block_rows);
    if (per_core_block_cnt == 0) {
        return;
    }

    constexpr uint32_t block_tiles = get_arg(args::per_core_block_tile_cnt);
    constexpr uint32_t sub_block_tiles = get_arg(args::sub_block_tiles);
    constexpr uint32_t block_rows = get_arg(args::block_rows);
    static_assert(block_tiles % sub_block_tiles == 0, "sub-blocks must tile the block");
    static_assert(sub_block_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "sub-block must fit in DEST");

    compute_kernel_hw_startup(dfb::in, dfb::out);
    DataflowBuffer in(dfb::in);
    DataflowBuffer out_even(dfb::out);
    DataflowBuffer out_odd(dfb::out_odd);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        const bool even = (r % 2) == 0;
        DataflowBuffer& out = even ? out_even : out_odd;
        const uint32_t out_id = even ? dfb::out : dfb::out_odd;
        // The packer's buffer descriptor names one output DFB, so it is reprogrammed per block.
        pack_untilize_init<sub_block_tiles, block_tiles>(dfb::in, out_id);
        out.reserve_back(block_rows);
        for (uint32_t b = 0; b < block_tiles / sub_block_tiles; ++b) {
            in.wait_front(sub_block_tiles);
            pack_untilize_block<sub_block_tiles, block_tiles>(dfb::in, 1, out_id, b);
            in.pop_front(sub_block_tiles);
        }
        out.push_back(r == per_core_block_cnt - 1 ? last_block_rows : block_rows);
        pack_untilize_uninit(out_id);
    }
}
