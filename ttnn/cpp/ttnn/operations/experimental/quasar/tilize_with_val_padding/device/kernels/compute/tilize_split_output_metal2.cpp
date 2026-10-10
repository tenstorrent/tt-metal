// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tilize with two output DFBs: even sub-blocks go to dfb::out and odd ones to dfb::out_odd, so two
// single-thread writer kernels drain them in parallel. Each input entry is one row segment of a
// sub-block of sub_block_tiles tiles, so a sub-block is block_rows entries.

#include <cstdint>

#include "api/compute/pack.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_sub_blocks = get_arg(args::num_sub_blocks);
    constexpr uint32_t sub_block_tiles = get_arg(args::sub_block_tiles);
    constexpr uint32_t block_rows = get_arg(args::block_rows);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    DataflowBuffer in(dfb::in);
    DataflowBuffer out_even(dfb::out);
    DataflowBuffer out_odd(dfb::out_odd);

    for (uint32_t s = 0; s < num_sub_blocks; ++s) {
        const bool even = (s % 2) == 0;
        DataflowBuffer& out = even ? out_even : out_odd;
        const uint32_t out_id = even ? dfb::out : dfb::out_odd;
        // The packer's buffer descriptor names one output DFB, so it is reprogrammed per sub-block.
        tilize_init(dfb::in, sub_block_tiles, out_id);
        pack_init(out_id);
        in.wait_front(block_rows);
        out.reserve_back(sub_block_tiles);
        tilize_block(dfb::in, sub_block_tiles, out_id);
        out.push_back(sub_block_tiles);
        in.pop_front(block_rows);
        tilize_uninit(dfb::in, out_id);
    }
}
