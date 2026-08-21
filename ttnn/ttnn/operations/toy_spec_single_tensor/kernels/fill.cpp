// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Generator half of the write-only program: materializes a constant bf16 tile in L1 with no NOC
// read at all, so the program reads no tensor. `fill_bits` is the bf16 encoding of the value.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t fill_bits = get_arg(args::fill_bits);

    DataflowBuffer dfb_out(dfb::out);
    const uint32_t words_per_tile = dfb_out.get_tile_size() / sizeof(uint32_t);
    const uint32_t pattern = (fill_bits << 16) | fill_bits;  // two bf16 lanes per 32-bit word

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_out.reserve_back(1);
        {
            auto lock = dfb_out.scoped_write_lock(1);
            auto entry = lock.get_ptr<uint32_t>();
            for (uint32_t w = 0; w < words_per_tile; ++w) {
                entry[w] = pattern;
            }
        }
        dfb_out.push_back(1);
    }
}
