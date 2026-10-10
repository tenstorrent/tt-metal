// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Writes every other tilized sub-block of this core, starting at first_sub_block, to the interleaved
// output. Each TXN_ID write drains one tile entry and acks it when it lands.
void kernel_main() {
    constexpr uint32_t sub_block_tiles = get_arg(args::sub_block_tiles);
    constexpr uint32_t sub_blocks_per_row = get_arg(args::sub_blocks_per_row);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);

    const uint32_t num_sub_blocks = get_arg(args::num_sub_blocks);
    const uint32_t first_sub_block = get_arg(args::first_sub_block);
    const uint32_t start_tile = get_arg(args::start_tile);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);
    const auto s = TensorAccessor(tensor::output);

    for (uint32_t sb = first_sub_block; sb < num_sub_blocks; sb += 2) {
        const uint32_t first_tile =
            start_tile + (sb / sub_blocks_per_row) * tiles_per_row + (sb % sub_blocks_per_row) * sub_block_tiles;
        for (uint32_t t = 0; t < sub_block_tiles; ++t) {
            noc.async_write<NocOptions::TXN_ID>(cb_out, s, {}, {.page_id = first_tile + t});
        }
    }
}
