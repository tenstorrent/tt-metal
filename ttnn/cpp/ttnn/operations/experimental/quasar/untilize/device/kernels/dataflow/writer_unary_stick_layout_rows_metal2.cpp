// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Writes untilized blocks to the row-major interleaved output. Each DFB entry is one full output
// row, so each TXN_ID write sends one entry to its page and acks it when it lands. This kernel's last
// block may hold only last_rows real rows.
void kernel_main() {
    const uint32_t num_blocks = get_arg(args::num_blocks);
    const uint32_t first_block = get_arg(args::first_block);  // tile-row index of this kernel's first block
    const uint32_t last_rows = get_arg(args::last_rows);
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t block_step = get_arg(args::block_step);  // tile rows between this kernel's blocks

    const auto s = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    for (uint32_t i = 0; i < num_blocks; ++i) {
        const uint32_t first_row = (first_block + i * block_step) * tile_height;
        const uint32_t num_rows = i == num_blocks - 1 ? last_rows : tile_height;
        for (uint32_t j = 0; j < num_rows; ++j) {
            noc.async_write<NocOptions::TXN_ID>(cb_out, s, {}, {.page_id = first_row + j});
        }
    }
}
