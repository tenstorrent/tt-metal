// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Writes untilized blocks into this core's row-major output shard. Each DFB entry is one shard row, so
// each TXN_ID write copies one entry to its row and acks it when it lands.
void kernel_main() {
    const uint32_t num_blocks = get_arg(args::num_blocks);
    const uint32_t first_block = get_arg(args::first_block);  // tile-row index of this kernel's first block
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t block_step = get_arg(args::block_step);  // tile rows between this kernel's blocks
    constexpr uint32_t row_bytes = get_arg(args::row_bytes);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);
    const uint32_t shard_base = TensorAccessor(tensor::output).get_bank_base_address();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    for (uint32_t i = 0; i < num_blocks; ++i) {
        const uint32_t first_row = (first_block + i * block_step) * tile_height;
        for (uint32_t j = 0; j < tile_height; ++j) {
            noc.async_write<NocOptions::TXN_ID>(
                cb_out,
                self_ep,
                {},
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = shard_base + (first_row + j) * row_bytes});
        }
    }
}
