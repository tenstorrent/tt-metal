// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Copies this core's input shard into the staging DFB for compute. Thread t of N copies the sub-blocks
// t, t + N, ... of sub_block_tiles consecutive tiles, the unit compute waits for: the ALL-pattern DFB
// gives each thread its own contiguous region and lets compute take sub-blocks from the threads in
// turn. Each TXN_ID read fills the next entry and posts its credit when it lands.
void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t sub_block_tiles = get_arg(args::sub_block_tiles);
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const uint32_t shard_base = TensorAccessor(tensor::input).get_bank_base_address();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];

    const uint32_t sub_block_step = get_num_threads() * sub_block_tiles;
    for (uint32_t first = get_my_thread_id() * sub_block_tiles; first < num_tiles; first += sub_block_step) {
        for (uint32_t t = first; t < first + sub_block_tiles && t < num_tiles; ++t) {
            noc.async_read<NocOptions::TXN_ID>(
                self_ep, cb_in, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = shard_base + t * tile_bytes}, {});
        }
    }
}
