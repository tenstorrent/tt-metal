// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_sharded_blocks.cpp.
//
// Block-by-block reader for sharded inputs. Reads from a borrowed-memory input
// region into a double-buffered DFB. Used when the shard CB can't be backed
// directly (e.g., uneven sharding).
//
// Bindings:
//   dfb::input               — DFB endpoint (PRODUCER)
//   ta::input                — TensorBinding (for the input buffer's L1 base address)
//   args::tiles_per_block    — CTA
//   args::num_blocks         — RTA

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto tiles_per_block = get_arg(args::tiles_per_block);
    auto num_blocks = get_arg(args::num_blocks);

    // ta::input provides the source buffer's L1 base address via implicit conversion.
    const uint32_t src_addr = TensorAccessor(ta::input).get_base_address();

    const uint32_t tile_size_bytes = get_tile_size(dfb::input);
    const uint32_t block_size_bytes = tiles_per_block * tile_size_bytes;

    Noc noc;
    DataflowBuffer cb_in(dfb::input);
    uint32_t l1_read_addr = src_addr;

    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_in.reserve_back(tiles_per_block);
        noc.async_read(
            UnicastEndpoint{},
            cb_in,
            block_size_bytes,
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
             .addr = l1_read_addr},
            {.offset_bytes = 0});
        l1_read_addr += block_size_bytes;
        noc.async_read_barrier();
        cb_in.push_back(tiles_per_block);
    }
}
