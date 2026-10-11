// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Feeds untilize's compute threads ("lanes") one tile per DFB entry with implicit sync. Reader thread t
// serves lane t alone: with as many readers as compute threads the strided DFB pairs producer t with
// consumer t. With COLUMN_LANES lane t takes tiles [t * lane_tiles, (t + 1) * lane_tiles) of every block
// (a block is one tile row); otherwise it takes every num_threads-th block whole. Tiles come from the
// interleaved input, or with LOCAL_SHARD from this core's input shard.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_lane_blocks = get_arg(args::num_lane_blocks);
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t lane_tiles = get_arg(args::lane_tiles);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const auto s = TensorAccessor(tensor::input);
#if LOCAL_SHARD
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);
    const uint32_t shard_base = s.get_bank_base_address();
    UnicastEndpoint self_ep;
    const uint32_t my_noc_x = my_x[noc.get_noc_id()];
    const uint32_t my_noc_y = my_y[noc.get_noc_id()];
#endif

    const uint32_t lane = get_my_thread_id();
    const uint32_t num_lanes = get_num_threads();
    for (uint32_t m = 0; m < num_lane_blocks; ++m) {
        const uint32_t block = COLUMN_LANES ? first_block + m : first_block + lane + num_lanes * m;
        const uint32_t first_tile = block * tiles_per_row + (COLUMN_LANES ? lane * lane_tiles : 0);
        for (uint32_t j = 0; j < lane_tiles; ++j) {
#if LOCAL_SHARD
            noc.async_read<NocOptions::TXN_ID>(
                self_ep,
                cb_in,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = shard_base + (first_tile + j) * tile_bytes},
                {});
#else
            noc.async_read<NocOptions::TXN_ID>(s, cb_in, {.page_id = first_tile + j}, {});
#endif
        }
    }
}
