// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t block_height_tiles = get_arg(args::block_height_tiles);
    const uint32_t block_width_tiles = get_arg(args::block_width_tiles);
    const uint32_t unpadded_block_height_tiles = get_arg(args::unpadded_block_height_tiles);
    const uint32_t unpadded_block_width_tiles = get_arg(args::unpadded_block_width_tiles);
    const uint32_t output_width_tiles =
        get_arg(args::output_width_tiles);                            // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg(args::block_num_tiles);  // block_height_tiles * block_width_tiles
    const uint32_t start_id_offset = get_arg(args::start_id_offset);
    const uint32_t start_id_base = get_arg(args::start_id_base);
    const uint32_t start_id = start_id_base + start_id_offset;

    // On Quasar a DataflowBuffer drains (waits for posted == acked) when destroyed, so the entry
    // size must come from the long-lived consumer: a temporary would block on tiles the reader
    // already posted, which only this kernel acks.
    DataflowBuffer cb_out(dfb::out);
    // single-tile ublocks
    const uint32_t tile_bytes = cb_out.get_entry_size();

    // The destination-buffer base address is bound via the tensor parameter (tensor::dst),
    // replacing the legacy buffer-address RTA slot 0.
    const auto s = TensorAccessor(tensor::dst);

    Noc noc;

    // Thread t of N writes shard tiles t, t + N, ...: the strided DFB hands it exactly those entries,
    // in order.
    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
#ifdef IMPLICIT_SYNC
    // Host enables this with a staging DFB that holds the block's unpadded tiles in row-major order:
    // each TXN_ID write drains the next staged tile and acks it when it lands.
    for (uint32_t k = thread_id; k < unpadded_block_height_tiles * unpadded_block_width_tiles; k += num_threads) {
        const uint32_t tile_id =
            start_id + (k / unpadded_block_width_tiles) * output_width_tiles + k % unpadded_block_width_tiles;
        noc.async_write<NocOptions::TXN_ID>(cb_out, s, {}, {.page_id = tile_id});
    }
#else
    // Explicit wait/pop move one TC per call, so entries are taken one at a time.
    for (uint32_t k = thread_id; k < block_num_tiles; k += num_threads) {
        const uint32_t h = k / block_width_tiles;
        const uint32_t w = k % block_width_tiles;
        cb_out.wait_front(1);
        if (h < unpadded_block_height_tiles && w < unpadded_block_width_tiles) {
            noc.async_write(cb_out, s, tile_bytes, {}, {.page_id = start_id + h * output_width_tiles + w});
            noc.async_writes_flushed();
        }
        cb_out.pop_front(1);
    }
    noc.async_write_barrier();
#endif
}
