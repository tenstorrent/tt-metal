// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#ifdef IMPLICIT_SYNC
#include "api/kernel_thread_globals.h"
#endif

void kernel_main() {
    // run-time args
    const uint32_t block_height_tiles = get_arg(args::block_height_tiles);
    const uint32_t block_width_tiles = get_arg(args::block_width_tiles);
    const uint32_t padded_offset = get_arg(args::padded_offset);
    const uint32_t block_width_padded_num_tiles = get_arg(args::block_width_padded_num_tiles);
    const uint32_t output_width_tiles = get_arg(args::output_width_tiles);
    const uint32_t start_id_offset = get_arg(args::start_id_offset);
    const uint32_t start_id_base = get_arg(args::start_id_base);

#ifdef IMPLICIT_SYNC
    const auto s = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    // Host enables this with a DFB that holds the block's tiles in row-major order, without the
    // shard's padding. Thread t of N writes tiles t, t + N, ...: the strided DFB hands it exactly
    // those, in order, and each TXN_ID write acks its entry when it lands.
    const uint32_t start_id = start_id_base + start_id_offset;
    for (uint32_t k = get_my_thread_id(); k < block_height_tiles * block_width_tiles; k += get_num_threads()) {
        const uint32_t tile_id = start_id + (k / block_width_tiles) * output_width_tiles + k % block_width_tiles;
        noc.async_write<NocOptions::TXN_ID>(cb_out, s, {}, {.page_id = tile_id});
    }
#else
    // single-tile ublocks
    const uint32_t tile_bytes = DataflowBuffer(dfb::out).get_entry_size();

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    uint32_t row_start_tile_id = start_id_base + start_id_offset;
    cb_out.wait_front(block_width_padded_num_tiles);
    uint32_t l1_read_offset = 0;
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = row_start_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc.async_write(
                cb_out, s, tile_bytes, {.offset_bytes = l1_read_offset}, {.page_id = tile_id, .offset_bytes = 0});
            tile_id++;
            l1_read_offset += tile_bytes;
        }
        l1_read_offset += padded_offset;
        row_start_tile_id += output_width_tiles;
    }
    noc.async_write_barrier();
    cb_out.pop_front(block_width_padded_num_tiles);
#endif
}
