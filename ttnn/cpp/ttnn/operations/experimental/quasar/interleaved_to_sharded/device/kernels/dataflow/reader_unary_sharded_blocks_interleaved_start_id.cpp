// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "tensix_types.h"
#include "experimental/kernel_args.h"

// Target 8KB of data before a single barrier for 8x8 grid of readers
template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    const uint32_t block_height_tiles = get_arg(args::block_height_tiles);
    const uint32_t block_width_tiles = get_arg(args::block_width_tiles);
    const uint32_t padded_offset_bytes =
        get_arg(args::padded_offset_bytes);  // input width in tiles - block width in tiles
    const uint32_t input_width_offset_tiles =
        get_arg(args::input_width_offset_tiles);                      // input width in tiles - block width in tiles
    const uint32_t block_num_tiles = get_arg(args::block_num_tiles);  // block_height_tiles * block_width_tiles
    const uint32_t start_id_offset = get_arg(args::start_id_offset);
    const uint32_t start_id_base = get_arg(args::start_id_base);
    const uint32_t start_id = start_id_base + start_id_offset;

    constexpr uint32_t num_readers = get_arg(args::num_readers);

    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);

    Noc noc;
    DataflowBuffer cb_in(dfb::in0);
    const auto s = TensorAccessor(tensor::src);

    // Thread t of N fills shard entries t, t + N, ...: the strided DFB gives each thread every N-th
    // entry, N entries apart in L1.
    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();
#ifdef IMPLICIT_SYNC
    // Host enables this only for unpadded blocks: each TXN_ID read fills the next DFB entry and posts
    // its credit when it lands.
    for (uint32_t k = thread_id; k < block_num_tiles; k += num_threads) {
        const uint32_t tile_id = start_id + (k / block_width_tiles) * input_width_offset_tiles + k % block_width_tiles;
        noc.async_read<NocOptions::TXN_ID>(s, cb_in, {.page_id = tile_id}, {});
    }
#else
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_readers>();
    // Entries span the padded shard width; the tiles past block_width_tiles stay unwritten.
    const uint32_t shard_width_tiles = block_width_tiles + padded_offset_bytes / tile_bytes;
    const uint32_t num_my_tiles = block_num_tiles > thread_id ? (block_num_tiles - thread_id - 1) / num_threads + 1 : 0;
    const uint32_t entry_stride_bytes = num_threads * tile_bytes;
    uint32_t barrier_count = 0;
    uint32_t l1_offset = 0;
    cb_in.reserve_back(num_my_tiles);
    for (uint32_t k = thread_id; k < block_num_tiles; k += num_threads) {
        const uint32_t h = k / shard_width_tiles;
        const uint32_t w = k % shard_width_tiles;
        if (w < block_width_tiles) {
            const uint32_t tile_id = start_id + h * input_width_offset_tiles + w;
            noc.async_read(s, cb_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = l1_offset});
            if (++barrier_count == barrier_threshold) {
                noc.async_read_barrier();
                barrier_count = 0;
            }
        }
        l1_offset += entry_stride_bytes;
    }
    noc.async_read_barrier();
    cb_in.push_back(num_my_tiles);
#endif
}
