// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_sharded_blocks_interleaved_start_id.cpp. Gathers a block of tiles out
// of an interleaved input tensor and lays it down as one core's shard in a DFB. Only the plumbing
// changes: the buffer-index compile-time arg becomes dfb::in, the accessor-args / base-address pair
// becomes the tensor::src binding, and the positional runtime args become named ones. The gather loop
// and its barrier accounting are untouched.
// Forked rather than converted in place because the legacy file is still bound by factories on the
// legacy positional-arg API.
//
// The binding names below (dfb::in, tensor::src) and the named argument set are this fork's interface:
// every later consumer inherits them, so they are taken from the kernel's own vocabulary rather than
// any one op's locals, and are not renamed once a consumer exists.

#include <stdint.h>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tensix_types.h"
#include "experimental/kernel_args.h"

// #include "api/debug/dprint.h"

// Target 8KB of data before a single barrier for 8x8 grid of readers
template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void kernel_main() {
    const uint32_t block_height_tiles = get_arg(args::block_height_tiles);
    const uint32_t block_width_tiles = get_arg(args::block_width_tiles);
    // input width in tiles - block width in tiles
    const uint32_t padded_offset_bytes = get_arg(args::padded_offset_bytes);
    // input width in tiles - block width in tiles
    const uint32_t input_width_offset_tiles = get_arg(args::input_width_offset_tiles);
    // block_height_tiles * block_width_tiles
    const uint32_t block_num_tiles = get_arg(args::block_num_tiles);
    const uint32_t start_id_offset = get_arg(args::start_id_offset);
    const uint32_t start_id_base = get_arg(args::start_id_base);
    const uint32_t start_id = start_id_base + start_id_offset;

    constexpr auto num_readers = get_arg(args::num_readers);
    // tile_bytes comes from a CTA, not the device-side get_tile_size(dfb::in): the latter is not
    // arch-portable to Quasar (it reads a DFB-descriptor slot that may be stale on a Quasar DM kernel,
    // yielding a wrong read size / L1 stride and a stray NOC write). The factory passes the correct
    // input/output tile size. Mirrors the Gen2-native experimental/quasar i2s reader.
    constexpr uint32_t tile_bytes = get_arg(args::tile_bytes);

    Noc noc;
    // dfb::in — this core's shard
    DataflowBuffer dfb_in(dfb::in);
    const auto s = TensorAccessor(tensor::src);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<tile_bytes, num_readers>();
    uint32_t barrier_count = 0;
    uint32_t curr_tile_id = start_id;
    uint32_t l1_offset = 0;
    dfb_in.reserve_back(block_num_tiles);
    for (uint32_t h = 0; h < block_height_tiles; h++) {
        uint32_t tile_id = curr_tile_id;
        for (uint32_t w = 0; w < block_width_tiles; w++) {
            noc.async_read(s, dfb_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = l1_offset});
            tile_id++;
            l1_offset += tile_bytes;
            if (++barrier_count == barrier_threshold) {
                noc.async_read_barrier();
                barrier_count = 0;
            }
        }
        l1_offset += padded_offset_bytes;
        curr_tile_id += input_width_offset_tiles;
    }
    noc.async_read_barrier();
    dfb_in.push_back(block_num_tiles);
}
