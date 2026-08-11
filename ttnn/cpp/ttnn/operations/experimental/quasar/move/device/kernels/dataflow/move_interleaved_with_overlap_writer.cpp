// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 cross-kernel split of move_interleaved_with_overlap.cpp (tiled / interleaved overlap
// move). The reader stages src -> the scratch DFB and, only after the all-cores read handshake,
// push_back()s it; this writer consumes the scratch DFB (wait_front) and drains it to dst. Splitting
// the producer (reader) and consumer (writer) across two kernels avoids a DM-kernel self-loop on
// dfb::scratch, which Metal 2.0 forbids. Because the reader's push_back happens only after the
// handshake, this wait_front cannot unblock until every core has finished reading src, preserving
// the overlap-safety invariant (no core writes dst until all cores have read src).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_tiles = get_arg(args::num_pages);

    Noc noc;
    DataflowBuffer cb(dfb::scratch);

    const auto dst_addrgen = TensorAccessor(tensor::output);

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t tile_bytes = cb.get_entry_size();

    // Drain the staged ublock (CB -> dst). wait_front unblocks only after the reader's
    // post-handshake push_back, i.e. after all cores have finished reading src.
    cb.wait_front(num_tiles);
    uint32_t cb_read_offset = 0;
    for (uint32_t i = start_id; i < start_id + num_tiles; i += ublock_size_tiles) {
        noc.async_write(
            cb, dst_addrgen, tile_bytes, {.offset_bytes = cb_read_offset}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        cb_read_offset += tile_bytes;
    }
    cb.pop_front(num_tiles);
}
