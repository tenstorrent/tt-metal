// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the block path of the tiled HC transpose. Compute emits the block's 32 output tiles in h order; the
// ones whose h lies past the logical height are padding and are dropped.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_block = get_arg(args::start_block);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t C = get_arg(args::C);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t Ct = (C + 31) / 32;
    constexpr uint32_t BLOCK_TILES = 32;
    constexpr uint32_t GROUP_TILES = 8;

    const auto s = TensorAccessor(tensor::output);
    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const uint32_t tile_bytes = dfb.get_entry_size();

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        uint32_t rem = block;
        const uint32_t wt = rem % Wt;
        rem /= Wt;
        const uint32_t ht = rem % Ht;
        rem /= Ht;
        const uint32_t ct = rem % Ct;
        const uint32_t n = rem / Ct;

        const uint32_t rows_in_block = (ht + 1) * 32 <= H ? 32 : H - ht * 32;
        for (uint32_t r0 = 0; r0 < BLOCK_TILES; r0 += GROUP_TILES) {
            dfb.wait_front(GROUP_TILES);
            uint32_t l1 = dfb.get_read_ptr();
            for (uint32_t r = r0; r < r0 + GROUP_TILES && r < rows_in_block; ++r) {
                const uint32_t h = ht * 32 + r;
                const uint32_t out_tile_id = ((n * H + h) * Ct + ct) * Wt + wt;
                noc.async_write(CoreLocalMem<uint32_t>(l1), s, tile_bytes, {}, {.page_id = out_tile_id});
                l1 += tile_bytes;
            }
            noc.async_writes_flushed();
            dfb.pop_front(GROUP_TILES);
        }
    }
    noc.async_write_barrier();
}
