// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the block path of the tiled HC transpose. A block is the 32 input tiles that share (n, h tile, w tile)
// and a 32-channel group: tile (n, c, ht, wt) for c = ct * 32 .. ct * 32 + 31. Those 32 tiles hold exactly the rows of
// the 32 output tiles (n, h, ct, wt) for the h values of the h tile, which compute rebuilds by untilize + tilize.
// When C is not a multiple of 32, the slots of channels past C are filled with the pad value instead.

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
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t pad_packed = get_arg(args::pad_packed);
    constexpr uint32_t Ct = (C + 31) / 32;
    constexpr uint32_t BLOCK_TILES = 32;

    const auto s = TensorAccessor(tensor::input);
    Noc noc;
    DataflowBuffer dfb(dfb::src);
    const uint32_t tile_bytes = dfb.get_entry_size();

    for (uint32_t block = start_block; block < start_block + num_blocks; ++block) {
        uint32_t rem = block;
        const uint32_t wt = rem % Wt;
        rem /= Wt;
        const uint32_t ht = rem % Ht;
        rem /= Ht;
        const uint32_t ct = rem % Ct;
        const uint32_t n = rem / Ct;

        dfb.reserve_back(BLOCK_TILES);
        uint32_t l1_addr = dfb.get_write_ptr();
        for (uint32_t i = 0; i < BLOCK_TILES; ++i) {
            const uint32_t c = ct * BLOCK_TILES + i;
            if (c < C) {
                const uint32_t tile_id = ((n * C + c) * Ht + ht) * Wt + wt;
                noc.async_read(s, CoreLocalMem<uint32_t>(l1_addr), tile_bytes, {.page_id = tile_id}, {});
            } else {
                auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
                for (uint32_t w = 0; w < tile_bytes / sizeof(uint32_t); ++w) {
                    words[w] = pad_packed;
                }
            }
            l1_addr += tile_bytes;
        }
        noc.async_read_barrier();
        dfb.push_back(BLOCK_TILES);
    }
}
