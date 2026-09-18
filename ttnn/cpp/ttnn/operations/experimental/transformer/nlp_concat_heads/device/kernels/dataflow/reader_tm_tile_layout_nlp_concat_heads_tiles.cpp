// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tile-split reader for the interleaved nlp_concat_heads: the core owns output tiles [start_tile, + num_tiles) of
// the flattened [B, 1, S, heads * head_dim] output (the writer streams them out contiguously) and gathers each from
// its [B, heads, S, head_dim] source position, CHUNK reads behind one barrier. Small-M shapes (SDXL: 32 tile rows)
// thus spread over the full grid instead of one tile row per core.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;
    uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_tile = get_arg(args::start_tile);
    constexpr uint32_t in0_h_tiles = get_arg(args::in0_h_tiles);
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);  // head_dim tiles
    constexpr uint32_t in0_c = get_arg(args::in0_c);              // heads
    constexpr uint32_t in0_HtWt = get_arg(args::in0_HtWt);
    constexpr uint32_t chunk = get_arg(args::chunk);
    constexpr uint32_t row_tiles = in0_c * in0_w_tiles;  // output tiles per tile row
    constexpr uint32_t in0_CHtWt = in0_c * in0_HtWt;

    DataflowBuffer dfb_in0(dfb::in0);
    const uint32_t tile_bytes = dfb_in0.get_entry_size();
    const auto s0 = TensorAccessor(tensor::src);

    uint32_t b = start_tile / row_tiles;      // output tile row
    uint32_t r = start_tile - b * row_tiles;  // (head, w) inside the row
    uint32_t batch = b / in0_h_tiles;
    uint32_t h = b - batch * in0_h_tiles;
    while (num_tiles > 0) {
        const uint32_t n = num_tiles < chunk ? num_tiles : chunk;
        dfb_in0.reserve_back(n);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t c = r / in0_w_tiles;
            const uint32_t w = r - c * in0_w_tiles;
            noc.async_read(
                s0,
                CoreLocalMem<uint32_t>(l1_write_addr),
                tile_bytes,
                {.page_id = batch * in0_CHtWt + c * in0_HtWt + h * in0_w_tiles + w},
                {});
            l1_write_addr += tile_bytes;
            if (++r == row_tiles) {
                r = 0;
                if (++h == in0_h_tiles) {
                    h = 0;
                    batch++;
                }
            }
        }
        noc.async_read_barrier();
        dfb_in0.push_back(n);
        num_tiles -= n;
    }
}
