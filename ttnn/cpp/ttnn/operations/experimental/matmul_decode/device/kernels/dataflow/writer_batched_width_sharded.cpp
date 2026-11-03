// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Scatter [Bc, M, Nc] output block to interleaved tile pages.
void kernel_main() {
    constexpr uint32_t Bc = get_arg(args::Bc);
    constexpr uint32_t M_tiles = get_arg(args::M_tiles);
    constexpr uint32_t Nc_tiles = get_arg(args::Nc_tiles);
    constexpr uint32_t N_tiles = get_arg(args::N_tiles);

    const uint32_t b_idx = get_arg(args::b_idx);
    const uint32_t n_idx = get_arg(args::n_idx);

    constexpr uint32_t out_num_tiles = Bc * M_tiles * Nc_tiles;

    Noc noc;
    CircularBuffer out_cb(dfb::out);
    const auto out = TensorAccessor(tensor::output);

    const uint32_t tile_size = out_cb.get_tile_size();
    out_cb.wait_front(out_num_tiles);

    uint32_t local = 0;
    for (uint32_t bc_i = 0; bc_i < Bc; ++bc_i) {
        const uint32_t b = b_idx * Bc + bc_i;
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            const uint32_t out_row = b * M_tiles + mt;
            for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
                const uint32_t out_col = n_idx * Nc_tiles + nc;
                const uint32_t page_id = out_row * N_tiles + out_col;
                noc.async_write(out_cb, out, tile_size, {.offset_bytes = local * tile_size}, {.page_id = page_id});
                ++local;
            }
        }
    }
    noc.async_write_barrier();
    out_cb.pop_front(out_num_tiles);
}
