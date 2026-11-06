// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_n = get_arg(args::start_n);
    uint32_t start_c = get_arg(args::start_c);
    uint32_t start_t = get_arg(args::start_t);
    uint32_t start_th = get_arg(args::start_th);
    uint32_t start_tw = get_arg(args::start_tw);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t n_stride = get_arg(args::n_stride);
    uint32_t c_stride = get_arg(args::c_stride);
    uint32_t N = get_arg(args::N);
    uint32_t C = get_arg(args::C);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);

    constexpr uint32_t onetile = 1;

    const auto src = TensorAccessor(tensor::input);
    Noc noc;
    DataflowBuffer dfb_src(dfb::src);
    const uint32_t src_tile_bytes = dfb_src.get_tile_size();

    uint32_t HtWt = Ht * Wt;

    // this is the INPUT tile offset
    uint32_t tile_offset = start_n * n_stride + start_c * c_stride;
    uint32_t next_batch_shift = n_stride - c_stride * C;
    uint32_t next_channel_shift = c_stride - HtWt;

    uint32_t num_tiles_read = 0;
    for (uint32_t n = start_n; n < N && num_tiles_read < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_read < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_read < num_tiles; ++th, start_tw = 0) {
                dfb_src.reserve_back(onetile);
                noc.async_read(src, dfb_src, src_tile_bytes, {.page_id = tile_offset + th}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_src.push_back(onetile);
                num_tiles_read += Wt - start_tw;
            }
            tile_offset += c_stride;
            // same as following logically
            // tile_offset += HtWt;
            // tile_offset += next_channel_shift;
        }
        tile_offset += next_batch_shift;
    }
}
