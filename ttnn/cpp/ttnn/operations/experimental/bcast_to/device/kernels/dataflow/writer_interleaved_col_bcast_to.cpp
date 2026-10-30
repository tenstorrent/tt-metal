// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
    uint32_t start_tile_id = get_arg(args::start_tile_id);

    constexpr uint32_t onetile = 1;

    const auto dst = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_dst(dfb::dst);
    const uint32_t dst_tile_bytes = dfb_dst.get_tile_size();

    uint32_t HtWt = Ht * Wt;
    uint32_t num_tiles_written = 0;
    for (uint32_t n = start_n; n < N && num_tiles_written < num_tiles; ++n, start_c = 0) {
        for (uint32_t c = start_c; c < C && num_tiles_written < num_tiles; ++c, start_th = 0) {
            for (uint32_t th = start_th; th < Ht && num_tiles_written < num_tiles; ++th, start_tw = 0) {
                dfb_dst.wait_front(onetile);
                for (uint32_t tw = start_tw; tw < Wt && num_tiles_written < num_tiles; ++tw, ++num_tiles_written) {
                    // write a tile to dst, since the dst shape is full, the tile offset simply grows linearly
                    noc.async_write(dfb_dst, dst, dst_tile_bytes, {}, {.page_id = start_tile_id + num_tiles_written});
                }
                noc.async_write_barrier();
                dfb_dst.pop_front(onetile);
            }
        }
    }
}
