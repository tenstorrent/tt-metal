// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t out_num_blocks_w_per_core = get_arg(args::out_num_blocks_w_per_core);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t out_num_blocks_h = get_arg(args::out_num_blocks_h);
    uint32_t out_total_blocks_w = get_arg(args::out_total_blocks_w);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(tensor::dst);

    DataflowBuffer dfb_out(dfb::out);
    const uint32_t tile_bytes = dfb_out.get_tile_size();

    for (uint32_t block_h_id = 0; block_h_id < out_num_blocks_h; block_h_id++) {
        uint32_t end_id = start_id + out_num_blocks_w_per_core;
        for (uint32_t i = start_id; i < end_id; ++i) {
            dfb_out.wait_front(onetile);
            noc.async_write(
                dfb_out,
                s,
                tile_bytes,
                {.offset_bytes = 0},
                {.page_id = (block_h_id * out_total_blocks_w) + i, .offset_bytes = 0});
            noc.async_write_barrier();
            dfb_out.pop_front(onetile);
        }
    }
}
