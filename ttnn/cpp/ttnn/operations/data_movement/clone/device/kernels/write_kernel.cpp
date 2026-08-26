// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_tiles = get_arg(args::num_tiles);
    auto start_id = get_arg(args::start_id);

    DataflowBuffer dst_dfb(dfb::dst);
    Noc noc;
    const auto s = TensorAccessor(tensor::output);
    const uint32_t tile_bytes = dst_dfb.get_tile_size();

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dst_dfb.wait_front(1);
        noc.async_write(dst_dfb, s, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dst_dfb.pop_front(1);
    }
}
