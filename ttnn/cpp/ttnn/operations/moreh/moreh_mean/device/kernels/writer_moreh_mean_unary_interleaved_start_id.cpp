// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const auto out_tile_bytes = dfb_out.get_tile_size();

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out.wait_front(onetile);
        noc.async_write(dfb_out, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
}
