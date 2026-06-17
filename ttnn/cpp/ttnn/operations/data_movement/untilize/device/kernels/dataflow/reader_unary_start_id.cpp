// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_page_id = get_arg(args::start_page_id);

    const auto s = TensorAccessor(ta::input);

    Noc noc;
    DataflowBuffer cb_in(dfb::in);
    const uint32_t tile_bytes = cb_in.get_tile_size();

    uint32_t end_page_id = start_page_id + num_tiles;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_in.reserve_back(1);
        noc.async_read(s, cb_in, tile_bytes, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in.push_back(1);
    }
}
