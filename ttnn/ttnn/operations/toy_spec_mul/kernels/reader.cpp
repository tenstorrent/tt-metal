// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_a(dfb::in_a);
    DataflowBuffer dfb_b(dfb::in_b);
    Noc noc;
    const auto acc_a = TensorAccessor(tensor::a);
    const auto acc_b = TensorAccessor(tensor::b);
    const uint32_t tile_bytes = dfb_a.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        dfb_a.reserve_back(1);
        dfb_b.reserve_back(1);
        noc.async_read(acc_a, dfb_a, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(acc_b, dfb_b, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_a.push_back(1);
        dfb_b.push_back(1);
    }
}
