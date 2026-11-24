// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t output_tiles = get_arg(args::output_tiles);
    const uint32_t first_output_tile = get_arg(args::first_output_tile);

    const auto output = TensorAccessor(tensor::output);
    DataflowBuffer pooled(dfb::pooled);
    Noc noc;
    constexpr uint32_t tile_bytes = 2048;
    const uint32_t tile_end = first_output_tile + output_tiles;
    for (uint32_t tile = first_output_tile; tile < tile_end; ++tile) {
        pooled.wait_front(1);
        noc.async_write(pooled, output, tile_bytes, {}, {.page_id = tile});
        noc.async_write_barrier();
        pooled.pop_front(1);
    }
}
