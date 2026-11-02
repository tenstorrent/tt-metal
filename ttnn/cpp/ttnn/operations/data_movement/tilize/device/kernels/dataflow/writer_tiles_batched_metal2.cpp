// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writes `num_blocks` blocks of `tiles_per_batch` consecutive tile pages with one write barrier per block.
// Each block is popped as a unit, so the producer's block granularity bounds how far the writer can run ahead.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_blocks = get_arg(args::num_blocks);
    const auto start_id = get_arg(args::start_id);
    constexpr auto tiles_per_batch = get_arg(args::tiles_per_batch);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t tile_bytes = dfb_out.get_entry_size();
    const auto s = TensorAccessor(tensor::dst);

    uint32_t page_id = start_id;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        dfb_out.wait_front(tiles_per_batch);
        for (uint32_t t = 0; t < tiles_per_batch; ++t) {
            noc.async_write(dfb_out, s, tile_bytes, {.offset_bytes = t * tile_bytes}, {.page_id = page_id + t});
        }
        noc.async_write_barrier();
        dfb_out.pop_front(tiles_per_batch);
        page_id += tiles_per_batch;
    }
}
