// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Shared by both programs: drains dfb::out to the one tensor the program owns. For the in-place
// program that tensor is the same one the reader read from; the DFB credits order the write of
// tile i after the read of tile i.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);

    DataflowBuffer dfb_out(dfb::out);
    Noc noc;
    const auto acc_out = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_out.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_tiles; ++i) {
        dfb_out.wait_front(1);
        noc.async_write(dfb_out, acc_out, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(1);
    }
}
