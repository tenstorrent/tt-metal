// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for toy_variance (interleaved output): writes the variance output tiles to DRAM.
// Output is one tile per row reduced -- Ht*NC tiles total.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out_tiles);
    const auto acc_out = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_out.get_tile_size();

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_out.wait_front(1);
        noc.async_write(dfb_out, acc_out, tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(1);
    }
}
