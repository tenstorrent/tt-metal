// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

TT_KERNEL void writer(uint32_t ct_start, uint32_t ct_count) {
    const auto output = TensorAccessor(tensor::output);
    DataflowBuffer packed_tile(dfb::packed_tile);
    Noc noc;
    const uint32_t tile_bytes = packed_tile.get_entry_size();
    for (uint32_t item = 0; item < ct_count; ++item) {
        packed_tile.wait_front(1);
        noc.async_write(packed_tile, output, tile_bytes, {}, {.page_id = ct_start + item});
        noc.async_write_barrier();
        packed_tile.pop_front(1);
    }
}
