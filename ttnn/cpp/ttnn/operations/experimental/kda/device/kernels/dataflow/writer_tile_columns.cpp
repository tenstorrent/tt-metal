// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Write tiles [first_tile, first_tile + num_tiles) of a window of `columns` tile columns from tile column first_column
// of a TILE tensor with row_tiles tiles per row, numbered row-major within the window, one tile per entry.
template <uint32_t columns, uint32_t row_tiles, uint32_t first_column>
TT_KERNEL void writer(uint32_t first_tile, uint32_t num_tiles) {
    const auto output = TensorAccessor(tensor::output);
    DataflowBuffer out(dfb::out);
    Noc noc;
    for (uint32_t tile = first_tile; tile < first_tile + num_tiles; ++tile) {
        out.wait_front(1);
        noc.async_write(
            out,
            output,
            out.get_entry_size(),
            {.offset_bytes = 0},
            {.page_id = tile / columns * row_tiles + first_column + tile % columns});
        // The entry is free once its data left L1.
        noc.async_writes_flushed();
        out.pop_front(1);
    }
    noc.async_write_barrier();
}
