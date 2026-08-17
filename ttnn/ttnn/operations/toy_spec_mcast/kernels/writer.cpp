// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t out_page = get_arg(args::out_page);

    Noc noc;
    DataflowBuffer dfb_tile(dfb::tile);
    const auto acc_out = TensorAccessor(tensor::out);
    const uint32_t tile_bytes = dfb_tile.get_tile_size();

    dfb_tile.wait_front(1);
    noc.async_write(dfb_tile, acc_out, tile_bytes, {.offset_bytes = 0}, {.page_id = out_page});
    noc.async_write_barrier();
    dfb_tile.pop_front(1);
}
