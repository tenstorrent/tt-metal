// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const bool output_is_dram = get_arg(args::output_is_dram) == 1;
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto Wt = get_arg(args::Wt);
    const auto tile_offset = get_arg(args::tile_offset);

    // Output tensor address + layout arrive via the tensor::output binding (no buffer-address RTA).
    const auto s = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_output(dfb::out);

    const auto start_tile_idx = tile_offset;
    const auto output_tile_bytes = dfb_output.get_tile_size();

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const auto tile_idx = start_tile_idx + row_idx * Wt + col_idx;
            dfb_output.wait_front(1);
            noc.async_write(dfb_output, s, output_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_idx});
            noc.async_write_barrier();
            dfb_output.pop_front(1);
        }
    }
}  // void kernel_main()
