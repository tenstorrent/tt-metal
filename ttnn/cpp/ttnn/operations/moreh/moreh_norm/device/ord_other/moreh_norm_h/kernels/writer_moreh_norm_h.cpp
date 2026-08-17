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
    const auto num_cols_per_core = get_arg(args::num_cols_per_core);
    const auto tile_offset = get_arg(args::tile_offset);

    const auto s = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_output(dfb::output);
    const auto output_tile_bytes = dfb_output.get_tile_size();

    auto output_tile_idx = tile_offset;
    for (uint32_t idx = 0; idx < num_cols_per_core; ++idx) {
        dfb_output.wait_front(1);
        noc.async_write(dfb_output, s, output_tile_bytes, {.offset_bytes = 0}, {.page_id = output_tile_idx});
        noc.async_write_barrier();
        dfb_output.pop_front(1);
        output_tile_idx++;
    }

}  // void kernel_main()
