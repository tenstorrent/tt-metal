// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const bool output_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{16};
    const auto cb_id_output = cb_id++;

    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(output_args, output_addr);

    const auto start_tile_idx = tile_offset / Wt;

    Noc noc;
    DataflowBuffer dfb_output(cb_id_output);
    const auto output_tile_bytes = get_tile_size(cb_id_output);

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        const auto tile_idx = start_tile_idx + row_idx;
        dfb_output.wait_front(1);
        noc.async_write(dfb_output, s, output_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_idx});
        noc.async_write_barrier();
        dfb_output.pop_front(1);
    }
}  // void kernel_main()
