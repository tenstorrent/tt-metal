// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer.  Drains cb_output (streamed, one tile at a time by compute's
// pass 2) and writes each tile-row's Wt tiles back to the output tensor in DRAM.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t row = 0; row < num_rows; ++row) {
        const uint32_t global_row = start_row + row;
        const uint32_t page_base = global_row * Wt;

        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb_wait_front(cb_output, 1);
            const uint32_t l1 = get_read_ptr(cb_output);
            noc_async_write_tile(page_base + wt, output_accessor, l1);
            noc_async_write_barrier();
            cb_pop_front(cb_output, 1);
        }
    }
}
