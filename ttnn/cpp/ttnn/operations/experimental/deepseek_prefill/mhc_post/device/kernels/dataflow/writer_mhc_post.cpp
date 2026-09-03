// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Writes the n output-stream tiles of each work unit back into the packed [T, n*C] layout.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t start_unit = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t col_tiles = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t page = get_local_cb_interface(cb_out).fifo_page_size;
    const auto s_out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cbo(cb_out);

    for (uint32_t w = 0; w < num_units; ++w) {
        const uint32_t unit = start_unit + w;
        const uint32_t t0 = unit / col_tiles;
        const uint32_t c0 = unit - t0 * col_tiles;
        const uint32_t out_base = t0 * n * col_tiles + c0;

        cbo.wait_front(n);
        for (uint32_t j = 0; j < n; ++j) {
            noc.async_write(cbo, s_out, page, {.offset_bytes = j * page}, {.page_id = out_base + j * col_tiles});
        }
        noc.async_write_barrier();
        cbo.pop_front(n);
    }
}
