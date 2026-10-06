// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr auto output_args = TensorAccessorArgs<2>();

    Noc noc;
    CircularBuffer out_cb(cb_out);
    const auto output_accessor = TensorAccessor(output_args, output_addr, out_cb.get_tile_size());

    // Compute emits output tiles row by row, which is the output's page order.
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        out_cb.wait_front(1);
        noc.async_write(out_cb, output_accessor, out_cb.get_tile_size(), {.offset_bytes = 0}, {.page_id = tile_id});
        noc.async_write_barrier();
        out_cb.pop_front(1);
    }
}
