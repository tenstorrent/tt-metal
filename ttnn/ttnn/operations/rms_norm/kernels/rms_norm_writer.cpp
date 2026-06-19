// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (both regimes).  Drains cb_output (streamed one tile at a time
// by compute's pass 2) and writes `num_tiles` contiguous output pages starting at
// `page_base`.  Regime A: a core's owned tile-rows are contiguous pages
// (start_row*Wt .. ).  Regime B: a core's W-shard is `Wt_s` contiguous pages at
// its shard base.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t page_base = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();

    const uint32_t tile_bytes = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_output, 1);
        const uint32_t l1 = get_read_ptr(cb_output);
        noc_async_write_tile(page_base + i, output_accessor, l1);
        noc_async_write_barrier();
        cb_pop_front(cb_output, 1);
    }
}
