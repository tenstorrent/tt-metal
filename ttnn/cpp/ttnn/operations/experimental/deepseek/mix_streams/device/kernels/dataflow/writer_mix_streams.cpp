// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Streams this core's slice of the mixed output back, one tile page at a time.
void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);

    constexpr uint32_t one_tile = 1;
    const uint32_t tile_size_bytes = out_cb.get_tile_size();

    for (uint32_t page = start_tile; page < start_tile + num_tiles; ++page) {
        out_cb.wait_front(one_tile);
        noc.async_write(out_cb, out, tile_size_bytes, {.offset_bytes = 0}, {.page_id = page});
        noc.async_write_barrier();
        out_cb.pop_front(one_tile);
    }
}
