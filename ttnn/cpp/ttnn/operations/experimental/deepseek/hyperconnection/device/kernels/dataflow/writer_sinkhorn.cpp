// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_o(cb_out);

    constexpr uint32_t one_tile = 1;
    const uint32_t tile_size_bytes = cb_o.get_tile_size();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_o.wait_front(one_tile);
        noc.async_write(cb_o, out, tile_size_bytes, {.offset_bytes = 0}, {.page_id = start_tile + i});
        noc.async_write_barrier();
        cb_o.pop_front(one_tile);
    }
}
