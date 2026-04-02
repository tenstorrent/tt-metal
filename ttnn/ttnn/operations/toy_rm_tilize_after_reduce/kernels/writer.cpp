// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_id = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr auto output_ta_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_tilized_out = 24;
    const uint32_t tile_bytes = get_tile_size(cb_tilized_out);
    const auto output_accessor = TensorAccessor(output_ta_args, output_addr, tile_bytes);

    for (uint32_t tile_id = start_id; tile_id < start_id + num_tiles; ++tile_id) {
        cb_wait_front(cb_tilized_out, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_tilized_out);
        noc_async_write_tile(tile_id, output_accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_tilized_out, 1);
    }
}
