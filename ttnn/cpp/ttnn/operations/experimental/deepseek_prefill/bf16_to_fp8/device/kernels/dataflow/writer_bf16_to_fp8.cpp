// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    uint32_t output_addr = get_arg_val<uint32_t>(0);
    const auto addr_gen = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_read = get_read_ptr(cb_out);
        noc_async_write_tile(t, addr_gen, l1_read);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
