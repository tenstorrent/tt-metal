// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();

    uint32_t input_addr = get_arg_val<uint32_t>(0);
    const auto addr_gen = TensorAccessor(input_args, input_addr, tile_bytes);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write = get_write_ptr(cb_in);
        noc_async_read_tile(t, addr_gen, l1_write);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
