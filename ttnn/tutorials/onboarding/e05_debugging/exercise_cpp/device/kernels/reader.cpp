// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t input_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t tile_size = get_tile_size(cb_in);

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input = TensorAccessor(input_args, input_addr, tile_size);

    constexpr uint32_t num_tiles_offset = input_args.next_compile_time_args_offset();
    constexpr uint32_t num_tiles = get_compile_time_arg_val(num_tiles_offset);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_addr = get_write_ptr(cb_in);

        noc_async_read_tile(i, input, l1_addr);
        noc_async_read_barrier();

        cb_push_back(cb_in, 1);
    }
}
