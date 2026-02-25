// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size = get_tile_size(cb_out);

    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto output = TensorAccessor(output_args, output_addr, tile_size);

    constexpr uint32_t num_tiles_offset = output_args.next_compile_time_args_offset();
    constexpr uint32_t num_tiles = get_compile_time_arg_val(num_tiles_offset);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out, 1);

        uint32_t l1_addr = get_read_ptr(cb_out);

        noc_async_write_tile(i, output, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
