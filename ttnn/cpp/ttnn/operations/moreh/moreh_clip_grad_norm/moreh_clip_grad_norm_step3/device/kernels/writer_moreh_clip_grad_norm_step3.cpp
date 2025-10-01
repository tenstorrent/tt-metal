// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_output = 16;

    constexpr uint32_t onetile = 1;

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);

    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(output_args, output_addr, output_tile_bytes);

    // output
    const auto output_l1_read_addr = get_read_ptr(cb_id_output);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        cb_wait_front(cb_id_output, onetile);
        noc_async_write_tile(tile_idx, s, output_l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_output, onetile);
    }

}  // void kernel_main()
