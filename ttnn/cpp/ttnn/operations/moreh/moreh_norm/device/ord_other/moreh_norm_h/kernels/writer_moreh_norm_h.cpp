// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const bool output_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{16};
    const auto cb_id_output = cb_id++;

    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);

    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(output_args, output_addr, output_tile_bytes);

    const auto output_l1_read_addr = get_read_ptr(cb_id_output);

    auto output_tile_idx = tile_offset;
    for (uint32_t idx = 0; idx < num_cols_per_core; ++idx) {
        cb_wait_front(cb_id_output, 1);
        noc_async_write_tile(output_tile_idx, s, output_l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_output, 1);
        output_tile_idx++;
    }

}  // void kernel_main()
