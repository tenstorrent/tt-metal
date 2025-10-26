// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    int i{0};
    const auto output_addr = get_arg_val<uint32_t>(i++);
    const bool output_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{16};
    const auto cb_id_output = cb_id++;

    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);

    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(output_args, output_addr, output_tile_bytes);

    const auto start_tile_idx = tile_offset;
    const auto output_l1_read_addr = get_read_ptr(cb_id_output);

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const auto tile_idx = start_tile_idx + row_idx * Wt + col_idx;
            cb_wait_front(cb_id_output, 1);
            noc_async_write_tile(tile_idx, s, output_l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_output, 1);
        }
    }
}  // void kernel_main()
