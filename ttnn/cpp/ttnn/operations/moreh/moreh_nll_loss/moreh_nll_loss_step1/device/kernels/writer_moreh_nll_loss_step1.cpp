// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_units_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    const uint32_t output_tile_bytes = get_tile_size(cb_output);

    constexpr auto output_args = TensorAccessorArgs<0>();

    const auto output_addrg = TensorAccessor(output_args, output_addr, output_tile_bytes);

    constexpr uint32_t onetile = 1;

    uint32_t end_id = start_id + num_units_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_output, onetile);
        uint32_t output_l1_write_addr = get_read_ptr(cb_output);
        noc_async_write_tile(i, output_addrg, output_l1_write_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_output, onetile);
    }
}
