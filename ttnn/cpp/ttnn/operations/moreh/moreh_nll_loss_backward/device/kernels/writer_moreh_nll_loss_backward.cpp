// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    auto input_grad_addr = get_arg_val<uint32_t>(i++);
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    const uint32_t input_grad_tile_bytes = get_tile_size(cb_input_grad);

    constexpr auto input_grad_args = TensorAccessorArgs<0>();

    const auto input_grad_addrg = TensorAccessor(input_grad_args, input_grad_addr, input_grad_tile_bytes);

    constexpr uint32_t onetile = 1;

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_input_grad, onetile);
        uint32_t input_grad_l1_write_addr = get_read_ptr(cb_input_grad);
        noc_async_write_tile(i, input_grad_addrg, input_grad_l1_write_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_input_grad, onetile);
    }
}
