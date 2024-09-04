// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    // compile time args
    constexpr bool input_grad_is_dram = (get_compile_time_arg_val(0) == 1);

    int i{0};
    const auto input_grad_addr = get_arg_val<uint32_t>(i++);
    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{16};
    const auto cb_id_input_grad = cb_id++;

    // input_grad
    const uint32_t input_grad_tile_bytes = get_tile_size(cb_id_input_grad);
    const auto input_grad_data_format = get_dataformat(cb_id_input_grad);

    const InterleavedAddrGenFast<input_grad_is_dram> input_grad_addrg = {
        .bank_base_address = input_grad_addr,
        .page_size = input_grad_tile_bytes,
        .data_format = input_grad_data_format};

    auto input_grad_tile_idx = tile_offset;
    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        cb_wait_front(cb_id_input_grad, 1);
        const auto input_grad_l1_read_addr = get_read_ptr(cb_id_input_grad);
        noc_async_write_tile(input_grad_tile_idx, input_grad_addrg, input_grad_l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_input_grad, 1);
        input_grad_tile_idx++;
    }

}  // void kernel_main()
