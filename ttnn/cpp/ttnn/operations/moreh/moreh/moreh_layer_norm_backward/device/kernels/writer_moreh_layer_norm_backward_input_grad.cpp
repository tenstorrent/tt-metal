// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    const auto input_grad_addr = get_arg_val<uint32_t>(0);
    const auto num_rows_per_core = get_arg_val<uint32_t>(1);
    const auto Wt = get_arg_val<uint32_t>(2);
    const auto tile_offset = get_arg_val<uint32_t>(3);

    constexpr bool input_grad_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_input_grad = 16;

    const uint32_t input_grad_tile_bytes = get_tile_size(cb_id_input_grad);
    const auto data_format = get_dataformat(cb_id_input_grad);

    const InterleavedAddrGenFast<input_grad_is_dram> input_grad_addrg = {
        .bank_base_address = input_grad_addr, .page_size = input_grad_tile_bytes, .data_format = data_format};

    uint32_t offs = 0;
    const auto NCHt = num_rows_per_core;
    constexpr uint32_t onetile = 1;

    const auto input_grad_l1_read_addr = get_read_ptr(cb_id_input_grad);

    for (uint32_t ncht = 0; ncht < num_rows_per_core; ncht++) {
        // input_grad (N, C, H, W)
        for (uint32_t wt = 0; wt < Wt; wt++) {
            cb_wait_front(cb_id_input_grad, onetile);
            noc_async_write_tile(offs + wt + tile_offset, input_grad_addrg, input_grad_l1_read_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_id_input_grad, onetile);
        }  // wt loop
        offs += Wt;

    }  // num_rows_per_core loop

}  // void kernel_main()
