// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_grad_addr = get_arg_val<uint32_t>(i++);
    const bool input_grad_is_dram = get_arg_val<uint32_t>(i++) == 1;

    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);

    constexpr uint32_t onetile = 1;

    uint32_t cb_id{16};
    const auto cb_id_input_grad = cb_id++;

    // input_grad
    const uint32_t input_grad_tile_bytes = get_tile_size(cb_id_input_grad);
    const auto input_grad_data_format = get_dataformat(cb_id_input_grad);

    const InterleavedAddrGenFast<true> dram_input_grad_addrg = {
        .bank_base_address = input_grad_addr,
        .page_size = input_grad_tile_bytes,
        .data_format = input_grad_data_format};

    const InterleavedAddrGenFast<false> l1_input_grad_addrg = {
        .bank_base_address = input_grad_addr,
        .page_size = input_grad_tile_bytes,
        .data_format = input_grad_data_format};

    const auto input_grad_l1_read_ptr = get_read_ptr(cb_id_input_grad);
    uint32_t input_grad_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            // input_grad (N, C, H, W)
            input_grad_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            cb_wait_front(cb_id_input_grad, onetile);
            if (input_grad_is_dram) {
                noc_async_write_tile(input_grad_tile_idx, dram_input_grad_addrg, input_grad_l1_read_ptr);
            } else {
                noc_async_write_tile(input_grad_tile_idx, l1_input_grad_addrg, input_grad_l1_read_ptr);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_input_grad, onetile);
        }  // inner_idx loop
    }      // outer_idx loop

}  // void kernel_main()
