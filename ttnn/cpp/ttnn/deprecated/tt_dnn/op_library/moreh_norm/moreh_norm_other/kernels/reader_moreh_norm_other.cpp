// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto decimal = get_arg_val<uint32_t>(i++);
    const auto recip_p_decimal = get_arg_val<uint32_t>(i++);
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto outer_stride = get_arg_val<uint32_t>(i++);
    const auto num_inner_tiles = get_arg_val<uint32_t>(i++);
    const auto num_reduced_tiles_along_dim = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_decimal = cb_id++;
    const auto cb_id_recip_p_decimal = cb_id++;

    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    Scalar one;
    one.f = 1.0f;
    fill_cb_with_value(cb_id_one, one.u);
    fill_cb_with_value(cb_id_decimal, decimal);
    fill_cb_with_value(cb_id_recip_p_decimal, recip_p_decimal);

    const auto input_l1_write_ptr = get_write_ptr(cb_id_input);

    auto start_output_tile_idx = tile_offset;
    const auto inner_stride = num_inner_tiles;
    for (uint32_t idx = 0; idx < num_output_tiles_per_core; ++idx) {
        const auto outer_idx = start_output_tile_idx / inner_stride;
        const auto inner_idx = start_output_tile_idx % inner_stride;

        auto input_tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < num_reduced_tiles_along_dim; ++d) {
            cb_reserve_back(cb_id_input, 1);
            if (input_is_dram) {
                noc_async_read_tile(input_tile_idx, dram_input_addrg, input_l1_write_ptr);
            } else {
                noc_async_read_tile(input_tile_idx, l1_input_addrg, input_l1_write_ptr);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_input, 1);
            input_tile_idx += inner_stride;
        }

        start_output_tile_idx++;
    }

}  // void kernel_main()
