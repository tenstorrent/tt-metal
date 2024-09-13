// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    uint32_t i = 0;
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto num_input_tiles = get_arg_val<uint32_t>(i++);
    const auto num_output_tiles = get_arg_val<uint32_t>(i++);
    const auto input_tile_stride = get_arg_val<uint32_t>(i++);
    const auto start_id = get_arg_val<uint32_t>(i++);
    const auto input_is_dram = (get_arg_val<uint32_t>(i++) == 1);
    const auto HtWt = get_arg_val<uint32_t>(i++);
    const auto inner_size = get_arg_val<uint32_t>(i++);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_id_in2 = tt::CB::c_in2;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    scaler.f = 1.0f / num_input_tiles;
    fill_cb_with_value(cb_id_in2, scaler.u, 1);

    uint32_t l1_write_addr_in0;
    uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const auto input_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        uint32_t hw_tile_id = i % HtWt;
        uint32_t inner_id = (i / HtWt) % inner_size * HtWt;
        uint32_t outer_id = (i / HtWt / inner_size) * inner_size * HtWt * num_input_tiles;

        auto read_tile_id = outer_id + inner_id + hw_tile_id;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            if (input_is_dram) {
                noc_async_read_tile(read_tile_id, dram_input_addrg, l1_write_addr_in0);
            } else {
                noc_async_read_tile(read_tile_id, l1_input_addrg, l1_write_addr_in0);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += input_tile_stride;
        }
    }
}
