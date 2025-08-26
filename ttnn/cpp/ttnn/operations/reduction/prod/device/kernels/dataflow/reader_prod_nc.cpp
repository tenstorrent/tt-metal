// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto num_input_tiles = get_arg_val<uint32_t>(1);
    const auto num_output_tiles = get_arg_val<uint32_t>(2);
    const auto input_tile_offset = get_arg_val<uint32_t>(3);
    const auto start_id = get_arg_val<uint32_t>(4);
    const auto input_is_dram = get_compile_time_arg_val(0) == 1;
    const auto HtWt = get_arg_val<uint32_t>(6);
    const auto CHtWt = get_arg_val<uint32_t>(7);
    const auto dim = get_compile_time_arg_val(1);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    uint32_t l1_write_addr_in0;
    uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const auto input_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<input_is_dram> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    uint32_t read_tile_id_temp = (dim == 0) ? (start_id) : (start_id / HtWt * CHtWt) + (start_id % HtWt);
    uint32_t start_tile_id = start_id / HtWt * CHtWt;
    uint32_t end_tile_id = start_tile_id + HtWt - 1;
    uint32_t read_tile_id = read_tile_id_temp;
    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        if constexpr (dim == 0) {
            read_tile_id = i;
        }
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(read_tile_id, dram_input_addrg, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += input_tile_offset;
        }
        if constexpr (dim != 0) {
            if (read_tile_id_temp == end_tile_id) {
                start_tile_id = start_tile_id + CHtWt;
                read_tile_id_temp = start_tile_id;
                end_tile_id = read_tile_id_temp + HtWt - 1;
            } else {
                read_tile_id_temp = read_tile_id_temp + 1;
            }
            read_tile_id = read_tile_id_temp;
        }
    }
}
