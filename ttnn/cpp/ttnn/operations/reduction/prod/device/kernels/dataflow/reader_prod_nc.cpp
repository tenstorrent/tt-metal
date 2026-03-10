// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto num_input_tiles = get_arg_val<uint32_t>(1);
    const auto num_output_tiles = get_arg_val<uint32_t>(2);
    const auto input_tile_offset = get_arg_val<uint32_t>(3);
    const auto start_id = get_arg_val<uint32_t>(4);
    const auto HtWt = get_arg_val<uint32_t>(5);
    const auto CHtWt = get_arg_val<uint32_t>(6);
    const auto dim = get_compile_time_arg_val(0);
    constexpr auto dram_input_addrg_args = TensorAccessorArgs<1>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(tt::CBIndex::c_0);

    uint32_t input_tile_bytes = get_tile_size(cb_in0.get_cb_id());
    const auto dram_input_addrg = TensorAccessor(dram_input_addrg_args, input_addr, input_tile_bytes);

    uint32_t read_tile_id_temp = (dim == 0) ? (start_id) : (start_id / HtWt * CHtWt) + (start_id % HtWt);
    uint32_t start_tile_id = start_id / HtWt * CHtWt;
    uint32_t end_tile_id = start_tile_id + HtWt - 1;
    uint32_t read_tile_id = read_tile_id_temp;
    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        if constexpr (dim == 0) {
            read_tile_id = i;
        }
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            cb_in0.reserve_back(onetile);
            noc.async_read(dram_input_addrg, cb_in0, input_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in0.push_back(onetile);
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
