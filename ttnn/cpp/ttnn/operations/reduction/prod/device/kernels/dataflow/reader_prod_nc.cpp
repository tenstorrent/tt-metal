// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);
    const auto input_tile_offset = get_arg(args::input_tile_offset);
    const auto start_id = get_arg(args::start_id);
    const auto HtWt = get_arg(args::HtWt);
    const auto CHtWt = get_arg(args::CHtWt);
    constexpr auto dim = get_arg(args::dim);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in);

    uint32_t input_tile_bytes = dfb_in0.get_tile_size();
    const auto dram_input_addrg = TensorAccessor(tensor::input);

    uint32_t read_tile_id_temp = (dim == 0) ? (start_id) : (start_id / HtWt * CHtWt) + (start_id % HtWt);
    uint32_t start_tile_id = start_id / HtWt * CHtWt;
    uint32_t end_tile_id = start_tile_id + HtWt - 1;
    uint32_t read_tile_id = read_tile_id_temp;
    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        if constexpr (dim == 0) {
            read_tile_id = i;
        }
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            dfb_in0.reserve_back(onetile);
            noc.async_read(dram_input_addrg, dfb_in0, input_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0.push_back(onetile);
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
