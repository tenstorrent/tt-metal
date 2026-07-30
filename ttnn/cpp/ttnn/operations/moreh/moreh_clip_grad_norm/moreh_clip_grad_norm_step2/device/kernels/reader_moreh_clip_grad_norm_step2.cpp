// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto decimal = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_decimal = cb_id++;

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(input_args, input_addr);

    DataflowBuffer dfb_decimal(cb_id_decimal);
    fill_cb_with_value(dfb_decimal, decimal);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_input(cb_id_input);
    const auto input_tile_bytes = get_tile_size(cb_id_input);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        dfb_input.reserve_back(onetile);
        noc.async_read(s, dfb_input, input_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_input.push_back(onetile);
    }

}  // void kernel_main()
