// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    int i{0};
    const auto input_addr = get_arg_val<uint32_t>(i++);
    const bool input_is_dram = get_arg_val<uint32_t>(i++) == 1;
    const auto decimal = get_arg_val<uint32_t>(i++);
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_decimal = cb_id++;
    const auto cb_id_mask_w = cb_id++;

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(input_args, input_addr);

    Scalar one;
    one.f = 1.0f;
    DataflowBuffer dfb_one(cb_id_one);
    DataflowBuffer dfb_decimal(cb_id_decimal);
    fill_cb_with_value(dfb_one, one.u);
    fill_cb_with_value(dfb_decimal, decimal);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_w) {
        DataflowBuffer dfb_mask_w(cb_id_mask_w);
        generate_mask_w(dfb_mask_w, mask_w);
    }

    Noc noc;
    DataflowBuffer dfb_input(cb_id_input);

    const auto start_tile_idx = tile_offset;
    const auto input_tile_bytes = get_tile_size(cb_id_input);

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const auto tile_idx = start_tile_idx + row_idx * Wt + col_idx;
            dfb_input.reserve_back(1);
            noc.async_read(s, dfb_input, input_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(1);
        }
    }

}  // void kernel_main()
