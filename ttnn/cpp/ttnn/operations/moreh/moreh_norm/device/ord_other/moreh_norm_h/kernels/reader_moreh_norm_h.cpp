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
    const auto num_cols_per_core = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);
    const auto Ht = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto origin_h = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_one = cb_id++;
    const auto cb_id_mask_h = cb_id++;

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(input_args, input_addr);

    Scalar one;
    one.f = 1.0f;
    DataflowBuffer dfb_one(cb_id_one);
    fill_cb_with_value(dfb_one, one.u);

    constexpr uint32_t TILE_H = 32;
    const bool do_mask_h = (origin_h % TILE_H) != 0;
    const auto mask_h = do_mask_h ? (origin_h % TILE_H) : TILE_H;

    if (do_mask_h) {
        DataflowBuffer dfb_mask_h(cb_id_mask_h);
        generate_mask_h(dfb_mask_h, mask_h);
    }

    Noc noc;
    DataflowBuffer dfb_input(cb_id_input);
    const auto input_tile_bytes = get_tile_size(cb_id_input);

    auto start_output_tile_idx = tile_offset;
    for (uint32_t col_idx = 0; col_idx < num_cols_per_core; ++col_idx) {
        const auto inner_idx = start_output_tile_idx % Wt;
        const auto outer_idx = start_output_tile_idx / Wt;

        auto input_tile_idx = outer_idx * Ht * Wt + inner_idx;
        for (uint32_t row_idx = 0; row_idx < Ht; ++row_idx) {
            dfb_input.reserve_back(1);
            noc.async_read(s, dfb_input, input_tile_bytes, {.page_id = input_tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(1);
            input_tile_idx += Wt;
        }

        start_output_tile_idx++;
    }

}  // void kernel_main()
