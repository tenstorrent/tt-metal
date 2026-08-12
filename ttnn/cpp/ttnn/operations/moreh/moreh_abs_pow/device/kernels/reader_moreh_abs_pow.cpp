// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const bool input_is_dram = get_arg(args::input_is_dram) == 1;
    const auto decimal = get_arg(args::decimal);
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto Wt = get_arg(args::Wt);
    const auto tile_offset = get_arg(args::tile_offset);
    const auto origin_w = get_arg(args::origin_w);

    // Input tensor address + layout arrive via the tensor::input binding (no buffer-address RTA).
    const auto s = TensorAccessor(tensor::input);

    Scalar one;
    one.f = 1.0f;
    DataflowBuffer dfb_one(dfb::one);
    DataflowBuffer dfb_decimal(dfb::decimal);
    fill_cb_with_value(dfb_one, one.u);
    fill_cb_with_value(dfb_decimal, decimal);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_w) {
        DataflowBuffer dfb_mask_w(dfb::mask_w);
        generate_mask_w(dfb_mask_w, mask_w);
    }

    Noc noc;
    DataflowBuffer dfb_input(dfb::in);

    const auto start_tile_idx = tile_offset;
    const auto input_tile_bytes = dfb_input.get_tile_size();

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
