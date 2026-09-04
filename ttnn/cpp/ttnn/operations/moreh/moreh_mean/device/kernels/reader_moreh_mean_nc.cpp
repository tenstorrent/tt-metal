// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_input_tiles = get_arg(args::num_input_tiles);
    const auto num_output_tiles = get_arg(args::num_output_tiles);
    const auto input_tile_stride = get_arg(args::input_tile_stride);
    const auto start_id = get_arg(args::start_id);
    const auto HtWt = get_arg(args::HtWt);
    const auto inner_size = get_arg(args::inner_size);

    constexpr uint32_t onetile = 1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    DataflowBuffer dfb_in1(dfb::in1);
    fill_cb_with_value(dfb_in1, scaler.u);

    scaler.f = 1.0f / num_input_tiles;
    DataflowBuffer dfb_in2(dfb::scalar);
    fill_cb_with_value(dfb_in2, scaler.u, 1);

    const auto s = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::input);
    const auto in0_tile_bytes = dfb_in0.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        uint32_t hw_tile_id = i % HtWt;
        uint32_t inner_id = (i / HtWt) % inner_size * HtWt;
        uint32_t outer_id = (i / HtWt / inner_size) * inner_size * HtWt * num_input_tiles;

        auto read_tile_id = outer_id + inner_id + hw_tile_id;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            dfb_in0.reserve_back(onetile);
            noc.async_read(s, dfb_in0, in0_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0.push_back(onetile);
            read_tile_id += input_tile_stride;
        }
    }
}
