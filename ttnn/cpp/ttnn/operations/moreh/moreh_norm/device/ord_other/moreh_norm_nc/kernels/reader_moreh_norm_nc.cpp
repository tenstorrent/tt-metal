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
    const auto num_output_tiles_per_core = get_arg(args::num_output_tiles_per_core);
    const auto tile_offset = get_arg(args::tile_offset);
    const auto outer_stride = get_arg(args::outer_stride);
    const auto num_inner_tiles = get_arg(args::num_inner_tiles);
    const auto num_reduced_tiles_along_dim = get_arg(args::num_reduced_tiles_along_dim);

    const auto s = TensorAccessor(tensor::input);

    Scalar one;
    one.f = 1.0f;
    DataflowBuffer dfb_one(dfb::one);
    fill_cb_with_value(dfb_one, one.u);

    Noc noc;
    DataflowBuffer dfb_input(dfb::input);
    const auto input_tile_bytes = dfb_input.get_tile_size();

    auto start_output_tile_idx = tile_offset;
    const auto inner_stride = num_inner_tiles;
    for (uint32_t idx = 0; idx < num_output_tiles_per_core; ++idx) {
        const auto outer_idx = start_output_tile_idx / inner_stride;
        const auto inner_idx = start_output_tile_idx % inner_stride;

        auto input_tile_idx = outer_idx * outer_stride + inner_idx;
        for (uint32_t d = 0; d < num_reduced_tiles_along_dim; ++d) {
            dfb_input.reserve_back(1);
            noc.async_read(s, dfb_input, input_tile_bytes, {.page_id = input_tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_input.push_back(1);
            input_tile_idx += inner_stride;
        }

        start_output_tile_idx++;
    }

}  // void kernel_main()
