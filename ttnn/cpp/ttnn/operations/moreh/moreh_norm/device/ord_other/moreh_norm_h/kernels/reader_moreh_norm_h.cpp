// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const bool input_is_dram = get_arg(args::input_is_dram) == 1;
    const auto num_cols_per_core = get_arg(args::num_cols_per_core);
    const auto tile_offset = get_arg(args::tile_offset);
    const auto Ht = get_arg(args::Ht);
    const auto Wt = get_arg(args::Wt);

    const auto s = TensorAccessor(tensor::input);

    using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::one>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();

    Noc noc;
    DataflowBuffer dfb_input(dfb::input);
    const auto input_tile_bytes = dfb_input.get_tile_size();

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
