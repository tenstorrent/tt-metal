// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Delivers the order the factory described to the planner with ReduceBlockSpec::output_group:
// strips of output_group columns, row by row within each strip.
void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t output_group = get_arg(args::output_group);

    const auto src = TensorAccessor(tensor::src);
    Noc noc;
    DataflowBuffer input(dfb::input);

    for (uint32_t strip = 0; strip < Wt; strip += output_group) {
        for (uint32_t row = 0; row < Ht; ++row) {
            for (uint32_t column = strip; column < strip + output_group; ++column) {
                input.reserve_back(1);
                noc.async_read(src, input, input.get_tile_size(), {.page_id = row * Wt + column}, {.offset_bytes = 0});
                noc.async_read_barrier();
                input.push_back(1);
            }
        }
    }
}
