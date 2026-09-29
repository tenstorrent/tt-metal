// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto tile_offset = get_arg(args::tile_offset);
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto num_inner_tiles = get_arg(args::num_inner_tiles);

    constexpr uint32_t onetile = 1;

    // input_grad
    const auto input_grad_addrg = TensorAccessor(tensor::input_grad);

    Noc noc;
    DataflowBuffer dfb_input_grad(dfb::input_grad);
    const auto input_grad_tile_bytes = dfb_input_grad.get_tile_size();

    uint32_t input_grad_tile_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < num_inner_tiles; ++inner_idx) {
            // input_grad (N, C, H, W)
            input_grad_tile_idx = tile_offset + outer_idx * num_inner_tiles + inner_idx;
            dfb_input_grad.wait_front(onetile);
            noc.async_write(
                dfb_input_grad,
                input_grad_addrg,
                input_grad_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = input_grad_tile_idx});
            noc.async_write_barrier();
            dfb_input_grad.pop_front(onetile);
        }  // inner_idx loop
    }  // outer_idx loop

}  // void kernel_main()
