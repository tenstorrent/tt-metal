// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // runtime args
    // input_grad base address is injected by its TensorBinding (TensorAccessor(tensor::input_grad)).
    const auto num_input_tiles_per_core = get_arg(args::num_input_tiles_per_core);
    const auto tile_offset = get_arg(args::tile_offset);

    // input_grad
    const auto input_grad_addrg = TensorAccessor(tensor::input_grad);

    Noc noc;
    DataflowBuffer dfb_input_grad(dfb::input_grad);
    const auto input_grad_tile_bytes = dfb_input_grad.get_tile_size();

    auto input_grad_tile_idx = tile_offset;
    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        dfb_input_grad.wait_front(1);
        noc.async_write(
            dfb_input_grad,
            input_grad_addrg,
            input_grad_tile_bytes,
            {.offset_bytes = 0},
            {.page_id = input_grad_tile_idx});
        noc.async_write_barrier();
        dfb_input_grad.pop_front(1);
        input_grad_tile_idx++;
    }

}  // void kernel_main()
