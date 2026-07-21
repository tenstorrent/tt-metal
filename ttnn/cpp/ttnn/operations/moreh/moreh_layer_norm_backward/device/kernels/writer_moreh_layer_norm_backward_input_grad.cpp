// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto input_grad_addr = get_arg_val<uint32_t>(0);
    const auto num_rows_per_core = get_arg_val<uint32_t>(1);
    const auto Wt = get_arg_val<uint32_t>(2);
    const auto tile_offset = get_arg_val<uint32_t>(3);

    constexpr auto input_grad_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_input_grad = 16;

    const auto input_grad_addrg = TensorAccessor(input_grad_args, input_grad_addr);

    uint32_t offs = 0;
    const auto NCHt = num_rows_per_core;
    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_input_grad(cb_id_input_grad);
    const auto input_grad_tile_bytes = get_tile_size(cb_id_input_grad);

    for (uint32_t ncht = 0; ncht < num_rows_per_core; ncht++) {
        // input_grad (N, C, H, W)
        for (uint32_t wt = 0; wt < Wt; wt++) {
            dfb_input_grad.wait_front(onetile);
            noc.async_write(
                dfb_input_grad,
                input_grad_addrg,
                input_grad_tile_bytes,
                {.offset_bytes = 0},
                {.page_id = offs + wt + tile_offset});
            noc.async_write_barrier();
            dfb_input_grad.pop_front(onetile);
        }  // wt loop
        offs += Wt;

    }  // num_rows_per_core loop

}  // void kernel_main()
