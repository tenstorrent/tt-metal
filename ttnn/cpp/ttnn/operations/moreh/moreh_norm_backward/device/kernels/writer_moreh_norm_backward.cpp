// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // compile time args
    constexpr auto input_grad_args = TensorAccessorArgs<0>();

    int i{0};
    const auto input_grad_addr = get_arg_val<uint32_t>(i++);
    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto tile_offset = get_arg_val<uint32_t>(i++);

    uint32_t cb_id{16};
    const auto cb_id_input_grad = cb_id++;

    // input_grad
    const auto input_grad_addrg = TensorAccessor(input_grad_args, input_grad_addr);

    Noc noc;
    DataflowBuffer dfb_input_grad(cb_id_input_grad);
    const auto input_grad_tile_bytes = get_tile_size(cb_id_input_grad);

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
