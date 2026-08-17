// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t i = 0;
    auto input_grad_addr = get_arg_val<uint32_t>(i++);
    auto num_tiles_per_core = get_arg_val<uint32_t>(i++);
    auto start_id = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    constexpr auto input_grad_args = TensorAccessorArgs<0>();

    const auto input_grad_addrg = TensorAccessor(input_grad_args, input_grad_addr);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_input_grad_obj(cb_input_grad);
    const auto input_grad_tile_bytes = get_tile_size(cb_input_grad);

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_input_grad_obj.wait_front(onetile);
        noc.async_write(
            dfb_input_grad_obj, input_grad_addrg, input_grad_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_input_grad_obj.pop_front(onetile);
    }
}
