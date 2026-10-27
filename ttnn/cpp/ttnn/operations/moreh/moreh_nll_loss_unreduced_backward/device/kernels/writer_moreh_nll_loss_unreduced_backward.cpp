// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    auto start_id = get_arg(args::start_id);

    const auto input_grad_addrg = TensorAccessor(tensor::input_grad);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_input_grad_obj(dfb::input_grad);
    const auto input_grad_tile_bytes = dfb_input_grad_obj.get_tile_size();

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_input_grad_obj.wait_front(onetile);
        noc.async_write(
            dfb_input_grad_obj, input_grad_addrg, input_grad_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_input_grad_obj.pop_front(onetile);
    }
}
