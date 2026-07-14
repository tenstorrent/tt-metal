// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t num_units_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    constexpr auto output_args = TensorAccessorArgs<0>();

    const auto output_addrg = TensorAccessor(output_args, output_addr);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_out(cb_output);
    const auto out_tile_bytes = get_tile_size(cb_output);

    uint32_t end_id = start_id + num_units_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out.wait_front(onetile);
        noc.async_write(dfb_out, output_addrg, out_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
}
