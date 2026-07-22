// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto input_grad_args = TensorAccessorArgs<0>();

    ArgFetcher arg_fetcher;
    const auto input_grad_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto start_id = arg_fetcher.get_next_arg_val<uint32_t>();

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;

    const auto input_grad_addrg = TensorAccessor(input_grad_args, input_grad_addr);

    Noc noc;
    DataflowBuffer dfb_out(cb_id_out);
    const auto out_tile_bytes = get_tile_size(cb_id_out);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        uint32_t write_tile_id = i;
        dfb_out.wait_front(onetile);

        noc.async_write(dfb_out, input_grad_addrg, out_tile_bytes, {.offset_bytes = 0}, {.page_id = write_tile_id});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
}
