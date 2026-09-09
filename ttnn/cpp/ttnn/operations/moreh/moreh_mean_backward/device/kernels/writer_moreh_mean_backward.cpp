// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto num_tiles = get_arg(args::num_tiles);
    const auto start_id = get_arg(args::start_id);

    constexpr uint32_t onetile = 1;

    const auto input_grad_addrg = TensorAccessor(tensor::input_grad);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);
    const auto out_tile_bytes = dfb_out.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        uint32_t write_tile_id = i;
        dfb_out.wait_front(onetile);

        noc.async_write(dfb_out, input_grad_addrg, out_tile_bytes, {.offset_bytes = 0}, {.page_id = write_tile_id});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
}
