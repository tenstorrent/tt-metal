// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // runtime args
    const auto num_tiles = get_arg(args::num_tiles);
    const auto start_id = get_arg(args::start_id);

    constexpr uint32_t onetile = 1;

    const auto output_addrg = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_out_obj(dfb::out);
    const auto out_tile_bytes = dfb_out_obj.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        uint32_t write_tile_id = i;
        dfb_out_obj.wait_front(onetile);

        noc.async_write(dfb_out_obj, output_addrg, out_tile_bytes, {.offset_bytes = 0}, {.page_id = write_tile_id});
        noc.async_write_barrier();
        dfb_out_obj.pop_front(onetile);
    }
}
