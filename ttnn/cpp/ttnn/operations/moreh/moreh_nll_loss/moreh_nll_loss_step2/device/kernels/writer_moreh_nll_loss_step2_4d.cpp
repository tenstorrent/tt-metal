// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    auto start_id = get_arg(args::start_id);

    const auto output_addrg = TensorAccessor(tensor::output);

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_out(dfb::output);
    const auto out_tile_bytes = dfb_out.get_tile_size();

    uint32_t end_id = start_id + num_tiles_per_core;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out.wait_front(onetile);
        noc.async_write(dfb_out, output_addrg, out_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
}
