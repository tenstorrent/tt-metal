// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const auto output_addr = get_arg_val<uint32_t>(0);
    const auto num_tiles = get_arg_val<uint32_t>(1);
    const auto start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;

    constexpr auto output_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(output_args, output_addr);

    Noc noc;
    DataflowBuffer dfb_out(cb_id_out);
    const auto out_tile_bytes = get_tile_size(cb_id_out);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        uint32_t write_tile_id = i;
        dfb_out.wait_front(onetile);

        noc.async_write(dfb_out, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = write_tile_id});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
}
