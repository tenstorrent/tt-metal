// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    // compile-time args
    constexpr auto output_args = TensorAccessorArgs<0>();

    // runtime args
    ArgFetcher arg_fetcher;
    const auto output_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto start_id = arg_fetcher.get_next_arg_val<uint32_t>();

    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;

    const auto output_addrg = TensorAccessor(output_args, output_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out_obj(cb_id_out);
    const auto out_tile_bytes = get_tile_size(cb_id_out);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        uint32_t write_tile_id = i;
        cb_out_obj.wait_front(onetile);

        noc.async_write(cb_out_obj, output_addrg, out_tile_bytes, {.offset_bytes = 0}, {.page_id = write_tile_id});
        noc.async_write_barrier();
        cb_out_obj.pop_front(onetile);
    }
}
