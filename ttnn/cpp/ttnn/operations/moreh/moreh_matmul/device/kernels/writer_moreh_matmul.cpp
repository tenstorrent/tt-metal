// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();

    ArgFetcher arg_fetcher;
    uint32_t output_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t num_output_tiles = arg_fetcher.get_next_arg_val<uint32_t>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_out = 16;

    const auto s = TensorAccessor(output_args, output_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out(cb_id_out);
    const auto out_tile_bytes = get_tile_size(cb_id_out);

    uint32_t end_id = start_id + num_output_tiles;
    for (uint32_t i = start_id; i < end_id; i++) {
        cb_out.wait_front(onetile);
        noc.async_write(cb_out, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_out.pop_front(onetile);
    }
}
