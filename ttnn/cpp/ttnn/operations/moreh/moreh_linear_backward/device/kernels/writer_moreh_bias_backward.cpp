// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    ArgFetcher arg_fetcher;
    uint32_t dst_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t num_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t start_id = arg_fetcher.get_next_arg_val<uint32_t>();

    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; i++) {
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
}
