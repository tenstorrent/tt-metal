// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    // compile-time args
    constexpr bool output_is_dram = (get_compile_time_arg_val(0) == 1);

    // runtime args
    ArgFetcher arg_fetcher;
    uint32_t output_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t num_output_tiles = arg_fetcher.get_next_arg_val<uint32_t>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_out = 16;
    const uint32_t output_tile_bytes = get_tile_size(cb_id_out);
    const DataFormat output_data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<output_is_dram> s = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    uint32_t end_id = start_id + num_output_tiles;
    for (uint32_t i = start_id; i < end_id; i++) {
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_tile(i, s, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
}
