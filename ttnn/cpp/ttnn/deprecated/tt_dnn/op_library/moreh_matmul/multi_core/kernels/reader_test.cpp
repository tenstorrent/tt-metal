// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
#include "debug/dprint.h"

void kernel_main() {
    // compile-time args
    constexpr bool input_is_dram = (get_compile_time_arg_val(0) == 1);

    // runtime args
    ArgFetcher arg_fetcher;
    const auto input_addr {arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto num_input_tiles{arg_fetcher.get_next_arg_val<uint32_t>()};
    const auto input_start_id{arg_fetcher.get_next_arg_val<uint32_t>()};

    constexpr uint32_t onetile{1};
    constexpr uint32_t cb_id_in0{0};

    uint32_t input_tile_bytes{static_cast<uint32_t>(get_tile_size(cb_id_in0))};
    const auto input_data_format{get_dataformat(cb_id_in0)};
    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr,
        .page_size = input_tile_bytes,
        .data_format = input_data_format};

    uint32_t l1_write_addr;
    for (uint32_t i = 0; i < num_input_tiles; ++i) {
        uint32_t read_input_tile_id{input_start_id + i};
        // read input tile
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(read_input_tile_id, input_addrg, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
