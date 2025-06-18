// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "accessor/sharded_accessor.h"

void kernel_main() {
    uint32_t bank_base_address = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr tt::CBIndex cb_id_out = (tt::CBIndex)get_compile_time_arg_val(0);
    constexpr uint32_t tile_elements = get_compile_time_arg_val(1);
    constexpr DataFormat data_format = static_cast<DataFormat>(get_compile_time_arg_val(2));
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t rank = get_compile_time_arg_val(4);
    constexpr uint32_t num_banks = get_compile_time_arg_val(5);
    constexpr uint32_t arg_index = 6;
    using output_dspec = distribution_spec_t<arg_index, rank, num_banks>;
    auto sharded_accessor = ShardedAccessor<output_dspec, page_size>{.bank_base_address = bank_base_address};

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint64_t curr_noc_addr = sharded_accessor.get_noc_addr(i);
        noc_async_write(curr_noc_addr, l1_read_addr, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, onetile);
    }
}
