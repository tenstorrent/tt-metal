// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t cache_addr  = get_arg_val<uint32_t>(0);
    const uint32_t Wt          = get_arg_val<uint32_t>(1);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(2);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(2);

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr,
        .page_size = cache_tile_bytes,
        .data_format = cache_data_format
    };

    cb_reserve_back(input_cb_id, Wt);
    cb_push_back(input_cb_id, Wt);

    uint32_t cache_id = cache_start_id;

    cb_reserve_back(cache_cb_id, Wt);
    uint32_t cache_l1_write_addr = get_write_ptr(cache_cb_id);
    for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
        noc_async_read_tile(curr_cache_id, s0, cache_l1_write_addr);
        cache_l1_write_addr += cache_tile_bytes;
    }

    noc_async_read_barrier();
    cb_push_back(cache_cb_id, Wt );
}
