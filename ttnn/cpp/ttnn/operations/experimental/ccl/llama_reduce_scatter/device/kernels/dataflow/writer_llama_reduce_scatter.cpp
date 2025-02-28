// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(2);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile = get_arg_val<uint32_t>(2);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
        cb_wait_front(cb_id_in0, onetile);
        uint32_t l1_read_ptr = get_read_ptr(cb_id_in0);
        noc_async_write_tile(tile, d, l1_read_ptr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_in0, onetile);
    }
}
