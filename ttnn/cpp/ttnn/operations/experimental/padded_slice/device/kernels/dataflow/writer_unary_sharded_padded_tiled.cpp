// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "hw/inc/dataflow_api.h"
#include "hw/inc/dataflow_api_addrgen.h"
#include "debug/dprint.h"
void kernel_main() {
    const uint32_t total_num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles_per_read = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_untilized_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(1);

    const uint32_t tile_size = get_tile_size(cb_out_id);
    const uint32_t read_size = tile_size * num_tiles_per_read;

    uint32_t write_addr = get_write_ptr(cb_out_id);
    uint32_t tiles_read = 0;
    uint32_t read_addr = get_read_ptr(cb_untilized_id);

#define DEBUG
#ifdef DEBUG
    DPRINT << "total_num_tiles: " << total_num_tiles << ", num_tiles_per_read: " << num_tiles_per_read
           << ", tile_size: " << tile_size << ", read_size: " << read_size << ENDL();
#endif
    noc_async_read_one_packet_set_state(get_noc_addr(read_addr), read_size);

    while (tiles_read < total_num_tiles) {
        cb_wait_front(cb_untilized_id, num_tiles_per_read);
        uint64_t noc_read_addr = get_noc_addr(get_read_ptr(cb_untilized_id));
        noc_async_read_one_packet_with_state<true>(noc_read_addr, write_addr);
        write_addr += read_size;
        cb_pop_front(cb_untilized_id, num_tiles_per_read);
        tiles_read += num_tiles_per_read;
    }
}
