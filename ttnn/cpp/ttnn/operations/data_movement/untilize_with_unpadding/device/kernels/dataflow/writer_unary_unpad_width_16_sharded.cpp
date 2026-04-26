// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"

// Special case writer for unpad width 16 tensors
// Skip untilize and just copy f0 and f2 from input tiles to output tiles
void kernel_main() {
    uint32_t num_unpadded_output_rows = get_arg_val<uint32_t>(0);
    uint32_t num_padded_tiles_per_core = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_untilize_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);

    constexpr uint32_t tile_size_in_bytes = get_tile_size(cb_id_out);
    constexpr uint32_t quarter_tile_size_in_bytes = tile_size_in_bytes / 4;

    experimental::CircularBuffer cb_untilize_out(cb_id_untilize_out);
    experimental::CircularBuffer cb_out(cb_id_out);

    const uint32_t batches_of_8 = num_padded_tiles_per_core / 8;
    const uint32_t remaining_tiles = num_padded_tiles_per_core % 8;

    cb_out.reserve_back(num_unpadded_output_rows);
    uint32_t l1_write_addr = cb_out.get_write_ptr();

    static_assert(quarter_tile_size_in_bytes <= NOC_MAX_BURST_SIZE);
    // set_state uses just x/y from the get_noc_addr, addr is ignored
    // Keep legacy NOC API for local L1 operations (sharded kernel)
    noc_async_read_one_packet_set_state(get_noc_addr(l1_write_addr), quarter_tile_size_in_bytes);

    for (uint32_t i = 0; i < batches_of_8; i++) {
        cb_untilize_out.wait_front(8);
        uint64_t noc_l1_read_addr = get_noc_addr(cb_untilize_out.get_read_ptr());

        for (uint32_t j = 0; j < 8; j++) {
            noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            noc_l1_read_addr += 2 * quarter_tile_size_in_bytes;
            l1_write_addr += quarter_tile_size_in_bytes;

            noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
            noc_l1_read_addr += 2 * quarter_tile_size_in_bytes;
            l1_write_addr += quarter_tile_size_in_bytes;
        }

        noc_async_read_barrier();
        cb_untilize_out.pop_front(8);
    }

    cb_untilize_out.wait_front(remaining_tiles);
    uint64_t noc_l1_read_addr = get_noc_addr(cb_untilize_out.get_read_ptr());
    for (uint32_t i = 0; i < remaining_tiles; i++) {
        noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
        noc_l1_read_addr += 2 * quarter_tile_size_in_bytes;
        l1_write_addr += quarter_tile_size_in_bytes;

        noc_async_read_one_packet_with_state<true>(noc_l1_read_addr, l1_write_addr);
        noc_l1_read_addr += 2 * quarter_tile_size_in_bytes;
        l1_write_addr += quarter_tile_size_in_bytes;
    }
    noc_async_read_barrier();
    cb_untilize_out.pop_front(remaining_tiles);

    cb_out.push_back(num_unpadded_output_rows);
}
