// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t ELEMENT_SIZE_BYTES = 2;
constexpr uint32_t STICK_SIZE = TILE_SIZE * ELEMENT_SIZE_BYTES;

void kernel_main() {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t BATCH_SIZE = 8;
    const uint32_t num_batches = total_tiles / BATCH_SIZE;
    const uint32_t leftover = total_tiles % BATCH_SIZE;

    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(0);
    constexpr uint32_t in_transpose_tile_size = get_tile_size(cb_in_transpose);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);

    cb_reserve_back(cb_out, 1);
    const uint32_t base_l1_write_addr = get_write_ptr(cb_out);
    noc_async_read_one_packet_set_state(get_noc_addr(get_read_ptr(cb_in_transpose)), STICK_SIZE);

    const uint32_t channel_size = total_tiles * STICK_SIZE;

    int tile_index = 0;
    for (uint32_t i = 0; i < num_batches; i++) {
        cb_wait_front(cb_in_transpose, BATCH_SIZE);
        uint64_t l1_read_addr_tile = get_noc_addr(get_read_ptr(cb_in_transpose));
        for (uint32_t b = 0; b < BATCH_SIZE; b++) {
            uint64_t l1_read_addr = l1_read_addr_tile;
            for (uint32_t j = 0; j < C; j++) {
                const uint32_t l1_write_addr = base_l1_write_addr + (j * channel_size) + (tile_index * STICK_SIZE);
                noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
                l1_read_addr += STICK_SIZE;
            }
            tile_index++;
            l1_read_addr_tile += in_transpose_tile_size;
        }
        cb_pop_front(cb_in_transpose, BATCH_SIZE);
    }

    for (uint32_t i = 0; i < leftover; i++) {
        cb_wait_front(cb_in_transpose, 1);
        uint64_t l1_read_addr = get_noc_addr(get_read_ptr(cb_in_transpose));
        for (uint32_t j = 0; j < C; j++) {
            const uint32_t l1_write_addr = base_l1_write_addr + (j * channel_size) + (tile_index * STICK_SIZE);
            noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
            l1_read_addr += STICK_SIZE;
        }
        tile_index++;
        cb_pop_front(cb_in_transpose, 1);
    }
    noc_async_read_barrier();
    cb_push_back(cb_out, 1);
}
