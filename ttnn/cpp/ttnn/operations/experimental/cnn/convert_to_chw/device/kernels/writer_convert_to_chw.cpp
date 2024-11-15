// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t ELEMENT_SIZE_BYTES = 2;
constexpr uint32_t STICK_SIZE = TILE_SIZE * ELEMENT_SIZE_BYTES;

void kernel_main() {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t C = get_compile_time_arg_val(2);

    const uint32_t base_l1_write_addr = get_write_ptr(cb_out);
    const uint64_t base_l1_read_addr = get_noc_addr(get_read_ptr(cb_in_transpose));
    noc_async_read_one_packet_set_state(base_l1_read_addr, STICK_SIZE);

    const uint32_t channel_size = total_tiles * STICK_SIZE;

    for (uint32_t i = 0; i < total_tiles; i++) {
        cb_wait_front(cb_in_transpose, 1);
        for (uint32_t j = 0; j < C; j++) {
            const uint32_t l1_read_addr = base_l1_read_addr + (j * STICK_SIZE);
            const uint32_t l1_write_addr = base_l1_write_addr + (j * channel_size) + (i * STICK_SIZE);
            noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
        }
        cb_pop_front(cb_in_transpose, 1);
        cb_push_back(cb_out, 1);
    }
}
