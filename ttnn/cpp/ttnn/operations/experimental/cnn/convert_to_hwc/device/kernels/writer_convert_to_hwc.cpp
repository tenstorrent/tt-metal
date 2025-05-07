// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_pages.h"

constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t ELEMENT_SIZE_BYTES = 2;
constexpr uint32_t STICK_SIZE = TILE_SIZE * ELEMENT_SIZE_BYTES;

FORCE_INLINE void copy_padded_sticks(
    uint64_t l1_read_addr,
    uint32_t& l1_write_addr,
    uint32_t num_sticks,
    uint32_t stick_size,
    uint32_t padded_stick_size) {
    for (uint32_t row = 0; row < num_sticks; row++) {
        noc_async_read(l1_read_addr, l1_write_addr, stick_size);
        l1_read_addr += padded_stick_size;
        l1_write_addr += stick_size;
    }
}

void kernel_main() {
    const uint32_t total_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t channels = get_compile_time_arg_val(2);

    constexpr uint32_t channel_size = channels * ELEMENT_SIZE_BYTES;

    const uint32_t base_l1_write_addr = get_write_ptr(cb_out);
    uint32_t l1_write_addr = base_l1_write_addr;
    for (uint32_t i = 0; i < total_tiles; i++) {
        cb_wait_front(cb_in_transpose, 1);
        const uint64_t l1_read_addr = get_noc_addr(get_read_ptr(cb_in_transpose));
        copy_padded_sticks(l1_read_addr, l1_write_addr, TILE_SIZE, channel_size, STICK_SIZE);
        cb_pop_front(cb_in_transpose, 1);
    }

    DPRINT << "done writer" << ENDL();
    noc_async_read_barrier();
}
