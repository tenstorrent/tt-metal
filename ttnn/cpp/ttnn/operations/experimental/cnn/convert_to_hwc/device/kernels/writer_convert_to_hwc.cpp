// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_pages.h"

constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t ELEMENT_SIZE_BYTES = 2;
constexpr uint32_t TILE_STICK_SIZE_BYTES = TILE_SIZE * ELEMENT_SIZE_BYTES;

template <uint32_t StickSize, uint32_t PaddedStickSize, uint32_t NumSticks>
FORCE_INLINE void copy_padded_sticks(uint64_t l1_read_addr, uint32_t& l1_write_addr) {
    noc_async_read_one_packet_set_state(l1_read_addr, StickSize);
    for (uint32_t row = 0; row < NumSticks; row++) {
        noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
        l1_read_addr += PaddedStickSize;
        l1_write_addr += StickSize;
    }
}

void kernel_main() {
    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t channels = get_compile_time_arg_val(2);  // stick size
    constexpr uint32_t hw = get_compile_time_arg_val(3);        // total number of sticks to copy into output
    constexpr uint32_t num_full_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t output_stride_sticks = get_compile_time_arg_val(5);
    constexpr uint32_t initial_l1_write_stick_offset = get_compile_time_arg_val(6);

    static_assert(hw % TILE_SIZE == 0, "Shard width must be multiple of tile width");

    constexpr uint32_t channel_size = channels * ELEMENT_SIZE_BYTES;
    constexpr uint32_t l1_write_addr_stride = output_stride_sticks * channel_size;
    constexpr uint32_t ini = output_stride_sticks * channel_size;
    constexpr uint32_t initial_l1_write_addr_offset = initial_l1_write_stick_offset * channel_size;

    const uint32_t base_l1_write_addr = get_write_ptr(cb_out) + initial_l1_write_addr_offset;

    uint32_t l1_write_addr = base_l1_write_addr;
    for (uint32_t i = 0; i < num_full_tiles; i++) {
        cb_wait_front(cb_in_transpose, 1);
        const uint64_t l1_read_addr = get_noc_addr(get_read_ptr(cb_in_transpose));
        copy_padded_sticks<channel_size, TILE_STICK_SIZE_BYTES, TILE_SIZE>(l1_read_addr, l1_write_addr);
        cb_pop_front(cb_in_transpose, 1);
        l1_write_addr += l1_write_addr_stride;  // skip some number of sticks if splitting writers across cores
    }
    noc_async_read_barrier();
}
