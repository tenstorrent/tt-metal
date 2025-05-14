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

template <uint32_t ReadStrideBytes, uint32_t WriteStrideBytes, uint32_t NumSticks>
FORCE_INLINE void copy_segment_from_dram(
    uint32_t copy_size,
    uint32_t base_write_addr,
    uint32_t write_offset,
    uint32_t bank_id,
    uint32_t base_read_addr,
    uint32_t read_offset) {
    uint32_t l1_write_addr = base_write_addr + write_offset;

    uint32_t read_addr = base_read_addr + read_offset;
    uint64_t noc_read_addr = get_noc_addr_from_bank_id<true>(bank_id, read_addr);

    for (uint32_t j = 0; j < NumSticks; ++j) {
        DPRINT << "stick=" << j << " read_addr=" << noc_read_addr << " write_addr=" << l1_write_addr
               << " size=" << copy_size << ENDL();
        noc_async_read(noc_read_addr, l1_write_addr, copy_size);
        l1_write_addr += WriteStrideBytes;
        noc_read_addr += ReadStrideBytes;
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
    constexpr uint32_t dram_write_stride_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t dram_read_stride_bytes = get_compile_time_arg_val(8);

    DPRINT << "dram strides " << dram_read_stride_bytes << " " << dram_write_stride_bytes << ENDL();

    constexpr uint32_t cb_in = 0;
    constexpr uint32_t sticks_per_core = 8;
    constexpr bool read_input_from_dram = true;

    static_assert(hw % TILE_SIZE == 0, "Shard width must be multiple of tile width");

    constexpr uint32_t channel_size = channels * ELEMENT_SIZE_BYTES;
    constexpr uint32_t l1_write_addr_stride = output_stride_sticks * channel_size;
    constexpr uint32_t initial_l1_write_addr_offset = initial_l1_write_stick_offset * channel_size;

    const uint32_t base_read_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_segments = get_arg_val<uint32_t>(1);
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
    uint32_t args_idx = 0;

    if constexpr (read_input_from_dram) {
        for (uint32_t i = 0; i < num_segments; ++i) {
            DPRINT << "segment " << i << ENDL();
            uint32_t copy_size = args[args_idx++];
            uint32_t write_offset = args[args_idx++];
            uint32_t bank_id = args[args_idx++];
            uint32_t read_offset = args[args_idx++];
            copy_segment_from_dram<dram_read_stride_bytes, dram_write_stride_bytes, sticks_per_core>(
                copy_size, get_write_ptr(cb_in), write_offset, bank_id, base_read_addr, read_offset);
        }
        noc_async_read_barrier();
    }

    cb_push_back(cb_in, sticks_per_core);

    const uint32_t base_l1_write_addr = get_write_ptr(cb_out) + initial_l1_write_addr_offset;

    uint32_t l1_write_addr = base_l1_write_addr;
    for (uint32_t i = 0; i < num_full_tiles; i++) {
        cb_wait_front(cb_in_transpose, 1);
        const uint64_t l1_read_addr = get_noc_addr(get_read_ptr(cb_in_transpose));
        copy_padded_sticks<channel_size, TILE_STICK_SIZE_BYTES, TILE_SIZE>(l1_read_addr, l1_write_addr);
        noc_async_read_barrier();
        cb_pop_front(cb_in_transpose, 1);
        l1_write_addr += l1_write_addr_stride;  // skip some number of sticks if splitting writers across cores
    }
}
