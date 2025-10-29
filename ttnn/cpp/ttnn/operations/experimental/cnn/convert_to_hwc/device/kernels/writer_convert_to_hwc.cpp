// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint_pages.h"
#include "debug/dprint.h"

constexpr uint32_t TILE_SIZE = 32;

template <uint32_t StickSize, uint32_t PaddedStickSize, uint32_t NumSticks>
FORCE_INLINE void copy_padded_sticks(uint32_t l1_read_addr, uint32_t& l1_write_addr) {
    noc_async_read_one_packet_set_state(get_noc_addr(l1_read_addr), StickSize);
    DPRINT << "copy padded sticks read_base=" << l1_read_addr << "  l1_write_base=" << l1_write_addr << ENDL();
    for (uint32_t row = 0; row < NumSticks; row++) {
        DPRINT << "copy stick from tile cb at " << l1_read_addr << "  to output cb at " << l1_write_addr << ENDL();
        // tt::data_movement::common::print_bf16_pages(l1_read_addr, 32, 32);
        noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
        l1_read_addr += PaddedStickSize;
        l1_write_addr += StickSize;
    }
}

template <uint32_t ReadStrideBytes, uint32_t WriteStrideBytes, uint32_t NumSticks, bool DRAM>
FORCE_INLINE void copy_segment(
    uint32_t copy_size,
    uint32_t base_write_addr,
    uint32_t write_offset,
    uint32_t bank_id,
    uint32_t base_read_addr,
    uint32_t read_offset) {
    uint32_t l1_write_addr = base_write_addr + write_offset;
    uint32_t read_addr = base_read_addr + read_offset;
    uint64_t noc_read_addr;
    if constexpr (DRAM) {
        noc_read_addr = get_noc_addr_from_bank_id<true>(bank_id, read_addr);
    } else {
        noc_read_addr = get_noc_addr(read_addr);
    }
    DPRINT << "STARTING COPY SEGMENT - BASE_READ=" << base_read_addr << " offset=" << read_offset
           << " - BASE_WRITE=" << base_write_addr << " offset=" << write_offset << ENDL();
    noc_async_read_one_packet_set_state(noc_read_addr, copy_size);
    for (uint32_t j = 0; j < NumSticks; ++j) {
        DPRINT << "copy stick from input at " << noc_read_addr << "  to output at " << l1_write_addr << ENDL();
        // tt::data_movement::common::print_bf16_pages(noc_read_addr, 32, 1);
        noc_async_read_one_packet_with_state<true>(noc_read_addr, l1_write_addr);
        l1_write_addr += WriteStrideBytes;
        noc_read_addr += ReadStrideBytes;
    }
}

void kernel_main() {
    constexpr uint32_t cb_full_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t channels = get_compile_time_arg_val(4);  // stick size
    constexpr uint32_t hw = get_compile_time_arg_val(5);        // total number of sticks to copy into output
    constexpr uint32_t num_full_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t output_stride_sticks = get_compile_time_arg_val(7);
    constexpr uint32_t initial_l1_write_stick_offset = get_compile_time_arg_val(8);
    constexpr uint32_t element_size_bytes = get_compile_time_arg_val(9);

    static_assert(hw % TILE_SIZE == 0, "Shard width must be multiple of tile width");

    constexpr bool is_input_in_dram = get_compile_time_arg_val(10);
    constexpr bool should_wait = get_compile_time_arg_val(11);
    constexpr uint32_t dram_write_stride_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t dram_read_stride_bytes = get_compile_time_arg_val(13);
    constexpr uint32_t input_sticks_per_core = get_compile_time_arg_val(14);
    constexpr uint32_t input_block_size_sticks_per_core = get_compile_time_arg_val(15);
    constexpr uint32_t input_num_blocks = get_compile_time_arg_val(16);
    constexpr uint32_t semaphore_id = get_compile_time_arg_val(17);
    constexpr uint32_t semaphore_id2 = get_compile_time_arg_val(18);

    const uint32_t x = NOC_X(my_x[0]);
    const uint32_t y = NOC_Y(my_y[0]);

    const uint32_t semaphore_addr = get_semaphore(semaphore_id);
    const uint64_t semaphore_noc_addr = get_noc_addr(x, y, semaphore_addr);
    volatile tt_l1_ptr uint32_t* semaphore_noc_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_noc_addr);

    const uint32_t semaphore_addr2 = get_semaphore(semaphore_id2);
    const uint64_t semaphore_noc_addr2 = get_noc_addr(x, y, semaphore_addr);
    volatile tt_l1_ptr uint32_t* semaphore_noc_addr_ptr2 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_noc_addr2);

    constexpr uint32_t channel_size = channels * element_size_bytes;

    const uint32_t dram_base_read_addr = is_input_in_dram ? get_arg_val<uint32_t>(0) : get_read_ptr(cb_full_input);
    const uint32_t num_segments = get_arg_val<uint32_t>(1);

    constexpr uint32_t tile_size_stick_bytes = TILE_SIZE * element_size_bytes;

    constexpr uint32_t l1_write_addr_stride = output_stride_sticks * channel_size;
    constexpr uint32_t initial_l1_write_addr_offset = initial_l1_write_stick_offset * channel_size;

    const uint32_t base_l1_write_addr = get_write_ptr(cb_out) + initial_l1_write_addr_offset;

    uint32_t l1_output_write_addr = base_l1_write_addr;

    for (uint32_t block_id = 0; block_id < input_num_blocks; block_id++) {
        if constexpr (should_wait) {
            noc_semaphore_wait(semaphore_noc_addr_ptr, block_id + 1);
        } else {
            cb_reserve_back(cb_in, 2 * input_block_size_sticks_per_core);
            noc_semaphore_set(semaphore_noc_addr_ptr, block_id + 1);
        }
        noc_async_read_barrier();

        tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
        uint32_t args_idx = 0;
        for (uint32_t i = 0; i < num_segments; ++i) {
            uint32_t copy_size = args[args_idx++];
            uint32_t write_offset = args[args_idx++];
            uint32_t bank_id = args[args_idx++];  // only used if source is in DRAM
            uint32_t read_offset = args[args_idx++] + (block_id * copy_size * (2 * input_block_size_sticks_per_core));
            copy_segment<
                dram_read_stride_bytes,
                dram_write_stride_bytes,
                input_block_size_sticks_per_core,
                is_input_in_dram>(
                copy_size, get_write_ptr(cb_in), write_offset, bank_id, dram_base_read_addr, read_offset);
        }
        noc_async_read_barrier();

        // One writer must wait until the other has pushed to avoid a race that will trigger a hang
        if constexpr (should_wait) {
            noc_semaphore_wait(semaphore_noc_addr_ptr2, block_id + 1);
        } else {
            cb_push_back(cb_in, 2 * input_block_size_sticks_per_core);
            noc_semaphore_set(semaphore_noc_addr_ptr2, block_id + 1);
        }
        noc_async_read_barrier();

        DPRINT << "pushed block " << block_id << ENDL();
        DPRINT << "   num tiles?=" << num_full_tiles << ENDL();
        DPRINT << "   l1_write_addr_stride=" << l1_write_addr_stride << ENDL();

        for (uint32_t i = 0; i < num_full_tiles; i++) {
            cb_wait_front(cb_in_transpose, 1);

            const uint32_t l1_read_addr = get_read_ptr(cb_in_transpose);

            // tt::data_movement::common::print_bf16_pages(l1_read_addr, 32, 32);

            copy_padded_sticks<channel_size, tile_size_stick_bytes, TILE_SIZE>(l1_read_addr, l1_output_write_addr);
            noc_async_read_barrier();
            cb_pop_front(cb_in_transpose, 1);

            // stride by a number of sticks when splitting writers across cores
            // l1_output_write_addr += l1_write_addr_stride;
        }
    }
}
