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
    for (uint32_t row = 0; row < NumSticks; row++) {
        noc_async_read_one_packet_with_state<true>(l1_read_addr, l1_write_addr);
        l1_read_addr += PaddedStickSize;
        l1_write_addr += StickSize;
    }
}


void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in_batch = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_transpose = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t num_output_channels_padded = get_compile_time_arg_val(4);  // padded output channels (min 8)
    constexpr uint32_t num_full_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t output_stride_sticks = get_compile_time_arg_val(6);
    constexpr uint32_t initial_l1_write_stick_offset = get_compile_time_arg_val(7);
    constexpr uint32_t element_size_bytes = get_compile_time_arg_val(8);
    constexpr bool is_input_in_dram = get_compile_time_arg_val(9);
    constexpr bool is_reader = get_compile_time_arg_val(10);
    constexpr uint32_t input_block_size_sticks_per_core = get_compile_time_arg_val(11);
    constexpr uint32_t input_num_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t l1_write_output_addr_stride = get_compile_time_arg_val(13);
    constexpr uint32_t block_size_bytes = get_compile_time_arg_val(14);

    constexpr uint32_t channel_size = num_output_channels_padded * element_size_bytes;

    DPRINT << "=== WRITER KERNEL START ===" << ENDL();
    DPRINT << "Compile-time args: num_output_channels_padded=" << num_output_channels_padded
           << ", num_full_tiles=" << num_full_tiles << ENDL();
    DPRINT << "output_stride_sticks=" << output_stride_sticks
           << ", initial_l1_write_stick_offset=" << initial_l1_write_stick_offset << ENDL();
    DPRINT << "element_size_bytes=" << element_size_bytes << ", is_input_in_dram=" << (uint32_t)is_input_in_dram
           << ENDL();
    DPRINT << "is_reader=" << (uint32_t)is_reader
           << ", input_block_size_sticks_per_core=" << input_block_size_sticks_per_core << ENDL();
    DPRINT << "input_num_blocks=" << input_num_blocks << ", l1_write_output_addr_stride=" << l1_write_output_addr_stride
           << ENDL();
    DPRINT << "block_size_bytes=" << block_size_bytes << ", channel_size=" << channel_size << ENDL();

    const uint32_t dram_base_read_addr = is_input_in_dram ? get_arg_val<uint32_t>(0) : get_read_ptr(cb_in);
    DPRINT << "dram_base_read_addr=" << dram_base_read_addr << ENDL();

    // Parse the new serialized blocked transfer format
    tt_l1_ptr uint32_t* args = (tt_l1_ptr uint32_t*)(get_arg_addr(1));
    uint32_t args_idx = 0;

    const uint32_t num_blocks = args[args_idx++];
    DPRINT << "Parsed num_blocks=" << num_blocks << "  input_num_blocks=" << input_num_blocks << ENDL();

    constexpr uint32_t tile_size_stick_bytes = TILE_SIZE * element_size_bytes;
    constexpr uint32_t initial_l1_write_addr_offset = initial_l1_write_stick_offset * channel_size;

    const uint32_t base_l1_write_addr = get_write_ptr(cb_out) + initial_l1_write_addr_offset;
    uint32_t l1_output_write_addr = base_l1_write_addr;

    // Process each blocked transfer group
    for (uint32_t block_id = 0; block_id < num_blocks && block_id < input_num_blocks; block_id++) {
        DPRINT << "Processing block_id=" << block_id << "/" << num_blocks << ENDL();

        if constexpr (is_reader) {
            cb_reserve_back(cb_in_batch, input_block_size_sticks_per_core);
            DPRINT << "Reserved cb_in_batch with " << input_block_size_sticks_per_core << " sticks" << ENDL();

            // Parse the number of transfers in this block group
            const uint32_t group_size = args[args_idx++];
            DPRINT << "block_id=" << block_id << " group_size=" << group_size << ENDL();

            // Process all transfers in this group
            for (uint32_t transfer_idx = 0; transfer_idx < group_size; transfer_idx++) {
                uint32_t src_x = args[args_idx++];
                uint32_t src_y = args[args_idx++];
                uint32_t src_offset_bytes = args[args_idx++];
                uint32_t dst_offset_bytes = args[args_idx++];
                uint32_t transfer_size_bytes = args[args_idx++];
                uint32_t bank_id = args[args_idx++];

                DPRINT << "Transfer " << transfer_idx << ": src=(" << src_x << "," << src_y << ")" << ENDL();
                DPRINT << "  src_offset_bytes=" << src_offset_bytes << ", dst_offset_bytes=" << dst_offset_bytes
                       << ENDL();
                DPRINT << "  transfer_size_bytes=" << transfer_size_bytes << ENDL();

                uint64_t src_addr_base = 0;
                if constexpr (is_input_in_dram) {
                    // For DRAM, use bank_id to compute NOC address from bank_id
                    src_addr_base = get_noc_addr_from_bank_id<true>(bank_id, dram_base_read_addr);
                    DPRINT << "DRAM transfer: src_addr_base=" << (uint32_t)src_addr_base << ENDL();
                } else {
                    src_addr_base = get_noc_addr(src_x, src_y, get_read_ptr(cb_in));
                    DPRINT << "L1 transfer: src_addr_base=" << (uint32_t)src_addr_base << ENDL();
                }

                const uint32_t dst_addr = get_write_ptr(cb_in_batch) + (dst_offset_bytes % block_size_bytes);
                noc_async_read(src_addr_base + src_offset_bytes, dst_addr, transfer_size_bytes);
            }

            noc_async_read_barrier();
            cb_push_back(cb_in_batch, input_block_size_sticks_per_core);
            DPRINT << "Completed transfers for block " << block_id << ENDL();
        } else {
            // For non-reader kernels, skip over the transfer data
            const uint32_t group_size = args[args_idx++];
            args_idx += group_size * 6;  // Skip 6 args per transfer (src_x, src_y, src_offset_bytes, dst_offset_bytes,
                                         // transfer_size_bytes, bank_id)
            DPRINT << "Non-reader: skipped group_size=" << group_size << " transfers" << ENDL();
        }

        DPRINT << "Processing " << num_full_tiles << " tiles for block " << block_id << ENDL();
        for (uint32_t i = 0; i < num_full_tiles; i++) {
            cb_wait_front(cb_in_transpose, 1);

            const uint32_t l1_read_addr = get_read_ptr(cb_in_transpose);
            DPRINT << "Tile " << i << ": l1_read_addr=" << l1_read_addr
                   << ", l1_output_write_addr=" << l1_output_write_addr << ENDL();

            copy_padded_sticks<channel_size, tile_size_stick_bytes, TILE_SIZE>(l1_read_addr, l1_output_write_addr);
            noc_async_read_barrier();
            cb_pop_front(cb_in_transpose, 1);

            // Stride by a number of sticks when splitting writers across cores
            l1_output_write_addr += l1_write_output_addr_stride;
        }
    }
    DPRINT << "=== WRITER KERNEL END ===" << ENDL();
}
