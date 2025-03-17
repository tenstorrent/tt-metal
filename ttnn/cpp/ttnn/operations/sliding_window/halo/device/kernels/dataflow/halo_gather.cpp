// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"

#define ENABLE_DEBUG 1

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

static inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        ptr[i] = val;
    }
    return true;
}

template <uint32_t STICK_SIZE_BYTES, uint32_t PAGE_SIZE, uint32_t BLOCK_HEIGHT_STICKS>
static inline void execute_transfer(
    uint32_t in_base_l1_addr,
    uint64_t out_base_l1_addr,
    uint16_t src_offset_id,
    uint16_t dst_offset_id,
    uint16_t transfer_size) {
    const uint32_t src_offset = (src_offset_id % BLOCK_HEIGHT_STICKS) *
                                PAGE_SIZE;  // Convert from global stick offset to local block stick offset
    const uint32_t dst_offset = dst_offset_id * STICK_SIZE_BYTES;
    const uint32_t size = transfer_size * STICK_SIZE_BYTES;
    const uint32_t src_addr = in_base_l1_addr + src_offset;
    const uint64_t dst_addr = out_base_l1_addr + dst_offset;
    DPRINT << "write: adj_src_offset=" << src_offset << " dst_offset=" << dst_offset << " size=" << size << ENDL();
    noc_async_write(src_addr, dst_addr, size);
}

template <
    uint32_t cb_id,
    uint32_t stick_nbytes,
    uint32_t input_aligned_page_size,
    uint32_t block_size_width_tiles,
    bool is_block_sharded,
    bool is_width_sharded,
    bool is_col_major>
static inline void execute_config(
    const tt_l1_ptr uint16_t* config,
    uint32_t in_base_l1_addr,
    uint32_t out_base_l1_addr,
    uint32_t my_noc_x,
    uint32_t my_noc_y) {
    uint16_t index = 0;
    const uint16_t total_number_of_segments = config[index++];

    uint16_t number_of_segments_remaining = total_number_of_segments;

    uint16_t destination_noc_x = 0;
    uint16_t destination_noc_y = 0;
    uint16_t transfers_remaining = 0;

    uint16_t src_offset = 0;
    uint16_t dst_offset = 0;
    uint16_t transfer_size = 0;

    uint64_t out_l1_addr = 0;

    const uint32_t block_height_sticks = 32;
    uint16_t block_id = 0;
    uint16_t block_boundary_offset = block_height_sticks;

    // Wait for the first set of tiles from compute before beginning
    cb_wait_front(cb_id, block_size_width_tiles);

    while (number_of_segments_remaining) {
        // Read header for to get destination for this route
        destination_noc_x = config[index++];
        destination_noc_y = config[index++];
        transfers_remaining = config[index++];
        DPRINT << "start of segment =" << number_of_segments_remaining << " x=" << destination_noc_x
               << " y=" << destination_noc_y << " transfers?=" << transfers_remaining << ENDL();

        const uint16_t noc_x = ((is_block_sharded && !is_col_major) || is_width_sharded) ? my_noc_x : destination_noc_x;
        const uint16_t noc_y = ((is_block_sharded && is_col_major) || is_width_sharded) ? my_noc_y : destination_noc_y;
        out_l1_addr = get_noc_addr(noc_x, noc_y, out_base_l1_addr);

        // Perform all transfers in this route
        while (transfers_remaining > 0) {
            src_offset = config[index++];
            dst_offset = config[index++];
            transfer_size = config[index++];
            DPRINT << "transfers rem " << transfers_remaining << " src_offset=" << src_offset
                   << " dst_offset=" << dst_offset << " transfer_size=" << transfer_size << ENDL();
            if (src_offset >= block_boundary_offset) {
                DPRINT << "new block! current blockid=" << block_id << " src_offset=" << src_offset
                       << " waiting for new tiles... " << ENDL();

                noc_async_read_barrier();
                noc_async_write_barrier();
                cb_pop_front(cb_id, block_size_width_tiles);

                cb_wait_front(cb_id, block_size_width_tiles);
                DPRINT << "got tiles for new block" << ENDL();
                block_id++;
                block_boundary_offset += block_height_sticks;
            }

            execute_transfer<stick_nbytes, input_aligned_page_size, block_height_sticks>(
                in_base_l1_addr, out_l1_addr, src_offset, dst_offset, transfer_size);

            transfers_remaining--;
        }
        number_of_segments_remaining--;
    }
    cb_pop_front(cb_id, block_size_width_tiles);
}

void kernel_main() {
    constexpr uint32_t padding_config_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_config_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t remote_config_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t blocking_local_config_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t blocking_remote_config_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(9);
    constexpr uint32_t in_nsticks = get_compile_time_arg_val(10);
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(11);
    constexpr bool is_block_sharded = get_compile_time_arg_val(12) == 1;
    constexpr bool remote_read = get_compile_time_arg_val(13) == 1;
    constexpr bool is_col_major = get_compile_time_arg_val(14) == 1;
    constexpr bool is_width_sharded = get_compile_time_arg_val(15) == 1;
    constexpr uint32_t input_aligned_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t block_size_width_tiles = get_compile_time_arg_val(17);

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);

    const uint16_t my_noc_x = NOC_X(my_x[noc_index]);
    const uint16_t my_noc_y = NOC_Y(my_y[noc_index]);

    const uint32_t in_base_l1_addr = get_read_ptr(in_cb_id);
    const uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);

    cb_reserve_back(src_cb_id, in_nsticks);
    cb_push_back(src_cb_id, in_nsticks);

    if constexpr (padding_config_cb_id) {
        cb_reserve_back(pad_cb_id, 1);
        const uint16_t pad_val = pad_val_u32;
        fill_with_val(get_write_ptr(pad_cb_id), stick_nbytes / elem_nbytes, pad_val);
        cb_push_back(pad_cb_id, 1);

        uint32_t padding_config_l1_addr = get_read_ptr(padding_config_cb_id);
        volatile tt_l1_ptr uint16_t* config_data =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);

        const uint64_t padding_l1_addr = get_noc_addr(my_noc_x, my_noc_y, get_read_ptr(pad_cb_id));
        const uint32_t dst_base_addr = out_base_l1_addr;

        uint16_t nsticks = 1;
        for (uint16_t j = 0; nsticks; j += 2) {
            uint16_t dst_local_idx = config_data[j + 0];
            nsticks = config_data[j + 1];
            uint64_t dst_addr = dst_base_addr + (dst_local_idx * stick_nbytes);

            for (uint16_t k = 0; k < nsticks; ++k) {
                noc_async_read(padding_l1_addr, dst_addr, stick_nbytes);
                dst_addr += stick_nbytes;
            }
        }
    }

    const uint32_t config_data_l1_addr = get_read_ptr(local_config_cb_id);
    const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
    execute_config<
        in_cb_id,
        stick_nbytes,
        input_aligned_page_size,
        block_size_width_tiles,
        is_block_sharded,
        is_width_sharded,
        is_col_major>(config_data, in_base_l1_addr, out_base_l1_addr, my_noc_x, my_noc_y);

    noc_async_read_barrier();
    noc_async_write_barrier();
    DPRINT << "done!" << ENDL();
}
