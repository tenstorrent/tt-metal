// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common utilities for pool operation kernels (rotate, upsample, grid_sample).
// It is intended for use in device kernels only, not host code.

#pragma once

#include <api/dataflow/dataflow_api.h>

#define ALWI inline __attribute__((always_inline))

#define TILE_HEIGHT 32
#define TILE_WIDTH 32
#define FACE_WIDTH 16
#define FACE_HEIGHT 16
#define FACE_SIZE (FACE_WIDTH * FACE_HEIGHT)
#define FACES_PER_TILE_WIDTH (TILE_WIDTH / FACE_WIDTH)

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
// WARNING: This function assumes n is even
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val, bool unconditionally = true) {
    // simplest impl:
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    uint32_t value = val | (val << 16);
    if (ptr[0] != value || unconditionally) {
        for (uint32_t i = 0; i < n / 2; ++i) {
            ptr[i] = (value);
        }
    }

    return true;
}

template <uint32_t cb_id, uint32_t clear_value_cb_id>
ALWI void clear_out_tiles() {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    const uint32_t num_pages = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_page_size / tile_size;
    const uint64_t clear_value_addr = get_noc_addr(get_read_ptr(clear_value_cb_id));
    uint64_t write_addr = get_noc_addr(get_write_ptr(cb_id));

    for (uint32_t i = 0; i < num_tiles * num_pages; ++i) {
        noc_async_read(clear_value_addr, write_addr, tile_size);
        write_addr += tile_size;
    }
    noc_async_read_barrier();
}

template <uint32_t clear_value_cb_id, uint32_t num_tiles>
ALWI void clear_out_tiles(uint64_t write_addr, uint64_t clear_value_addr) {
    constexpr uint32_t tile_size = get_tile_size(clear_value_cb_id);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc_async_read(clear_value_addr, write_addr, tile_size);
        write_addr += tile_size;
    }
    noc_async_read_barrier();
}

template <uint32_t config_dram_addr, uint32_t config_page_size, uint32_t tensor_args_index, uint32_t cb_reader_index>
ALWI void load_config_tensor_if_in_dram(uint32_t core_index) {
    // TODO: Instead of all cores reading from dram, only the first column reads, and does an MCAST to all the other
    // cores in the row.
    constexpr auto config_tensor_args = TensorAccessorArgs<tensor_args_index>();
    const auto config_accessor = TensorAccessor(config_tensor_args, config_dram_addr, config_page_size);
    uint64_t src_noc_addr = get_noc_addr(core_index, config_accessor);

    noc_async_read(src_noc_addr, get_write_ptr(cb_reader_index), config_page_size);
    noc_async_read_barrier();
    cb_push_back(cb_reader_index, 1);
}

template <
    bool one_scalar_per_core,
    uint32_t in_scalar_cb_id,
    uint32_t reader_nindices,
    bool split_reader,
    uint32_t multi_buffering_factor>
ALWI void fill_scalar(
    uint32_t& scalar_start,
    uint32_t& scalar_end,
    uint32_t& scalar_value,
    uint32_t& scalar_index,
    uint32_t& counter,
    volatile uint16_t* config_ptr) {
    constexpr uint32_t num_readers = split_reader ? 2 : 1;
    cb_reserve_back(in_scalar_cb_id, 1);

    while (counter >= scalar_end && scalar_end < reader_nindices) {
        scalar_index++;
        scalar_start = scalar_end;
        scalar_value = config_ptr[3 * scalar_index + 1];
        scalar_end = config_ptr[3 * scalar_index + 2];
    }

    // We want to fill the scalar CB the fewest times possible, this will be min(scalar_end - scalar_start, num_readers
    // * multi_buffering_factor)
    if (counter < scalar_start + num_readers * multi_buffering_factor) {
        // Fill only the first FACE_WIDTH, since we set reload_srcB = true in unpack_tilizeA_B_block, meaning the values
        // for the remaining faces will be reused from the first one. This is safe here because there’s no difference
        // between the first and second face.
        fill_with_val(get_write_ptr(in_scalar_cb_id), FACE_WIDTH, scalar_value, false);
    }
    counter += num_readers;

    cb_push_back(in_scalar_cb_id, 1);
}

template <uint32_t cb_id>
ALWI void zero_out_page(uint32_t write_addr) {
    const uint32_t page_size = get_local_cb_interface(cb_id).fifo_page_size;
    const uint32_t num_zeros_reads = page_size / MEM_ZEROS_SIZE;
    const uint32_t remainder_bytes = page_size % MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    if (remainder_bytes > 0) {
        noc_async_read(zeros_noc_addr, write_addr, remainder_bytes);
    }
}

ALWI void zero_out_nbytes(uint32_t write_addr, uint32_t nbytes) {
    const uint32_t num_zeros_reads = nbytes / MEM_ZEROS_SIZE;
    const uint32_t remainder_bytes = nbytes % MEM_ZEROS_SIZE;
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);

    noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        noc_async_read_one_packet_with_state<true>(zeros_noc_addr, write_addr);
        write_addr += MEM_ZEROS_SIZE;
    }
    if (remainder_bytes > 0) {
        noc_async_read(zeros_noc_addr, write_addr, remainder_bytes);
    }
}
