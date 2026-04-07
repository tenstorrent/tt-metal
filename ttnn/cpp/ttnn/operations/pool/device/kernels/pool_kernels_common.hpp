// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This file contains common utilities for pool operation kernels (rotate, upsample, grid_sample).
// It is intended for use in device kernels only, not host code.

#pragma once

#include <api/dataflow/dataflow_api.h>
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define ALWI inline __attribute__((always_inline))

#define TILE_HEIGHT 32
#define TILE_WIDTH 32
#define FACE_WIDTH 16
#define FACE_HEIGHT 16
#define FACE_SIZE (FACE_WIDTH * FACE_HEIGHT)
#define FACES_PER_TILE_WIDTH (TILE_WIDTH / FACE_WIDTH)

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val, bool unconditionally = true) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    uint32_t value = val | (val << 16);
    if (ptr[0] != value || unconditionally) {
        // Process pairs of uint16_t as uint32_t for performance
        uint32_t num_pairs = n / 2;
        for (uint32_t i = 0; i < num_pairs; ++i) {
            ptr[i] = value;
        }
        // Handle odd case: write the final uint16_t
        if (n & 1) {
            volatile tt_l1_ptr uint16_t* ptr16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
            ptr16[n - 1] = val;
        }
    }

    return true;
}

template <uint32_t cb_id, uint32_t clear_value_cb_id>
ALWI void clear_out_tiles(experimental::Noc noc, experimental::CB cb, experimental::CB clear_cb) {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    const uint32_t num_pages = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_page_size / tile_size;

    experimental::UnicastEndpoint self_ep;
    const auto src = experimental::local_addr(clear_cb.get_read_ptr());

    for (uint32_t i = 0; i < num_tiles * num_pages; ++i) {
        noc.async_read(self_ep, cb, tile_size, src, {.offset_bytes = i * tile_size});
    }
    noc.async_read_barrier();
}

template <uint32_t clear_value_cb_id>
ALWI void clear_out_tiles(
    experimental::Noc noc, experimental::CB dst_cb, experimental::CB clear_value_cb, uint32_t num_tiles) {
    constexpr uint32_t tile_size = get_tile_size(clear_value_cb_id);

    experimental::UnicastEndpoint self_ep;
    const auto src = experimental::local_addr(clear_value_cb.get_read_ptr());

    for (uint32_t i = 0; i < num_tiles; ++i) {
        noc.async_read(self_ep, dst_cb, tile_size, src, {.offset_bytes = i * tile_size});
    }
    noc.async_read_barrier();
}

// Zero out all tiles for a given circular buffer.
template <uint32_t cb_id>
ALWI void zero_out_tiles(experimental::Noc noc, experimental::CB cb) {
    constexpr uint32_t tile_size = get_tile_size(cb_id);
    const uint32_t num_tiles = get_local_cb_interface(cb_id).fifo_num_pages;
    const uint32_t num_zeros_reads = (tile_size / MEM_ZEROS_SIZE) * num_tiles;

    constexpr uint32_t packet_size = MEM_ZEROS_SIZE;

    experimental::set_read_state<packet_size>(noc, MEM_ZEROS_BASE);

    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        experimental::read_with_state(noc, cb, MEM_ZEROS_BASE, {.offset_bytes = i * packet_size});
    }
    noc.async_read_barrier();
}

template <uint32_t config_dram_addr, uint32_t config_page_size, uint32_t tensor_args_index, uint32_t cb_reader_index>
ALWI void load_config_tensor_if_in_dram(experimental::Noc noc, experimental::CB reader_cb, uint32_t core_index) {
    constexpr auto config_tensor_args = TensorAccessorArgs<tensor_args_index>();
    const auto config_accessor = TensorAccessor(config_tensor_args, config_dram_addr, config_page_size);

    noc.async_read(config_accessor, reader_cb, config_page_size, {.page_id = core_index}, {});
    noc.async_read_barrier();
    reader_cb.push_back(1);
}

template <
    bool one_scalar_per_core,
    uint32_t in_scalar_cb_id,
    uint32_t reader_nindices,
    bool split_reader,
    uint32_t multi_buffering_factor>
ALWI void fill_scalar(
    experimental::CB scalar_cb,
    uint32_t& scalar_start,
    uint32_t& scalar_end,
    uint32_t& scalar_value,
    uint32_t& scalar_index,
    uint32_t& counter,
    volatile uint16_t* config_ptr) {
    constexpr uint32_t num_readers = split_reader ? 2 : 1;
    scalar_cb.reserve_back(1);

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
        fill_with_val(scalar_cb.get_write_ptr(), FACE_WIDTH, scalar_value, false);
    }
    counter += num_readers;

    scalar_cb.push_back(1);
}

template <uint32_t cb_id>
ALWI void zero_out_page(experimental::Noc noc, experimental::CB cb) {
    const uint32_t page_size = get_local_cb_interface(cb_id).fifo_page_size;
    const uint32_t num_zeros_reads = page_size / MEM_ZEROS_SIZE;
    const uint32_t remainder_bytes = page_size % MEM_ZEROS_SIZE;
    constexpr uint32_t packet_size = MEM_ZEROS_SIZE;

    experimental::set_read_state<packet_size>(noc, MEM_ZEROS_BASE);
    for (uint32_t i = 0; i < num_zeros_reads; ++i) {
        experimental::read_with_state(noc, cb, MEM_ZEROS_BASE, {.offset_bytes = i * packet_size});
    }
    if (remainder_bytes > 0) {
        experimental::UnicastEndpoint self_ep;
        noc.async_read(
            self_ep,
            cb,
            remainder_bytes,
            experimental::local_addr(MEM_ZEROS_BASE),
            {.offset_bytes = num_zeros_reads * packet_size});
    }
}
