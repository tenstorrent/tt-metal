// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"
#include <tt-metalium/constants.hpp>

template <uint32_t tile_bytes, uint32_t num_readers>
constexpr uint32_t get_barrier_read_threshold() {
    return ((512 / num_readers) * (1024 + 128)) / tile_bytes;
}

void fill_zeros_async(uint32_t write_addr, uint32_t tile_bytes) {
    // volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);
    uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    // Fill tile with zeros
    uint32_t bytes_left = tile_bytes;
    for (;;) {
        uint32_t read_size = bytes_left > MEM_ZEROS_SIZE ? MEM_ZEROS_SIZE : bytes_left;
        noc_async_read(zeros_noc_addr, write_addr, read_size);
        write_addr += read_size;
        bytes_left -= read_size;
        if (bytes_left == 0) {
            break;
        }
    }
}

template <uint32_t tile_bytes>
void fill_tile_zeros(uint32_t cb_id, uint32_t tile_id) {
    static_assert(tile_bytes % 4 == 0, "tile_bytes must be a multiple of 4");
    uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
    fill_zeros_async(write_addr, tile_bytes);
}

class TensorTileShape {
public:
    uint32_t shape[4];
    uint32_t strides[4];
    // Constructor to initialize with 4D shape
    TensorTileShape(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
        shape[0] = d0;
        shape[1] = d1;
        shape[2] = d2;
        shape[3] = d3;

        // Calculate strides (row-major order)
        strides[3] = 1;
        strides[2] = strides[3] * shape[3];
        strides[1] = strides[2] * shape[2];
        strides[0] = strides[1] * shape[1];
    }

    // Get flattened index from 4D coordinates
    uint32_t id_of(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) const {
        return i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
    }
};

template <bool is_dram = true, uint32_t tile_bytes>
uint32_t async_read_chunk_with_padding(
    const InterleavedAddrGenFast<is_dram>& reader,
    const uint32_t cb_id,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t barrier_threshold,
    const bool transpose = false) {
    /*
    Method always reads tiles from memory in row-major order.
    It assumes that the block of rows x cols in stored in contiguous tile order.
    That means, it won't work if the chunk to read is a slice of the last dimension.

    This handles the case where the dst CB is larger than the src CB, with some padding on the
    rows or cols of the DST CB.
    */
    // Read Q chunk
    const uint32_t num_tiles = dst_rows * dst_cols;
    cb_reserve_back(cb_id, num_tiles);
    const uint32_t base_write_ptr = get_write_ptr(cb_id);
    uint32_t outer_ptr_stride = transpose ? tile_bytes : dst_cols * tile_bytes;
    uint32_t inner_ptr_stride = transpose ? tile_bytes * dst_rows : tile_bytes;

    uint32_t barrier_count = 0;
    for (uint32_t row = 0; row < src_rows; ++row) {
        uint32_t write_ptr = base_write_ptr + row * outer_ptr_stride;
        for (uint32_t col = 0; col < src_cols; ++col) {
            noc_async_read_tile(start_tile_id, reader, write_ptr);
            start_tile_id += 1;
            write_ptr += inner_ptr_stride;

            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
    }

    // Zero out the padding
    for (uint32_t row = 0; row < dst_rows; ++row) {
        for (uint32_t col = 0; col < dst_cols; ++col) {
            if (row < src_rows && col < src_cols) {
                continue;
            }
            uint32_t tile_id = transpose ? col * dst_rows + row : row * dst_cols + col;
            fill_tile_zeros<tile_bytes>(cb_id, tile_id);
        }
    }

    return num_tiles;
}

template <bool is_dram = true, uint32_t tile_bytes>
void read_chunk_with_padding(
    const InterleavedAddrGenFast<is_dram>& reader,
    const uint32_t cb_id,
    uint32_t start_tile_id,
    const uint32_t src_rows,
    const uint32_t src_cols,
    const uint32_t dst_rows,
    const uint32_t dst_cols,
    const uint32_t barrier_threshold,
    const bool transpose = false) {
    auto num_tiles = async_read_chunk_with_padding<is_dram, tile_bytes>(
        reader, cb_id, start_tile_id, src_rows, src_cols, dst_rows, dst_cols, barrier_threshold, transpose);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_tiles);
}

template <uint32_t tile_bytes>
void copy_tile(uint64_t noc_read_addr_base, uint32_t q_write_ptr_base, uint32_t src_tile_id, uint32_t dst_tile_id) {
    noc_async_read(
        noc_read_addr_base + src_tile_id * tile_bytes, q_write_ptr_base + dst_tile_id * tile_bytes, tile_bytes);
}

template <uint32_t tile_bytes>
void fill_neginf_tile_bfp4(uint32_t cb_id, uint32_t tile_id) {
    constexpr uint32_t num_exponents = tt::constants::FACE_HEIGHT * (tt::constants::TILE_HW / tt::constants::FACE_HW);
    constexpr uint32_t num_mantissas = tt::constants::TILE_HW / 2;
    static_assert(
        tile_bytes == num_exponents + num_mantissas, "tile_bytes must be equal to bfp4 num_exponents + num_mantissas");

    uint32_t write_addr = get_write_ptr(cb_id) + tile_id * tile_bytes;
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

    // Fill the first 64 bytes (16 uint32_t values) with 0xFFFFFFFF for exponents
    constexpr uint32_t NEG_INF_EXP = 0xFFFFFFFF;
    constexpr uint32_t exp_words = num_exponents / sizeof(uint32_t);  // 16 words

    for (uint32_t i = 0; i < exp_words; i++) {
        ptr[i] = NEG_INF_EXP;
    }

    // Fill the next 512 bytes (128 uint32_t values) with 0xCCCCCCCC for mantissas
    constexpr uint32_t NEG_INF_MANT = 0xCCCCCCCC;
    constexpr uint32_t mant_words = num_mantissas / sizeof(uint32_t);  // 128 words

    for (uint32_t i = exp_words; i < exp_words + mant_words; i++) {
        ptr[i] = NEG_INF_MANT;
    }
}
