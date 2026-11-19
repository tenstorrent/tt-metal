// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "named_args.hpp"

constexpr uint32_t BF16_ONE_PAIR = 0x3F803F80;
constexpr uint32_t FACE_BYTES = 512;
constexpr uint32_t FACE_ROW_BYTES = 32;

inline void fill_reduce_scaler(const Noc& noc, uint32_t cb_id, uint32_t tile_bytes) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(1);
    noc.async_write_zeros(cb, tile_bytes);
    noc.write_zeros_l1_barrier();
    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
    for (uint32_t face = 0; face < 4; face++) {
        for (uint32_t i = 0; i < FACE_ROW_BYTES / sizeof(uint32_t); i++) {
            tile[face * (FACE_BYTES / sizeof(uint32_t)) + i] = BF16_ONE_PAIR;
        }
    }
    cb.push_back(1);
}

// Only row 0 of the first two faces of a norm weight tile is read by the compute kernel, so a tile's weight is kept as
// two face rows, each at the 64-byte alignment of its DRAM source.
constexpr uint32_t CACHED_FACE_ROW_STRIDE = 64;
constexpr uint32_t WEIGHT_TILE_BYTES = 2 * CACHED_FACE_ROW_STRIDE;

template <typename WeightAccessor>
void load_norm_weights(const Noc& noc, const WeightAccessor& weight, uint32_t cache_cb_id, uint32_t row_tiles) {
    const uint32_t cache = CircularBuffer(cache_cb_id).get_write_ptr();
    for (uint32_t t = 0; t < row_tiles; t++) {
        for (uint32_t face = 0; face < 2; face++) {
            noc.async_read(
                weight,
                CoreLocalMem<uint32_t>(cache + t * WEIGHT_TILE_BYTES + face * CACHED_FACE_ROW_STRIDE),
                FACE_ROW_BYTES,
                {.page_id = t, .offset_bytes = face * FACE_BYTES},
                {});
        }
    }
    noc.async_read_barrier();
}

inline void push_norm_weights(
    uint32_t cache_cb_id, uint32_t cb_weight_id, uint32_t first_tile, uint32_t tiles, uint32_t tile_bytes) {
    CircularBuffer cb_weight(cb_weight_id);
    const uint32_t cache = CircularBuffer(cache_cb_id).get_write_ptr();
    cb_weight.reserve_back(tiles);
    const uint32_t base = cb_weight.get_write_ptr();
    for (uint32_t t = 0; t < tiles; t++) {
        for (uint32_t face = 0; face < 2; face++) {
            volatile tt_l1_ptr uint32_t* dst =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * tile_bytes + face * FACE_BYTES);
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                cache + (first_tile + t) * WEIGHT_TILE_BYTES + face * CACHED_FACE_ROW_STRIDE);
            for (uint32_t i = 0; i < FACE_ROW_BYTES / sizeof(uint32_t); i++) {
                dst[i] = src[i];
            }
        }
    }
    cb_weight.push_back(tiles);
}

template <typename HAccessor>
void read_h_row(
    const Noc& noc,
    const HAccessor& h,
    uint32_t cb_id,
    uint32_t tile_row,
    uint32_t M_logical,
    uint32_t row_tiles,
    uint32_t k_tile,
    uint32_t tiles,
    uint32_t tile_bytes) {
    CircularBuffer cb(cb_id);
    cb.reserve_back(tiles);
    if (tile_row < M_logical) {
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < tiles; t++) {
            noc.async_read(
                h,
                CoreLocalMem<uint32_t>(base + t * tile_bytes),
                tile_bytes,
                {.page_id = tile_row * row_tiles + k_tile + t},
                {});
        }
        noc.async_read_barrier();
    }
    cb.push_back(tiles);
}
