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
    uint32_t cache_cb_id, uint32_t cb_weight_id, uint32_t row_tiles, uint32_t group, uint32_t tile_bytes) {
    CircularBuffer cb_weight(cb_weight_id);
    const uint32_t cache = CircularBuffer(cache_cb_id).get_write_ptr();
    for (uint32_t first = 0; first < row_tiles; first += group) {
        cb_weight.reserve_back(group);
        const uint32_t base = cb_weight.get_write_ptr();
        for (uint32_t t = 0; t < group; t++) {
            for (uint32_t face = 0; face < 2; face++) {
                volatile tt_l1_ptr uint32_t* dst =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + t * tile_bytes + face * FACE_BYTES);
                volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    cache + (first + t) * WEIGHT_TILE_BYTES + face * CACHED_FACE_ROW_STRIDE);
                for (uint32_t i = 0; i < FACE_ROW_BYTES / sizeof(uint32_t); i++) {
                    dst[i] = src[i];
                }
            }
        }
        cb_weight.push_back(group);
    }
}

template <typename HAccessor>
void read_norm_rows(
    const Noc& noc,
    const HAccessor& h,
    uint32_t cb_in_id,
    uint32_t row_tiles,
    uint32_t chunk,
    uint32_t tile_bytes,
    uint32_t first_row,
    uint32_t row_stride,
    uint32_t rows,
    uint32_t cache_cb_id,
    uint32_t cb_weight_id,
    uint32_t group) {
    CircularBuffer cb_in(cb_in_id);
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t row_first_tile = (first_row + r * row_stride) * row_tiles;
        cb_in.reserve_back(row_tiles);
        const uint32_t base = cb_in.get_write_ptr();
        for (uint32_t first = 0; first < row_tiles; first += chunk) {
            for (uint32_t t = first; t < first + chunk; t++) {
                noc.async_read(
                    h, CoreLocalMem<uint32_t>(base + t * tile_bytes), tile_bytes, {.page_id = row_first_tile + t}, {});
            }
            noc.async_read_barrier();
            cb_in.push_back(chunk);
        }
        push_norm_weights(cache_cb_id, cb_weight_id, row_tiles, group, tile_bytes);
    }
}

template <typename Accessor>
void write_norm_rows(
    const Noc& noc,
    const Accessor& n,
    uint32_t cb_id,
    uint32_t row_tiles,
    uint32_t group,
    uint32_t tile_bytes,
    uint32_t first_row,
    uint32_t row_stride,
    uint32_t rows) {
    CircularBuffer cb(cb_id);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t first = 0; first < row_tiles; first += group) {
            cb.wait_front(group);
            const uint32_t base = cb.get_read_ptr();
            for (uint32_t t = 0; t < group; t++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(base + t * tile_bytes),
                    n,
                    tile_bytes,
                    {},
                    {.page_id = (first_row + r * row_stride) * row_tiles + first + t});
            }
            noc.async_writes_flushed();
            cb.pop_front(group);
        }
    }
}
