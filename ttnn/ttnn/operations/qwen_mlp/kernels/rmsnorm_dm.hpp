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

template <typename HAccessor, typename WeightAccessor>
void read_norm_rows(
    const Noc& noc,
    const HAccessor& h,
    const WeightAccessor& weight,
    uint32_t cb_in_id,
    uint32_t cb_weight_id,
    uint32_t row_tiles,
    uint32_t chunk,
    uint32_t tile_bytes,
    uint32_t first_row,
    uint32_t row_stride,
    uint32_t rows) {
    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_weight(cb_weight_id);
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t row_first_tile = (first_row + r * row_stride) * row_tiles;
        for (uint32_t pass = 0; pass < 2; pass++) {
            for (uint32_t first = 0; first < row_tiles; first += chunk) {
                cb_in.reserve_back(chunk);
                const uint32_t in_base = cb_in.get_write_ptr();
                for (uint32_t t = 0; t < chunk; t++) {
                    noc.async_read(
                        h,
                        CoreLocalMem<uint32_t>(in_base + t * tile_bytes),
                        tile_bytes,
                        {.page_id = row_first_tile + first + t},
                        {});
                }
                if (pass == 1) {
                    cb_weight.reserve_back(chunk);
                    const uint32_t weight_base = cb_weight.get_write_ptr();
                    for (uint32_t t = 0; t < chunk; t++) {
                        for (uint32_t face = 0; face < 2; face++) {
                            noc.async_read(
                                weight,
                                CoreLocalMem<uint32_t>(weight_base + t * tile_bytes + face * FACE_BYTES),
                                FACE_ROW_BYTES,
                                {.page_id = first + t, .offset_bytes = face * FACE_BYTES},
                                {});
                        }
                    }
                }
                noc.async_read_barrier();
                cb_in.push_back(chunk);
                if (pass == 1) {
                    cb_weight.push_back(chunk);
                }
            }
        }
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
