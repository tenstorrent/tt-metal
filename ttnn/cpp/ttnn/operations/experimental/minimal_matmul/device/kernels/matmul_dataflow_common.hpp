// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>
#include "dataflow_api.h"
#include <tt-metalium/constants.hpp>

struct TensorShape2D {
    uint32_t d0;
    uint32_t d1;
    // Constructor to initialize with 2D shape
    TensorShape2D(uint32_t _d0, uint32_t _d1) : d0(_d0), d1(_d1) {}
};

template <typename TensorAccessorType>
void read_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t write_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    for (uint32_t i = d0_start; i < d0_end; i++) {
        for (uint32_t j = d1_start; j < d1_end; j++) {
            uint32_t tile_id = i * shape.d1 + j;
            noc_async_read_tile(tile_id, tensor_accessor, write_ptr);
            write_ptr += tile_size_bytes;
        }
    }
    noc_async_read_barrier();
}

template <typename TensorAccessorType>
void write_block_sync(
    const TensorAccessorType& tensor_accessor,
    const TensorShape2D& shape,
    uint32_t read_ptr,
    uint32_t tile_size_bytes,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    for (uint32_t i = d0_start; i < d0_end; i++) {
        for (uint32_t j = d1_start; j < d1_end; j++) {
            uint32_t tile_id = i * shape.d1 + j;
            noc_async_write_tile(tile_id, tensor_accessor, read_ptr);
            read_ptr += tile_size_bytes;
        }
    }
    noc_async_writes_flushed();
}
