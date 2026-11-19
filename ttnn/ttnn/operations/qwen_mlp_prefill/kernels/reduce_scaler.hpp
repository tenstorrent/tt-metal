// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

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
