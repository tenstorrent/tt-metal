// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

namespace sharded_rmsnorm_detail {

// Copies `bytes` (a multiple of 4) between local L1 addresses.
inline void copy_l1_words(uint32_t dst_addr, uint32_t src_addr, uint32_t bytes) {
    auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
    const auto* src = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src_addr);
    for (uint32_t i = 0; i < bytes / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

// Scatters 1x32 row tiles [first_tile, first_tile + num_tiles) from cb_row into row 0 of
// zeroed 32x32 output tiles in cb_tiles (at tiles_addr). Safe when num_tiles == 0.
inline void scatter_row_to_tiles(
    uint32_t cb_row,
    uint32_t w_tiles,
    uint32_t cb_tiles,
    uint32_t tiles_addr,
    uint32_t first_tile,
    uint32_t num_tiles) {
    if (num_tiles == 0) {
        return;
    }
    constexpr uint32_t kTileBytes = 32 * 32 * sizeof(uint16_t);
    constexpr uint32_t kFaceRowBytes = 16 * sizeof(uint16_t);
    constexpr uint32_t kFaceBytes = 16 * kFaceRowBytes;

    Noc noc;
    CircularBuffer cb_out_tiles(cb_tiles);
    noc.async_write_zeros(cb_out_tiles, num_tiles * kTileBytes, {.offset_bytes = first_tile * kTileBytes});
    noc.write_zeros_l1_barrier();

    CircularBuffer cb_y(cb_row);
    cb_y.wait_front(w_tiles);
    const uint32_t row = cb_y.get_read_ptr();
    noc_async_write_one_packet_set_state(get_noc_addr(tiles_addr), kFaceRowBytes, noc_index);
    for (uint32_t n = first_tile; n < first_tile + num_tiles; ++n) {
        const uint32_t src = row + n * 2 * kFaceRowBytes;
        const uint32_t dst = tiles_addr + n * kTileBytes;
        noc_async_write_one_packet_with_state(src, dst, noc_index);
        noc_async_write_one_packet_with_state(src + kFaceRowBytes, dst + kFaceBytes, noc_index);
    }
    noc_async_write_barrier(noc_index);
}

}  // namespace sharded_rmsnorm_detail
