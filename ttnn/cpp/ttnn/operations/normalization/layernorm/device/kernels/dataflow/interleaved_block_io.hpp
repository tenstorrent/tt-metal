// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Block transfers between an interleaved tile tensor and one core's width block of a width split.
// The block is block_ht tile rows of block_w tiles, starting at tile column width_start of a tensor
// Kt tiles wide; only the first num_valid_tiles of each row exist in the tensor (the last block can
// overhang the width), and the rest of the buffer is left as it is.

// Issues the reads of the block into `dfb` without waiting for them; the caller barriers and pushes.
template <typename Accessor, typename DFB>
inline void issue_block_reads(
    const Noc& noc,
    const Accessor& src,
    DFB& dfb,
    uint32_t block_ht,
    uint32_t block_w,
    uint32_t Kt,
    uint32_t width_start,
    uint32_t num_valid_tiles) {
    const uint32_t tile_bytes = dfb.get_tile_size();
    dfb.reserve_back(static_cast<uint16_t>(block_ht * block_w));
    for (uint32_t h = 0; h < block_ht; ++h) {
        for (uint32_t w = 0; w < num_valid_tiles; ++w) {
            noc.async_read(
                src,
                dfb,
                tile_bytes,
                {.page_id = h * Kt + width_start + w},
                {.offset_bytes = (h * block_w + w) * tile_bytes});
        }
    }
}

template <typename Accessor, typename DFB>
inline void write_block(
    const Noc& noc,
    const Accessor& dst,
    DFB& dfb,
    uint32_t block_ht,
    uint32_t block_w,
    uint32_t Kt,
    uint32_t width_start,
    uint32_t num_valid_tiles) {
    const uint32_t tile_bytes = dfb.get_tile_size();
    const uint16_t num_tiles = static_cast<uint16_t>(block_ht * block_w);
    dfb.wait_front(num_tiles);
    for (uint32_t h = 0; h < block_ht; ++h) {
        for (uint32_t w = 0; w < num_valid_tiles; ++w) {
            noc.async_write(
                dfb,
                dst,
                tile_bytes,
                {.offset_bytes = (h * block_w + w) * tile_bytes},
                {.page_id = h * Kt + width_start + w});
        }
    }
    noc.async_write_barrier();
    dfb.pop_front(num_tiles);
}
