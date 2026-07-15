// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

// Gather one out-block of a ROW_MAJOR interleaved DRAM tensor into `cb`, row by row, so the
// compute kernel can tilize it on-core. Shared by the TILIZE_IN input read and the UNTILIZE_OUT
// output re-read in the legacy mcast sender/receiver readers, and by the Welford readers.
//
// Two calling conventions:
//  - legacy: one call per group -- `block_w` is the group's tile span and `index_g_offset` its
//    column offset; the last group can run past the channels, so columns are clamped.
//  - Welford: one call for the whole per-core batch -- `block_w` is per_core_N and
//    `index_g_offset` is 0; the clamp is then inert because per_core_N <= num_channels_tiles.
//
// tile_width / tile_height / block_w / datum_size_bytes are compile-time so row_chunk_bytes stays
// constexpr. `accessor` and `base_start_id` select the source tensor (input vs previous output).
template <uint32_t tile_width, uint32_t tile_height, uint32_t block_w, uint32_t datum_size_bytes, typename AccessorT>
void groupnorm_gather_rm_block(
    Noc& noc,
    const AccessorT& accessor,
    CircularBuffer& cb,
    uint32_t base_start_id,
    uint32_t out_block_start_id_offset,
    uint32_t index_b_offset,
    uint32_t index_g_offset,
    uint32_t num_channels_tiles,
    uint32_t out_block_h_actual,
    uint32_t out_block_hw_normal) {
    constexpr uint32_t row_chunk_bytes = tile_width * datum_size_bytes;
    uint32_t l1_write_addr = cb.get_write_ptr();
    cb.reserve_back(out_block_hw_normal);
    for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
        for (uint32_t r = 0; r < tile_height; r++) {
            for (uint32_t nt = 0; nt < block_w; nt++) {
                // Clamp out-of-range columns (last group) to the last valid column (masked out
                // downstream); avoids a NOC address-overflow fault on L1-interleaved buffers.
                const uint32_t abs_col = index_g_offset + nt;
                const uint32_t col = abs_col < num_channels_tiles ? abs_col : num_channels_tiles - 1;
                const uint32_t page_id_tile =
                    base_start_id + out_block_start_id_offset + (mt * num_channels_tiles) + index_b_offset + col;
                const uint32_t tile_row = page_id_tile / num_channels_tiles;
                const uint32_t tile_col = page_id_tile % num_channels_tiles;
                const uint32_t rm_row = (tile_row * tile_height) + r;
                const uint32_t col_off_bytes = tile_col * row_chunk_bytes;
                noc.async_read(
                    accessor,
                    CoreLocalMem<uint32_t>(l1_write_addr),
                    row_chunk_bytes,
                    {.page_id = rm_row, .offset_bytes = col_off_bytes},
                    {});
                l1_write_addr += row_chunk_bytes;
            }
        }
        noc.async_read_barrier();
    }
    cb.push_back(out_block_hw_normal);
}
