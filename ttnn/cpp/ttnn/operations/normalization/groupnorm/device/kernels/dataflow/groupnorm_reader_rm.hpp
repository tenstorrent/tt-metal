// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

// Read one out-block of a ROW_MAJOR tensor into `dfb` row by row, for the compute kernel to tilize.
template <uint32_t tile_width, uint32_t tile_height, uint32_t block_w, uint32_t datum_size_bytes, typename AccessorT>
void groupnorm_gather_rm_block(
    Noc& noc,
    const AccessorT& accessor,
    DataflowBuffer& dfb,
    uint32_t base_start_id,
    uint32_t out_block_start_id_offset,
    uint32_t index_b_offset,
    uint32_t index_g_offset,
    uint32_t num_channels_tiles,
    uint32_t out_block_h_actual,
    uint32_t out_block_hw_normal) {
    constexpr uint32_t row_chunk_bytes = tile_width * datum_size_bytes;
    uint32_t l1_write_addr = dfb.get_write_ptr();
    dfb.reserve_back(out_block_hw_normal);
    for (uint32_t mt = 0; mt < out_block_h_actual; mt++) {
        for (uint32_t r = 0; r < tile_height; r++) {
            for (uint32_t nt = 0; nt < block_w; nt++) {
                // Clamp past-the-end columns in the last group; they get masked out later anyway.
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
    dfb.push_back(out_block_hw_normal);
}
