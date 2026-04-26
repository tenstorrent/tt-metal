// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reads border tiles from the local L1 shard into cb_data_in using NOC
 * reads addressed to this core's own L1. No cross-core NOC access.
 *
 * CT args:
 *   [0] W_tiles         – width of the shard in tiles (= full tensor width for HEIGHT_SHARDED)
 *   [1] has_right_pad   – 1 if tensor width % 32 != 0
 *   [2] elem_size       – 2 or 4 bytes per element
 *   [3] cb_data_in_idx  – CB index for data-in (= 0)
 *
 * RT args:
 *   [0] shard_l1_base_addr  – L1 base address of this core's shard buffer
 *   [1] shard_H_tiles       – active height tiles in this shard
 *   [2] has_bottom_pad_core – 1 if this is the last active shard and tensor has bottom padding
 *   [3] num_work            – 0 → no border tiles on this core; >0 → work to do
 *   [4] local_right_col     – local column index of the right-border tile within this shard
 *                             (= W_tiles-1 for fully-packed shards; may be smaller for the
 *                              rightmost shard when W_tiles_tensor % pages_per_shard_x != 0)
 *
 * Tile ordering matches fill_pad_compute.cpp exactly:
 *   Mode A (has_bottom_pad_core == 0):
 *     right column: (row, local_right_col) for row = 0..shard_H_tiles-1
 *   Mode B (has_bottom_pad_core == 1):
 *     right non-corner: (row, local_right_col) for row = 0..shard_H_tiles-2   [if has_right_pad]
 *     bottom row:       (shard_H_tiles-1, col) for col = 0..local_right_col
 */

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t W_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t has_right_pad = get_compile_time_arg_val(1);
    constexpr uint32_t elem_size = get_compile_time_arg_val(2);
    constexpr uint32_t cb_data_in_idx = get_compile_time_arg_val(3);

    constexpr uint32_t tile_bytes = get_tile_size(cb_data_in_idx);

    const uint32_t shard_l1_base = get_arg_val<uint32_t>(0);
    const uint32_t shard_H_tiles = get_arg_val<uint32_t>(1);
    const uint32_t has_bottom_pad_core = get_arg_val<uint32_t>(2);
    const uint32_t num_work = get_arg_val<uint32_t>(3);
    const uint32_t local_right_col = get_arg_val<uint32_t>(4);

    // get_noc_addr(addr) encodes the local core's physical NOC coordinates
    // automatically — no need to call my_x[]/my_y[] directly.

    constexpr uint32_t row_stride_bytes = W_tiles * tile_bytes;

    experimental::Noc noc;
    experimental::CircularBuffer cb_data_in(cb_data_in_idx);

    // Local-L1 self-read: no address-generator trait is applicable, so fall back
    // to raw noc_async_read(get_noc_addr(addr), ...). CB reservations and the
    // read barrier still go through the experimental API.

    if (has_bottom_pad_core) {
        // ---- Mode B: right non-corner tiles, then full bottom row ----

        // Right non-corner tiles: rows 0..shard_H_tiles-2, col local_right_col.
        // addr steps by row_stride_bytes each iter.
        if constexpr (has_right_pad) {
            uint32_t addr = shard_l1_base + local_right_col * tile_bytes;
            for (uint32_t r = 0; r < shard_H_tiles - 1u; r++) {
                cb_data_in.reserve_back(1);
                noc_async_read(get_noc_addr(addr), cb_data_in.get_write_ptr(), tile_bytes);
                noc.async_read_barrier();
                cb_data_in.push_back(1);
                addr += row_stride_bytes;
            }
        }

        // Bottom row: all valid columns (including corner at col local_right_col).
        // addr steps by tile_bytes each iter.
        {
            uint32_t addr = shard_l1_base + (shard_H_tiles - 1u) * row_stride_bytes;
            for (uint32_t c = 0; c <= local_right_col; c++) {
                cb_data_in.reserve_back(1);
                noc_async_read(get_noc_addr(addr), cb_data_in.get_write_ptr(), tile_bytes);
                noc.async_read_barrier();
                cb_data_in.push_back(1);
                addr += tile_bytes;
            }
        }

    } else {
        // ---- Mode A: right-column tiles only ----

        if constexpr (has_right_pad) {
            uint32_t addr = shard_l1_base + local_right_col * tile_bytes;
            for (uint32_t r = 0; r < shard_H_tiles; r++) {
                cb_data_in.reserve_back(1);
                noc_async_read(get_noc_addr(addr), cb_data_in.get_write_ptr(), tile_bytes);
                noc.async_read_barrier();
                cb_data_in.push_back(1);
                addr += row_stride_bytes;
            }
        }
    }
}
