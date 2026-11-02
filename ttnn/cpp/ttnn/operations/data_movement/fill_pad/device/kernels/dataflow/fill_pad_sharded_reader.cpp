// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reads border tiles from the local L1 shard into dfb::data_in using NOC
 * reads addressed to this core's own L1. No cross-core NOC access.
 *
 * Metal 2.0 named resources:
 *   CTAs:  W_tiles (shard width in tiles), has_right_pad, elem_size (elem_size unused).
 *   DFB:   dfb::data_in (this reader is its PRODUCER).
 *   tensor: tensor::src — this core's L1 shard, read via a LocalTensorAccessor as the
 *           loopback NOC source (byte offsets into the shard are unchanged).
 *   RTAs:  shard_H_tiles, has_bottom_pad_core, num_work (num_work is inert in the reader),
 *          local_right_col.
 *
 * Tile ordering matches fill_pad_compute.cpp exactly:
 *   Mode A (has_bottom_pad_core == 0):
 *     right column: (row, local_right_col) for row = 0..shard_H_tiles-1
 *   Mode B (has_bottom_pad_core == 1):
 *     right non-corner: (row, local_right_col) for row = 0..shard_H_tiles-2   [if has_right_pad]
 *     bottom row:       (shard_H_tiles-1, col) for col = 0..local_right_col
 */

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);
    constexpr auto has_right_pad = get_arg(args::has_right_pad);
    [[maybe_unused]] constexpr auto elem_size = get_arg(args::elem_size);

    const auto shard_H_tiles = get_arg(args::shard_H_tiles);
    const auto has_bottom_pad_core = get_arg(args::has_bottom_pad_core);
    [[maybe_unused]] const auto num_work = get_arg(args::num_work);
    const auto local_right_col = get_arg(args::local_right_col);

    // This core's shard; the self-read byte-offset arithmetic below is unchanged from the legacy kernel.
    const LocalTensorAccessor<std::uint32_t> s(tensor::src);

    Noc noc;
    DataflowBuffer dfb_data_in(dfb::data_in);
    const std::uint32_t tile_bytes = dfb_data_in.get_entry_size();

    // The LocalTensorAccessor as the NOC read source is a loopback to this core's
    // own physical NOC coordinates (my_x[]/my_y[]), so each read targets local L1.

    const std::uint32_t row_stride_bytes = W_tiles * tile_bytes;

    // Local-L1 self-read via the LocalTensorAccessor endpoint at a byte offset into the
    // shard. CB reservations and the read barrier use the Device 2.0 API.

    if (has_bottom_pad_core) {
        // ---- Mode B: right non-corner tiles, then full bottom row ----

        // Right non-corner tiles: rows 0..shard_H_tiles-2, col local_right_col.
        // offset steps by row_stride_bytes each iter.
        if constexpr (has_right_pad) {
            std::uint32_t offset = local_right_col * tile_bytes;
            for (std::uint32_t r = 0; r < shard_H_tiles - 1u; r++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(s, dfb_data_in, tile_bytes, {.offset_bytes = offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_data_in.push_back(1);
                offset += row_stride_bytes;
            }
        }

        // Bottom row: all valid columns (including corner at col local_right_col).
        // offset steps by tile_bytes each iter.
        {
            std::uint32_t offset = (shard_H_tiles - 1u) * row_stride_bytes;
            for (std::uint32_t c = 0; c <= local_right_col; c++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(s, dfb_data_in, tile_bytes, {.offset_bytes = offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_data_in.push_back(1);
                offset += tile_bytes;
            }
        }

    } else {
        // ---- Mode A: right-column tiles only ----

        if constexpr (has_right_pad) {
            std::uint32_t offset = local_right_col * tile_bytes;
            for (std::uint32_t r = 0; r < shard_H_tiles; r++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(s, dfb_data_in, tile_bytes, {.offset_bytes = offset}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_data_in.push_back(1);
                offset += row_stride_bytes;
            }
        }
    }
}
