// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reads border tiles from the local L1 shard into dfb::data_in using NOC
 * reads addressed to this core's own L1. No cross-core NOC access.
 *
 * Named compile-time args:
 *   W_tiles         – width of the shard in tiles (= full tensor width for HEIGHT_SHARDED)
 *   has_right_pad   – 1 if tensor width % 32 != 0
 *   elem_size       – 2 or 4 bytes per element (declared, unused here)
 * Named runtime args:
 *   shard_H_tiles       – active height tiles in this shard
 *   has_bottom_pad_core – 1 if this is the last active shard and tensor has bottom padding
 *   num_work            – 0 → no border tiles on this core; >0 → work to do
 *   local_right_col     – local column index of the right-border tile within this shard
 *                         (= W_tiles-1 for fully-packed shards; may be smaller for the
 *                          rightmost shard when W_tiles_tensor % pages_per_shard_x != 0)
 * Resource bindings: dfb::data_in (produced); tensor::src — the input tensor. This is a
 *   Case 2 (raw pointer) binding: the shard L1 base address is pulled from the TensorAccessor
 *   via get_bank_base_address(), then used directly in UnicastEndpoint address arithmetic.
 *
 * Tile ordering matches fill_pad_compute.cpp exactly:
 *   Mode A (has_bottom_pad_core == 0):
 *     right column: (row, local_right_col) for row = 0..shard_H_tiles-1
 *   Mode B (has_bottom_pad_core == 1):
 *     right non-corner: (row, local_right_col) for row = 0..shard_H_tiles-2   [if has_right_pad]
 *     bottom row:       (shard_H_tiles-1, col) for col = 0..local_right_col
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);
    constexpr auto has_right_pad = get_arg(args::has_right_pad);
    constexpr auto elem_size = get_arg(args::elem_size);  // unused here (preserved arg)

    const uint32_t shard_H_tiles = get_arg(args::shard_H_tiles);
    const uint32_t has_bottom_pad_core = get_arg(args::has_bottom_pad_core);
    const uint32_t num_work = get_arg(args::num_work);
    const uint32_t local_right_col = get_arg(args::local_right_col);

    Noc noc;
    DataflowBuffer dfb_data_in(dfb::data_in);
    const uint32_t tile_bytes = dfb_data_in.get_tile_size();

    // Case 2 binding: pull this core's shard L1 base address from the TensorAccessor.
    // (The sharded reader does raw UnicastEndpoint arithmetic on the base, not accessor iteration.)
    const auto s = TensorAccessor(tensor::src);
    const uint32_t shard_l1_base = s.get_bank_base_address();

    // The UnicastEndpoint below carries this core's own physical NOC
    // coordinates (my_x[]/my_y[]) so each read targets local L1.

    const uint32_t row_stride_bytes = W_tiles * tile_bytes;

    // Local-L1 self-read via the Noc wrapper's UnicastEndpoint form: no
    // address-generator trait is applicable, so the endpoint carries explicit
    // noc_x/noc_y/addr. CB reservations and the read barrier use the Device 2.0 API.

    if (has_bottom_pad_core) {
        // ---- Mode B: right non-corner tiles, then full bottom row ----

        // Right non-corner tiles: rows 0..shard_H_tiles-2, col local_right_col.
        // addr steps by row_stride_bytes each iter.
        if constexpr (has_right_pad) {
            uint32_t addr = shard_l1_base + local_right_col * tile_bytes;
            for (uint32_t r = 0; r < shard_H_tiles - 1u; r++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(
                    UnicastEndpoint{},
                    dfb_data_in,
                    tile_bytes,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = addr},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_data_in.push_back(1);
                addr += row_stride_bytes;
            }
        }

        // Bottom row: all valid columns (including corner at col local_right_col).
        // addr steps by tile_bytes each iter.
        {
            uint32_t addr = shard_l1_base + (shard_H_tiles - 1u) * row_stride_bytes;
            for (uint32_t c = 0; c <= local_right_col; c++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(
                    UnicastEndpoint{},
                    dfb_data_in,
                    tile_bytes,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = addr},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_data_in.push_back(1);
                addr += tile_bytes;
            }
        }

    } else {
        // ---- Mode A: right-column tiles only ----

        if constexpr (has_right_pad) {
            uint32_t addr = shard_l1_base + local_right_col * tile_bytes;
            for (uint32_t r = 0; r < shard_H_tiles; r++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(
                    UnicastEndpoint{},
                    dfb_data_in,
                    tile_bytes,
                    {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                     .addr = addr},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_data_in.push_back(1);
                addr += row_stride_bytes;
            }
        }
    }
}
