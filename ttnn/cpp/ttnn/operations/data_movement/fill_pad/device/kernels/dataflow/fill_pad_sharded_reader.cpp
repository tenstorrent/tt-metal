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
 *   tensor: tensor::src — bound only to recover this core's shard L1 base address
 *           (Case 2: get_bank_base_address()); the raw self-read arithmetic is unchanged.
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
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);
    constexpr auto has_right_pad = get_arg(args::has_right_pad);
    [[maybe_unused]] constexpr auto elem_size = get_arg(args::elem_size);

    const auto shard_H_tiles = get_arg(args::shard_H_tiles);
    const auto has_bottom_pad_core = get_arg(args::has_bottom_pad_core);
    [[maybe_unused]] const auto num_work = get_arg(args::num_work);
    const auto local_right_col = get_arg(args::local_right_col);

    // Case 2: recover this core's shard L1 base from the tensor binding; the raw
    // self-read address arithmetic below is unchanged from the legacy kernel.
    const auto s = TensorAccessor(tensor::src);
    const std::uint32_t shard_l1_base = s.get_bank_base_address();

    Noc noc;
    DataflowBuffer dfb_data_in(dfb::data_in);
    const std::uint32_t tile_bytes = dfb_data_in.get_entry_size();

    // The UnicastEndpoint below carries this core's own physical NOC
    // coordinates (my_x[]/my_y[]) so each read targets local L1.

    const std::uint32_t row_stride_bytes = W_tiles * tile_bytes;

    // Local-L1 self-read via the Noc wrapper's UnicastEndpoint form: no
    // address-generator trait is applicable, so the endpoint carries explicit
    // noc_x/noc_y/addr. CB reservations and the read barrier use the Device 2.0 API.

    if (has_bottom_pad_core) {
        // ---- Mode B: right non-corner tiles, then full bottom row ----

        // Right non-corner tiles: rows 0..shard_H_tiles-2, col local_right_col.
        // addr steps by row_stride_bytes each iter.
        if constexpr (has_right_pad) {
            std::uint32_t addr = shard_l1_base + local_right_col * tile_bytes;
            for (std::uint32_t r = 0; r < shard_H_tiles - 1u; r++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(
                    UnicastEndpoint{},
                    dfb_data_in,
                    tile_bytes,
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
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
            std::uint32_t addr = shard_l1_base + (shard_H_tiles - 1u) * row_stride_bytes;
            for (std::uint32_t c = 0; c <= local_right_col; c++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(
                    UnicastEndpoint{},
                    dfb_data_in,
                    tile_bytes,
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
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
            std::uint32_t addr = shard_l1_base + local_right_col * tile_bytes;
            for (std::uint32_t r = 0; r < shard_H_tiles; r++) {
                dfb_data_in.reserve_back(1);
                noc.async_read(
                    UnicastEndpoint{},
                    dfb_data_in,
                    tile_bytes,
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
                     .addr = addr},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_data_in.push_back(1);
                addr += row_stride_bytes;
            }
        }
    }
}
