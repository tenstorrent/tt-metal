// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 – Mask generation (before main loop):
 *   Same mask-tile generation as fill_pad_writer.cpp. Pushes right-mask and/or
 *   bottom-mask tiles to their DFBs once; the compute kernel reuses them persistently.
 *
 * Phase 2 – Write-back loop:
 *   Reads masked tiles from dfb::data_out and writes them back to the correct
 *   positions in this core's local L1 shard via NOC (local self-write).
 *   No cross-core NOC access.
 *
 * Metal 2.0 named resources:
 *   CTAs:  W_tiles; W_mod32 (only when the right mask is bound), H_mod32 (only
 *          when the bottom mask is bound).
 *   Defines: FILL_PAD_HAS_RIGHT_PAD / FILL_PAD_HAS_BOTTOM_PAD gate the
 *            conditionally-bound right / bottom mask DFBs (per compute group).
 *   DFBs:  dfb::right_mask (PRODUCER, conditional), dfb::bot_mask (PRODUCER,
 *          conditional), dfb::data_out (this writer is its CONSUMER).
 *   tensor: tensor::input — bound only to recover this core's shard L1 base
 *           (Case 2: get_bank_base_address()); raw self-write arithmetic unchanged.
 *   RTAs:  shard_H_tiles, has_bottom_pad_core, num_work, local_right_col.
 *
 * Tile ordering mirrors fill_pad_sharded_reader.cpp and fill_pad_compute.cpp exactly.
 */

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "fill_pad_dataflow_common.hpp"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);

    // has_right_pad is carried as a preprocessor define (not a CTA) because it gates
    // references to the conditionally-bound right mask DFB.
#ifdef FILL_PAD_HAS_RIGHT_PAD
    constexpr std::uint32_t has_right_pad = 1;
#else
    constexpr std::uint32_t has_right_pad = 0;
#endif

    const auto shard_H_tiles = get_arg(args::shard_H_tiles);
    const auto has_bottom_pad_core = get_arg(args::has_bottom_pad_core);
    const auto num_work = get_arg(args::num_work);
    const auto local_right_col = get_arg(args::local_right_col);

    if (num_work == 0) {
        return;
    }

    // Case 2: recover this core's shard L1 base from the tensor binding; the raw
    // self-write address arithmetic below is unchanged from the legacy kernel.
    const auto ta = TensorAccessor(tensor::input);
    const std::uint32_t shard_l1_base = ta.get_bank_base_address();

    Noc noc;
#ifdef FILL_PAD_HAS_RIGHT_PAD
    DataflowBuffer dfb_right_mask(dfb::right_mask);
#endif
#ifdef FILL_PAD_HAS_BOTTOM_PAD
    DataflowBuffer dfb_bot_mask(dfb::bot_mask);
#endif
    DataflowBuffer dfb_data_out(dfb::data_out);
    const std::uint32_t tile_bytes = dfb_data_out.get_entry_size();

    // ---- Phase 1: generate and push mask tile(s) ----
#if defined(FILL_PAD_HAS_RIGHT_PAD) || defined(FILL_PAD_HAS_BOTTOM_PAD)
    using mask_t = MASK_ELEM_UINT;
    constexpr std::uint32_t TILE = 32;
#endif
#ifdef FILL_PAD_HAS_RIGHT_PAD
    constexpr auto W_mod32 = get_arg(args::W_mod32);
    push_right_mask_tile<mask_t, W_mod32, TILE>(dfb_right_mask, static_cast<mask_t>(MASK_VALUE));
#endif
#ifdef FILL_PAD_HAS_BOTTOM_PAD
    constexpr auto H_mod32 = get_arg(args::H_mod32);
    if (has_bottom_pad_core) {
        push_bottom_mask_tile<mask_t, H_mod32, TILE>(dfb_bot_mask, static_cast<mask_t>(MASK_VALUE));
    }
#endif

    // ---- Phase 2: write-back loop ----
    // Tiles arrive in the same order as the reader and compute kernels.
    //
    // Local-L1 self-write via the Noc wrapper's UnicastEndpoint form: no
    // address-generator trait is applicable, so the endpoint carries explicit
    // noc_x/noc_y/addr. CB wait/pop and the writes-flushed barrier use the Device 2.0 API.

    if (has_bottom_pad_core) {
        // ---- Mode B ----

        // Step 1: right non-corner tiles (rows 0..shard_H_tiles-2, col local_right_col)
        if constexpr (has_right_pad) {
            for (std::uint32_t r = 0; r < shard_H_tiles - 1u; r++) {
                const std::uint32_t dst = shard_l1_base + (r * W_tiles + local_right_col) * tile_bytes;
                dfb_data_out.wait_front(1);
                noc.async_write(
                    dfb_data_out,
                    UnicastEndpoint{},
                    tile_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
                     .addr = dst});
                noc.async_writes_flushed();
                dfb_data_out.pop_front(1);
            }
        }

        // Step 2: bottom row
        if constexpr (has_right_pad) {
            // Non-corner bottom tiles: cols 0..local_right_col-1
            for (std::uint32_t c = 0; c < local_right_col; c++) {
                const std::uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + c) * tile_bytes;
                dfb_data_out.wait_front(1);
                noc.async_write(
                    dfb_data_out,
                    UnicastEndpoint{},
                    tile_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
                     .addr = dst});
                noc.async_writes_flushed();
                dfb_data_out.pop_front(1);
            }
            // Corner tile: col local_right_col
            const std::uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + local_right_col) * tile_bytes;
            dfb_data_out.wait_front(1);
            noc.async_write(
                dfb_data_out,
                UnicastEndpoint{},
                tile_bytes,
                {.offset_bytes = 0},
                {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                 .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
                 .addr = dst});
            noc.async_writes_flushed();
            dfb_data_out.pop_front(1);
        } else {
            for (std::uint32_t c = 0; c <= local_right_col; c++) {
                const std::uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + c) * tile_bytes;
                dfb_data_out.wait_front(1);
                noc.async_write(
                    dfb_data_out,
                    UnicastEndpoint{},
                    tile_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
                     .addr = dst});
                noc.async_writes_flushed();
                dfb_data_out.pop_front(1);
            }
        }

    } else {
        // ---- Mode A: right-column tiles only ----

        if constexpr (has_right_pad) {
            for (std::uint32_t r = 0; r < shard_H_tiles; r++) {
                const std::uint32_t dst = shard_l1_base + (r * W_tiles + local_right_col) * tile_bytes;
                dfb_data_out.wait_front(1);
                noc.async_write(
                    dfb_data_out,
                    UnicastEndpoint{},
                    tile_bytes,
                    {.offset_bytes = 0},
                    {.noc_x = (std::uint32_t)my_x[noc.get_noc_id()],
                     .noc_y = (std::uint32_t)my_y[noc.get_noc_id()],
                     .addr = dst});
                noc.async_writes_flushed();
                dfb_data_out.pop_front(1);
            }
        }
    }

    noc.async_write_barrier();
}
