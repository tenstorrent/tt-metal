// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 – Mask generation (before main loop):
 *   Same mask-tile generation as fill_pad_writer.cpp. Pushes right-mask and/or
 *   bottom-mask tiles to their DFBs once; the compute kernel reuses them persistently.
 *
 * Phase 2 – Write-back loop:
 *   Reads masked tiles from dfb::data_out and writes them back to the
 *   correct positions in this core's local L1 shard via NOC (local self-write).
 *   No cross-core NOC access.
 *
 * The right / bottom mask pads are conditionally-bound DFBs: HAS_RIGHT_PAD /
 * HAS_BOTTOM_PAD are #defines (per (rp_idx, has_bottom_pad_core) KernelSpec). The
 * legacy runtime `has_bottom_pad_core` gate is now the HAS_BOTTOM_PAD compile define.
 *
 * Named compile-time args:
 *   W_tiles, W_mod32 (right-mask threshold), H_mod32 (bottom-mask threshold)
 * Named runtime args:
 *   shard_H_tiles, num_work, local_right_col
 * Resource bindings: dfb::data_out (consumed); dfb::right_mask / dfb::bot_mask
 *   (produced, conditional); tensor::dst — the input tensor (Case 2: shard L1 base pulled
 *   via TensorAccessor::get_bank_base_address(), used directly in UnicastEndpoint arithmetic).
 *
 * Tile ordering mirrors fill_pad_sharded_reader.cpp and fill_pad_compute.cpp exactly.
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "fill_pad_dataflow_common.hpp"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);
    // Mask thresholds are read only where their mask is bound (avoids an unused-arg warning
    // in single-pad builds; the arg is still declared on the host for both).
#ifdef HAS_RIGHT_PAD
    constexpr auto W_mod32 = get_arg(args::W_mod32);
#endif
#ifdef HAS_BOTTOM_PAD
    constexpr auto H_mod32 = get_arg(args::H_mod32);
#endif

    const uint32_t shard_H_tiles = get_arg(args::shard_H_tiles);
    const uint32_t num_work = get_arg(args::num_work);
    const uint32_t local_right_col = get_arg(args::local_right_col);

    if (num_work == 0) {
        return;
    }

    Noc noc;
    DataflowBuffer dfb_data_out(dfb::data_out);
#ifdef HAS_RIGHT_PAD
    DataflowBuffer dfb_right_mask(dfb::right_mask);
#endif
#ifdef HAS_BOTTOM_PAD
    DataflowBuffer dfb_bot_mask(dfb::bot_mask);
#endif

    const uint32_t tile_bytes = dfb_data_out.get_tile_size();

    // Case 2 binding: pull this core's shard L1 base address from the TensorAccessor.
    const auto s = TensorAccessor(tensor::dst);
    const uint32_t shard_l1_base = s.get_bank_base_address();

    // ---- Phase 1: generate and push mask tile(s) ----
    using mask_t = MASK_ELEM_UINT;
    constexpr uint32_t TILE = 32;
#ifdef HAS_RIGHT_PAD
    push_right_mask_tile<mask_t, W_mod32, TILE>(dfb_right_mask, static_cast<mask_t>(MASK_VALUE));
#endif
#ifdef HAS_BOTTOM_PAD
    push_bottom_mask_tile<mask_t, H_mod32, TILE>(dfb_bot_mask, static_cast<mask_t>(MASK_VALUE));
#endif

    // ---- Phase 2: write-back loop ----
    // Tiles arrive in the same order as the reader and compute kernels.
    //
    // Local-L1 self-write via the Noc wrapper's UnicastEndpoint form: no
    // address-generator trait is applicable, so the endpoint carries explicit
    // noc_x/noc_y/addr. CB wait/pop and the writes-flushed barrier use the Device 2.0 API.

#ifdef HAS_BOTTOM_PAD
    // ---- Mode B ----

    // Step 1: right non-corner tiles (rows 0..shard_H_tiles-2, col local_right_col)
#ifdef HAS_RIGHT_PAD
    for (uint32_t r = 0; r < shard_H_tiles - 1u; r++) {
        const uint32_t dst = shard_l1_base + (r * W_tiles + local_right_col) * tile_bytes;
        dfb_data_out.wait_front(1);
        noc.async_write(
            dfb_data_out,
            UnicastEndpoint{},
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = dst});
        noc.async_writes_flushed();
        dfb_data_out.pop_front(1);
    }
#endif

    // Step 2: bottom row
#ifdef HAS_RIGHT_PAD
    // Non-corner bottom tiles: cols 0..local_right_col-1
    for (uint32_t c = 0; c < local_right_col; c++) {
        const uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + c) * tile_bytes;
        dfb_data_out.wait_front(1);
        noc.async_write(
            dfb_data_out,
            UnicastEndpoint{},
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = dst});
        noc.async_writes_flushed();
        dfb_data_out.pop_front(1);
    }
    // Corner tile: col local_right_col
    {
        const uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + local_right_col) * tile_bytes;
        dfb_data_out.wait_front(1);
        noc.async_write(
            dfb_data_out,
            UnicastEndpoint{},
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = dst});
        noc.async_writes_flushed();
        dfb_data_out.pop_front(1);
    }
#else
    for (uint32_t c = 0; c <= local_right_col; c++) {
        const uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + c) * tile_bytes;
        dfb_data_out.wait_front(1);
        noc.async_write(
            dfb_data_out,
            UnicastEndpoint{},
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = dst});
        noc.async_writes_flushed();
        dfb_data_out.pop_front(1);
    }
#endif

#else
    // ---- Mode A: right-column tiles only ----

#ifdef HAS_RIGHT_PAD
    for (uint32_t r = 0; r < shard_H_tiles; r++) {
        const uint32_t dst = shard_l1_base + (r * W_tiles + local_right_col) * tile_bytes;
        dfb_data_out.wait_front(1);
        noc.async_write(
            dfb_data_out,
            UnicastEndpoint{},
            tile_bytes,
            {.offset_bytes = 0},
            {.noc_x = (uint32_t)my_x[noc.get_noc_id()], .noc_y = (uint32_t)my_y[noc.get_noc_id()], .addr = dst});
        noc.async_writes_flushed();
        dfb_data_out.pop_front(1);
    }
#endif

#endif

    noc.async_write_barrier();
}
