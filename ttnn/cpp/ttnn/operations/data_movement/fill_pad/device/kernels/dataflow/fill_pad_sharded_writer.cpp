// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 – Mask generation (before main loop):
 *   Same mask-tile generation as fill_pad_writer.cpp. Pushes right-mask and/or
 *   bottom-mask tiles to their CBs once; the compute kernel reuses them persistently.
 *
 * Phase 2 – Write-back loop:
 *   Reads masked tiles from CB[cb_data_out_idx] and writes them back to the
 *   correct positions in this core's local L1 shard via NOC (local self-write).
 *   No cross-core NOC access.
 *
 * CT args:
 *   [0] W_tiles
 *   [1] has_right_pad
 *   [2] W_mod32        – width modulo 32 (right-mask threshold)
 *   [3] H_mod32        – height modulo 32 (bottom-mask threshold)
 *   [4] cb_right_mask_idx (= 1)
 *   [5] cb_bot_mask_idx   (= 2)
 *   [6] cb_data_out_idx   (= 16)
 *
 * RT args:
 *   [0] shard_l1_base_addr
 *   [1] shard_H_tiles
 *   [2] has_bottom_pad_core
 *   [3] num_work
 *   [4] local_right_col – local column index of the right-border tile within this shard
 *                         (= W_tiles-1 for fully-packed shards; may be smaller for the
 *                          rightmost shard when W_tiles_tensor % pages_per_shard_x != 0)
 *
 * Tile ordering mirrors fill_pad_sharded_reader.cpp and fill_pad_compute.cpp exactly.
 */

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "fill_pad_dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t W_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t has_right_pad = get_compile_time_arg_val(1);
    constexpr uint32_t W_mod32 = get_compile_time_arg_val(2);
    constexpr uint32_t H_mod32 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_right_mask_idx = get_compile_time_arg_val(4);
    constexpr uint32_t cb_bot_mask_idx = get_compile_time_arg_val(5);
    constexpr uint32_t cb_data_out_idx = get_compile_time_arg_val(6);

    constexpr uint32_t tile_bytes = get_tile_size(cb_data_out_idx);

    const uint32_t shard_l1_base = get_arg_val<uint32_t>(0);
    const uint32_t shard_H_tiles = get_arg_val<uint32_t>(1);
    const uint32_t has_bottom_pad_core = get_arg_val<uint32_t>(2);
    const uint32_t num_work = get_arg_val<uint32_t>(3);
    const uint32_t local_right_col = get_arg_val<uint32_t>(4);

    if (num_work == 0) {
        return;
    }

    experimental::Noc noc;
    experimental::CircularBuffer cb_right_mask(cb_right_mask_idx);
    experimental::CircularBuffer cb_bot_mask(cb_bot_mask_idx);
    experimental::CircularBuffer cb_data_out(cb_data_out_idx);

    // ---- Phase 1: generate and push mask tile(s) ----
    using mask_t = MASK_ELEM_UINT;
    if constexpr (has_right_pad) {
        cb_right_mask.reserve_back(1);
        generate_mask_tile<mask_t, W_mod32, TILE>(
            reinterpret_cast<mask_t*>(cb_right_mask.get_write_ptr()), static_cast<mask_t>(MASK_VALUE));
        cb_right_mask.push_back(1);
    }
    if (has_bottom_pad_core) {
        cb_bot_mask.reserve_back(1);
        generate_mask_tile<mask_t, TILE, H_mod32>(
            reinterpret_cast<mask_t*>(cb_bot_mask.get_write_ptr()), static_cast<mask_t>(MASK_VALUE));
        cb_bot_mask.push_back(1);
    }

    // ---- Phase 2: write-back loop ----
    // Tiles arrive in the same order as the reader and compute kernels.
    //
    // Local-L1 self-write: no address-generator trait is applicable, so fall back
    // to raw noc_async_write(..., get_noc_addr(addr), ...). CB wait/pop and the
    // writes-flushed barrier still go through the experimental API.

    if (has_bottom_pad_core) {
        // ---- Mode B ----

        // Step 1: right non-corner tiles (rows 0..shard_H_tiles-2, col local_right_col)
        if constexpr (has_right_pad) {
            for (uint32_t r = 0; r < shard_H_tiles - 1u; r++) {
                const uint32_t dst = shard_l1_base + (r * W_tiles + local_right_col) * tile_bytes;
                cb_data_out.wait_front(1);
                noc_async_write(cb_data_out.get_read_ptr(), get_noc_addr(dst), tile_bytes);
                noc.async_writes_flushed();
                cb_data_out.pop_front(1);
            }
        }

        // Step 2: bottom row
        if constexpr (has_right_pad) {
            // Non-corner bottom tiles: cols 0..local_right_col-1
            for (uint32_t c = 0; c < local_right_col; c++) {
                const uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + c) * tile_bytes;
                cb_data_out.wait_front(1);
                noc_async_write(cb_data_out.get_read_ptr(), get_noc_addr(dst), tile_bytes);
                noc.async_writes_flushed();
                cb_data_out.pop_front(1);
            }
            // Corner tile: col local_right_col
            const uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + local_right_col) * tile_bytes;
            cb_data_out.wait_front(1);
            noc_async_write(cb_data_out.get_read_ptr(), get_noc_addr(dst), tile_bytes);
            noc.async_writes_flushed();
            cb_data_out.pop_front(1);
        } else {
            for (uint32_t c = 0; c <= local_right_col; c++) {
                const uint32_t dst = shard_l1_base + ((shard_H_tiles - 1u) * W_tiles + c) * tile_bytes;
                cb_data_out.wait_front(1);
                noc_async_write(cb_data_out.get_read_ptr(), get_noc_addr(dst), tile_bytes);
                noc.async_writes_flushed();
                cb_data_out.pop_front(1);
            }
        }

    } else {
        // ---- Mode A: right-column tiles only ----

        if constexpr (has_right_pad) {
            for (uint32_t r = 0; r < shard_H_tiles; r++) {
                const uint32_t dst = shard_l1_base + (r * W_tiles + local_right_col) * tile_bytes;
                cb_data_out.wait_front(1);
                noc_async_write(cb_data_out.get_read_ptr(), get_noc_addr(dst), tile_bytes);
                noc.async_writes_flushed();
                cb_data_out.pop_front(1);
            }
        }
    }

    noc.async_write_barrier();
}
