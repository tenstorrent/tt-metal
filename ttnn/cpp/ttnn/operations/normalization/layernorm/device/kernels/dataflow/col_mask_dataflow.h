// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file col_mask_dataflow.h
 * @brief On-device column-mask generation shared by the sharded layernorm writer kernels.
 */

#pragma once

#include "ttnn/kernel/dataflow/moreh_common.hpp"  // generate_mask_w<T>
#include "api/debug/assert.h"

// Generate this core's column mask on-device: block_w tiles, one per tile across the shard width, with
// the padding columns of the boundary tile (and any all-padding tiles) zeroed. The core's width position
// comes from width_shard_tile_start_id (its first tile index along the normalized dimension); block_w is
// the per-core width in tiles and logical_K the un-padded normalized width in columns.
FORCE_INLINE void generate_col_mask(
    uint32_t cb_col_mask, uint32_t block_w, uint32_t logical_K, uint32_t width_shard_tile_start_id) {
    constexpr uint32_t tile_w = 32;
    DataflowBuffer dfb_col_mask_obj(cb_col_mask);
    // A width shard always starts on a block boundary, so this offset is an exact multiple of block_w.
    ASSERT(width_shard_tile_start_id % block_w == 0);
    const uint32_t core_start_col = width_shard_tile_start_id * tile_w;
    for (uint32_t wt = 0; wt < block_w; wt++) {
        const uint32_t tile_start_col = core_start_col + wt * tile_w;
        uint32_t mask_w;
        if (logical_K <= tile_start_col) {
            mask_w = 0;
        } else if (logical_K - tile_start_col >= tile_w) {
            mask_w = tile_w;
        } else {
            mask_w = logical_K - tile_start_col;
        }
        // The mask holds only 1.0 or 0.0 (exact in bfloat16) and is always generated as bfloat16;
        // each masking multiply reconfigures the unpacker so SrcB reads it in this format.
        generate_mask_w<uint16_t>(dfb_col_mask_obj, mask_w);
    }
}
