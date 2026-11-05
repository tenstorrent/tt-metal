// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/kernel_helper_functions/pad_tile.hpp"

// The host defines these for bf16 and fp32 only, with TOPK_PAD_VALUE the +/-inf that can never win the sort.
#if TOPK_PAD_FP32
using topk_pad_t = uint32_t;
constexpr DataFormat topk_pad_format = DataFormat::Float32;
#else
using topk_pad_t = uint16_t;
constexpr DataFormat topk_pad_format = DataFormat::Float16_b;
#endif

#ifdef TOPK_PAD_LAST_TILE_W
// Writes TOPK_PAD_VALUE into columns TOPK_PAD_LAST_TILE_W..31 of the 32x32 tile at l1_tile_ptr.
FORCE_INLINE void topk_fill_width_padding(uint32_t l1_tile_ptr) {
    fill_pad_tile<topk_pad_t, TOPK_PAD_LAST_TILE_W, tt::constants::TILE_HEIGHT>(l1_tile_ptr, TOPK_PAD_VALUE);
    drain_pad_stores<topk_pad_format>(l1_tile_ptr);
}
#endif

#ifdef TOPK_PAD_TILES
// Fills the whole 32x32 tile at l1_tile_ptr with TOPK_PAD_VALUE: a pad tile appended to a row.
FORCE_INLINE void topk_fill_pad_tile(uint32_t l1_tile_ptr) {
    auto* tile_ptr = reinterpret_cast<topk_pad_t*>(l1_tile_ptr);
    for (uint32_t i = 0; i < tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH; ++i) {
        tile_ptr[i] = TOPK_PAD_VALUE;
    }
    drain_pad_stores<topk_pad_format>(l1_tile_ptr);
}
#endif
