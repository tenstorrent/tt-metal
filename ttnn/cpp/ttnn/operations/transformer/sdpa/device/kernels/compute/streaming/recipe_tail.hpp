// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef SDPA_RECIPE_VALID_K_TILES
#ifdef TRISC_PACK
#include "sfpu/ckernel_sfpu_fill.h"
#endif

static uint32_t recipe_k_tile_offset;

// Stamp chunk padding before packing QK, so neither maxima nor the denominator
// see dummy keys. PACK owns SFPU while MATH overlaps the next matmul; using a
// math-thread fill here would race the pack-thread exponential. No math-counter
// reset is needed. Aligned recipes compile without this hook.
ALWI void mask_recipe_tail(uint32_t col_offset, uint32_t width, uint32_t height) {
#ifdef TRISC_PACK
    if (recipe_k_tile_offset + col_offset + width <= SDPA_RECIPE_VALID_K_TILES) {
        return;
    }
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            if (recipe_k_tile_offset + col_offset + col >= SDPA_RECIPE_VALID_K_TILES) {
                const uint32_t dst = row * width + col;
                SFPU_UNARY_CALL(
                    DST_SYNC_MODE,
                    DST_ACCUM_MODE,
                    _calculate_fill_bitcast_,
                    (APPROX, 8),
                    dst,
                    VectorMode::RC,
                    0xff800000);
            }
        }
    }
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
#endif
}
#endif
