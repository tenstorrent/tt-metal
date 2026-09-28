// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(SDPA_RECIPE_K_PRIMARY_ROWS) || defined(SDPA_RECIPE_RING)
#ifdef TRISC_PACK
#include "sfpu/ckernel_sfpu_fill.h"
namespace ckernel::sfpu {
inline void mask_recipe_columns(uint32_t valid_columns) {
    sfpi::vUInt local_column = sfpi::vConstTileId & sfpi::vUInt(0xe);
    const sfpi::vFloat negative_infinity = Converter::as_float(0xff800000);
    for (uint32_t i = 0; i < 32; ++i) {
        // One vector covers four rows and eight even/odd columns of one face.
        const uint32_t offset = ((i / 8) % 2) * 16 + i % 2;
        v_if(local_column + offset >= valid_columns) { sfpi::dst_reg[0] = negative_infinity; }
        v_endif;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

static uint32_t recipe_k_tile_offset;
#ifdef SDPA_RECIPE_RING
static uint32_t recipe_k_valid_rows;
// Rows in one full K chunk; the ring hook masks only chunks with fewer valid rows.
static uint32_t recipe_k_chunk_rows = 512;
#endif

ALWI uint32_t recipe_valid_k_columns(uint32_t tile) {
#ifdef SDPA_RECIPE_RING
    const uint32_t remaining = tile * 32 < recipe_k_valid_rows ? recipe_k_valid_rows - tile * 32 : 0;
#else
    constexpr uint32_t primary_padded = ((SDPA_RECIPE_K_PRIMARY_ROWS + 31) / 32) * 32;
    const uint32_t row = tile * 32;
    const uint32_t remaining = row < primary_padded ? SDPA_RECIPE_K_PRIMARY_ROWS - row
                               : row < primary_padded + SDPA_RECIPE_K_JOINT_ROWS
                                   ? SDPA_RECIPE_K_JOINT_ROWS - (row - primary_padded)
                                   : 0;
#endif
    return remaining < 32 ? remaining : 32;
}

// Stamp chunk padding before packing QK, so neither maxima nor the denominator
// see dummy keys. PACK owns SFPU while MATH overlaps the next matmul; using a
// math-thread fill here would race the pack-thread exponential. No math-counter
// reset is needed. Aligned recipes compile without this hook.
#ifdef SDPA_RECIPE_RING
__attribute__((noinline, noclone))
#else
ALWI
#endif
void mask_recipe_tail(uint32_t col_offset, uint32_t width, uint32_t height) {
#ifdef TRISC_PACK
    bool masked = false;
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t col = 0; col < width; ++col) {
            const uint32_t valid = recipe_valid_k_columns(recipe_k_tile_offset + col_offset + col);
            if (valid < 32) {
                if (!masked) {
                    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
                    masked = true;
                }
                const uint32_t dst = row * width + col;
                if (valid != 0) {
                    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                        DST_SYNC_MODE, DST_ACCUM_MODE, mask_recipe_columns, dst, VectorMode::None, valid);
                } else {
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
    }
    if (masked) {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
    }
#endif
}
#endif
