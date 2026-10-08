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

#ifdef SDPA_RECIPE_RING_CAUSAL
// A QK tile on the causal diagonal (query row r sees key column c <= r): -inf above the diagonal. Face 1 (rows
// 0-15, columns 16-31) is masked whole, face 2 not at all; faces 0 and 3 hold the diagonal. A vector holds four
// rows (lane row (id >> 4) & 3) and the even or odd columns (id & 0xe, + 1) of one face; per 4-row group the
// threshold on column - row moves by 4.
inline void mask_recipe_diagonal() {
    const sfpi::vInt id = sfpi::vConstTileId;
    const sfpi::vInt column_minus_row = (id & sfpi::vInt(0xe)) - ((id >> 4) & sfpi::vInt(3));
    const sfpi::vFloat negative_infinity = Converter::as_float(0xff800000);
#pragma GCC unroll 1
    for (uint32_t face = 0; face < 4; face += 3) {
        sfpi::vInt v = column_minus_row;
#pragma GCC unroll 1
        for (uint32_t group = 0; group < 4; ++group) {
            // Even columns: masked where v > 0; odd columns (one further right): where v >= 0.
            v_if(v > 0) { sfpi::dst_reg[0] = negative_infinity; }
            v_endif;
            sfpi::dst_reg++;
            v_if(v >= 0) { sfpi::dst_reg[0] = negative_infinity; }
            v_endif;
            sfpi::dst_reg++;
            v -= 4;
        }
        if (face == 0) {
            // Face 1 masked whole, face 2 untouched.
#pragma GCC unroll 1
            for (uint32_t i = 0; i < 16; ++i) {
                if (i < 8) {
                    sfpi::dst_reg[0] = negative_infinity;
                }
                sfpi::dst_reg++;
            }
        }
    }
}
#endif
}  // namespace ckernel::sfpu
#endif

static uint32_t recipe_k_tile_offset;
#ifdef SDPA_RECIPE_RING
static uint32_t recipe_k_valid_rows;
// Rows in one full K chunk; the ring hook masks only chunks with fewer valid rows.
static uint32_t recipe_k_chunk_rows = 512;
#endif
#ifdef SDPA_RECIPE_RING_CAUSAL
// Ring causal step on the local shard (streaming/recipe_ring.hpp): whether this K chunk crosses the Q chunk's
// diagonal, its first K tile minus the Q chunk's first tile row (both in the shard's frame), and the first Q tile
// row of the subblock being packed (set by the QK pack sites). QK tile (r, c) is masked whole when its K tile
// lies past its Q tile, and above the diagonal when they are equal.
static bool recipe_causal_edge;
static int32_t recipe_causal_tile_delta;
static uint32_t recipe_causal_row0;
#ifdef SDPA_RECIPE_RING_CHUNKED
// Chunked prefill (streaming/recipe_ring.hpp): every step masks in the sequence's frame. A device's K cache holds its
// slab of each chunk group (SDPA_RECIPE_RING_CHUNKED Q tiles per device, SDPA_RECIPE_RING_GROUP_TILES per group)
// back to back, so local K tile t of device d is global tile (t / slab) * group + d * slab + t % slab. The tile delta
// above is then minus the Q chunk's global first tile, plus this K chunk's first local tile and its device.
static uint32_t recipe_causal_k_tile0;
static uint32_t recipe_causal_ring_id;
ALWI uint32_t recipe_chunked_k_tile(uint32_t ring_id, uint32_t tile) {
    constexpr uint32_t slab = SDPA_RECIPE_RING_CHUNKED;
    return (tile / slab) * SDPA_RECIPE_RING_GROUP_TILES + ring_id * slab + tile % slab;
}
#endif
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

#ifdef SDPA_RECIPE_RING
// Whether this K chunk's QK tiles need the pack-thread mask: a key tail or (ring causal) the diagonal.
ALWI bool recipe_ring_chunk_masked() {
#ifdef SDPA_RECIPE_RING_CAUSAL
    return recipe_k_valid_rows < recipe_k_chunk_rows || recipe_causal_edge;
#else
    return recipe_k_valid_rows < recipe_k_chunk_rows;
#endif
}
#endif

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
            uint32_t valid = recipe_valid_k_columns(recipe_k_tile_offset + col_offset + col);
#ifdef SDPA_RECIPE_RING_CAUSAL
#ifdef SDPA_RECIPE_RING_CHUNKED
            const uint32_t k_tile =
                recipe_chunked_k_tile(recipe_causal_ring_id, recipe_causal_k_tile0 + col_offset + col);
#else
            const uint32_t k_tile = col_offset + col;
#endif
            const int32_t delta = recipe_causal_edge ? recipe_causal_tile_delta + static_cast<int32_t>(k_tile) -
                                                           static_cast<int32_t>(recipe_causal_row0 + row)
                                                     : -1;
            valid = delta > 0 ? 0 : valid;
            if (delta == 0) {
                if (!masked) {
                    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
                    masked = true;
                }
                SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, mask_recipe_diagonal, row * width + col, VectorMode::None);
            }
#endif
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
