// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "recipe_config.hpp"

// Key masking in the pack thread: the ring kernels (recipe::ring: per-chunk key padding and positional masks, see
// RecipeRingMask) and joint dense recipes with unaligned key lengths (recipe::k_tail). PACK owns the SFPU here; SFPI
// exists in the MATH and PACK builds only.
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"

// A masked ring score: -2^100 rather than -inf (as the dense key-range masks, recipe_key_range.hpp), so a row whose keys
// are all masked in a Q chunk's first K chunk takes a finite maximum that the first visible key's rescale wipes out.
constexpr uint32_t kRecipeRingMaskedBits = 0xF1800000u;

#ifdef TRISC_PACK
#include "ckernel_sfpu_fill.h"
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

// Ring: one QK tile with key column c and query row r masked (-2^100) where c - r > hi, c - r < lo or c >= valid. A
// vector holds four rows (lane row (id >> 4) & 3) and the even or odd columns (id & 0xe, + 1) of one face; iteration i
// covers face i / 8 (faces 0-3: rows 0-15 / 16-31 x columns 0-15 / 16-31), row group (i / 2) % 4 and parity i % 2.
inline void mask_recipe_band(int32_t lo, int32_t hi, int32_t valid) {
#ifdef ARCH_WORMHOLE
    // Wormhole has no arithmetic vector shift (SFPI defines vInt >> on Blackhole only): the lane id is nonnegative,
    // so compute unsigned and reinterpret.
    const sfpi::vUInt column_u = sfpi::vConstTileId & sfpi::vUInt(0xe);
    const sfpi::vUInt row_u = (sfpi::vConstTileId & sfpi::vUInt(0x30)) >> 4;
    const sfpi::vInt column = sfpi::as<sfpi::vInt>(column_u);
    const sfpi::vInt column_minus_row = sfpi::as<sfpi::vInt>(column_u - row_u);
#else
    const sfpi::vInt id = sfpi::vConstTileId;
    const sfpi::vInt column = id & sfpi::vInt(0xe);
    const sfpi::vInt column_minus_row = column - ((id >> 4) & sfpi::vInt(3));
#endif
    const sfpi::vFloat masked = Converter::as_float(kRecipeRingMaskedBits);
#pragma GCC unroll 1
    for (uint32_t i = 0; i < 32; ++i) {
        const int32_t c0 = static_cast<int32_t>((i & 1) + ((i >> 3) & 1) * 16);
        const int32_t r0 = static_cast<int32_t>(((i >> 1) & 3) * 4 + (i >> 4) * 16);
        v_if(column_minus_row > hi - c0 + r0) { sfpi::dst_reg[0] = masked; }
        v_endif;
        v_if(column_minus_row < lo - c0 + r0) { sfpi::dst_reg[0] = masked; }
        v_endif;
        v_if(column >= valid - c0) { sfpi::dst_reg[0] = masked; }
        v_endif;
        sfpi::dst_reg++;
    }
}

// Dense attn_mask: one mask tile in DST times 1 / scale (FP32 bits), the whole tile per call (VectorMode::None).
inline void scale_recipe_mask(uint32_t inverse_scale_bits) {
    const sfpi::vFloat inverse_scale = Converter::as_float(inverse_scale_bits);
#pragma GCC unroll 8
    for (uint32_t i = 0; i < 32; ++i) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * inverse_scale;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

// Dense attn_mask: multiply the `tiles` mask tiles MATH copied to DST by 1 / scale before PACK adds them onto the
// scores (recipe_add_attn_mask). PACK owns the SFPU (see mask_recipe_tail_body).
ALWI void scale_recipe_mask_tiles(uint32_t tiles) {
    if constexpr (recipe::mask_scale != 0) {
        PACK({
            addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
            for (uint32_t i = 0; i < tiles; ++i) {
                SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                    DST_SYNC_MODE, DST_ACCUM_MODE, scale_recipe_mask, i, VectorMode::None, recipe::mask_scale);
            }
            TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
        })
    }
}

static uint32_t recipe_k_tile_offset;

// Ring kernels (recipe::ring): the current K chunk's mask, set by the ring wrapper (recipe_ring.hpp) before each step.
// A key is masked when it is padding (at or past valid_rows in the chunk) or, with a positional mask (recipe::ring_mask),
// when its position is not visible to the query row's: past it (causal), at or before it minus the window
// (recipe::ring_window), past the logical length or the cache (recipe::ring_rotation), or a Q row maps to no position.
struct RecipeRingMask {
    uint32_t valid_rows = 0;  // keys [0, valid_rows) of the chunk exist
    bool masked = false;      // some QK tile of the chunk is masked: the pack hook runs (conservative)
    bool positional = false;  // this chunk takes the positional mask
    uint32_t q_tile0 = 0;     // the Q chunk's first tile row (shard or sequence frame; rotation: its slab tile)
    uint32_t k_tile0 = 0;     // the chunk's first local K tile
    uint32_t k_seq_tile0 = 0;  // circular KV: the chunk's first sequence tile
    uint32_t ring_id = 0;     // the K shard's device (sequence frame)
};
static RecipeRingMask recipe_ring_mask;
// Per-call geometry of the sequence frame (recipe::ring_mask == 2), set once by the ring wrapper: the chunk group's Q
// tiles, the KV region tiles per device and group, the local K tiles (rotation's bound), the logical length in tiles
// (rotation) and the Q slab's wrap segments (rotation).
struct RecipeRingFrame {
    uint32_t group_tiles = 0;
    uint32_t region_tiles = 1;
    uint32_t local_tiles = 0;
    uint32_t logical_tiles = 0;
    KVPadRotationContext rotation = {};
};
static RecipeRingFrame recipe_ring_frame;
// The first Q tile row of the subblock being packed (set by the QK pack sites).
static uint32_t recipe_mask_row0;

ALWI uint32_t recipe_valid_k_columns(uint32_t tile) {
    uint32_t remaining;
    if constexpr (recipe::ring) {
        remaining = tile * 32 < recipe_ring_mask.valid_rows ? recipe_ring_mask.valid_rows - tile * 32 : 0;
    } else {
        constexpr uint32_t primary_padded = ((recipe::k_primary_rows + 31) / 32) * 32;
        const uint32_t row = tile * 32;
        remaining = row < primary_padded                          ? recipe::k_primary_rows - row
                    : row < primary_padded + recipe::k_joint_rows ? recipe::k_joint_rows - (row - primary_padded)
                                                                  : 0;
    }
    return remaining < 32 ? remaining : 32;
}

// Ring: whether this K chunk's QK tiles need the pack-thread mask.
ALWI bool recipe_ring_chunk_masked() { return recipe_ring_mask.masked; }

// Masks no key in a tile (c - r spans [-31, 31]).
constexpr int32_t kRecipeBandOpen = 64;

// Ring positional mask of QK tile (q_tile, k_tile) of the chunk: the band [lo, hi] of visible c - r, and the valid
// columns (clipped by the logical length and the cache with KV-pad rotation).
ALWI void recipe_ring_band(uint32_t q_tile, uint32_t k_tile, int32_t& lo, int32_t& hi, uint32_t& valid) {
    uint32_t q = recipe_ring_mask.q_tile0 + q_tile;
    uint32_t k;
    if constexpr (recipe::ring_mask == 1) {
        k = recipe_ring_mask.k_tile0 + k_tile;
    } else {
        if constexpr (recipe::ring_rotation) {
            q = q_global_tile_for_mask_row<true>(q_tile, recipe_ring_mask.q_tile0, recipe_ring_frame.rotation);
            if (q == KV_PAD_ROTATION_INVALID_TILE || recipe_ring_mask.k_tile0 + k_tile >= recipe_ring_frame.local_tiles) {
                valid = 0;
                return;
            }
        }
        if constexpr (recipe::ring_circular) {
            k = recipe_ring_mask.k_seq_tile0 + k_tile;
        } else {
            k = chunked_kv_global_tile_for_local(
                recipe_ring_mask.ring_id,
                recipe_ring_mask.k_tile0 + k_tile,
                recipe_ring_frame.group_tiles,
                recipe_ring_frame.region_tiles);
        }
        if constexpr (recipe::ring_rotation) {
            if (k >= recipe_ring_frame.logical_tiles) {
                valid = 0;
                return;
            }
        }
    }
    // Key k * 32 + c is visible to query q * 32 + r iff its offset 32 (k - q) + c - r is <= 0 (and, with a window w,
    // > -w).
    const int32_t offset = 32 * (static_cast<int32_t>(k) - static_cast<int32_t>(q));
    hi = -offset;
    if constexpr (recipe::ring_window != 0) {
        lo = 1 - static_cast<int32_t>(recipe::ring_window) - offset;
    }
}

// Stamp chunk padding (and, on rings, positional masks) before packing QK, so neither maxima nor the denominator see
// masked keys. PACK owns SFPU while MATH overlaps the next matmul; using a math-thread fill here would race the
// pack-thread exponential. No math-counter reset is needed. Aligned recipes compile without this hook.
ALWI void mask_recipe_tail_body(uint32_t col_offset, uint32_t width, uint32_t height) {
    PACK({
        bool masked = false;
        for (uint32_t row = 0; row < height; ++row) {
            for (uint32_t col = 0; col < width; ++col) {
                uint32_t valid = recipe_valid_k_columns(recipe_k_tile_offset + col_offset + col);
                const uint32_t dst = row * width + col;
                if constexpr (recipe::ring) {
                    int32_t lo = -kRecipeBandOpen, hi = kRecipeBandOpen;
                    if constexpr (recipe::ring_mask != 0) {
                        if (recipe_ring_mask.positional && valid != 0) {
                            recipe_ring_band(recipe_mask_row0 + row, col_offset + col, lo, hi, valid);
                        }
                    }
                    if (valid >= 32 && hi >= 31 && lo <= -31) {
                        continue;
                    }
                    if (!masked) {
                        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
                        masked = true;
                    }
                    if (valid == 0 || hi < -31 || lo > 31) {
                        SFPU_UNARY_CALL(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            _calculate_fill_bitcast_,
                            (APPROX, 8),
                            dst,
                            VectorMode::RC,
                            kRecipeRingMaskedBits);
                    } else {
                        SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
                            DST_SYNC_MODE,
                            DST_ACCUM_MODE,
                            mask_recipe_band,
                            dst,
                            VectorMode::None,
                            lo,
                            hi,
                            static_cast<int32_t>(valid));
                    }
                } else if (valid < 32) {
                    if (!masked) {
                        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
                        masked = true;
                    }
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
    })
}

// The ring kernels call one out-of-line copy (a template, so builds that never call it carry no copy).
template <bool = true>
__attribute__((noinline, noclone)) void mask_recipe_tail_outlined(
    uint32_t col_offset, uint32_t width, uint32_t height) {
    mask_recipe_tail_body(col_offset, width, height);
}

ALWI void mask_recipe_tail(uint32_t col_offset, uint32_t width, uint32_t height) {
    if constexpr (recipe::ring) {
        mask_recipe_tail_outlined(col_offset, width, height);
    } else {
        mask_recipe_tail_body(col_offset, width, height);
    }
}
