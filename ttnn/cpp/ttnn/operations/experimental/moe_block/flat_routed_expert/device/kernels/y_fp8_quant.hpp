// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fp8 y (FP8_Y_PLAN.md): quantize one DEST pass of `TILES` fp32 tiles (a block: 32 token rows x TILES x 32 values) to
// e4m3 with one power-of-two scale a (token row, block), on the PACK thread's SFPU, in place in the packer's DEST half;
// the packer then packs the tiles (pack_untilize_dest into an Fp8_e4m3 CB: its fp32 -> e4m3 conversion truncates, which
// is exact after the rounding here).
//
//   amax = max |y| over the row's TILES x 32 values of the block
//   s    = 2^ceil(log2(max(amax, 1e-4) * fp32(1/448)))      (fp32 multiply, then the exact power-of-two ceil)
//   q    = RNE_e4m3(y / s)                                   (y / s: exact, an exponent shift)
//
// |q| <= amax / s <= 448 (1 + 3 x 2^-24) < 464 (the 448 / 480 midpoint): no saturation needed.
//
// The out CB's rows must have room past a pass's tiles: pack_untilize_dest of an odd block width into an 8-bit format
// writes one tile more a row (Blackhole, measured: a block of 3 e4m3 tiles writes 128 bytes a row) - give it a
// full_ct_dim of columns + one pass (the row stride; the y writer sends the first `columns` tiles of each row).
//
// DEST layout (fp32, a 32x32 tile = faces 0..3 of 16 rows x 16 columns, 64 DEST rows): token row r < 16 is DEST rows r
// (face 0, columns 0..15) and 16 + r (face 1, columns 16..31); r >= 16 is rows 32 + (r - 16) and 48 + (r - 16). An
// SFPU load at row address A (A % 4 == 0) reads DEST rows A..A+3, even columns (A + 2: odd columns): lane l is row
// A + l / 8, column 2 (l % 8) (+1). So the 12 loads of a 4-row group (3 tiles x 2 faces x 2 column parities) put each
// token row in its own 8 lanes: a lane-wise max over them, then a max over the row's 8 lanes, is the row's amax.
//
// The scales go out through DEST: the SFPU stores each row's scale (fp32 bits) in tile TILES of the half (the spare
// tile; TILES < the half's 4 fp32 tiles), odd columns; TRISC2 reads them through its RISC view of DEST (Blackhole
// 0xFFBD8000, 16 fp32 words a DEST row) into the scale CB.
#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"

#ifdef TRISC_PACK
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_dest.h"

namespace ckernel::sfpu {

// The max of |m| over each row's 8 lanes, in every lane of the row: a butterfly over 8-lane rotations
sfpi_inline sfpi::vFloat y_fp8_row_max8(sfpi::vFloat m) {
    sfpi::vFloat r = sfpi::subvec_shflror1(m);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    m = sfpi::max(m, r);
    r = sfpi::subvec_shflror1(m);
    r = sfpi::subvec_shflror1(r);
    m = sfpi::max(m, r);
    r = sfpi::subvec_shflror1(m);
    return sfpi::max(m, r);
}

// dst_reg offset (a tile is 32, a face 8, odd columns +1) of vector i of a tile's 4-row group: left face even, odd,
// right face even, odd
constexpr uint32_t y_fp8_at(uint32_t tile, uint32_t i) { return tile * 32 + (i / 2) * 8 + i % 2; }

// One 4-row group of a pass (the dst_reg pointer at the group's rows in tile 0), block by block (BLOCK_TILES tiles a
// block): amax, scale, quantize in place; block b's scale into the spare tile (tile TILES) at slot b: left face even
// / odd columns, right face even / odd (y_fp8_at(TILES, b)).
template <uint32_t TILES, uint32_t BLOCK_TILES>
sfpi_inline void y_fp8_quantize_group() {
#pragma GCC unroll 4
    for (uint32_t block = 0; block < TILES / BLOCK_TILES; ++block) {
        const uint32_t first = block * BLOCK_TILES;
        sfpi::vFloat m = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[y_fp8_at(first, 0)]));
#pragma GCC unroll 12
        for (uint32_t i = 1; i < BLOCK_TILES * 4; ++i) {
            m = sfpi::max(m, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[y_fp8_at(first + i / 4, i % 4)])));
        }
        m = y_fp8_row_max8(m);
        const sfpi::vFloat v = sfpi::max(m, sfpi::vFloat(1e-4f)) * sfpi::vFloat(1.0f / 448.0f);
        const sfpi::vUInt scale = (sfpi::as<sfpi::vUInt>(v) + 0x7FFFFFu) & 0x7F800000u;  // 2^ceil(log2 v), fp32 bits
        sfpi::dst_reg[y_fp8_at(TILES, block)] = sfpi::as<sfpi::vFloat>(scale);
        const sfpi::vFloat inv = sfpi::as<sfpi::vFloat>(sfpi::vUInt(254u << 23) - scale);  // 1 / s, exact
#pragma GCC unroll 12
        for (uint32_t i = 0; i < BLOCK_TILES * 4; ++i) {
            const uint32_t at = y_fp8_at(first + i / 4, i % 4);
            const sfpi::vFloat x = sfpi::dst_reg[at] * inv;
            const sfpi::vFloat ax = sfpi::abs(x);
            // e4m3 subnormals (|x| < 2^-6, code c = RNE(|x| 2^9)): the packer maps an fp32 of e4m3 exponent 0 (2^-7
            // <= |v| < 2^-6) to its 3 mantissa bits, without the implicit 1 (and smaller exponents to 0). So round
            // |x| + 2^-6 = 2^-6 + c 2^-9 to 3 mantissa bits and halve it: 2^-7 + c 2^-10 = (1 + c / 8) 2^-7, the fp32
            // that packs as code c (c = 8: 2^-6, the smallest normal)
            sfpi::vFloat t = x;
            v_if(ax < 0.015625f) { t = sfpi::copysgn(ax + 0.015625f, x); }
            v_endif;
            sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(t);
            bits = (bits + 0x7FFFFu + ((bits >> 20) & 1u)) & 0xFFF00000u;  // round to 3 mantissa bits, nearest even
            sfpi::vFloat r = sfpi::as<sfpi::vFloat>(bits);
            v_if(ax < 0.015625f) { r = r * 0.5f; }
            v_endif;
            sfpi::dst_reg[at] = r;
        }
    }
}

// The pass at the SFPU's DEST base (the packer's half, tile 0): 8 groups of 4 token rows
template <uint32_t TILES, uint32_t BLOCK_TILES>
inline void calculate_y_fp8_quantize() {
#pragma GCC unroll 0
    for (uint32_t face_pair = 0; face_pair < 2; ++face_pair) {
#pragma GCC unroll 0
        for (uint32_t group = 0; group < 4; ++group) {
            y_fp8_quantize_group<TILES, BLOCK_TILES>();
            sfpi::dst_reg += 2;  // the next 4 rows
        }
        sfpi::dst_reg += 3;  // past the right face's rows: the next face pair (+8; an increment is at most 7 rows)
        sfpi::dst_reg += 3;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace y_fp8 {

// Once, on every thread (does its work on PACK): the SFPU's init, and the RISC view of DEST for fp32 reads
ALWI void init() {
    PACK((SFPU_UNARY_INIT(unused)));
    PACK((ckernel::configure_dest_access<ckernel::PackThreadId>(DataFormat::Float32, true)));
    PACK((ckernel::tensix_sync()));  // (the config writes done)
}

// The packer's half (after tile_regs_wait): quantize its first TILES tiles in place, in blocks of BLOCK_TILES tiles
// (32 x BLOCK_TILES values a row), and write the 32 rows' block scales (fp32) to the L1 words
// scales[row * stride_words + block]. PACK only (a no-op on the other threads).
template <uint32_t TILES, uint32_t BLOCK_TILES = TILES>
ALWI void quantize_block([[maybe_unused]] uint32_t scales_l1, [[maybe_unused]] uint32_t stride_words) {
    static_assert(TILES >= 1 && TILES <= 3, "1..3 fp32 tiles a pass (the half's 4th tile holds the scales)");
    static_assert(BLOCK_TILES >= 1 && TILES % BLOCK_TILES == 0);
    constexpr uint32_t blocks = TILES / BLOCK_TILES;
#ifdef TRISC_PACK
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_y_fp8_quantize, (TILES, BLOCK_TILES), 0, VectorMode::RC_custom);
    // the SFPU's stores done before the RISC reads DEST (and before the pack: the packer waits on the SFPU)
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
    ckernel::tensix_sync();
    // the half's spare tile, in the RISC view (fp32 rows of 16 words; the half: 4 tiles of 64 rows)
    volatile uint32_t* dest = reinterpret_cast<volatile uint32_t*>(RISCV_DEST_START_ADDR) +
                              ((ckernel::dest_offset_id ? 4u : 0u) + TILES) * 64u * 16u;
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(scales_l1);
#pragma GCC unroll 2
    for (uint32_t row = 0; row < 32; ++row) {
        const uint32_t dest_row = row < 16 ? row : row + 16;  // faces 0 / 2 (left), + 16: faces 1 / 3 (right)
#pragma GCC unroll 3
        for (uint32_t block = 0; block < blocks; ++block) {
            // slot b: left / right face (b / 2), even / odd columns (b % 2): column b % 2 of the face's row
            out[row * stride_words + block] = dest[(dest_row + (block / 2) * 16) * 16 + block % 2];
        }
    }
#endif
}

}  // namespace y_fp8
