// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The producer side of a quantized y row (common/kernels/quantized_row.hpp): quantize one DEST pass of the down compute
// - TILES fp32 tiles of 32 token rows - in place in the packer's DEST half, on the PACK thread's SFPU, and write each
// row's exponent bytes into its cell. The packer then packs the pass into a UInt8 out CB (pack_untilize_dest): every
// DEST word holds its final byte as an int32 in 0 .. 255, which the UInt8 packer writes as is (the convention of
// ckernel_sfpu_quant.h's int8 output).
//
// One power-of-two scale s a (token row, block of BLOCK_TILES x 32 values):
//
//   Fp8E4m3  s = 2^ceil(log2(fp32(max(amax, 1e-4) x fp32(1/448))))   byte = e4m3(RNE_3(y / s)), |y / s| < 2^-6 -> 0
//   Int8     s = 2^ceil(log2(fp32(max(amax, 2^-100) x fp32(1/127))))  byte = sign | RNE(|y / s|) (0 -> 0x00, never
//   0x80)
//
// y / s is exact (an exponent shift). |y / s| <= amax / s <= 448 (127) x (1 + 3 x 2^-24): below the 448 / 480 midpoint
// (127.5): no saturation needed. The e4m3 bits are built here, not by the packer (its fp32 -> e4m3 conversion gets
// subnormals wrong); e4m3 subnormals are flushed to 0 (the consumer's unpacker flushes them anyway).
//
// The exponent bytes (quantized_row.hpp: one a 16 values, a block's repeated): Fp8E4m3 log2(s) + 127, Int8 log2(s) +
// 133. Pass p of a row writes the row cell's bytes 2 TILES p .. 2 TILES (p + 1) - 1.
//
// DEST layout (fp32, a 32x32 tile = faces 0..3 of 16 rows x 16 columns, 64 DEST rows): token row r < 16 is DEST rows r
// (face 0, columns 0..15) and 16 + r (face 1, columns 16..31); r >= 16 is rows 32 + (r - 16) and 48 + (r - 16). An
// SFPU load at row address A (A % 4 == 0) reads DEST rows A..A+3, even columns (A + 2: odd columns): lane l is row
// A + l / 8, column 2 (l % 8) (+1). So the loads of a 4-row group (tiles x 2 faces x 2 column parities) put each token
// row in its own 8 lanes: a lane-wise max over them, then a max over the row's 8 lanes, is the row's amax.
//
// The scales leave through DEST: the SFPU stores each (row, block)'s scale bits in tile TILES of the half (the spare
// tile: TILES < the half's 4 fp32 tiles) at slot b (left / right face, even / odd columns); TRISC2 reads them through
// its RISC view of DEST (Blackhole 0xFFBD8000, 16 words a DEST row).
//
// The out CB's rows need room past a pass's tiles: pack_untilize_dest of an odd block width into an 8-bit format writes
// one tile more a row (Blackhole, measured) - give it a full_ct_dim of columns + one pass (the row stride; the y writer
// sends the first `columns` tiles of each row).
//
// Usage (PACK work; call on every thread, a no-op off PACK):
//     y_quant::init();                                      // once
//     tile_regs_wait();
//     y_quant::quantize_pass<format, pass_tiles, block_tiles>(cells, cell_stride_bytes, pass);
//     pack_untilize_dest<pass_tiles, row_stride_tiles>(out_cb, 1, pass);
#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/quantized_row.hpp"

#ifdef TRISC_PACK
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_dest.h"

namespace ckernel::sfpu {

// The max of |m| over each row's 8 lanes, in every lane of the row: a butterfly over 8-lane rotations
sfpi_inline sfpi::vFloat y_quant_row_max8(sfpi::vFloat m) {
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
constexpr uint32_t y_quant_at(uint32_t tile, uint32_t i) { return tile * 32 + (i / 2) * 8 + i % 2; }

// One 4-row group of a pass (the dst_reg pointer at the group's rows in tile 0), block by block
template <quantized_row::Format format, uint32_t TILES, uint32_t BLOCK_TILES>
sfpi_inline void y_quant_group() {
    constexpr bool fp8 = format == quantized_row::Format::Fp8E4m3;
#pragma GCC unroll 4
    for (uint32_t block = 0; block < TILES / BLOCK_TILES; ++block) {
        const uint32_t first = block * BLOCK_TILES;
        sfpi::vFloat m = sfpi::abs(sfpi::vFloat(sfpi::dst_reg[y_quant_at(first, 0)]));
#pragma GCC unroll 12
        for (uint32_t i = 1; i < BLOCK_TILES * 4; ++i) {
            m = sfpi::max(m, sfpi::abs(sfpi::vFloat(sfpi::dst_reg[y_quant_at(first + i / 4, i % 4)])));
        }
        m = y_quant_row_max8(m);
        const sfpi::vFloat v = fp8 ? sfpi::max(m, sfpi::vFloat(1e-4f)) * sfpi::vFloat(1.0f / 448.0f)
                                   : sfpi::max(m, sfpi::vFloat(0x1p-100f)) * sfpi::vFloat(1.0f / 127.0f);
        const sfpi::vUInt scale = (sfpi::as<sfpi::vUInt>(v) + 0x7FFFFFu) & 0x7F800000u;  // 2^ceil(log2 v), fp32 bits
        sfpi::dst_reg[y_quant_at(TILES, block)] = sfpi::as<sfpi::vFloat>(scale);
        const sfpi::vFloat inv = sfpi::as<sfpi::vFloat>(sfpi::vUInt(254u << 23) - scale);  // 1 / s, exact
#pragma GCC unroll 12
        for (uint32_t i = 0; i < BLOCK_TILES * 4; ++i) {
            const uint32_t at = y_quant_at(first + i / 4, i % 4);
            const sfpi::vFloat x = sfpi::dst_reg[at] * inv;
            const sfpi::vUInt sign = (sfpi::as<sfpi::vUInt>(x) >> 31) << 7;
            sfpi::vInt byte;
            if constexpr (fp8) {
                sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(x);
                bits = (bits + 0x7FFFFu + ((bits >> 20) & 1u)) & 0x7FF00000u;  // |x| to 3 mantissa bits, nearest even
                // e4m3 = (fp32 exponent - 120) << 3 | 3 mantissa bits = (bits >> 20) - (120 << 3); below 2^-6 (fp32
                // exponent < 121): 0
                byte = sfpi::as<sfpi::vInt>(bits >> 20) - 960;
                v_if(byte < 8) { byte = 0; }
                v_endif;
            } else {
                // |x| + 1.5 x 2^23 rounds |x| (<= 127) to an integer, nearest even: its bits less the magic's
                byte = sfpi::as<sfpi::vInt>(sfpi::abs(x) + 12582912.0f) - 0x4B400000;
            }
            v_if(byte != 0) { byte = byte | sfpi::as<sfpi::vInt>(sign); }
            v_endif;
            sfpi::dst_reg[at] = byte;  // (an int32: the UInt8 packer writes its low byte)
        }
    }
}

// The pass at the SFPU's DEST base (the packer's half, tile 0): 8 groups of 4 token rows
template <quantized_row::Format format, uint32_t TILES, uint32_t BLOCK_TILES>
inline void calculate_y_quant() {
#pragma GCC unroll 0
    for (uint32_t face_pair = 0; face_pair < 2; ++face_pair) {
#pragma GCC unroll 0
        for (uint32_t group = 0; group < 4; ++group) {
            y_quant_group<format, TILES, BLOCK_TILES>();
            sfpi::dst_reg += 2;  // the next 4 rows
        }
        sfpi::dst_reg += 3;  // past the right face's rows: the next face pair (+8; an increment is at most 7 rows)
        sfpi::dst_reg += 3;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
#endif

namespace y_quant {

// Once, on every thread (does its work on PACK): the SFPU's init, and the RISC view of DEST for fp32 reads
ALWI void init() {
    PACK((SFPU_UNARY_INIT(unused)));
    PACK((ckernel::configure_dest_access<ckernel::PackThreadId>(DataFormat::Float32, true)));
}

// The packer's half (after tile_regs_wait): quantize its first TILES tiles in place (every word its byte, as an int32)
// in blocks of BLOCK_TILES tiles, and write each of the 32 rows' exponent bytes of this pass into its cell: row r's at
// L1 cells + r x cell_stride_bytes + 2 TILES pass (TILES halfwords: a tile's two face rows of 16). PACK only.
template <quantized_row::Format format, uint32_t TILES, uint32_t BLOCK_TILES = TILES>
ALWI void quantize_pass(
    [[maybe_unused]] uint32_t cells, [[maybe_unused]] uint32_t cell_stride_bytes, [[maybe_unused]] uint32_t pass) {
    static_assert(format != quantized_row::Format::Bf16, "y_quant: a quantized format");
    static_assert(TILES >= 1 && TILES <= 3, "1..3 fp32 tiles a pass (the half's 4th tile holds the scales)");
    static_assert(BLOCK_TILES >= 1 && TILES % BLOCK_TILES == 0);
#ifdef TRISC_PACK
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_y_quant, (format, TILES, BLOCK_TILES), 0, VectorMode::RC_custom);
    // the SFPU's stores done before the RISC reads DEST (and before the pack: the packer waits on the SFPU)
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
    ckernel::tensix_sync();
    // the half's spare tile, in the RISC view (fp32 rows of 16 words; the half: 4 tiles of 64 rows)
    volatile uint32_t* dest = reinterpret_cast<volatile uint32_t*>(RISCV_DEST_START_ADDR) +
                              ((ckernel::dest_offset_id ? 4u : 0u) + TILES) * 64u * 16u;
    constexpr uint32_t bias = format == quantized_row::Format::Int8 ? 6u : 0u;  // (Int8: Bfp8_b's exponent)
#pragma GCC unroll 2
    for (uint32_t row = 0; row < 32; ++row) {
        const uint32_t dest_row = row < 16 ? row : row + 16;  // faces 0 / 2 (left), + 16: faces 1 / 3 (right)
        volatile uint16_t* cell =
            reinterpret_cast<volatile uint16_t*>(cells + row * cell_stride_bytes + 2 * TILES * pass);
#pragma GCC unroll 3
        for (uint32_t tile = 0; tile < TILES; ++tile) {
            const uint32_t block = tile / BLOCK_TILES;  // its scale: slot `block`, column block % 2
            const uint32_t e = (dest[(dest_row + (block / 2) * 16) * 16 + block % 2] >> 23) + bias;
            cell[tile] = static_cast<uint16_t>(e | (e << 8));
        }
    }
#endif
}

}  // namespace y_quant
