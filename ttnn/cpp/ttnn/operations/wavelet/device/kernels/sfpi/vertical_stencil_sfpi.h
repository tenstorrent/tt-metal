// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Register allocation:
//   LREG0: f_0 (input1)
//   LREG1: f_1 (input2)
//   LREG2: f_2 (input3)
//   LREG3: f_3 (input4)
//   LREG4: g_0 (output1)
//   LREG5: g_1 (output2)
//   LREG6: g_2 (output3)
//   LREG7: tmp (broadcast coefficient)
#pragma once

#include <algorithm>
#include <array>

#include "lwt_sfpi_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline uint32_t _get_dst_base(
    const uint32_t tile_index, const uint32_t face_index, const uint32_t row_index, const uint32_t col_index) {
    // addr uint10: XTTTFFRRCX
    // T: tile index (0-7) << 6
    // F: face index (0-3) << 4
    // R: row index (0,4,8,12)
    // C: column index (0 for even cols, 1 for odd cols)*2
    return (tile_index << 6) + (face_index << 4) + row_index + (col_index << 1);
}

inline uint32_t _get_block(
    const uint32_t tile1, const uint32_t tile2, const uint32_t row_index, const uint32_t col_index) {
    const uint32_t tile = (row_index < 32) ? tile1 : tile2;
    const uint32_t col_face = (col_index < 2) ? 0 : 1;
    const uint32_t row_face = ((row_index % 32) < 16) ? 0 : 1;
    const uint32_t face = row_face * 2 + col_face;
    const uint32_t row_offset = row_index % 16;
    const uint32_t col_offset = col_index % 2;

    return _get_dst_base(tile, face, row_offset, col_offset);
}

inline void _vertical_stencil_init() {
    _init_sfpu_config_reg();

    addr_mod_t addr_mod{.dest = {.incr = 0}};
    addr_mod.set(ADDR_MOD_3);
    math::reset_counters(p_setrwc::SET_ABD_F);

    // Enable all lanes
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
}

// _vertical_stencil_rotate_(): 1-element up shift within subvectors of LReg0-LReg3.
inline void _vertical_stencil_rotate_() {
    TT_SFPTRANSP(0, 0, 0, 0);
    TT_SFPSHFT2(0, 0, 0, sfpi::SFPSHFT2_MOD1_SUBVEC_CHAINED_COPY4);
    TTI_SFPNOP;
    TT_SFPTRANSP(0, 0, 0, 0);
}

// Arguments:
//   h_packed: array of K uint32_t values, each being the bit-cast of h[j] as float32
//   f0: first 4x8 block
//   f1: second 4x8 block (next 4 rows)
//   f2: third 4x8 block (next 4 rows)
//   f3: fourth 4x8 block (next 4 rows)
//   g0: first 4x8 block output
//   g1: second 4x8 block output
//   g2: third 4x8 block output
//   base0: first 4x8 block base
//   base1: second 4x8 block base
//   base2: third 4x8 block base
template <uint8_t K>
inline void _vertical_stencil_block(
    const uint32_t h_packed[K],
    const uint32_t dst_f0,
    const uint32_t dst_f1,
    const uint32_t dst_f2,
    const uint32_t dst_f3,
    const uint32_t dst_tail0,
    const uint32_t dst_tail1,
    const uint32_t dst_tail2,
    const uint32_t dst_tail3,
    const uint32_t dst_g0,
    const uint32_t dst_g1,
    const uint32_t dst_g2,
    const uint32_t dst_base0 = 512,
    const uint32_t dst_base1 = 512,
    const uint32_t dst_base2 = 512) {
    static_assert(K > 0 && K <= 17, "Vertical stencil supports 1..17 coefficients");

    // Register allocations
    const auto& f_0 = p_sfpu::LREG0;
    const auto& f_1 = p_sfpu::LREG1;
    const auto& f_2 = p_sfpu::LREG2;
    const auto& f_3 = p_sfpu::LREG3;
    const auto& g_0 = p_sfpu::LREG4;
    const auto& g_1 = p_sfpu::LREG5;
    const auto& g_2 = p_sfpu::LREG6;
    const auto& tmp = p_sfpu::LREG7;

    // Load inputs into LRegs, for odd columns offset of 2 is used
    TT_SFPLOAD(f_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f0);
    TT_SFPLOAD(f_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f1);
    TT_SFPLOAD(f_2, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f2);
    TT_SFPLOAD(f_3, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_f3);

    // Load base from Dst
    if (dst_base0 < 512) {
        TT_SFPLOAD(g_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_base0);
    } else {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, g_0, 0);
    }
    if (K < 10 && dst_base1 < 512) {
        TT_SFPLOAD(g_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_base1);
    } else if (K < 10) {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, g_1, 0);
    }
    if (K < 6 && dst_base2 < 512) {
        TT_SFPLOAD(g_2, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_base2);
    } else if (K < 6) {
        TTI_SFPMOV(0, p_sfpu::LCONST_0, g_2, 0);
    }

#pragma GCC unroll 13
    for (uint8_t j = 0; j < std::min<uint8_t>(K, 13); j++) {
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_UPPER, (h_packed[(K - 1) - j]) >> 16);
        TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_LOWER, (h_packed[(K - 1) - j]) & 0xFFFF);
        if constexpr (K < 6) {
            TTI_SFPMAD(f_0, tmp, g_0, g_0, 0);
            TTI_SFPMAD(f_1, tmp, g_1, g_1, 0);
            TTI_SFPMAD(f_2, tmp, g_2, g_2, 0);
        } else if constexpr (K < 10) {
            TTI_SFPMAD(f_0, tmp, g_0, g_0, 0);
            TTI_SFPMAD(f_1, tmp, g_1, g_1, 0);
        } else {
            TTI_SFPMAD(f_0, tmp, g_0, g_0, 0);
        }
        TTI_SFPNOP;
        if (j + 1 < std::min<uint8_t>(K, 13)) {
            _vertical_stencil_rotate_();
        }
    }

    if constexpr (K > 13) {
        // Four output rows with K=14..17 require at most 20 contiguous
        // source rows. Keep g_0 live, reload a tail window beginning at
        // source row + 12, and rotate once so tap 13 observes row + 13.
        // No partial accumulator is materialized, and coefficients remain in
        // the same ordered FP32 MAD sequence as the K<=13 path.
        TT_SFPLOAD(f_0, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_tail0);
        TT_SFPLOAD(f_1, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_tail1);
        TT_SFPLOAD(f_2, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_tail2);
        TT_SFPLOAD(f_3, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, dst_tail3);
        _vertical_stencil_rotate_();

#pragma GCC unroll 4
        for (uint8_t j = 13; j < K; ++j) {
            TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_UPPER, (h_packed[(K - 1) - j]) >> 16);
            TT_SFPLOADI(tmp, sfpi::SFPLOADI_MOD0_LOWER, (h_packed[(K - 1) - j]) & 0xFFFF);
            TTI_SFPMAD(f_0, tmp, g_0, g_0, 0);
            TTI_SFPNOP;
            if (j + 1 < K) {
                _vertical_stencil_rotate_();
            }
        }
    }

    TT_SFPSTORE(g_0, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g0);
    if (K < 10 && dst_g1 < 512) {
        TT_SFPSTORE(g_1, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g1);
    }
    if (K < 6 && dst_g2 < 512) {
        TT_SFPSTORE(g_2, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, dst_g2);
    }
}

// Arguments:
//   h_packed: array of K uint32_t values, each being the bit-cast of h[j] as float32
//   input1: first tile index in dst register
//   input2: second tile index in dst register
//   output: tile index in dst register for output (can be same as either input)
template <uint8_t K>
inline void _vertical_stencil(
    const uint32_t h_packed[K],
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t output,
    const uint32_t base = 8) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    // We use addr mod 3, so base=0
    _lwt_clear_addr_mod_base_();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    constexpr uint32_t ROW_STRIDE = (K >= 10) ? 4 : (K >= 6) ? 8 : 12;

#pragma GCC unroll 8
    for (uint32_t row = 0; row < 32; row += ROW_STRIDE) {
#pragma GCC unroll 4
        for (uint32_t col = 0; col < 4; col += 1) {
            _vertical_stencil_block<K>(
                h_packed,
                _get_block(input1, input2, row, col),
                _get_block(input1, input2, row + 4, col),
                _get_block(input1, input2, row + 8, col),
                _get_block(input1, input2, row + 12, col),
                _get_block(input1, input2, row + 12, col),
                _get_block(input1, input2, row + 16, col),
                _get_block(input1, input2, row + 20, col),
                _get_block(input1, input2, row + 24, col),
                _get_block(output, 8, row, col),
                _get_block(output, 8, row + 4, col),
                _get_block(output, 8, row + 8, col),
                _get_block(base, base, row, col),
                _get_block(base, base, row + 4, col),
                _get_block(base, base, row + 8, col));
        }
    }

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

}  // namespace sfpu
}  // namespace ckernel

inline void vstencil_init() { MATH((ckernel::sfpu::_vertical_stencil_init())); }

template <uint8_t K>
inline void vstencil_tile(
    std::array<uint32_t, K> h_packed,
    const uint32_t input1,
    const uint32_t input2,
    const uint32_t output,
    const uint32_t base = 8) {
    MATH((ckernel::sfpu::_vertical_stencil<K>(h_packed.data(), input1, input2, output, base)));
}
