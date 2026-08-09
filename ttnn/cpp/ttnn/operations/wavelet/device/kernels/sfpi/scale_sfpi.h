// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "lwt_sfpi_common.h"

namespace ckernel::sfpu {

inline void _lwt_scale_register_(
    const std::uint32_t value_reg, const std::uint32_t scalar_reg, const std::uint32_t product_reg) {
    TTI_SFPMUL(value_reg, scalar_reg, p_sfpu::LCONST_0, product_reg, 0);
    TTI_NOP;
    TTI_SFPMOV(0, product_reg, value_reg, 0);
}

template <DstTileShape TileShape, uint32_t FaceCount>
inline void _scale_lwt_tile(const uint32_t tile, const uint32_t scalar_packed) {
    _init_sfpu_config_reg();

    addr_mod_t addr_mod{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    };
    addr_mod.set(ADDR_MOD_3);

    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::set_dst_write_addr<TileShape, UnpackDestination::SrcRegs>(0);
    _lwt_clear_addr_mod_base_();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    const auto& value = p_sfpu::LREG0;
    const auto& scalar = p_sfpu::LREG1;
    const auto& product = p_sfpu::LREG2;
    TT_SFPLOADI(scalar, sfpi::SFPLOADI_MOD0_UPPER, scalar_packed >> 16);
    TT_SFPLOADI(scalar, sfpi::SFPLOADI_MOD0_LOWER, scalar_packed & 0xFFFF);

#pragma GCC unroll 4
    for (uint32_t face = 0; face < FaceCount; ++face) {
        const uint32_t face_base =
            TileShape == DstTileShape::Tile32x16 ? _lwt_narrow_dst_base(tile, face) : _lwt_dst_base(tile, face);
#pragma GCC unroll 8
        for (uint32_t row = 0; row < 16; row += 4) {
            TT_SFPLOAD(value, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row);
            TTI_SFPMUL(value, scalar, p_sfpu::LCONST_0, product, 0);
            TTI_NOP;
            TT_SFPSTORE(product, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row);

            TT_SFPLOAD(value, sfpi::SFPLOAD_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row + 2);
            TTI_SFPMUL(value, scalar, p_sfpu::LCONST_0, product, 0);
            TTI_NOP;
            TT_SFPSTORE(product, sfpi::SFPSTORE_MOD0_FMT_FP32, ADDR_MOD_3, face_base + row + 2);
        }
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
}

}  // namespace ckernel::sfpu

inline void scale_tile(const uint32_t tile, const uint32_t scalar_packed) {
    MATH((ckernel::sfpu::_scale_lwt_tile<DstTileShape::Tile32x32, 4>(tile, scalar_packed)));
}

inline void scale_narrow_tile(const uint32_t tile, const uint32_t scalar_packed) {
    MATH((ckernel::sfpu::_scale_lwt_tile<DstTileShape::Tile32x16, 2>(tile, scalar_packed)));
}
