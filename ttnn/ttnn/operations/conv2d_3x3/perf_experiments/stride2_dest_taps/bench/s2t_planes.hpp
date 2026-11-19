// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Bench-only variants of conv2d_3x3_s2_taps.hpp's section (one output tile, one channel tile, one kernel row): planes
// built by one-row SrcB -> DEST moves so the packer reads more taps with 4-row PACRs.
//   mode 1 (oplane): the pixel buffer as the op (4-row moves; E packed with masks 0101) plus an O plane (odd pixels by
//     one-row moves, the previous tile's pixel 31 one row ahead): O = 16 4-row PACRs, kw = 0 = per face 1000, 3 x 1111,
//     0111 (5 PACRs).
//   mode 2 (eoplane): E and O planes only (one-row moves), E / O 4-row, kw = 0 as mode 1.
// Every block (channel half) is HS = 72 rows after the previous one (one packer Z stride).

#pragma once

#include "s2t_mask.hpp"

namespace conv2d_3x3::s2_planes {

using conv2d_3x3::s2_taps_mask::HS;
using conv2d_3x3::s2_taps_mask::R0;
// mode 1: O plane block 0 rows [OB1 - 4, OB1 + 32) past the pixel buffer; mode 2: E at EB2, O at OB2.
constexpr uint32_t OB1 = 2 * HS + 4;  // 148 (block 1: 220)
constexpr uint32_t EB2 = 0;
constexpr uint32_t OB2 = 40;

#ifdef TRISC_MATH
using conv2d_3x3::s2_taps_mask::ADDR_MOD_ZERO;
using conv2d_3x3::s2_taps_mask::dest_row;
using conv2d_3x3::s2_taps_mask::release_srcb;

// SrcB row r of a tile (face f = r / 16: pixel p = 16 (f / 2) + r % 16, half f % 2): even pixels to eb + HS h + 16 k +
// p / 2 (if eb != NONE), odd pixels to ob + HS h + 16 k + p / 2.
constexpr uint32_t NONE = 0xFFFF;
template <uint32_t eb, uint32_t ob, uint32_t k, uint32_t r>
ALWI void plane_row() {
    constexpr uint32_t f = r / 16;
    constexpr uint32_t p = 16 * (f / 2) + r % 16;
    constexpr uint32_t h = f % 2;
    if constexpr (p % 2 == 0) {
        if constexpr (eb != NONE) {
            TTI_MOVB2D(0, r, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, eb + HS * h + 16 * k + p / 2);
        }
    } else {
        TTI_MOVB2D(0, r, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, ob + HS * h + 16 * k + p / 2);
    }
}
template <uint32_t eb, uint32_t ob, uint32_t k, uint32_t... R>
ALWI void plane_rows(std::integer_sequence<uint32_t, R...>) {
    (plane_row<eb, ob, k, R>(), ...);
}
// The previous tile's pixel 31 (SrcB rows 47, 63) to the O plane's row -1.
template <uint32_t ob>
ALWI void plane_prev() {
    TTI_MOVB2D(0, 47, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, ob - 1);
    TTI_MOVB2D(0, 63, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, ob - 1 + HS);
}
template <uint32_t mode, uint32_t k>
ALWI void move_tile() {
    if constexpr (mode == 1) {
        conv2d_3x3::s2_taps_mask::move_pair_tile<k>();  // pixel buffer (dest_row 0)
        plane_rows<NONE, OB1, k>(std::make_integer_sequence<uint32_t, 64>{});
    } else {
        dest_row(0);
        plane_rows<EB2, OB2, k>(std::make_integer_sequence<uint32_t, 64>{});
    }
}
template <uint32_t mode>
ALWI void move_prev() {
    if constexpr (mode == 1) {
        conv2d_3x3::s2_taps_mask::move_prev();  // dest_row 0
        plane_prev<OB1>();
    } else {
        dest_row(0);
        plane_prev<OB2>();
    }
}
#endif

#ifdef TRISC_PACK
#define S2P_PACR(intf, last)            \
    TTI_PACR(                           \
        p_pacr::CFG_CTXT_0,             \
        p_pacr::NO_ROW_PAD_ZERO,        \
        p_pacr::DST_ACCESS_NORMAL_MODE, \
        ADDR_MOD_0,                     \
        p_pacr::ADDR_CNT_CTXT_0,        \
        0,                              \
        intf,                           \
        0,                              \
        0,                              \
        0,                              \
        0,                              \
        last)
// Replay (after s2_taps::pack_init's [0, 28)): [28] 4-row, [29] 4-row last ... only 4 slots left, so the plane taps
// issue their PACRs directly (no replay).
template <uint32_t intf, uint32_t cnt>
ALWI void pacrs() {
    for (uint32_t i = 0; i < cnt; ++i) {
        S2P_PACR(intf, 0);
    }
}
// A 4-row tap (16 PACRs) from window w.
ALWI void pack_tap4(uint32_t out_id, uint32_t w, uint32_t off) {
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, w);
    program_packer_destination(get_output_tile_address<true, PackMode::Default>(out_id, off));
    pacrs<p_pacr::ALL_INTF_ACTIVE, 4>();
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);
    pacrs<p_pacr::ALL_INTF_ACTIVE, 4>();
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);  // Y 16
    pacrs<p_pacr::ALL_INTF_ACTIVE, 4>();
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 16);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);
    pacrs<p_pacr::ALL_INTF_ACTIVE, 3>();
    S2P_PACR(p_pacr::ALL_INTF_ACTIVE, 1);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
}
// The kw = 0 tap from the O plane (row -1 = the previous pixel) with window base w = O plane base - 4: per face 1000,
// 3 x 1111, 0111.
ALWI void pack_tapk(uint32_t out_id, uint32_t w, uint32_t off) {
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, w);
    program_packer_destination(get_output_tile_address<true, PackMode::Default>(out_id, off));
    auto face = [](uint32_t last) {
        S2P_PACR(0b1000, 0);
        pacrs<p_pacr::ALL_INTF_ACTIVE, 3>();
        if (last) {
            S2P_PACR(p_pacr::THREE_INTFS_ACTIVE, 1);
        } else {
            S2P_PACR(p_pacr::THREE_INTFS_ACTIVE, 0);
        }
    };
    face(0);                                          // F0: rows -1..14 (Y 0..16)
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
    face(0);                                          // F1
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);  // Z 0
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 16);
    face(0);  // F2: rows 15..30
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 16);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
    face(1);                                          // F3
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
}
template <uint32_t mode>
ALWI void pack_taps(uint32_t out_id, uint32_t off0, uint32_t kw_step) {
    using conv2d_3x3::s2_taps_mask::pack_tap;
    using conv2d_3x3::s2_taps_mask::RE;
    using conv2d_3x3::s2_taps_mask::RE_LAST;
    if constexpr (mode == 1) {
        pack_tapk(out_id, OB1 - 4, off0);
        pack_tap<RE, 8, RE_LAST>(out_id, R0, off0 + kw_step);
        pack_tap4(out_id, OB1, off0 + 2 * kw_step);
    } else {
        pack_tapk(out_id, OB2 - 4, off0);
        pack_tap4(out_id, EB2, off0 + kw_step);
        pack_tap4(out_id, OB2, off0 + 2 * kw_step);
    }
}
#undef S2P_PACR
#endif

// run_row of conv2d_3x3_s2_taps.hpp with the plane modes (the bench's diag: 1 pack only, 2 no pack).
template <uint32_t mode, uint32_t cb_out, uint32_t diag = 0, typename TileL1, typename OutOff>
ALWI void run_row(uint32_t c0, uint32_t n, bool live, uint32_t kw_step, const TileL1& tile_l1, const OutOff& out_off) {
    using namespace conv2d_3x3::s2_taps_mask;
    [[maybe_unused]] uint32_t out_id = 0;
    PACK((out_id = get_output_id(cb_out)));
    for (uint32_t u = 0; u < n; ++u) {
        MATH((llk_math_wait_for_dest_available()));
        if (!live || (u == 0 && c0 == 0)) {
            MATH((_llk_math_clear_dest_section_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
        }
        if (live && diag != 1) {
            const uint32_t c = c0 + u;
            if (u == 0 && c > 0) {
                UNPACK((load(tile_l1(2 * c - 1))));
            }
            if (c > 0) {
                MATH((s2_planes::move_prev<mode>()));
                MATH((release_srcb()));
            }
            UNPACK((load(tile_l1(2 * c))));
            MATH((s2_planes::move_tile<mode, 0>()));
            MATH((release_srcb()));
            UNPACK((load(tile_l1(2 * c + 1))));
            MATH((s2_planes::move_tile<mode, 1>()));
            if (u + 1 == n) {
                MATH((release_srcb()));
            }
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        if constexpr (diag != 2) {
            PACK((s2_planes::pack_taps<mode>(out_id, out_off(u), kw_step)));
        }
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

}  // namespace conv2d_3x3::s2_planes
