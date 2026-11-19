// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf 11 `tap_shift_rate` candidate 2 (rb2): the row-buffer tap shift in fp32 DEST (no DEST / packer format switch).
//
// For one (kernel row kh, channel tile ch) and a group of n <= G consecutive units of one image row (columns
// c0 .. c0 + n - 1), one DEST section holds a slot: the pixels [32 c0 - d, 32 (c0 + n) + d) of input image row h, laid
// out pixel-contiguous per channel half (block hf at slot row hf * Hs). Every input tile of the group (and the d
// edge pixels of its row neighbours) is unpacked ONCE and moved into the slot; out-of-row pixels stay zero (DEST is
// cleared on every section release). The packer then reads each tap tile (unit u, kw) as the 32-pixel window of both
// blocks that starts at the window's first pixel: W = window row, Z = block (Hs rows), Y = row in the window; faces
// F0 (Z 0, Y 0..15), F1 (Z 1, Y 0..15), F2 (Z 0, Y 16..31), F3 (Z 1, Y 16..31).
//
// Hardware rules (Blackhole, measured in this bench):
//   * a 4-row PACR (all four interfaces) reads exactly only from a window row that is a multiple of 4; a 2-row PACR
//     (two interfaces) from a multiple of 2;
//   * a 4-row MOVB2D is exact only when its SrcB and DEST rows agree mod 8 (the DEST offset register included);
//   * in fp32 DEST the packer's Y / Z / W strides count 16 bytes per DEST row.
// Two layouts follow:
//   A (d % 4 == 0): pixel p at block row s + p, s = d rounded up to 8: every tile's rows land 8-aligned (16 aligned
//     4-row MOVB2D per tile); both windows (s + 32 u + (kw - 1) d) start on a multiple of 4.
//   B (otherwise): pixel p at block row d + p: kw = 0 windows start at 32 u (4-row PACR), kw = 2 at 32 u + 2 d
//     (4-row PACR for even d, 2-row for odd d); the rows move one at a time (4 replays of a 16-move run per tile).

#pragma once

#include <cstdint>
#include <utility>

#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../op_base/kernels/conv2d_3x3_padded_slab.hpp"

namespace conv2d_3x3::rb2 {

template <int32_t d>
constexpr bool layout_a() {
    return d % 4 == 0;
}
// Rows of one block for n units.
template <int32_t d>
constexpr uint32_t block_rows(uint32_t n) {
    return layout_a<d>() ? ((((d + 7) / 8) * 8 + 32 * n + d + 7) / 8) * 8 : ((32 * n + 2 * d + 3) / 4) * 4;
}
// Block row of pixel 0 of the group's first tile.
template <int32_t d>
constexpr uint32_t row0() {
    return layout_a<d>() ? static_cast<uint32_t>(((d + 7) / 8) * 8) : static_cast<uint32_t>(d);
}
// Units per slot: the two blocks of one slot fit a DEST half (256 fp32 rows).
template <int32_t d, uint32_t tpr, uint32_t gmax>
constexpr uint32_t group_max() {
    uint32_t g = gmax < tpr ? gmax : tpr;
    while (g > 1 && 2 * block_rows<d>(g) > 256) {
        --g;
    }
    return g;
}

#ifdef TRISC_UNPACK
inline void unpack_init() {
    TT_SETADCXX(p_setadc::UNP_B, 32 * 32 - 1, 0x0);
    ckernel::ckernel_template tmp(1, 1, TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}
inline void load(uint32_t l1) { _llk_unpack_AB_<BroadcastType::NONE>((l1 >> 4) - 1, (l1 >> 4) - 1); }
#endif

#ifdef TRISC_MATH
constexpr uint32_t ADDR_MOD_ZERO = ADDR_MOD_7;
constexpr uint32_t ADDR_MOD_STEP = ADDR_MOD_6;
ALWI void dest_row(uint32_t row) { TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, row + get_dest_buffer_base()); }
ALWI void release_srcb() { TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); }

template <int32_t d>
inline void math_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_ZERO);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 1}, .dest = {.incr = 1}}.set(ADDR_MOD_STEP);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
    if constexpr (!layout_a<d>()) {
        load_replay_buf(0, 16, [] {
            for (int k = 0; k < 16; ++k) {
                TTI_MOVB2D(0, 0, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
        });
    }
    TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_ZERO, 0);
}

// Layout A, 4-row groups g of SrcB rows [4g, 4g + 4) (face f = g / 4, rows j0 = 4 (g % 4)) with pixel
// q = 16 (f / 2) + j0 in [q_lo, q_hi), into block row Hs (f % 2) + base + q (base relative to the DEST offset).
template <uint32_t Hs, uint32_t base, uint32_t q_lo, uint32_t q_hi, uint32_t g>
ALWI void a_group() {
    constexpr uint32_t f = g / 4;
    constexpr uint32_t q = 16 * (f / 2) + 4 * (g % 4);
    if constexpr (q >= q_lo && q < q_hi) {
        TTI_MOVB2D(0, 4 * g, ADDR_MOD_ZERO, p_movb2d::MOV_4_ROWS, Hs * (f % 2) + base + q);
    }
}
template <uint32_t Hs, uint32_t base, uint32_t q_lo, uint32_t q_hi, uint32_t... G>
ALWI void a_moves(std::integer_sequence<uint32_t, G...>) {
    (a_group<Hs, base, q_lo, q_hi, G>(), ...);
}
// Layout B, single rows r (face f = r / 16) with pixel q = 16 (f / 2) + r % 16 in [q_lo, q_hi).
template <uint32_t Hs, uint32_t base, uint32_t q_lo, uint32_t q_hi, uint32_t r>
ALWI void b_row() {
    constexpr uint32_t f = r / 16;
    constexpr uint32_t q = 16 * (f / 2) + r % 16;
    if constexpr (q >= q_lo && q < q_hi) {
        TTI_MOVB2D(0, r, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, Hs * (f % 2) + base + q);
    }
}
template <uint32_t Hs, uint32_t base, uint32_t q_lo, uint32_t q_hi, uint32_t... R>
ALWI void b_rows(std::integer_sequence<uint32_t, R...>) {
    (b_row<Hs, base, q_lo, q_hi, R>(), ...);
}
// A whole tile at slot row `at` (its pixel 0): layout A 16 aligned 4-row moves; layout B 4 replays of 16 one-row moves.
template <int32_t d, uint32_t Hs>
ALWI void move_tile(uint32_t at) {
    if constexpr (layout_a<d>()) {
        dest_row(at);
        a_moves<Hs, 0, 0, 32>(std::make_integer_sequence<uint32_t, 16>{});
    } else {
        dest_row(at);
        lltt::replay(0, 16);  // F0 -> block 0 pixels 0..15
        dest_row(at + Hs);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        lltt::replay(0, 16);  // F1 -> block 1 pixels 0..15
        dest_row(at + 16);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        lltt::replay(0, 16);  // F2 -> block 0 pixels 16..31
        dest_row(at + Hs + 16);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        lltt::replay(0, 16);  // F3 -> block 1 pixels 16..31
    }
    release_srcb();
}
// The left neighbour's last d pixels, its tile placed with pixel 0 at slot row `at` - 32 (DEST offset at `at` - r0,
// the slot base, so no offset is negative).
template <int32_t d, uint32_t Hs>
ALWI void move_prev(uint32_t slot) {
    constexpr uint32_t r0 = row0<d>();
    dest_row(slot);
    if constexpr (layout_a<d>()) {
        a_moves<Hs, r0 - 32, 32 - d, 32>(std::make_integer_sequence<uint32_t, 16>{});
    } else {
        b_rows<Hs, r0 - 32, 32 - d, 32>(std::make_integer_sequence<uint32_t, 64>{});
    }
    release_srcb();
}
// The right neighbour's first d pixels (its pixel 0 at slot row `at`).
template <int32_t d, uint32_t Hs>
ALWI void move_next(uint32_t at) {
    dest_row(at);
    if constexpr (layout_a<d>()) {
        a_moves<Hs, 0, 0, d>(std::make_integer_sequence<uint32_t, 16>{});
    } else {
        b_rows<Hs, 0, 0, d>(std::make_integer_sequence<uint32_t, 64>{});
    }
    release_srcb();
}
#endif

#ifdef TRISC_PACK
#define RB2_PACR(am, intf, last)        \
    TTI_PACR(                           \
        p_pacr::CFG_CTXT_0,             \
        p_pacr::NO_ROW_PAD_ZERO,        \
        p_pacr::DST_ACCESS_NORMAL_MODE, \
        am,                             \
        p_pacr::ADDR_CNT_CTXT_0,        \
        0,                              \
        intf,                           \
        0,                              \
        0,                              \
        0,                              \
        0,                              \
        last)
// Replay [0, 22): one 4-row-PACR window tap. [22, 30): 8 two-row PACRs (one face); [30]: the tile's last 2-row PACR.
constexpr uint32_t R4 = 0, R4_LEN = 22, R2 = 22, R2_LAST = 30;
template <uint32_t Hs>
inline void pack_init(uint32_t out_id) {
    // One DEST row = 16 datums of the packer's source format (as set_packer_strides).
    const uint32_t row = FACE_C_DIM * datum_size_in_bytes(pack_src_format[out_id]);
    const uint32_t y_stride = row, z_stride = Hs * row, w_stride = row;
    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
    addr_mod_pack_t{.y_src = {.incr = 2}}.set(ADDR_MOD_3);  // 2-row PACR step (ADDR_MOD_0: 4 rows, llk_pack_init)
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    load_replay_buf(R4, R4_LEN + 9, [] {
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F0: Z 0, Y 0..15
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);   // Y 0
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);   // Z 1
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F1: Z 1, Y 0..15
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);   // Z 0 (Y is 16)
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F2: Z 0, Y 16..31
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 16);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);   // Z 1
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F3: Z 1, Y 16..31
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        RB2_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 1);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
        for (int k = 0; k < 8; ++k) {
            RB2_PACR(ADDR_MOD_3, p_pacr::TWO_INTFS_ACTIVE, 0);
        }
        RB2_PACR(ADDR_MOD_3, p_pacr::TWO_INTFS_ACTIVE, 1);
    });
}
// One tap: the window at slot row w (of the packer's current DEST half) into output tile `off` of cb_out.
template <bool two_row>
ALWI void pack_tap(uint32_t out_id, uint32_t w, uint32_t off) {
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, w);
    program_packer_destination(get_output_tile_address<true, PackMode::Default>(out_id, off));
    if constexpr (!two_row) {
        lltt::replay(R4, R4_LEN);
    } else {
        lltt::replay(R2, 8);                              // F0
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
        lltt::replay(R2, 8);                              // F1
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);  // Z 0
        lltt::replay(R2, 8);                              // F2
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 16);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
        lltt::replay(R2, 7);                              // F3
        lltt::replay(R2_LAST, 1);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
    }
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);  // Z 0
}
inline void pack_uninit(uint32_t cb_out) {
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
    llk_pack_init(cb_out);  // default strides, ADDR_MODs and MOP
}
#endif

template <
    uint32_t gmax,
    uint32_t diag,
    uint32_t ct,
    uint32_t in0_k,
    uint32_t cb_in,
    uint32_t cb_out,
    uint32_t output_width,
    int32_t input_height,
    int32_t input_width,
    int32_t padding,
    int32_t dilation,
    uint32_t group_h,
    uint32_t num_groups>
ALWI void run(
    DataflowBuffer& in,
    DataflowBuffer& out,
    uint32_t rows,
    uint32_t o_first,
    uint32_t o_end,
    const conv2d_3x3::padded_slab::Plan& plan,
    const uint8_t (&kh_lo)[num_groups],
    const uint8_t (&kh_hi)[num_groups]) {
    static_assert(dilation > 0 && dilation < 32 && padding == dilation && input_width % 32 == 0);
    constexpr int32_t d = dilation;
    constexpr uint32_t tpr = static_cast<uint32_t>(input_width / 32);
    constexpr uint32_t G = group_max<d, tpr, gmax>();
    constexpr uint32_t Hs = block_rows<d>(G);
    static_assert(2 * Hs <= 256, "one slot per fp32 DEST half");
    constexpr uint32_t r0 = row0<d>();
    // Window rows (slot-relative) of unit u: kw = 0 at r0 + 32 u - d, kw = 2 at r0 + 32 u + d.
    constexpr bool kw2_two_row = !layout_a<d>() && (d % 2 == 1);
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_out);
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_in, cb_in)));
    UNPACK((unpack_init()));
    MATH((math_init<d>()));
    PACK((llk_pack_init(cb_out)));
    [[maybe_unused]] uint32_t out_id = 0;
    PACK((out_id = get_output_id(cb_out)));
    PACK((pack_init<Hs>(out_id)));
    [[maybe_unused]] uint32_t ring_l1 = 0;
    [[maybe_unused]] uint32_t tile_bytes = 0;
    UNPACK((ring_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    out.reserve_back(rows * in0_k);
    // Slab row of input tile i = i + off[kh] (a band's tiles are one run of the ring).
    uint32_t waited = 0;
    auto slab_row = [&](uint32_t i) {
        uint32_t base = 0;
        for (uint32_t sg = 0; sg < plan.num_segs; ++sg) {
            if (i >= plan.seg_first[sg] && i < plan.seg_first[sg] + plan.seg_count[sg]) {
                return base + (i - plan.seg_first[sg]);
            }
            base += plan.seg_count[sg];
        }
        return base;
    };
    auto dead = [&](uint32_t t, int32_t kh) {
        const uint32_t g = t / group_h;
        return kh < kh_lo[g] || kh >= kh_hi[g];
    };
    for (int32_t kh = 0; kh < 3; ++kh) {
        uint32_t t = 0;
        while (t < rows) {
            if (dead(t, kh)) {
                ++t;
                continue;
            }
            const uint32_t o = o_first + t * 32;
            const uint32_t ho0 = o / output_width;
            const uint32_t c0 = (o % output_width) / 32;
            const int32_t h_in = static_cast<int32_t>(ho0) + kh * dilation - padding;
            const bool in_block = o < o_end;
            const bool live = in_block && h_in >= 0 && h_in < input_height;
            uint32_t n = 1;
            while (n < G && t + n < rows && c0 + n < tpr && !dead(t + n, kh) &&
                   ((o_first + (t + n) * 32 < o_end) == in_block)) {
                ++n;
            }
            const bool has_prev = live && c0 > 0;
            const bool has_next = live && c0 + n < tpr;
            uint32_t k_first = 0;
            if (live) {
                const uint32_t i0 = static_cast<uint32_t>(h_in) * tpr + c0;
                k_first = slab_row(has_prev ? i0 - 1 : i0);
                const uint32_t need = slab_row(has_next ? i0 + n : i0 + n - 1) + 1;
                if (need > waited) {
                    in.wait_front(need * ct);
                    waited = need;
                }
            }
            for (uint32_t ch = 0; ch < ct; ++ch) {
                MATH((llk_math_wait_for_dest_available()));
                if (live && diag != 1) {
                    uint32_t k = k_first;
                    const uint32_t nl = (has_prev ? 1 : 0) + n + (has_next ? 1 : 0);
                    if constexpr (diag == 2 || diag == 4) {
                        for (uint32_t l = 0; l < nl; ++l, ++k) {
                            UNPACK((load(ring_l1 + (k * ct + ch) * tile_bytes)));
#ifdef TRISC_MATH
                            TTI_MOVB2D(0, 0, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, 0);
                            release_srcb();
#endif
                        }
                    } else {
                        if (has_prev) {
                            UNPACK((load(ring_l1 + (k * ct + ch) * tile_bytes)));
                            MATH((move_prev<d, Hs>(0)));
                            ++k;
                        }
                        for (uint32_t u = 0; u < n; ++u, ++k) {
                            UNPACK((load(ring_l1 + (k * ct + ch) * tile_bytes)));
                            MATH((move_tile<d, Hs>(r0 + 32 * u)));
                        }
                        if (has_next) {
                            UNPACK((load(ring_l1 + (k * ct + ch) * tile_bytes)));
                            MATH((move_next<d, Hs>(r0 + 32 * n)));
                        }
                    }
                }
                MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                PACK((llk_packer_wait_for_math_done()));
                if constexpr (diag != 3 && diag != 4) {
                    for (uint32_t u = 0; u < n; ++u) {
                        const uint32_t unit = (t + u) * in0_k + kh * 2 * ct + ch;
                        PACK((pack_tap<false>(out_id, r0 + 32 * u - d, unit)));
                        PACK((pack_tap<kw2_two_row>(out_id, r0 + 32 * u + d, unit + ct)));
                    }
                }
                PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
            }
            t += n;
        }
    }
    out.push_back(rows * in0_k);
    MATH((math::clear_dst_reg_addr()));
    PACK((pack_uninit(cb_out)));
}

}  // namespace conv2d_3x3::rb2
