// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf 11 `tap_shift_rate` bench candidate: conv2d_3x3::tap_shift::run (HEAD) with knobs.
//   B16   : run the shift with DEST in 16-bit mode (ALU Fp32_enabled = 0, packer Read_32b = 0, Float16_b src
//           strides), as the BH fast tilize does under fp32 DEST. The values are bf8_b (a subset of bf16): the
//           16-bit DEST holds them exactly. Restored at the end.
//   MOVE4 : 4-row MOVB2D wherever a run of rows with one source offset is >= 4 long (overlapping the run's tail);
//           one-row moves for shorter runs. Immediates only (counters stay 0).
//   SEC   : DEST tiles per section (4 at fp32; up to 8 at 16-bit).
//   DIAG  : 0 = real; 1 = pack only (no unpack, no moves: packs the cleared DEST); 2 = unpack + one move + release
//           per load (no row moves).
// Loop, waits, unit order and pack offsets are HEAD's.

#pragma once

#include <cstdint>
#include <utility>

#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../op_base/kernels/conv2d_3x3_tap_shift.hpp"

#ifndef TSR_F32ROW
#define TSR_F32ROW 16
#endif
#ifndef TSR_YDIV
#define TSR_YDIV 2
#endif
namespace conv2d_3x3::tsr {

using conv2d_3x3::tap_shift::from_nb;
using conv2d_3x3::tap_shift::srcrow;

struct MovePlan {
    uint32_t n;
    uint8_t src[64];
    uint8_t dst[64];
    uint8_t four[64];
};

constexpr MovePlan make_plan(int s, bool nb, bool use4) {
    MovePlan p{};
    p.n = 0;
    uint32_t i = 0;
    while (i < 64) {
        if (from_nb(s, i) != nb) {
            ++i;
            continue;
        }
        uint32_t e = i + 1;
        while (e < 64 && from_nb(s, e) == nb && srcrow(s, e) == srcrow(s, i) + (e - i)) {
            ++e;
        }
        const uint32_t base = srcrow(s, i);
        if (use4 && e - i >= 4) {
            for (uint32_t k = i; k < e; k += 4) {
                const uint32_t st = (k + 4 <= e) ? k : e - 4;
                p.src[p.n] = static_cast<uint8_t>(base + (st - i));
                p.dst[p.n] = static_cast<uint8_t>(st);
                p.four[p.n] = 1;
                ++p.n;
            }
        } else {
            for (uint32_t k = i; k < e; ++k) {
                p.src[p.n] = static_cast<uint8_t>(base + (k - i));
                p.dst[p.n] = static_cast<uint8_t>(k);
                p.four[p.n] = 0;
                ++p.n;
            }
        }
        i = e;
    }
    return p;
}

template <int s, bool nb, bool use4>
struct Moves {
    static constexpr MovePlan p = make_plan(s, nb, use4);
};

#ifdef TRISC_MATH
constexpr uint32_t ADDR_MOD_ZERO = ADDR_MOD_7;
template <int s, bool nb, bool use4, uint32_t K>
ALWI void emit_one() {
    constexpr const MovePlan& p = Moves<s, nb, use4>::p;
    if constexpr (K < p.n) {
        if constexpr (p.four[K]) {
            TTI_MOVB2D(0, p.src[K], ADDR_MOD_ZERO, p_movb2d::MOV_4_ROWS, p.dst[K]);
        } else {
            TTI_MOVB2D(0, p.src[K], ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, p.dst[K]);
        }
    }
}
template <int s, bool nb, bool use4, uint32_t... K>
ALWI void emit(std::integer_sequence<uint32_t, K...>) {
    (emit_one<s, nb, use4, K>(), ...);
}
template <int s, bool nb, bool use4>
ALWI void pass() {
    emit<s, nb, use4>(std::make_integer_sequence<uint32_t, 64>{});
}
ALWI void release_srcb() { TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); }
ALWI void dest_tile(uint32_t j) { math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(j); }

template <bool b16>
inline void math_begin() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_ZERO);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
    if constexpr (b16) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(0);
        TTI_ZEROACC(p_zeroacc::CLR_ALL, 0, 0, ADDR_MOD_ZERO, 0);
    } else {
        TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_ZERO, 0);
    }
}
template <bool b16>
inline void math_end() {
    math::clear_dst_reg_addr();
    if constexpr (b16) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
        cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(DST_ACCUM_MODE);
    }
}
#endif

#ifdef TRISC_PACK
template <bool b16>
inline void pack_begin(uint32_t cb_out) {
    if constexpr (b16) {
        const uint32_t id = get_output_id(cb_out);
        const uint32_t tile_size = SCALE_DATUM_SIZE(pack_dst_format[id], 32 * 32);
        constexpr uint32_t bf16 = ckernel::to_underlying(DataFormat::Float16_b);
        reconfig_packer_data_format<DST_ACCUM_MODE>(bf16, pack_dst_format[id], tile_size, 32, 4, false);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
        cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(0);
        set_packer_strides<PackMode::Default>(bf16, 32);
    }
}
template <bool b16>
inline void pack_end(uint32_t cb_out) {
    if constexpr (b16) {
        const uint32_t id = get_output_id(cb_out);
        const uint32_t tile_size = SCALE_DATUM_SIZE(pack_dst_format[id], 32 * 32);
        reconfig_packer_data_format<DST_ACCUM_MODE>(pack_src_format[id], pack_dst_format[id], tile_size, 32, 4, false);
        set_packer_strides<PackMode::Default>(pack_src_format[id], 32);
    }
}
#endif

template <
    bool b16,
    bool move4,
    uint32_t sec,
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
    static_assert(sec % 2 == 0 && sec <= (b16 ? 8 : 4));
    constexpr int32_t tpr = input_width / 32;
    constexpr int d = dilation;
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_out);
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_in, cb_in)));
    UNPACK((conv2d_3x3::tap_shift::unpack_init()));
    MATH((math_begin<b16>()));
    PACK((llk_pack_init(cb_out)));
    PACK((pack_begin<b16>(cb_out)));
    [[maybe_unused]] uint32_t ring_l1 = 0;
    [[maybe_unused]] uint32_t tile_bytes = 0;
    UNPACK((ring_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    [[maybe_unused]] uint32_t out_id = 0;
    PACK((out_id = get_output_id(cb_out)));
    out.reserve_back(rows * in0_k);
    uint32_t j = 0;
    uint32_t dst_off[sec];
    uint32_t waited = 0;
    auto close = [&]() {
        if (j == 0) {
            return;
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        for (uint32_t i = 0; i < j; ++i) {
            PACK((_llk_pack_<DST_SYNC_MODE, false, PackMode::Default>(
                i, get_output_tile_address<true, PackMode::Default>(out_id, dst_off[i]))));
        }
        PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, b16 ? false : DST_ACCUM_MODE>()));
        j = 0;
    };
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
    for (int32_t kh = 0; kh < 3; ++kh) {
        for (uint32_t t = 0; t < rows; ++t) {
            const uint32_t g = t / group_h;
            if (kh < kh_lo[g] || kh >= kh_hi[g]) {
                continue;
            }
            const uint32_t o = o_first + t * 32;
            const int32_t ho0 = o / output_width;
            const int32_t c = (o % output_width) / 32;
            const int32_t h_in = ho0 + kh * dilation - padding;
            const bool live = o < o_end && h_in >= 0 && h_in < input_height;
            const bool has_prev = live && c > 0;
            const bool has_next = live && c + 1 < tpr;
            uint32_t k_cur = 0, k_prev = 0, k_next = 0;
            if (live) {
                const uint32_t i_cur = static_cast<uint32_t>(h_in * tpr + c);
                k_cur = slab_row(i_cur);
                k_prev = has_prev ? slab_row(i_cur - 1) : k_cur;
                k_next = has_next ? slab_row(i_cur + 1) : k_cur;
                uint32_t need = k_cur > k_prev ? k_cur : k_prev;
                need = (need > k_next ? need : k_next) + 1;
                if (need > waited) {
                    in.wait_front(need * ct);
                    waited = need;
                }
            }
            const uint32_t unit = t * in0_k + kh * 2 * ct;
            for (uint32_t ch = 0; ch < ct; ++ch) {
                if (j == 0) {
                    MATH((llk_math_wait_for_dest_available()));
                }
                dst_off[j] = unit + ch;
                dst_off[j + 1] = unit + ct + ch;
                if (live && diag != 1) {
                    UNPACK((conv2d_3x3::tap_shift::load(ring_l1 + (k_cur * ct + ch) * tile_bytes)));
#ifdef TRISC_MATH
                    if constexpr (diag == 2) {
                        dest_tile(j);
                        TTI_MOVB2D(0, 0, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, 0);
                        release_srcb();
                        if (has_prev) {
                            TTI_MOVB2D(0, 0, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, 0);
                            release_srcb();
                        }
                        if (has_next) {
                            TTI_MOVB2D(0, 0, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, 0);
                            release_srcb();
                        }
                    } else {
                        dest_tile(j);
                        pass<-d, false, move4>();
                        dest_tile(j + 1);
                        pass<d, false, move4>();
                        release_srcb();
                        if (has_prev) {
                            dest_tile(j);
                            pass<-d, true, move4>();
                            release_srcb();
                        }
                        if (has_next) {
                            dest_tile(j + 1);
                            pass<d, true, move4>();
                            release_srcb();
                        }
                    }
#endif
                    if (has_prev) {
                        UNPACK((conv2d_3x3::tap_shift::load(ring_l1 + (k_prev * ct + ch) * tile_bytes)));
                    }
                    if (has_next) {
                        UNPACK((conv2d_3x3::tap_shift::load(ring_l1 + (k_next * ct + ch) * tile_bytes)));
                    }
                }
                j += 2;
                if (j == sec) {
                    close();
                }
            }
        }
    }
    close();
    out.push_back(rows * in0_k);
    MATH((math_end<b16>()));
    PACK((pack_end<b16>(cb_out)));
}

// ---------------------------------------------------------------------------------------------------------------
// Row-buffer shift (rowbuf): the shift moves from the math thread to the packer.
//
// For one (kernel row, channel tile) and up to G consecutive units of one image row (columns c0 .. c0 + n - 1),
// DEST holds a slot: the input tiles c0 - 1 .. c0 + G laid out pixel-contiguous per channel half (16-bit DEST):
// pixel q of tile position k (k = c - c0 + 1) and channel half hf at slot row hf * Hs + 32 k + q, Hs = 32 (G + 2).
// Each loaded tile is 16 aligned 4-row MOVB2D (source and DEST rows agree mod 8: Hs, 32 k and the slot base are
// multiples of 8), so the math thread no longer depends on d. Positions outside the image row stay zero (DEST is
// cleared on every section release). The tap tile of unit u, kw is then the 32-pixel window starting at slot row
// 32 (u + 1) + (kw - 1) d of each channel-half block: the packer reads it with a custom replay (W = window start,
// one row per W; Z = channel-half block, Hs rows; Y = row in the window), faces F0 (Z 0, Y 0..15), F1 (Z 1, Y 0..15),
// F2 (Z 0, Y 16..31), F3 (Z 1, Y 16..31). Every input tile is unpacked once per (kernel row, channel tile, group).
// ---------------------------------------------------------------------------------------------------------------

#ifdef TRISC_PACK
#define TSR_PACR(last)                  \
    TTI_PACR(                           \
        p_pacr::CFG_CTXT_0,             \
        p_pacr::NO_ROW_PAD_ZERO,        \
        p_pacr::DST_ACCESS_NORMAL_MODE, \
        ADDR_MOD_0,                     \
        p_pacr::ADDR_CNT_CTXT_0,        \
        0,                              \
        p_pacr::ALL_INTF_ACTIVE,        \
        0,                              \
        0,                              \
        0,                              \
        0,                              \
        last)
constexpr uint32_t RB_REPLAY = 0;
constexpr uint32_t RB_REPLAY_LEN = 22;
template <uint32_t Hs, bool b16>
inline void rowbuf_pack_begin(uint32_t cb_out) {
    pack_begin<b16>(cb_out);
    // ADDR_MOD_0 (llk_pack_init): Y += 4 per PACR (4 rows, one per interface).
    constexpr uint32_t row_bytes = b16 ? 32 : TSR_F32ROW;  // one DEST row
    constexpr uint32_t y_stride = row_bytes, z_stride = Hs * row_bytes, w_stride = row_bytes;
    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    load_replay_buf(RB_REPLAY, RB_REPLAY_LEN, [] {
        TSR_PACR(0);  // F0: Z 0, Y 0..15
        TSR_PACR(0);
        TSR_PACR(0);
        TSR_PACR(0);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
        TSR_PACR(0);                                      // F1: Z 1, Y 0..15
        TSR_PACR(0);
        TSR_PACR(0);
        TSR_PACR(0);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);  // Z 0 (Y is 16)
        TSR_PACR(0);                                      // F2: Z 0, Y 16..31
        TSR_PACR(0);
        TSR_PACR(0);
        TSR_PACR(0);
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 16);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
        TSR_PACR(0);                                      // F3: Z 1, Y 16..31
        TSR_PACR(0);
        TSR_PACR(0);
        TSR_PACR(1);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0, Z 0 for the next tap
    });
}
template <uint32_t Hs, uint32_t pmode>
ALWI void rowbuf_pack_tap(uint32_t out_id, uint32_t w, uint32_t off) {
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, w);
    program_packer_destination(get_output_tile_address<true, PackMode::Default>(out_id, off));
    if constexpr (pmode == 0) {
        lltt::replay(RB_REPLAY, RB_REPLAY_LEN);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
    } else {
        ckernel::ckernel_template::run();
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    }
}
template <bool b16>
inline void rowbuf_pack_end(uint32_t cb_out) {
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
    pack_end<b16>(cb_out);
    llk_pack_init(cb_out);
}
#endif

#ifdef TRISC_MATH
// The 64 rows of one SrcB tile into its slot position (DEST offset at the position's row 0 of block 0).
template <uint32_t Hs, uint32_t... K>
ALWI void rowbuf_moves(std::integer_sequence<uint32_t, K...>) {
    // K = 4-row group 0..15: source rows 4K; face f = K / 4: F0 -> block 0 rows 0..15, F1 -> block 1 rows 0..15,
    // F2 -> block 0 rows 16..31, F3 -> block 1 rows 16..31.
    ((TTI_MOVB2D(0, 4 * K, ADDR_MOD_ZERO, p_movb2d::MOV_4_ROWS, ((K / 4) % 2) * Hs + ((K / 4) / 2) * 16 + 4 * (K % 4))),
     ...);
}
ALWI void dest_row(uint32_t row) { TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, row + get_dest_buffer_base()); }
#endif

// SrcB placed at row offset delta (the unpacker's CH1 Y): input row r at SrcB row (r + delta) % 64 (if it wraps).
// 4-row moves where the 4 source rows do not wrap, one-row moves elsewhere.
constexpr MovePlan make_rowbuf_plan(uint32_t Hs, uint32_t delta) {
    MovePlan p{};
    p.n = 0;
    for (uint32_t g = 0; g < 16; ++g) {
        const uint32_t f = g / 4;
        const uint32_t fp = (f % 2) * Hs + (f / 2) * 16;
        const uint32_t src = 4 * g + delta;
        if (src + 3 < 64) {
            p.src[p.n] = static_cast<uint8_t>(src);
            p.dst[p.n] = static_cast<uint8_t>(fp + 4 * (g % 4));
            p.four[p.n] = 1;
            ++p.n;
        } else {
            for (uint32_t j = 0; j < 4; ++j) {
                p.src[p.n] = static_cast<uint8_t>((src + j) % 64);
                p.dst[p.n] = static_cast<uint8_t>(fp + 4 * (g % 4) + j);
                p.four[p.n] = 0;
                ++p.n;
            }
        }
    }
    return p;
}
template <uint32_t Hs, uint32_t delta>
struct RbMoves {
    static constexpr MovePlan p = make_rowbuf_plan(Hs, delta);
};
#ifdef TRISC_MATH
template <uint32_t Hs, uint32_t delta, uint32_t K>
ALWI void rb_emit_one() {
    constexpr const MovePlan& p = RbMoves<Hs, delta>::p;
    if constexpr (K < p.n) {
        if constexpr (p.four[K]) {
            TTI_MOVB2D(0, p.src[K], ADDR_MOD_ZERO, p_movb2d::MOV_4_ROWS, p.dst[K]);
        } else {
            TTI_MOVB2D(0, p.src[K], ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, p.dst[K]);
        }
    }
}
template <uint32_t Hs, uint32_t delta, uint32_t... K>
ALWI void rb_emit(std::integer_sequence<uint32_t, K...>) {
    (rb_emit_one<Hs, delta, K>(), ...);
}
#endif
#ifdef TRISC_UNPACK
// SrcB rows written from row `delta` (CH1 Y stride = one row of the SrcB format). Returns the old Y stride word.
inline uint32_t unpack_offset_begin(uint32_t cb_in, uint32_t delta) {
    const uint32_t fmt = unpack_dst_format[get_operand_id(cb_in)] & 0xF;
    const uint32_t datum = (fmt == (uint32_t)DataFormat::Float32 || fmt == (uint32_t)DataFormat::Tf32) ? 4 : 2;
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    const uint32_t old = cfg[UNP1_ADDR_CTRL_XY_REG_1_Ystride_ADDR32];
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_XY_REG_1_Ystride_RMW>(16 * datum / TSR_YDIV);
    TT_SETADCXY(p_setadc::UNP_B, delta, 0, 0, 0, 0b1000);
    return old;
}
inline void unpack_offset_end(uint32_t old) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_XY_REG_1_Ystride_RMW>(old >> UNP1_ADDR_CTRL_XY_REG_1_Ystride_SHAMT);
    TTI_SETADCXY(p_setadc::UNP_B, 0, 0, 0, 0, 0b1000);
}
#endif

template <
    uint32_t G,
    uint32_t rdiag,
    bool b16,
    uint32_t pmode,
    bool uoff,
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
ALWI void run_rowbuf(
    DataflowBuffer& in,
    DataflowBuffer& out,
    uint32_t rows,
    uint32_t o_first,
    uint32_t o_end,
    const conv2d_3x3::padded_slab::Plan& plan,
    const uint8_t (&kh_lo)[num_groups],
    const uint8_t (&kh_hi)[num_groups]) {
    static_assert(dilation > 0 && dilation < 32 && padding == dilation && input_width % 32 == 0);
    constexpr int32_t tpr = input_width / 32;
    constexpr uint32_t Gm = G < static_cast<uint32_t>(tpr) ? G : static_cast<uint32_t>(tpr);
    // sigma: the slot's data offset (rows), so both windows start on a multiple of 4 (even d only).
    constexpr uint32_t sigma = (4 - static_cast<uint32_t>(dilation) % 4) % 4;
    constexpr uint32_t Hs = 32 * (Gm + 2) + (sigma ? 8 : 0);
    constexpr uint32_t SLOT = 2 * Hs;
    constexpr uint32_t SLOTS = (b16 ? 512 : 256) / SLOT;
    static_assert(SLOTS >= 1);
    constexpr uint32_t MAX_TAPS = SLOTS * 2 * Gm;
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_out);
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_in, cb_in)));
    UNPACK((conv2d_3x3::tap_shift::unpack_init()));
    [[maybe_unused]] uint32_t old_ystride = 0;
    if constexpr (uoff) {
        static_assert(dilation % 2 == 0, "uoff: even d");
        UNPACK((old_ystride = unpack_offset_begin(cb_in, sigma)));
    }
    MATH((math_begin<b16>()));
    PACK((llk_pack_init(cb_out)));
    PACK((rowbuf_pack_begin<Hs, b16>(cb_out)));
    [[maybe_unused]] uint32_t ring_l1 = 0;
    [[maybe_unused]] uint32_t tile_bytes = 0;
    UNPACK((ring_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    [[maybe_unused]] uint32_t out_id = 0;
    PACK((out_id = get_output_id(cb_out)));
    out.reserve_back(rows * in0_k);
    uint32_t nslots = 0, ntaps = 0;
    [[maybe_unused]] uint16_t tap_w[MAX_TAPS];
    [[maybe_unused]] uint16_t tap_off[MAX_TAPS];
    uint32_t waited = 0;
    auto close = [&]() {
        if (nslots == 0) {
            return;
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        if constexpr (rdiag != 3) {
            for (uint32_t i = 0; i < ntaps; ++i) {
                PACK((rowbuf_pack_tap<Hs, pmode>(out_id, tap_w[i], tap_off[i])));
            }
        }
        PACK((_llk_pack_dest_section_done_<DST_SYNC_MODE, b16 ? false : DST_ACCUM_MODE>()));
        nslots = 0;
        ntaps = 0;
    };
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
            const int32_t ho0 = o / output_width;
            const int32_t c0 = (o % output_width) / 32;
            const int32_t h_in = ho0 + kh * dilation - padding;
            const bool live = o < o_end && h_in >= 0 && h_in < input_height;
            // The group: up to Gm units of image row ho0 (consecutive columns), all live or all not.
            uint32_t n = 1;
            while (n < Gm && t + n < rows && !dead(t + n, kh) && c0 + static_cast<int32_t>(n) < tpr &&
                   ((o_first + (t + n) * 32 < o_end) == (o < o_end))) {
                ++n;
            }
            const int32_t cl = c0 > 0 ? c0 - 1 : 0;
            const int32_t cr = c0 + static_cast<int32_t>(n) < tpr ? c0 + static_cast<int32_t>(n) : tpr - 1;
            uint32_t i_l = 0;
            if (live) {
                i_l = static_cast<uint32_t>(h_in * tpr + cl);
                const uint32_t need = slab_row(static_cast<uint32_t>(h_in * tpr + cr)) + 1;
                if (need > waited) {
                    in.wait_front(need * ct);
                    waited = need;
                }
            }
            for (uint32_t ch = 0; ch < ct; ++ch) {
                if (nslots == 0) {
                    MATH((llk_math_wait_for_dest_available()));
                }
                const uint32_t B = nslots * SLOT;
                if (live && rdiag != 1) {
                    for (int32_t c = cl; c <= cr; ++c) {
                        UNPACK((conv2d_3x3::tap_shift::load(
                            ring_l1 + (slab_row(i_l + static_cast<uint32_t>(c - cl)) * ct + ch) * tile_bytes)));
#ifdef TRISC_MATH
                        dest_row(B + sigma + 32 * static_cast<uint32_t>(c - c0 + 1));
                        if constexpr (rdiag == 2) {
                            TTI_MOVB2D(0, 0, ADDR_MOD_ZERO, p_movb2d::MOV_4_ROWS, 0);
                        } else if constexpr (uoff) {
                            rb_emit<Hs, sigma>(std::make_integer_sequence<uint32_t, 64>{});
                        } else {
                            rowbuf_moves<Hs>(std::make_integer_sequence<uint32_t, 16>{});
                        }
                        release_srcb();
#endif
                    }
                }
                for (uint32_t u = 0; u < n; ++u) {
                    const uint32_t unit = (t + u) * in0_k + kh * 2 * ct + ch;
                    tap_w[ntaps] = static_cast<uint16_t>(B + sigma + 32 * (u + 1) - dilation);
                    tap_off[ntaps] = static_cast<uint16_t>(unit);
                    ++ntaps;
                    tap_w[ntaps] = static_cast<uint16_t>(B + sigma + 32 * (u + 1) + dilation);
                    tap_off[ntaps] = static_cast<uint16_t>(unit + ct);
                    ++ntaps;
                }
                if (++nslots == SLOTS) {
                    close();
                }
            }
            t += n;
        }
    }
    close();
    out.push_back(rows * in0_k);
    MATH((math_end<b16>()));
    PACK((rowbuf_pack_end<b16>(cb_out)));
    if constexpr (uoff) {
        UNPACK((unpack_offset_end(old_ystride)));
    }
}

}  // namespace conv2d_3x3::tsr
