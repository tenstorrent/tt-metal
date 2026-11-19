// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stride-2 taps from DEST (the descriptor's S2_TAPS): the three kw taps of a stride-2 (d 1, p 1) output tile are built
// straight from the input's TILE slab on the FPU and the packer, with no row-major round trip (no untilize, no window
// gather, no strided tilize).
//
// Output tile (image row ho, column tile c) of kernel row kh reads input row h = 2 ho + kh - 1, pixels 64 c - 1 + kw
// + 2 r (r = 0..31): kw = 1 the even pixels of input tiles 2c, 2c + 1 (E), kw = 2 their odd pixels (O), kw = 0 the odd
// pixels shifted down one row (pixel 31 of tile 2c - 1 first, zero at the image's left edge). W_in % 64 == 0: the pair
// is two whole input tiles of one image row.
//
// DEST (fp32, one half-sync section per output tile, one block per channel half, HS rows apart): the pair's pixels
// 0..63 at block rows R0.. (16 4-row MOVB2D per tile, SrcB and DEST rows agreeing mod 8), and an O plane of its odd
// pixels at block rows OB.. (one-row MOVB2D), the previous tile's pixel 31 one row ahead of it (OB - 1). The packer
// reads E (kw = 1) from the pixel buffer with interface mask 0101 (interfaces 0 and 2 read rows y, y + 2 of the
// 4-aligned group at y: the BH tilize pack's own unswizzle), 8 PACRs per face; O (kw = 2) as a 4-row window of the O
// plane, 4 PACRs per face; kw = 0 as the O plane window one row earlier: per face one 1-row PACR (interface 3), 3 4-row
// PACRs and one 3-row PACR (interfaces 0..2). Values move unchanged (bf8_b / bf16 -> SrcB -> fp32 DEST -> the same
// format; each 16-datum row keeps its datums and so its bf8_b shared exponent): bit-identical to the untilize / window
// tilize round trip.
//
// SrcB banks: within a run of consecutive output tiles of one image row, input tile 2c + 1 stays in its bank across the
// section boundary and gives the next tile's previous pixels, so a run of n tiles loads 2 n tiles (+ 1 when c0 > 0).
// A section whose packer reads rows no move writes (an input row outside the image, the left image edge's previous
// pixel) is cleared by math after it is acquired; every section is cleared again on release.
//
// RAW LLK (capability): no helper decimates rows. compute_kernel_lib::tilize / untilize read and write whole tiles or
// rows at CB pointers; none unpacks a whole tile into one Src bank outside the matmul, places rows at a DEST row
// offset, or packs a tile from a DEST window with a packer interface mask. The helper path is the untilize + window
// gather + fast tilize this replaces. Measured (Blackhole p150b, LoFi, fp32 DEST, bit-identical): isolated (one core,
// slab resident, perf_experiments/stride2_dest_taps/) 40.4 ns per tap tile (bf8_b, 4 channel tiles; 46.7 at 1, 37.5
// bf16), pack and math nearly level (pack only 39.8, loads + moves only 36.2); the first design, every tap from the
// pixel buffer with interface masks (no O plane), 47.3; the untilize + window tilize it replaces ~57 ns per in0 tile on
// res3.0. In the op (unzoned, 2 alternating pairs): res3.0 @ LoFi 17.45 -> 13.41 us, res4.0 @ LoFi 15.02 -> 12.53 us.
// See conv2d_3x3_tap_shift.hpp for the shared hardware rules (4-row MOVB2D mod 8, packer strides in source-format
// rows, SETRWC vs SrcB valid).

#pragma once

#include <cstdint>
#include <utility>

#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

namespace conv2d_3x3::s2_taps {

constexpr uint32_t R0 = 8;           // block row of the pair's pixel 0
constexpr uint32_t HS = 72;          // block stride (the packer's Z): blocks 0 / 1 of both regions
constexpr uint32_t OB = 2 * HS + 4;  // block row of the O plane's pixel 1 (block 1: OB + HS + 63 < 256 rows)

#ifdef TRISC_UNPACK
inline void unpack_init() {
    TTI_SETADCXX(p_setadc::UNP_B, 32 * 32 - 1, 0x0);
    ckernel::ckernel_template tmp(1, 1, TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}
// One whole tile at byte address `l1` into the next SrcB bank.
inline void load(uint32_t l1) { _llk_unpack_AB_<BroadcastType::NONE>((l1 >> 4) - 1, (l1 >> 4) - 1); }
#endif

#ifdef TRISC_MATH
constexpr uint32_t ADDR_MOD_ZERO = ADDR_MOD_7;
ALWI void dest_row(uint32_t row) { TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, row + get_dest_buffer_base()); }
ALWI void release_srcb() { TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); }

inline void math_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_ZERO);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// 4-row group g (SrcB rows [4 g, 4 g + 4), face f = g / 4, pixels q..q + 3) to block row HS (f % 2) + base + q.
template <uint32_t base, uint32_t q_lo, uint32_t q_hi, uint32_t g>
ALWI void group() {
    constexpr uint32_t f = g / 4;
    constexpr uint32_t q = 16 * (f / 2) + 4 * (g % 4);
    if constexpr (q >= q_lo && q < q_hi) {
        TTI_MOVB2D(0, 4 * g, ADDR_MOD_ZERO, p_movb2d::MOV_4_ROWS, HS * (f % 2) + base + q);
    }
}
template <uint32_t base, uint32_t q_lo, uint32_t q_hi, uint32_t... G>
ALWI void moves(std::integer_sequence<uint32_t, G...>) {
    (group<base, q_lo, q_hi, G>(), ...);
}
// SrcB row r of a tile (face f = r / 16: pixel p = 16 (f / 2) + r % 16, half f % 2): an odd pixel to the O plane row
// OB + HS (f % 2) + 16 k + p / 2.
template <uint32_t k, uint32_t r>
ALWI void odd_row() {
    constexpr uint32_t f = r / 16;
    constexpr uint32_t p = 16 * (f / 2) + r % 16;
    if constexpr (p % 2 == 1) {
        TTI_MOVB2D(0, r, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, OB + HS * (f % 2) + 16 * k + p / 2);
    }
}
template <uint32_t k, uint32_t... R>
ALWI void odd_rows(std::integer_sequence<uint32_t, R...>) {
    (odd_row<k, R>(), ...);
}
// The pair's tile k (0: 2c, 1: 2c + 1): pixels to block row R0 + 32 k.., odd pixels to the O plane.
template <uint32_t k>
ALWI void move_pair_tile() {
    dest_row(0);
    moves<R0 + 32 * k, 0, 32>(std::make_integer_sequence<uint32_t, 16>{});
    odd_rows<k>(std::make_integer_sequence<uint32_t, 64>{});
}
// The previous tile's pixel 31 (SrcB rows 47, 63) to the O plane's row OB - 1.
ALWI void move_prev() {
    dest_row(0);
    TTI_MOVB2D(0, 47, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, OB - 1);
    TTI_MOVB2D(0, 63, ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, OB - 1 + HS);
}
#endif

#ifdef TRISC_PACK
#define CONV2D_3X3_S2_PACR(intf, last)  \
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
// Replay: [0, 8) one E face (8 x 0101), [8] E last; [9, 13) one O face (4 x 1111), [13] O last; [14, 19) one kw = 0
// face (1000, 3 x 1111, 0111), [19] kw = 0 last.
constexpr uint32_t RE = 0, RE_LAST = 8, RO = 9, RO_LAST = 13, RK = 14, RK_LAST = 19;
constexpr uint32_t INTF_EVEN = 0b0101, INTF_3 = 0b1000, INTF_012 = 0b0111;

inline void pack_init(uint32_t out_id) {
    // One DEST row = 16 datums of the packer's source format. Y = row, Z = block, W = window base.
    const uint32_t row = FACE_C_DIM * datum_size_in_bytes(pack_src_format[out_id]);
    const uint32_t y_stride = row, z_stride = HS * row, w_stride = row;
    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
    // ADDR_MOD_0 (llk_pack_init): Y += 4 per PACR, whatever its interface mask.
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    load_replay_buf(RE, 20, [] {
        for (int k = 0; k < 8; ++k) {
            CONV2D_3X3_S2_PACR(INTF_EVEN, 0);
        }
        CONV2D_3X3_S2_PACR(INTF_EVEN, 1);
        for (int k = 0; k < 4; ++k) {
            CONV2D_3X3_S2_PACR(p_pacr::ALL_INTF_ACTIVE, 0);
        }
        CONV2D_3X3_S2_PACR(p_pacr::ALL_INTF_ACTIVE, 1);
        CONV2D_3X3_S2_PACR(INTF_3, 0);
        for (int k = 0; k < 3; ++k) {
            CONV2D_3X3_S2_PACR(p_pacr::ALL_INTF_ACTIVE, 0);
        }
        CONV2D_3X3_S2_PACR(INTF_012, 0);
        CONV2D_3X3_S2_PACR(INTF_012, 1);
    });
}

// One tap from window base w (block row of the face's first group) into output tile `off` of the CB: four faces of len
// PACRs (replay `face`; Y += 4 each), the second half's rows from Y = y2, the tile closed by replay `last`.
template <uint32_t face, uint32_t len, uint32_t last, uint32_t y2>
ALWI void pack_tap(uint32_t out_id, uint32_t w, uint32_t off) {
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, w);
    program_packer_destination(get_output_tile_address<true, PackMode::Default>(out_id, off));
    lltt::replay(face, len);                          // F0: Z 0, Y 0..
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
    lltt::replay(face, len);                          // F1
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);  // Z 0
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, y2);
    lltt::replay(face, len);  // F2: Y y2..
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, y2);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);  // Z 1
    lltt::replay(face, len - 1);                      // F3
    lltt::replay(last, 1);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);  // Z 0
}
// The three taps of the section's output tile: kw = 0, 1, 2 at `off0`, `off0 + kw_step`, `off0 + 2 kw_step`. E's second
// half is pixels 32..63 of the pair (Y 32); the O plane's is O pixels 16..31 (Y 16; kw = 0: 15..30).
ALWI void pack_taps(uint32_t out_id, uint32_t off0, uint32_t kw_step) {
    pack_tap<RK, 5, RK_LAST, 16>(out_id, OB - 4, off0);
    pack_tap<RE, 8, RE_LAST, 32>(out_id, R0, off0 + kw_step);
    pack_tap<RO, 4, RO_LAST, 16>(out_id, OB, off0 + 2 * kw_step);
}
// Default strides, ADDR_MODs and MOP for cb_out (llk_pack_init), counters at 0.
inline void pack_uninit(uint32_t cb_out) {
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
    llk_pack_init(cb_out);
}
#undef CONV2D_3X3_S2_PACR
#endif

// Setup for a run of sections: formats (cb_in -> SrcB, DEST -> cb_out), unpack / math / pack init.
template <uint32_t cb_in, uint32_t cb_out>
ALWI void init() {
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_out);
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_in, cb_in)));
    UNPACK((unpack_init()));
    MATH((math_init()));
    PACK((llk_pack_init(cb_out)));
    PACK((pack_init(get_output_id(cb_out))));
}
template <uint32_t cb_out>
ALWI void uninit() {
    MATH((math::clear_dst_reg_addr()));
    PACK((pack_uninit(cb_out)));
}

// One run of n consecutive output tiles (columns c0 .. c0 + n - 1) of one image row, one channel tile, one kernel row:
// n sections. tile_l1(j): the L1 address of input tile j of the row (UNPACK only); out_off(u): the CB offset of tile
// u's kw = 0 tap (kw_step further for kw = 1, 2); ready(u): waits until tile u's input tiles landed (a live run).
// live: the input row is in the image (else zero taps). Bench diag: 1 = pack only, 2 = no pack.
template <uint32_t cb_out, uint32_t diag = 0, typename TileL1, typename OutOff, typename Ready>
ALWI void run_row(
    uint32_t c0,
    uint32_t n,
    bool live,
    uint32_t kw_step,
    const TileL1& tile_l1,
    const OutOff& out_off,
    const Ready& ready) {
    [[maybe_unused]] uint32_t out_id = 0;
    PACK((out_id = get_output_id(cb_out)));
    for (uint32_t u = 0; u < n; ++u) {
        if (live) {
            ready(u);
        }
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
                MATH((move_prev()));
                MATH((release_srcb()));
            }
            UNPACK((load(tile_l1(2 * c))));
            MATH((move_pair_tile<0>()));
            MATH((release_srcb()));
            UNPACK((load(tile_l1(2 * c + 1))));
            MATH((move_pair_tile<1>()));
            if (u + 1 == n) {
                MATH((release_srcb()));  // else the next tile's previous pixels stay in this bank
            }
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        if constexpr (diag != 2) {
            PACK((pack_taps(out_id, out_off(u), kw_step)));
        }
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// TILE slab plan of one M-block (conv2d_3x3_slab_dataflow.hpp tile_slab_plan with block_footprint at stride 2,
// padding 1, dilation 1, band_last_tap 2): the input tile-rows the fill streams, segment by segment, back to back.
struct SlabPlan {
    uint32_t num_segs;
    uint32_t seg_first[3];
    uint32_t seg_count[3];
};
template <uint32_t out_w, int32_t in_w, int32_t num_in_px>
inline void slab_plan(uint32_t o_first, uint32_t o_end, SlabPlan& p) {
    constexpr int32_t TILE_DIM = 32;
    const int32_t ho_first = o_first / out_w;
    const int32_t wo_first = o_first % out_w;
    const int32_t ho_last = (o_end - 1) / out_w;
    const int32_t wo_last = (o_end - 1) % out_w;
    const int32_t f = (2 * ho_first - 1) * in_w + 2 * wo_first - 1;
    const int32_t band_len = (ho_last - ho_first) * 2 * in_w + (wo_last - wo_first) * 2 + 3;
    p.num_segs = 0;
    bool have_prev = false;
    int32_t prev_t_hi = 0;
    for (int32_t kh = 0; kh < 3; ++kh) {
        const int32_t v_lo = f + kh * in_w;
        const int32_t lo = v_lo > 0 ? v_lo : 0;
        const int32_t hi = v_lo + band_len < num_in_px ? v_lo + band_len : num_in_px;
        if (lo >= hi) {
            continue;
        }
        const int32_t t_lo = lo / TILE_DIM;
        const int32_t t_hi = (hi + TILE_DIM - 1) / TILE_DIM;
        const int32_t first_new = (have_prev && t_lo < prev_t_hi) ? prev_t_hi : t_lo;
        if (t_hi > first_new) {
            p.seg_first[p.num_segs] = first_new;
            p.seg_count[p.num_segs] = t_hi - first_new;
            ++p.num_segs;
        }
        prev_t_hi = t_hi;
        have_prev = true;
    }
}
// Slab row of input tile-row i (the plan's segments back to back; a band's tiles are consecutive slab rows).
ALWI uint32_t slab_row(const SlabPlan& p, uint32_t i) {
    uint32_t base = 0;
    for (uint32_t s = 0; s < p.num_segs; ++s) {
        if (i >= p.seg_first[s] && i < p.seg_first[s] + p.seg_count[s]) {
            return base + (i - p.seg_first[s]);
        }
        base += p.seg_count[s];
    }
    return base;  // not reached for a tile the build reads
}

// Every tap of the block's valid output tile-rows (output pixels [o_first, o_end), whole tile-rows) x 3 kernel rows,
// from the input slab (cb_in: the plan's tile-rows, ct tiles each, resident from its read pointer) into the reserved
// block of cb_out: tap (t, kh, kw, ch) at out_off(t, kh, ch) + kw * ct. dead(t, kh): the unit is never read (not
// built); the tile-rows past o_end are not built either (their output rows are dropped, as the gathered units there
// were never written). Units go in input-row order (h = 2 ho + kh - 1 ascending; the fill streams the slab in flat
// input-pixel order), each output row's tiles of one input row as one run, so the build follows the fill; no ordering
// state.
template <
    uint32_t ct,
    uint32_t cb_in,
    uint32_t cb_out,
    uint32_t out_w,
    int32_t in_h,
    int32_t in_w,
    int32_t num_in_px,
    uint32_t diag = 0,
    typename OutOff,
    typename Dead>
ALWI void build(DataflowBuffer& in, uint32_t o_first, uint32_t o_end, const OutOff& out_off, const Dead& dead) {
    static_assert(in_w % 64 == 0 && out_w * 2 == static_cast<uint32_t>(in_w));
    constexpr uint32_t tpr = in_w / 32;
    SlabPlan plan;
    slab_plan<out_w, in_w, num_in_px>(o_first, o_end, plan);
    init<cb_in, cb_out>();
    [[maybe_unused]] uint32_t in_l1 = 0;
    [[maybe_unused]] uint32_t tile_bytes = 0;
    UNPACK((in_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    const int32_t ho_first = static_cast<int32_t>(o_first / out_w);
    const int32_t ho_last = static_cast<int32_t>((o_end - 1) / out_w);
    uint32_t waited = 0;  // slab rows waited for
    for (int32_t h = 2 * ho_first - 1; h <= 2 * ho_last + 1; ++h) {
        // Odd h: kernel row 2 of output row (h - 1) / 2, then kernel row 0 of (h + 1) / 2; even h: kernel row 1 of h
        // / 2.
        for (int32_t kh = (h & 1) ? 2 : 1; kh >= 0; kh -= 2) {
            const int32_t ho = (h + 1 - kh) / 2;
            if (ho < ho_first || ho > ho_last) {
                continue;
            }
            // The block's tiles of output row ho: t in [t_lo, t_hi), columns from c_lo.
            const uint32_t row_first = static_cast<uint32_t>(ho) * out_w;
            const uint32_t p_lo = row_first > o_first ? row_first : o_first;
            const uint32_t p_hi = row_first + out_w < o_end ? row_first + out_w : o_end;
            const uint32_t t_lo = (p_lo - o_first) / 32;
            const uint32_t n = (p_hi - p_lo) / 32;
            const uint32_t c_lo = (p_lo - row_first) / 32;
            const bool live = h >= 0 && h < in_h;
            uint32_t u = 0;
            while (u < n) {
                if (dead(t_lo + u, kh)) {
                    ++u;
                    continue;
                }
                uint32_t m = 1;  // the sub-run of units the walk reads
                while (u + m < n && !dead(t_lo + u + m, kh)) {
                    ++m;
                }
                const uint32_t c = c_lo + u;
                const uint32_t row0 = static_cast<uint32_t>(h) * tpr;
                [[maybe_unused]] uint32_t k_first = 0;  // tile j of the row at slab row k_first + j
                if (live) {
                    const uint32_t j_first = c > 0 ? 2 * c - 1 : 0;
                    k_first = slab_row(plan, row0 + j_first) - j_first;
                }
                // Tile v of the run reads input tiles up to 2 (c + v) + 1: consecutive slab rows from k_first.
                auto ready = [&](uint32_t v) {
                    const uint32_t need = k_first + 2 * (c + v) + 2;
                    if (need > waited) {
#if S2_OVERLAP == 9
                        MaybeDeviceZoneScope("compute_s2_fill_wait");
#endif
                        in.wait_front(need * ct);
                        waited = need;
                    }
                };
                for (uint32_t ch = 0; ch < ct; ++ch) {
                    auto tile_l1 = [&](uint32_t j) { return in_l1 + ((k_first + j) * ct + ch) * tile_bytes; };
                    auto off = [&](uint32_t v) { return out_off(t_lo + u + v, kh, ch); };
                    run_row<cb_out, diag>(c, m, live, ct, tile_l1, off, ready);
                }
                u += m;
            }
            if (kh == 0) {
                break;
            }
        }
    }
    uninit<cb_out>();
}

// Perf 13 experiment (s2_build_walk_overlap): `build`'s units, in its order, restricted to the sections (one per unit
// and channel tile: ch outer, the run's tiles inner) with index in [sec_lo, sec_hi), so a K-block's build can be cut
// into pieces between matmul subblocks. `waited` (slab rows waited for) persists across the pieces. A piece that starts
// inside a run reloads the run's previous tile (run_row's u == 0). Returns the number of sections enumerated.
template <
    uint32_t ct,
    uint32_t cb_in,
    uint32_t cb_out,
    uint32_t out_w,
    int32_t in_h,
    int32_t in_w,
    int32_t num_in_px,
    typename OutOff,
    typename Dead>
ALWI uint32_t build_part(
    DataflowBuffer& in,
    uint32_t o_first,
    uint32_t o_end,
    const OutOff& out_off,
    const Dead& dead,
    uint32_t sec_lo,
    uint32_t sec_hi,
    uint32_t& waited) {
    static_assert(in_w % 64 == 0 && out_w * 2 == static_cast<uint32_t>(in_w));
    constexpr uint32_t tpr = in_w / 32;
    SlabPlan plan;
    slab_plan<out_w, in_w, num_in_px>(o_first, o_end, plan);
    bool inited = false;
    [[maybe_unused]] uint32_t in_l1 = 0;
    [[maybe_unused]] uint32_t tile_bytes = 0;
    UNPACK((in_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    const int32_t ho_first = static_cast<int32_t>(o_first / out_w);
    const int32_t ho_last = static_cast<int32_t>((o_end - 1) / out_w);
    uint32_t s = 0;  // sections enumerated
    for (int32_t h = 2 * ho_first - 1; h <= 2 * ho_last + 1 && s < sec_hi; ++h) {
        for (int32_t kh = (h & 1) ? 2 : 1; kh >= 0; kh -= 2) {
            const int32_t ho = (h + 1 - kh) / 2;
            if (ho < ho_first || ho > ho_last) {
                continue;
            }
            const uint32_t row_first = static_cast<uint32_t>(ho) * out_w;
            const uint32_t p_lo = row_first > o_first ? row_first : o_first;
            const uint32_t p_hi = row_first + out_w < o_end ? row_first + out_w : o_end;
            const uint32_t t_lo = (p_lo - o_first) / 32;
            const uint32_t n = (p_hi - p_lo) / 32;
            const uint32_t c_lo = (p_lo - row_first) / 32;
            const bool live = h >= 0 && h < in_h;
            uint32_t u = 0;
            while (u < n) {
                if (dead(t_lo + u, kh)) {
                    ++u;
                    continue;
                }
                uint32_t m = 1;
                while (u + m < n && !dead(t_lo + u + m, kh)) {
                    ++m;
                }
                if (s + ct * m <= sec_lo || s >= sec_hi) {
                    s += ct * m;
                    u += m;
                    continue;
                }
                const uint32_t c = c_lo + u;
                const uint32_t row0 = static_cast<uint32_t>(h) * tpr;
                [[maybe_unused]] uint32_t k_first = 0;
                if (live) {
                    const uint32_t j_first = c > 0 ? 2 * c - 1 : 0;
                    k_first = slab_row(plan, row0 + j_first) - j_first;
                }
                for (uint32_t ch = 0; ch < ct; ++ch, s += m) {
                    const uint32_t a = sec_lo > s ? (sec_lo - s < m ? sec_lo - s : m) : 0;
                    const uint32_t b = sec_hi > s ? (sec_hi - s < m ? sec_hi - s : m) : 0;
                    if (a >= b) {
                        continue;
                    }
                    if (!inited) {
                        init<cb_in, cb_out>();
                        inited = true;
                    }
                    auto ready = [&](uint32_t v) {
                        const uint32_t need = k_first + 2 * (c + a + v) + 2;
                        if (need > waited) {
                            in.wait_front(need * ct);
                            waited = need;
                        }
                    };
                    auto tile_l1 = [&](uint32_t j) { return in_l1 + ((k_first + j) * ct + ch) * tile_bytes; };
                    auto off = [&](uint32_t v) { return out_off(t_lo + u + a + v, kh, ch); };
                    run_row<cb_out>(c + a, b - a, live, ct, tile_l1, off, ready);
                }
                u += m;
            }
            if (kh == 0) {
                break;
            }
        }
    }
    if (inited) {
        uninit<cb_out>();
    }
    return s;
}

}  // namespace conv2d_3x3::s2_taps
