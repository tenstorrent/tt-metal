// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tap shift (the descriptor's TAP_SHIFT, on the padded-slab N-slice prelude): the kw = 0 / kw = 2 tap tiles are built
// straight from the input ring's TILE slab on the FPU, with no row-major round trip (no untilize, no padded slab).
//
// With W % 32 == 0 every kernel-row offset is a whole number of input tile-rows; only the +-d pixel shift is sub-tile.
// The kw = 2 tap of output tile-row t (image row ho, column tile c) for kernel row kh is input row h = ho + kh d - p,
// pixels [32 c + d, 32 c + 32 + d); kw = 0 is pixels [32 c - d, 32 c + 32 - d); pixels outside the row (and rows
// outside the image) are zero.
//
// Row buffer: for one (kernel row, channel tile) and a group of n <= 2 consecutive units of one image row (columns
// c0 .. c0 + n - 1), one DEST section (fp32, one half: 256 rows) holds the input row's pixels [32 c0 - d,
// 32 (c0 + n) + d), pixel-contiguous per channel half: channel half hf at block row hf * Hs, pixel p at row r0 + p
// (p from the group's first pixel). Each of the group's input tiles, and the d edge pixels of its row neighbours, is
// unpacked ONCE (one full-tile UNPACR into SrcB) and moved into the slot; out-of-row pixels stay zero (DEST is cleared
// on every section release). The packer then reads each tap (unit u, kw) as the 32-row window of both blocks starting
// at its first pixel: W = window row (one DEST row per W), Z = block (Hs rows), Y = row in the window; faces F0 (Z 0,
// Y 0..15), F1 (Z 1, Y 0..15), F2 (Z 0, Y 16..31), F3 (Z 1, Y 16..31). The shift costs the packer nothing: a tap is
// one tile pack, as the tilize it replaced. HEAD's shift (one-row SrcB -> DEST moves building every tap tile, three
// tile unpacks per tap pair) cost ~60 ns per tap; this costs 34-42 (isolated, below).
//
// Hardware rules (Blackhole, measured, perf_experiments/tap_shift_rate/):
//   * a 4-row PACR (all four interfaces) reads a window exactly only from a DEST row that is a multiple of 4; a 2-row
//     PACR (two interfaces) from a multiple of 2;
//   * a 4-row MOVB2D is exact only when its SrcB and DEST rows agree mod 8 (the DEST offset register included; also in
//     16-bit DEST): a misaligned offset or source row is not fixed by the other;
//   * the packer's Y / Z / W strides count one DEST row as 16 datums of its source format (set_packer_strides' Y);
//   * SETRWC does not wait for SrcB valid: every pass starts with a move (which does);
//   * (rejected) a 16-bit DEST shift (ALU Fp32_enabled = 0, packer Read_32b = 0, as the BH fast tilize) is exact but
//     its bf8_b pack is slower than the fp32 one (43 vs 32 ns per tile here); the unpacker's SrcB row offset (CH1 Y)
//     does place a tile lower in SrcB, but 4-row MOVB2D across a 16-row SrcB boundary reads the wrong rows.
// Two layouts follow:
//   A (d % 4 == 0): r0 = d rounded up to 8: every tile row lands 8-aligned (16 aligned 4-row MOVB2D per tile), and both
//     windows (r0 + 32 u -+ d) start on a multiple of 4 (4-row PACR).
//   B (otherwise): r0 = d: kw = 0 windows start at 32 u (4-row PACR), kw = 2 at 32 u + 2 d (4-row PACR for even d,
//     2-row for odd d); rows move one at a time (4 replays of a 16-move run per tile).
// The values move unchanged (bf8_b / bf16 -> SrcB -> fp32 DEST -> the same format; a bf8_b 16-datum row keeps its
// datums and so its shared exponent): bit-identical to the untilize / padded tilize round trip.
//
// RAW LLK (capability): no helper builds a tile from row ranges of two tiles or packs a tile from a DEST window that
// is not a whole DEST tile (compute_kernel_lib::tilize / untilize read and write at CB pointers and have no row shift;
// pack_tile packs whole DEST tiles), and none unpacks a whole tile into one Src bank outside the matmul. Measured
// (Blackhole p150b, LoFi, fp32 DEST, bf8_b, one core, one M-block of 8 tile-rows, W = 64, ns per tap tile, HEAD ->
// this): d = 1 60.5 -> 40.6, d = 2 59.7 -> 35.7, d = 4 58.1 -> 33.6, d = 8 59.9 -> 41.4, d = 12 58.8 -> 39.7,
// d = 18 58.4 -> 42.0. In the op (unzoned, medians of 3 alternating pairs, bit-identical): res4.x 12079 -> 11379 ns,
// aspp.convs.2 45326 -> 42095, aspp.convs.3 21213 -> 20689, res5.1 32725 -> 32394, res5.2 flat; on the Perf 10
// carve-out (now taken): aspp.convs.1 51360 -> 48102, res5.0 33339 -> 32512.

#pragma once

#include <cstdint>
#include <utility>

#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "conv2d_3x3_padded_slab.hpp"

namespace conv2d_3x3::tap_shift {

template <int32_t d>
constexpr bool layout_a() {
    return d % 4 == 0;
}
// Block row of the group's first pixel (pixel 0 of tile c0).
template <int32_t d>
constexpr uint32_t row0() {
    return layout_a<d>() ? static_cast<uint32_t>(((d + 7) / 8) * 8) : static_cast<uint32_t>(d);
}
// Rows of one block for n units (a multiple of 8 for layout A's aligned moves, of 4 for the packer's windows).
template <int32_t d>
constexpr uint32_t block_rows(uint32_t n) {
    return layout_a<d>() ? ((row0<d>() + 32 * n + d + 7) / 8) * 8 : ((32 * n + 2 * d + 3) / 4) * 4;
}
// Units per slot: both blocks fit one fp32 DEST half (256 rows).
template <int32_t d, uint32_t tpr>
constexpr uint32_t group_max() {
    uint32_t g = tpr < 2 ? tpr : 2;
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
// One whole tile at byte address `l1` into the next SrcB bank (the SrcA address is unused).
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
    // DEST starts zero: the pixels outside the image row must read as zero.
    TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_ZERO, 0);
}

// Layout A: 4-row group g (SrcB rows [4 g, 4 g + 4), face f = g / 4) of pixel q = 16 (f / 2) + 4 (g % 4) in
// [q_lo, q_hi) to block row Hs (f % 2) + base + q (relative to the DEST offset; base wraps for the left neighbour).
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
// Layout B: SrcB row r (face f = r / 16) of pixel q = 16 (f / 2) + r % 16 in [q_lo, q_hi).
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
// A whole tile with its pixel 0 at slot row `at`.
template <int32_t d, uint32_t Hs>
ALWI void move_tile(uint32_t at) {
    dest_row(at);
    if constexpr (layout_a<d>()) {
        a_moves<Hs, 0, 0, 32>(std::make_integer_sequence<uint32_t, 16>{});
    } else {
        // The SrcB counter runs 0..63 over the four replays; DEST restarts at each face's block row.
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
// The left neighbour's last d pixels (its pixel 0 at slot row r0 - 32; the DEST offset stays at the slot base).
template <int32_t d, uint32_t Hs>
ALWI void move_prev() {
    constexpr uint32_t r0 = row0<d>();
    dest_row(0);
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
#define CONV2D_3X3_TS_PACR(am, intf, last) \
    TTI_PACR(                              \
        p_pacr::CFG_CTXT_0,                \
        p_pacr::NO_ROW_PAD_ZERO,           \
        p_pacr::DST_ACCESS_NORMAL_MODE,    \
        am,                                \
        p_pacr::ADDR_CNT_CTXT_0,           \
        0,                                 \
        intf,                              \
        0,                                 \
        0,                                 \
        0,                                 \
        0,                                 \
        last)
// Replay [0, 22): one 4-row-PACR window tap. [22, 30): one face of 2-row PACRs. [30]: a tile's last 2-row PACR.
constexpr uint32_t R4 = 0, R4_LEN = 22, R2 = 22, R2_LAST = 30;
template <uint32_t Hs>
inline void pack_init(uint32_t out_id) {
    // One DEST row = 16 datums of the packer's source format (as set_packer_strides). Y = row, Z = block, W = window.
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
    // ADDR_MOD_0 (llk_pack_init): Y += 4 per 4-row PACR. ADDR_MOD_3 (only the fast tilize reads it, and its init sets
    // it): Y += 2 per 2-row PACR.
    addr_mod_pack_t{.y_src = {.incr = 2}}.set(ADDR_MOD_3);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    load_replay_buf(R4, R4_LEN + 9, [] {
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F0: Z 0, Y 0..15
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);             // Y 0
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);             // Z 1
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F1: Z 1, Y 0..15
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);             // Z 0 (Y is 16)
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F2: Z 0, Y 16..31
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_Y, 16);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 1, 0b0001);             // Z 1
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);  // F3: Z 1, Y 16..31
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0);
        CONV2D_3X3_TS_PACR(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 1);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0010);  // Y 0
        for (int k = 0; k < 8; ++k) {
            CONV2D_3X3_TS_PACR(ADDR_MOD_3, p_pacr::TWO_INTFS_ACTIVE, 0);
        }
        CONV2D_3X3_TS_PACR(ADDR_MOD_3, p_pacr::TWO_INTFS_ACTIVE, 1);
    });
}
// One tap: the window at slot row w (of the packer's current DEST half) into output tile `off` of the CB.
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
// Default strides, ADDR_MODs and MOP for cb_out (llk_pack_init), counters at 0.
inline void pack_uninit(uint32_t cb_out) {
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
    llk_pack_init(cb_out);
}
#undef CONV2D_3X3_TS_PACR
#endif

// The prelude's taps 0 and 2 of every live unit, from the input ring (cb_in, the plan's slab rows back to back from
// its base, ct tiles per row) into the reserved in0 block of cb_out ([rows, in0_k], unit (t, kh) taps at
// t * in0_k + kh * 2 ct + {0, ct}). Dead units (kh outside [kh_lo, kh_hi) of t's ring row group) are not written.
// Kernel-row major: each group waits only for the slab rows it reads, so the prelude runs behind the fill.
template <
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
    constexpr uint32_t G = group_max<d, tpr>();
    constexpr uint32_t Hs = block_rows<d>(G);
    static_assert(2 * Hs <= 256, "one slot per fp32 DEST half");
    constexpr uint32_t r0 = row0<d>();
    // Window rows of unit u: kw = 0 at r0 + 32 u - d, kw = 2 at r0 + 32 u + d (2-row PACR: layout B, odd d).
    constexpr bool kw2_two_row = !layout_a<d>() && (d % 2 == 1);
#ifdef CONV2D_3X3_ABLATE_TILIZE
    out.reserve_back(rows * in0_k);
    out.push_back(rows * in0_k);
    return;
#endif
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
    [[maybe_unused]] uint32_t tile_bytes = 0;  // the input tile's page (bf8_b or bf16)
    UNPACK((ring_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    out.reserve_back(rows * in0_k);
    uint32_t waited = 0;  // slab rows waited for
    // Slab row of input tile-row i (the plan's segments back to back). A group's tiles are consecutive slab rows: a
    // band's tiles are one run of the ring (a band that shares tiles with the previous one continues its segment).
    auto slab_row = [&](uint32_t i) {
        uint32_t base = 0;
        for (uint32_t sg = 0; sg < plan.num_segs; ++sg) {
            if (i >= plan.seg_first[sg] && i < plan.seg_first[sg] + plan.seg_count[sg]) {
                return base + (i - plan.seg_first[sg]);
            }
            base += plan.seg_count[sg];
        }
        return base;  // not reached for a row the prelude reads
    };
    auto dead = [&](uint32_t t, int32_t kh) {
        const uint32_t g = t / group_h;
        return kh < kh_lo[g] || kh >= kh_hi[g];  // the ring walk never reads it
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
            // The group: up to G live (or all not live) units of image row ho0, consecutive columns.
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
                if (live) {
                    uint32_t k = k_first;
                    if (has_prev) {
                        UNPACK((load(ring_l1 + (k * ct + ch) * tile_bytes)));
                        MATH((move_prev<d, Hs>()));
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
                MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                PACK((llk_packer_wait_for_math_done()));
                for (uint32_t u = 0; u < n; ++u) {
                    const uint32_t unit = (t + u) * in0_k + kh * 2 * ct + ch;
                    PACK((pack_tap<false>(out_id, r0 + 32 * u - d, unit)));
                    PACK((pack_tap<kw2_two_row>(out_id, r0 + 32 * u + d, unit + ct)));
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

}  // namespace conv2d_3x3::tap_shift
