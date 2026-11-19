// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tap shift (the descriptor's TAP_SHIFT, on the padded-slab N-slice prelude): the kw = 0 / kw = 2 tap tiles are built
// straight from the input ring's TILE slab on the FPU, with no row-major round trip (no untilize, no padded slab).
//
// With W % 32 == 0 every kernel-row offset is a whole number of input tile-rows; only the +-d pixel shift is sub-tile.
// The kw = 2 tap of output tile-row t (image row ho, column tile c) for kernel row kh is input tile (h, c) shifted up
// by d pixels, its last d pixels taken from tile (h, c + 1) (zero at the row's end); kw = 0 is (h, c) shifted down by
// d, its first d pixels from (h, c - 1) (zero at the row's start). h = ho + kh d - p; a tap outside the image is zero.
//
// Per (tile-row, kernel row, channel tile): the unpacker loads cur = (h, c) whole into SrcB (one full-tile UNPACR, one
// bank flip), the math thread moves its rows into the two DEST tiles (kw = 0 and kw = 2) with one-row MOVB2D, then
// loads each non-zero neighbour and moves its d rows per channel half. DEST rows never written stay zero (DEST is
// cleared at the start and by every section release). 4 DEST tiles per section, each packed by the standard pack to
// its in0 offset (the whole [rows, in0_k] block is reserved once and pushed once). Kernel-row major: each tap waits
// only for the slab rows it reads, so the prelude runs behind the fill instead of after it.
//
// The values move unchanged (bf8_b / bf16 -> SrcB -> fp32 DEST -> the same format; a bf8_b 16-datum row keeps its
// datums and so its shared exponent): bit-identical to the untilize / padded tilize round trip.
//
// RAW LLK (capability): no helper builds a tile from row ranges of two tiles. Mechanism notes (Blackhole, measured):
//   * the stock AB unpack flips the Src bank per face (the eltwise math clears per face), so a bank never holds a whole
//     tile; the full-tile UNPACR (X end 1023, as the BH matmul's) puts F0..F3 in rows 0..63 of one bank;
//   * one-row MOVA2D with fp32 DEST leaves the rows of faces F1 / F3 unwritten; one-row MOVB2D is exact, so every row
//     goes through SrcB; 8- / 4-row moves need row-aligned shifts (d % 8 / d % 4), not the general case.
//   * SETRWC does not wait for SrcB valid: every pass starts with a move (which does).

#pragma once

#include <cstdint>
#include <utility>

#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "conv2d_3x3_padded_slab.hpp"

namespace conv2d_3x3::tap_shift {

// DEST row i of a 32x32 tile (fp32): F0 [0, 16) pixels 0..15 channels 0..15, F1 [16, 32) pixels 0..15 channels
// 16..31, F2 [32, 48) pixels 16..31 channels 0..15, F3 [48, 64). An unpacked SrcB tile has the same layout.
constexpr int pix_of_row(uint32_t i) { return static_cast<int>(i < 32 ? (i % 16) : 16 + (i % 16)); }
constexpr int fpos(int p) { return p < 16 ? p : p + 16; }
// Output pixel r takes pixel r + s of cur, or of the neighbour when that leaves the tile (|s| < 32).
constexpr bool from_nb(int s, uint32_t i) {
    const int p = pix_of_row(i) + s;
    return p < 0 || p >= 32;
}
constexpr uint32_t off_of(int s, uint32_t i) {
    const int r = pix_of_row(i);
    int p = r + s;
    p = p < 0 ? p + 32 : (p >= 32 ? p - 32 : p);
    return static_cast<uint32_t>((fpos(p) - fpos(r)) & 63);
}
constexpr uint32_t srcrow(int s, uint32_t i) { return (off_of(s, i) + i) & 63; }

#ifdef TRISC_UNPACK
inline void unpack_init() {
    TT_SETADCXX(p_setadc::UNP_B, 32 * 32 - 1, 0x0);
    ckernel::ckernel_template tmp(1, 1, TT_OP_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}
// One whole bf8_b tile at byte address `l1` into the next SrcB bank (the SrcA address is unused).
inline void load(uint32_t l1) { _llk_unpack_AB_<BroadcastType::NONE>((l1 >> 4) - 1, (l1 >> 4) - 1); }
#endif

#ifdef TRISC_MATH
constexpr uint32_t ADDR_MOD_ZERO = ADDR_MOD_7;
constexpr uint32_t ADDR_MOD_STEP = ADDR_MOD_6;

// With d % 8 == 0 every run boundary (16 - d, d, 32 - d) is a multiple of 8, so each aligned group of 4 rows has one
// source and one offset: one 4-row MOVB2D per group (16 per tile instead of 64). Measured: 4-row MOVB2D is exact only
// when the SrcB and DEST rows agree mod 8 (d = 8 bit-identical; d = 4 and 12, rows 4 apart mod 8, differ), so d % 8 ==
// 4 keeps the one-row moves.
template <int s, bool nb, uint32_t i>
ALWI void row() {
    constexpr int a = s < 0 ? -s : s;
    if constexpr (a % 8 == 0) {
        if constexpr (i % 4 == 0 && from_nb(s, i) == nb) {
            static_assert(from_nb(s, i) == from_nb(s, i + 3) && off_of(s, i) == off_of(s, i + 3));
            TTI_MOVB2D(0, srcrow(s, i), ADDR_MOD_ZERO, p_movb2d::MOV_4_ROWS, i);
        }
    } else if constexpr (from_nb(s, i) == nb) {
        TTI_MOVB2D(0, srcrow(s, i), ADDR_MOD_ZERO, p_movb2d::MOV_1_ROW, i);
    }
}
template <int s, bool nb, uint32_t... I>
ALWI void move_rows(std::integer_sequence<uint32_t, I...>) {
    (row<s, nb, I>(), ...);
}

// Replayed rows: runs of rows with the same source offset are one replay of a recorded run of one-row MOVB2D that step
// the SrcB and DEST counters by 1 (rows not in a pass are skipped with INCRWC). Rows 0..15 and 16..31 have the same
// offsets, as do 32..47 and 48..63. Replay buffer [0, 32): R1 x (16 - e), R2 x e, R3 x (16 - e), R4 x e.
//   d < 16 (e = d): offsets R1 = d, R2 = 16 + d, R3 = 64 - d, R4 = 48 - d. kw = 2 per 16-row block: R1 then R2 (blocks
//     0, 1 from cur; blocks 2, 3: R2's rows from nb); kw = 0: R4's rows (nb in blocks 0, 1, cur in 2, 3) then R3 (cur).
//   16 < d < 32 (e = d - 16): offsets R1 = e + 32, R2 = e + 48, R3 = 32 - e, R4 = 16 - e. kw = 2: R1 then R2 (cur then
//     nb in blocks 0, 1; nb in 2, 3); kw = 0: R4 then R3 (nb in blocks 0, 1; nb then cur in 2, 3).
// d % 8 == 0 takes the 4-row moves instead.
template <int d>
constexpr bool replayed() {
    return d % 8 != 0;
}
template <int d>
inline void math_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_ZERO);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 1}, .dest = {.incr = 1}}.set(ADDR_MOD_STEP);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
    if constexpr (replayed<d>()) {
        constexpr int e = d < 16 ? d : d - 16;
        constexpr int o1 = d < 16 ? d : e + 32;
        constexpr int o2 = d < 16 ? 16 + d : e + 48;
        constexpr int o3 = d < 16 ? 64 - d : 32 - e;
        constexpr int o4 = d < 16 ? 48 - d : 16 - e;
        load_replay_buf(0, 32, [] {
            for (int k = 0; k < 16 - e; ++k) {
                TTI_MOVB2D(0, o1, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
            for (int k = 0; k < e; ++k) {
                TTI_MOVB2D(0, o2, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
            for (int k = 0; k < 16 - e; ++k) {
                TTI_MOVB2D(0, o3, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
            for (int k = 0; k < e; ++k) {
                TTI_MOVB2D(0, o4, ADDR_MOD_STEP, p_movb2d::MOV_1_ROW, 0);
            }
        });
    }
    // DEST starts zero: the rows a tap never writes (neighbour or out-of-image rows) must read as zero.
    TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_ZERO, 0);
}

// Skip n rows (both counters).
template <int n>
ALWI void skip() {
    if constexpr (n > 15) {
        TTI_INCRWC(0, 15, 15, 0);
        skip<n - 15>();
    } else if constexpr (n > 0) {
        TTI_INCRWC(0, n, n, 0);
    }
}

// The cur (nb = false) or neighbour (nb = true) rows of the DEST tile at the current write offset (counters from 0,
// back to 0 after).
template <int d, bool kw2, bool nb>
ALWI void pass() {
    if constexpr (!replayed<d>()) {
        move_rows<kw2 ? d : -d, nb>(std::make_integer_sequence<uint32_t, 64>{});
    } else {
        constexpr int e = d < 16 ? d : d - 16;
        constexpr uint32_t R1 = 0, R2 = 16 - e, R3 = 16, R4 = 32 - e;
        constexpr int a = 16 - e;  // R1 / R3 run
        if constexpr (d < 16 && kw2 && !nb) {
            lltt::replay(R1, a);
            lltt::replay(R2, e);
            lltt::replay(R1, a);
            lltt::replay(R2, e);
            lltt::replay(R1, a);
            skip<e>();
            lltt::replay(R1, a);
        } else if constexpr (d < 16 && kw2 && nb) {
            skip<32 + a>();
            lltt::replay(R2, e);
            skip<a>();
            lltt::replay(R2, e);
        } else if constexpr (d < 16 && !kw2 && !nb) {
            skip<e>();
            lltt::replay(R3, a);
            skip<e>();
            lltt::replay(R3, a);
            lltt::replay(R4, e);
            lltt::replay(R3, a);
            lltt::replay(R4, e);
            lltt::replay(R3, a);
        } else if constexpr (d < 16 && !kw2 && nb) {
            lltt::replay(R4, e);
            skip<a>();
            lltt::replay(R4, e);
        } else if constexpr (kw2 && !nb) {  // d > 16
            lltt::replay(R1, a);
            skip<e>();
            lltt::replay(R1, a);
        } else if constexpr (kw2 && nb) {
            skip<a>();
            lltt::replay(R2, e);
            skip<a>();
            lltt::replay(R2, e);
            lltt::replay(R1, a);
            lltt::replay(R2, e);
            lltt::replay(R1, a);
            lltt::replay(R2, e);
        } else if constexpr (!kw2 && !nb) {
            skip<32 + e>();
            lltt::replay(R3, a);
            skip<e>();
            lltt::replay(R3, a);
        } else {  // kw = 0, nb, d > 16
            lltt::replay(R4, e);
            lltt::replay(R3, a);
            lltt::replay(R4, e);
            lltt::replay(R3, a);
            lltt::replay(R4, e);
            skip<a>();
            lltt::replay(R4, e);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
    }
}
template <int d, bool kw2>
ALWI void cur_pass() {
    pass<d, kw2, false>();
}
ALWI void release_srcb() { TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); }
ALWI void dest_tile(uint32_t j) { math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(j); }
#endif

// The prelude's taps 0 and 2 of every live unit, from the input ring (cb_in, the plan's slab rows back to back from
// its base, ct tiles per row) into the reserved in0 block of cb_out ([rows, in0_k], unit (t, kh) taps at
// t * in0_k + kh * 2 ct + {0, ct}). Dead units (kh outside [kh_lo, kh_hi) of t's ring row group) are not written.
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
    constexpr int32_t tpr = input_width / 32;
    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(cb_out);
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(cb_in, cb_in)));
    UNPACK((unpack_init()));
    MATH((math_init<dilation>()));
    PACK((llk_pack_init(cb_out)));
    [[maybe_unused]] uint32_t ring_l1 = 0;
    [[maybe_unused]] uint32_t tile_bytes = 0;  // the input tile's page (bf8_b or bf16)
    UNPACK((ring_l1 = get_local_cb_interface(get_operand_id(cb_in)).fifo_rd_ptr << 4));
    UNPACK((tile_bytes = get_local_cb_interface(get_operand_id(cb_in)).fifo_page_size << 4));
    out.reserve_back(rows * in0_k);
    uint32_t j = 0;       // DEST tiles in the open section
    uint32_t dst_off[4];  // their in0 offsets
    uint32_t waited = 0;  // slab rows waited for
    auto close = [&]() {
        if (j == 0) {
            return;
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        for (uint32_t i = 0; i < j; ++i) {
            PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(i, cb_out, dst_off[i])));
        }
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
        j = 0;
    };
    // Slab row of input tile-row i (the plan's segments back to back).
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
    for (int32_t kh = 0; kh < 3; ++kh) {
        for (uint32_t t = 0; t < rows; ++t) {
            const uint32_t g = t / group_h;
            if (kh < kh_lo[g] || kh >= kh_hi[g]) {
                continue;  // dead unit: the ring walk never reads it
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
                if (live) {
                    UNPACK((load(ring_l1 + (k_cur * ct + ch) * tile_bytes)));
#ifdef TRISC_MATH
                    dest_tile(j);
                    cur_pass<dilation, false>();
                    dest_tile(j + 1);
                    cur_pass<dilation, true>();
                    release_srcb();
                    if (has_prev) {
                        dest_tile(j);
                        pass<dilation, false, true>();
                        release_srcb();
                    }
                    if (has_next) {
                        dest_tile(j + 1);
                        pass<dilation, true, true>();
                        release_srcb();
                    }
#endif
                    if (has_prev) {
                        UNPACK((load(ring_l1 + (k_prev * ct + ch) * tile_bytes)));
                    }
                    if (has_next) {
                        UNPACK((load(ring_l1 + (k_next * ct + ch) * tile_bytes)));
                    }
                }
                j += 2;
                if (j == 4) {
                    close();
                }
            }
        }
    }
    close();
    out.push_back(rows * in0_k);
    MATH((math::clear_dst_reg_addr()));
}

}  // namespace conv2d_3x3::tap_shift
