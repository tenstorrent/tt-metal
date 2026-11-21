// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// C_out fold on the K-blocked tap-direct ROW_MAJOR ring walk (the descriptor's cout_fold_capable: stride 1, dilation 1,
// one column of padding each side so W_out = W, W % 32 == 0, C_out at most FOLD_LS lanes, bias).
//
// A conv with C_out = 3 multiplies a whole 32-lane N tile per output tile-row and K-block. The fold puts kw in the N
// lanes instead: the weights' N tile is W'[(kh, c), FOLD_LS kw + co] = W[co, c, kh, kw] (the writer builds it), the
// walk runs 3 K-blocks (kh) of C_in tiles, and each K-block's unit is the kw = 0 tap only. Output tile-row o's unit for
// kernel row kh is the row-major pixel stream's [q + kh W - 1, + 32), q = 32 o (the reader's flat units: across an
// image-row end too), so DEST row j holds U[q + j - 1] of kernel row kh, lane group kw = the partial pixel q + j - 1
// adds to output pixel q + j - kw. Packing DEST into a 16-lane fp32 plane at -kw rows and -kw lane groups (-kw (64 + 4
// FOLD_LS) B) puts every kw group's partial at lanes [0, C_out) of its output row: 9 packs per tile-row (3 per K-block)
// through the packer's fp32 L1 accumulation sum the 9 taps, in the base walk's (kh, kw) order. The other lanes collect
// the other groups (garbage, masked on the output pack). 3 K-blocks of bk tile-MACs and 3 units per tile-row instead
// of 9.
//
// Row-major wrap: at an image-row start (o % FOLD_R == 0) DEST row 0 is the previous row's last pixel. Its kw = 1, 2
// partials belong to that row's last two output pixels (right), its kw = 0 partial would land on this row's first pixel
// (padding: 0), and DEST row 1's kw = 2 partial on the previous row's last pixel (padding: 0). The packer's edge masks
// zero those two rows of those two packs (row 0 of face 0 for kw = 0, row 1 of face 0 for kw = 2).
//
// Plane: FOLD_S = bm + 1 slots of 32 rows x 16 fp32 lanes (2 KB), core-local tile-row o in slot o % FOLD_S. Each tile's
// kw = 1, 2 packs spill 1-2 rows (and a lane group) into the previous slot: the previous tile-row is complete only once
// the next one is packed, so the final pass trails by one tile-row and the M-block's last tile-row waits for the next
// block (bm + 1 slots). Slot 0's spills go below the plane (FOLD_PRE bytes) and are packed again from the alias slot
// FOLD_S, so that they land in slot FOLD_S - 1. K-block 0's kw = 0 pack writes each slot's 32 rows first (overwrite);
// every other pack accumulates. The walk runs one tile-row past the core's range (the reader's and this walk's
// extended M range), whose rows 0-1 complete the core's last tile-row; it is never output.
//
// Final pass: each completed tile-row's 32 plane rows are faces 0 and 2 of a tile (srcA, the kw-fold final pass's two
// 2-face runs: faces 1 and 3 read the right half-plane region FOLD_LEFT further, never written and masked), plus the
// bias row (srcB from the bf16 bias tile), packed with the edge mask that keeps lanes [0, C_out) of faces 0 and 2 (the
// lanes the base walk leaves at bias 0 + 0 come out 0). Same fp32 partials (each tap's K sum in DEST, as the base
// K-block of that tap) and same final add; the packer adds the 9 partials in the base order except on the 1-2 rows a
// tile-row takes from the next one at an M-block boundary.
//
// Raw LLK (capability): compute_kernel_lib::matmul_block packs whole tiles at a CB's write pointer; here a subblock's
// tiles go as rows at computed plane addresses (kw-fold's replayed 4-row PACR streams, the left half-plane only), with
// per-pack edge masks. compute_kernel_lib::add_bias_bcast_rows reads tiles at the CB read pointer; the final pass reads
// face pairs at computed addresses (kw-fold's fs_unpack_run) and packs with a lane mask. No helper sets edge masks.

#pragma once

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_kw_fold.hpp"

namespace conv2d_3x3::cout_fold {

constexpr uint32_t SLOT = 32 * 16 * 4;  // one tile-row of the plane: 32 rows of 16 fp32 lanes
constexpr uint32_t SBH = 4;             // subblock rows (fp32 DEST half)
constexpr uint32_t FINAL_BATCH = 4;     // output tiles per DEST section in the final pass

struct State {
    uint32_t fin = 0;  // core-local tile-rows output so far
    bool bias_waited = false;
};

#ifdef TRISC_PACK
// Edge masks (tt-isa-documentation BlackholeA0 Packers/EdgeMasking.md): mask sections SEC0 pass (0 during the final
// pass), SEC1 zero, SEC2 lanes [0, C_out); row tables T0 every row -> SEC0, T1 row 0 -> SEC1, T2 row 1 -> SEC1, T3
// every row -> SEC2; face tables F0 every face -> T0, F1 face 0 -> T1, F2 face 0 -> T2, F3 faces 0, 2 -> T3 and 1, 3 ->
// T0. Face lookup is on only inside this file's packs (with pack_reads_per_xy_plane 16, the tile position's Z is the
// face count of the PACR stream); T0 / SEC0 are the hardware defaults every other pack uses.
inline void mask_tables_init(uint32_t cout_mask) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_EDGE_OFFSET_SEC1_mask_ADDR32, 0, 0xffff>(0);
    cfg_reg_rmw_tensix<PCK_EDGE_OFFSET_SEC2_mask_ADDR32, 0, 0xffff>(cout_mask);
    cfg_reg_rmw_tensix<TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32, 0, 0xffffffff>(0x1);
    cfg_reg_rmw_tensix<TILE_ROW_SET_MAPPING_2_row_set_mapping_0_ADDR32, 0, 0xffffffff>(0x4);
    cfg_reg_rmw_tensix<TILE_ROW_SET_MAPPING_3_row_set_mapping_0_ADDR32, 0, 0xffffffff>(0xaaaaaaaa);
    cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32, 0, 0xffffffff>(0);
    cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_1_face_set_mapping_0_ADDR32, 0, 0xffffffff>(0x1);
    cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_2_face_set_mapping_0_ADDR32, 0, 0xffffffff>(0x2);
    cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_3_face_set_mapping_0_ADDR32, 0, 0xffffffff>(0x33);
}
inline void mask_select(uint32_t face_table) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0x100 | face_table);
}
inline void masks_on() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(16);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0x100);
}
// Back to the defaults (pack_reads_per_xy_plane 1 is what a packer data-format reconfig expects).
inline void masks_off() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);
}
inline void final_masks_on() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_EDGE_OFFSET_SEC0_mask_ADDR32, 0, 0xffff>(0);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(16);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0x103);
}
inline void final_masks_off() {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_EDGE_OFFSET_SEC0_mask_ADDR32, 0, 0xffff>(0xffff);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);
}

// kw-fold's face-split stream (kw_fold::pack_fst), left half-plane only: DEST tiles t0 .. t0 + g - 1, faces 0 and 2 of
// each, as 32 g consecutive 64 B rows from byte address addr. Entry and exit state: packer ch0 Y / Z at 0.
inline void pack_init() {
    addr_mod_pack_t{.y_src = {.incr = 4}}.set(ADDR_MOD_0);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(2 * FACE_R_DIM * FACE_C_DIM * 4);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(FACE_R_DIM * FACE_C_DIM * 4);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    load_replay_buf(kw_fold::FST_REPLAY, 8, [] {
        kw_fold::fst_pacr_face(ADDR_MOD_2, 0);
        kw_fold::fst_pacr_face(ADDR_MOD_1, 1);
    });
}
inline void pack_left(uint32_t addr, uint32_t t0, uint32_t g) {
    program_packer_destination((addr >> 4) - 1);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 4 * t0);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(kw_fold::FST_REPLAY, 4);
    }
    lltt::replay(kw_fold::FST_REPLAY + 4, 4);
}

// One subblock's three kw packs: DEST tiles 0 .. h - 1 are core-local tile-rows o0 .. (global g0 ..) of kernel row
// kb. A stream is one run of tile-rows in consecutive slots; a tile-row at an image-row start begins its own (its
// masked row is the stream's face 0). l1acc: the packer's L1 accumulation state.
template <uint32_t R, uint32_t S, uint32_t LS, uint32_t PRE>
inline void pack_subblock(uint32_t plane, uint32_t o0, uint32_t h, uint32_t g0, bool first_kb, uint32_t& l1acc) {
    for (uint32_t kw = 0; kw < 3; ++kw) {
        // Every stream but K-block 0's kw = 0 accumulates onto rows an earlier stream wrote: its reads follow them.
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
        const uint32_t acc = first_kb && kw == 0 ? 0 : 1;
        if (acc != l1acc) {
            llk_pack_reconfig_l1_acc(acc);
            l1acc = acc;
        }
        const uint32_t shift = kw * (64 + LS * 4);
        uint32_t slot = o0 % S;
        uint32_t gr = g0 % R;
        for (uint32_t j = 0; j < h;) {
            uint32_t e = j + 1;
            uint32_t es = slot + 1 == S ? 0 : slot + 1;
            uint32_t er = gr + 1 == R ? 0 : gr + 1;
            while (e < h && es != 0 && er != 0) {
                ++e;
                es = es + 1 == S ? 0 : es + 1;
                er = er + 1 == R ? 0 : er + 1;
            }
            const bool special = kw != 1 && gr == 0;
            if (special) {
                mask_select(kw == 0 ? 1 : 2);
            }
            pack_left(plane + PRE + slot * SLOT - shift, j, e - j);
            if (kw != 0 && slot == 0) {
                // Slot 0's spill rows, again from the alias slot: into slot S - 1.
                pack_left(plane + PRE + S * SLOT - shift, j, 1);
            }
            if (special) {
                mask_select(0);
            }
            j = e;
            slot = es;
            gr = er;
        }
    }
}
#endif

// One M-block of the fold walk. pre_k_block: the tilize hook + resident-weights wait (K-block kb's bm units of the kw =
// 0 tap of kernel row kb into cb_in0). m_tile_start: the core's first output tile-row; ext: its walk tile-rows (the
// output tile-rows + 1).
template <
    uint32_t cb_in0,
    uint32_t cb_w,
    uint32_t cb_plane,
    uint32_t cb_token,
    uint32_t cb_out,
    uint32_t cb_bias,
    uint32_t bm,
    uint32_t bk,
    bool relu,
    typename PreKBlockFn>
ALWI void block(const PreKBlockFn& pre_k_block, State& st, uint32_t m_block_idx, uint32_t m_tile_start, uint32_t ext) {
    constexpr uint32_t R = FOLD_R;
    constexpr uint32_t S = FOLD_S;
    constexpr uint32_t LS = FOLD_LS;
    constexpr uint32_t PRE = FOLD_PRE;
    constexpr uint32_t LEFT = FOLD_LEFT;
    constexpr uint32_t ZK = LEFT / 4 <= 0xffff ? 1 : 2;
    static_assert(S == bm + 1 && LEFT % (4 * ZK) == 0 && LEFT / (4 * ZK) <= 0xffff && 3 * LS <= 16);
    CircularBuffer in0(cb_in0);
    CircularBuffer token(cb_token);
    CircularBuffer bias(cb_bias);

    const uint32_t t_first = m_block_idx * bm;
    const uint32_t valid = ext - t_first < bm ? ext - t_first : bm;
    [[maybe_unused]] uint32_t plane_wr = 0;
    PACK((plane_wr = get_local_cb_interface(get_output_id(cb_plane)).fifo_wr_ptr << 4));
    {
        MaybeDeviceZoneScope("compute_fold_walk");
        for (uint32_t kb = 0; kb < 3; ++kb) {
            pre_k_block(kb, 3, kb == 2);
            // Matmul state after the hook (the tilize reconfigures unpack / math / pack); its pack left L1 acc off.
            reconfig_data_format(cb_w, cb_in0);
            pack_reconfig_data_format(cb_plane);
            PACK((pack_init()));
            PACK((masks_on()));
            [[maybe_unused]] uint32_t l1acc = 0;
            uint32_t init_rows = SBH;
            ckernel::matmul_block_init(cb_in0, cb_w, false, 1, SBH, bk);
            in0.wait_front(bm * bk);
            const uint32_t ng = kw_fold::row_groups<SBH>(valid);
            for (uint32_t k = 0; k < ng; ++k) {
                uint32_t i0 = 0, h = 0;
                kw_fold::row_group<SBH>(0, valid, k, i0, h);
                if ((h == 1) != (init_rows == 1)) {
                    // ct = 1: the unpack / math MOPs change only where rt = 1 flips ct >= rt.
                    ckernel::matmul_block_init(cb_in0, cb_w, false, 1, h, bk);
                }
                init_rows = h;
                tile_regs_acquire();
                kw_fold::matmul_k_rows<1, SBH, bk>(cb_in0, cb_w, i0 * bk, kb * bk, 1, 0, h);
                tile_regs_commit();
                tile_regs_wait();
                PACK((pack_subblock<R, S, LS, PRE>(
                    plane_wr, t_first + i0, h, m_tile_start + t_first + i0, kb == 0, l1acc)));
                tile_regs_release();
            }
            in0.pop_front(bm * bk);
            PACK((masks_off()));
            PACK((llk_pack_reconfig_l1_acc(0)));
        }
        // The final pass unpacks the plane: a CB push from the pack thread is stored only after the packer drained.
        token.reserve_back(1);
        token.push_back(1);
    }

    // ---- output tile-rows [fin, t_first + valid - 1): plane rows + bias, lanes [0, C_out) of faces 0 and 2.
    const uint32_t end_fin = t_first + valid - 1;
    token.wait_front(1);
    if (end_fin > st.fin) {
        MaybeDeviceZoneScope("compute_fold_final");
        if (!st.bias_waited) {
            MaybeDeviceZoneScope("compute_bias_wait");
            bias.wait_front(1);
            st.bias_waited = true;
        }
        [[maybe_unused]] uint32_t plane_rd = 0;
        [[maybe_unused]] uint32_t bias_rd = 0;
        UNPACK((plane_rd = get_local_cb_interface(get_operand_id(cb_plane)).fifo_rd_ptr << 4));
        UNPACK((bias_rd = get_local_cb_interface(get_operand_id(cb_bias)).fifo_rd_ptr << 4));
        reconfig_data_format(cb_plane, cb_bias);
        pack_reconfig_data_format(cb_out);
        if constexpr (relu) {
            pack_relu_config(ReluConfig::zero());
        }
        UNPACK((kw_fold::fs_unpack_init<ZK>(LEFT)));
        MATH((kw_fold::fs_math_init()));
        PACK((llk_pack_init<PackMode::Default>(cb_out)));
        PACK((final_masks_on()));
        const uint32_t total = end_fin - st.fin;
        UNPACK(({
            uint32_t u = st.fin;
            while (u < end_fin) {
                const uint32_t slot = u % S;
                const uint32_t n = end_fin - u < S - slot ? end_fin - u : S - slot;
                kw_fold::fs_unpack_run<ZK>(plane_rd + PRE + slot * SLOT, n, bias_rd);
                u += n;
            }
        }));
        MATH(({
            for (uint32_t f0 = 0; f0 < total; f0 += FINAL_BATCH) {
                const uint32_t cnt = total - f0 < FINAL_BATCH ? total - f0 : FINAL_BATCH;
                llk_math_wait_for_dest_available();
                for (uint32_t j = 0; j < cnt; ++j) {
                    kw_fold::fs_math_tile(j);
                }
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }
        }));
        PACK(({
            for (uint32_t f0 = 0; f0 < total; f0 += FINAL_BATCH) {
                const uint32_t cnt = total - f0 < FINAL_BATCH ? total - f0 : FINAL_BATCH;
                llk_packer_wait_for_math_done();
                for (uint32_t j = 0; j < cnt; ++j) {
                    // One tile per push: the writer drains in groups that never straddle the CB's end.
                    cb_reserve_back(cb_out, 1);
                    llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(j, cb_out, 0);
                    cb_push_back(cb_out, 1);
                }
                llk_pack_dest_section_done<DST_ACCUM_MODE>();
            }
        }));
        PACK((final_masks_off()));
        UNPACK((kw_fold::fs_unpack_uninit()));
        if constexpr (relu) {
            pack_relu_config(ReluConfig::none());
        }
        st.fin = end_fin;
    }
    token.pop_front(1);
}

}  // namespace conv2d_3x3::cout_fold
