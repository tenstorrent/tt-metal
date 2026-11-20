// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated bench (compute only, one core, Blackhole, fp32 DEST half-sync): the kw-fold Q pack, i.e. fp32 DEST tiles
// packed as rows onto an fp32 accumulator plane at an arbitrary row offset (overwrite or packer L1 accumulation).
//   layout 0 (zst, HEAD): [rows, 32] planes of 128 B rows; one strided 2-interface PACR per row, one stream per run.
//   layout 1 (fst, face-split): each plane two [rows, 16] half-planes of 64 B rows; two streams per run (left half:
//     faces 0, 2; right half: faces 1, 3) of normal-mode 4-interface PACRs (4 rows each).
// shape 0: 4 x 1 column runs (one 4-tile run into one plane, stem.conv2's Q_1); shape 1: 1 x 4 (four 1-tile runs, one
// per plane, the 1 x gw subblocks and the fused side rows). Each rep is one DEST section of 4 tiles. DEST holds garbage
// (math does nothing): perf only. RAW LLK (bench): the op's pack streams (conv2d_3x3_kw_fold.hpp).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

namespace bench {
#ifdef TRISC_PACK
constexpr uint32_t REPLAY = ckernel::packer::replay_buf_offset;
inline void zst_init() {
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    load_replay_buf(REPLAY, 16, [] {
        for (uint32_t i = 0; i < 15; ++i) {
            TTI_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_STRIDED_MODE,
                ADDR_MOD_1,
                p_pacr::ADDR_CNT_CTXT_0,
                0,
                p_pacr::TWO_INTFS_ACTIVE,
                0,
                0,
                p_pacr::NO_CTXT_CTRL,
                0,
                0);
        }
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_STRIDED_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::TWO_INTFS_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0);
    });
}
inline void pack_zst(uint32_t addr, uint32_t t0, uint32_t g) {
    program_packer_destination((addr >> 4) - 1);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(REPLAY, 16);
    }
    lltt::replay(REPLAY, 15);
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::TWO_INTFS_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    set_dst_write_addr(0);
}

inline void fst_init() {
    addr_mod_pack_t{.y_src = {.incr = 4}}.set(ADDR_MOD_0);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(2 * FACE_R_DIM * FACE_C_DIM * 4);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(FACE_R_DIM * FACE_C_DIM * 4);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    load_replay_buf(REPLAY, 4, [] {
        for (uint32_t i = 0; i < 3; ++i) {
            TTI_PACR(
                p_pacr::CFG_CTXT_0,
                p_pacr::NO_ROW_PAD_ZERO,
                p_pacr::DST_ACCESS_NORMAL_MODE,
                ADDR_MOD_0,
                p_pacr::ADDR_CNT_CTXT_0,
                0,
                p_pacr::ALL_INTF_ACTIVE,
                0,
                0,
                p_pacr::NO_CTXT_CTRL,
                0,
                0);
        }
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_2,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0);
    });
}
inline void fst_last() {
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        1);
}
inline void pack_fst(uint32_t addr, uint32_t half, uint32_t t0, uint32_t g) {
    for (uint32_t s = 0; s < 2; ++s) {
        program_packer_destination(((addr + s * half) >> 4) - 1);
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0001);
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 4 * t0 + s);
        TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
            lltt::replay(REPLAY, 4);
        }
        lltt::replay(REPLAY, 3);
        fst_last();
    }
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    set_dst_write_addr(0);
}
// fst2: one destination program per run. The left stream's last PACR steps the packer's ch1 (output) Y counter by 4
// (ch1 Y stride HALF / 4: a 16-bit byte stride), so the right stream starts HALF further; the right stream's last
// PACR clears it. Replay buffer: [face | last face, left | last face, right].
inline void fst2_init(uint32_t half) {
    addr_mod_pack_t{.y_src = {.incr = 4}}.set(ADDR_MOD_0);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .y_dst = {.incr = 4}, .z_src = {.incr = 0, .clr = 1}}.set(
        ADDR_MOD_3);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .y_dst = {.incr = 0, .clr = 1}, .z_src = {.incr = 0, .clr = 1}}.set(
        ADDR_MOD_1);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(2 * FACE_R_DIM * FACE_C_DIM * 4);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(FACE_R_DIM * FACE_C_DIM * 4);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(half / 4);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111);
    load_replay_buf(REPLAY, 12, [] {
        for (uint32_t f = 0; f < 3; ++f) {
            for (uint32_t i = 0; i < 3; ++i) {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_0,
                    p_pacr::ADDR_CNT_CTXT_0,
                    0,
                    p_pacr::ALL_INTF_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);
            }
            if (f == 0) {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_2,
                    p_pacr::ADDR_CNT_CTXT_0,
                    0,
                    p_pacr::ALL_INTF_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    0);
            } else if (f == 1) {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_3,
                    p_pacr::ADDR_CNT_CTXT_0,
                    0,
                    p_pacr::ALL_INTF_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    1);
            } else {
                TTI_PACR(
                    p_pacr::CFG_CTXT_0,
                    p_pacr::NO_ROW_PAD_ZERO,
                    p_pacr::DST_ACCESS_NORMAL_MODE,
                    ADDR_MOD_1,
                    p_pacr::ADDR_CNT_CTXT_0,
                    0,
                    p_pacr::ALL_INTF_ACTIVE,
                    0,
                    0,
                    p_pacr::NO_CTXT_CTRL,
                    0,
                    1);
            }
        }
    });
}
// Entry and exit state: packer ch0 Y / Z and ch1 Y at 0.
inline void pack_fst2(uint32_t addr, uint32_t t0, uint32_t g) {
    program_packer_destination((addr >> 4) - 1);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 4 * t0);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(REPLAY, 4);
    }
    lltt::replay(REPLAY + 4, 4);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 4 * t0 + 1);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(REPLAY, 4);
    }
    lltt::replay(REPLAY + 8, 4);
}
// fst3: the right stream's destination by CFGSHIFTMASK (L1_Dest_addr += SCRATCH_SEC2 = HALF / 16) after the left
// stream's Last; each stream's last PACR clears Y and Z (no per-stream counter resets). Replay buffer:
// [face | last face].
inline void fst3_pacr_face(uint32_t last_addr_mod, uint32_t last) {
    for (uint32_t i = 0; i < 3; ++i) {
        TTI_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            0,
            p_pacr::ALL_INTF_ACTIVE,
            0,
            0,
            p_pacr::NO_CTXT_CTRL,
            0,
            0);
    }
    TT_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        last_addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        last);
}
inline void fst3_init(uint32_t half) {
    addr_mod_pack_t{.y_src = {.incr = 4}}.set(ADDR_MOD_0);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_2);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(2 * FACE_R_DIM * FACE_C_DIM * 4);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(FACE_R_DIM * FACE_C_DIM * 4);
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TT_SETDMAREG(0, LOWER_HALFWORD(half / 16), 0, LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(half / 16), 0, HI_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR_OFFSET, 0, SCRATCH_SEC2_val_ADDR32);
    TTI_NOP;
    load_replay_buf(REPLAY, 8, [] {
        fst3_pacr_face(ADDR_MOD_2, 0);
        fst3_pacr_face(ADDR_MOD_1, 1);
    });
}
inline void pack_fst3(uint32_t addr, uint32_t t0, uint32_t g) {
    program_packer_destination((addr >> 4) - 1);
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 4 * t0);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(REPLAY, 4);
    }
    lltt::replay(REPLAY + 4, 4);
    TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    TTI_NOP;
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 4 * t0 + 1);
    for (uint32_t k = 0; k + 1 < 2 * g; ++k) {
        lltt::replay(REPLAY, 4);
    }
    lltt::replay(REPLAY + 4, 4);
}
#endif
}  // namespace bench

void kernel_main() {
    constexpr uint32_t layout = get_compile_time_arg_val(0);
    constexpr uint32_t shape = get_compile_time_arg_val(1);
    constexpr uint32_t l1acc = get_compile_time_arg_val(2);
    constexpr uint32_t reps = get_compile_time_arg_val(3);
    constexpr uint32_t plane = get_compile_time_arg_val(4);  // bytes; 4 planes
    constexpr uint32_t rows = plane / 128;
    constexpr uint32_t check = get_compile_time_arg_val(5);  // DEST from cb 1's 4 tiles (copy_tile): layout check
    constexpr uint32_t cb_acc = 0;
    constexpr uint32_t cb_in = 1;
    constexpr uint32_t d = 1;
    if constexpr (check) {
        compute_kernel_hw_startup(cb_in, cb_in, cb_acc);
        copy_init(cb_in);
    } else {
        compute_kernel_hw_startup(cb_acc, cb_acc, cb_acc);
    }
    pack_reconfig_data_format(cb_acc);
    if constexpr (layout == 0) {
        pack_untilize_dest_init<1, 1>(cb_acc);
        PACK((bench::zst_init()));
    } else if constexpr (layout == 1) {
        PACK((bench::fst_init()));
    } else if constexpr (layout == 2) {
        PACK((bench::fst2_init(plane / 2)));
    } else {
        PACK((bench::fst3_init(plane / 2)));
    }
    pack_reconfig_l1_acc(l1acc);
    [[maybe_unused]] uint32_t acc = 0;
    PACK((acc = get_local_cb_interface(get_output_id(cb_acc)).fifo_wr_ptr << 4));
    constexpr uint32_t PITCH = layout == 0 ? 128 : 64;
    for (uint32_t r = 0; r < reps; ++r) {
        tile_regs_acquire();
        if constexpr (check) {
            for (uint32_t j = 0; j < 4; ++j) {
                copy_tile(cb_in, j, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        // Rows advance like a walk: each rep's tiles at the next tile-row band (wrapping), d-row offset.
        [[maybe_unused]] const uint32_t row = 2 * d + (r * 32 * (shape == 0 ? 4 : 1)) % (rows - 4 * 32 - 2 * d);
        if constexpr (shape == 0) {
            if constexpr (layout == 0) {
                PACK((bench::pack_zst(acc + row * PITCH, 0, 4)));
            } else if constexpr (layout == 1) {
                PACK((bench::pack_fst(acc + row * PITCH, plane / 2, 0, 4)));
            } else if constexpr (layout == 2) {
                PACK((bench::pack_fst2(acc + row * PITCH, 0, 4)));
            } else {
                PACK((bench::pack_fst3(acc + row * PITCH, 0, 4)));
            }
        } else if constexpr (shape == 1) {
            for (uint32_t j = 0; j < 4; ++j) {
                if constexpr (layout == 0) {
                    PACK((bench::pack_zst(acc + j * plane + row * PITCH, j, 1)));
                } else if constexpr (layout == 1) {
                    PACK((bench::pack_fst(acc + j * plane + row * PITCH, plane / 2, j, 1)));
                } else if constexpr (layout == 2) {
                    PACK((bench::pack_fst2(acc + j * plane + row * PITCH, j, 1)));
                } else {
                    PACK((bench::pack_fst3(acc + j * plane + row * PITCH, j, 1)));
                }
            }
        } else {
            // The fused side row (gw 2): Q_0 n0, n1 at +d into planes 0, 1, a packer drain, Q_2 n0, n1 at -d.
            for (uint32_t j = 0; j < 4; ++j) {
                if (j == 2) {
                    PACK((TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK)));
                }
                [[maybe_unused]] const uint32_t a = acc + (j & 1) * plane + (j < 2 ? row + d : row - d) * PITCH;
                if constexpr (layout == 0) {
                    PACK((bench::pack_zst(a, j, 1)));
                } else if constexpr (layout == 1) {
                    PACK((bench::pack_fst(a, plane / 2, j, 1)));
                } else if constexpr (layout == 2) {
                    PACK((bench::pack_fst2(a, j, 1)));
                } else {
                    PACK((bench::pack_fst3(a, j, 1)));
                }
            }
        }
        tile_regs_release();
    }
    pack_reconfig_l1_acc(0);
}
