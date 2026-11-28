// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_BLACKHOLE)
#ifdef TRISC_UNPACK
namespace ckernel {

ALWI void sdh_ctx_wait() {
    volatile std::uint32_t tt_reg_ptr* sem = pc_buf_base + PC_BUF_SEMAPHORE_BASE + semaphore::UNPACK_SYNC;
    while (static_cast<std::uint8_t>(*sem) >= 2) {
    }
}

ALWI void sdh_ctx_post() {
    volatile std::uint32_t tt_reg_ptr* sem = pc_buf_base + PC_BUF_SEMAPHORE_BASE + semaphore::UNPACK_SYNC;
    *sem = 0;
}

ALWI void sdh_ctx_done() {
    TTI_SEMGET(semaphore::t6_sem(semaphore::UNPACK_SYNC));
    if (unp_cfg_context == 0) {
        unp_cfg_context = 1;
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    } else {
        unp_cfg_context = 0;
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    }
}

ALWI void sdh_unpack_mm_init(const std::uint32_t transpose, const std::uint32_t kt_dim) {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXX(p_setadc::UNP_A, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP_B, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TT_SETDMAREG(0, LOWER_HALFWORD(kt_dim), 0, LO_16(p_gpr_unpack::KT_DIM));
}

ALWI void sdh_unpack_mm_step_a() {
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    if (unp_cfg_context == 0) {
        TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
        TTI_NOP;
    } else {
        TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
        TTI_NOP;
    }
}

ALWI void sdh_unpack_mm_step_b() {
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    if (unp_cfg_context == 0) {
        TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_address_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_address_ADDR32);
        TTI_NOP;
    } else {
        TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
        TTI_NOP;
    }
}

ALWI void sdh_unpack_ab_init() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCXX(p_setadc::UNP_AB, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

template <BroadcastType B>
ALWI void sdh_unpack_ab(
    const std::uint32_t cb0, const std::uint32_t cb1, const std::uint32_t i0, const std::uint32_t i1) {
    static_assert(B == BroadcastType::NONE || B == BroadcastType::COL);
    const std::uint32_t id0 = get_operand_id(cb0);
    const std::uint32_t id1 = get_operand_id(cb1);
    const std::uint32_t a0 =
        get_local_cb_interface(id0).fifo_rd_ptr - 1 + get_local_cb_interface(id0).fifo_page_size * i0;
    const std::uint32_t a1 =
        get_local_cb_interface(id1).fifo_rd_ptr - 1 + get_local_cb_interface(id1).fifo_page_size * i1;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t* cfg = get_cfg_pointer();
    sdh_ctx_wait();
    if (unp_cfg_context == 0) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = a0;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = a1;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a0;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = a1;
    }
    sdh_ctx_post();
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (std::uint32_t r = 0; r < 2; r++) {
        if constexpr (B == BroadcastType::COL) {
            TTI_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001);
        } else {
            TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        }
        TTI_NOP;
    }
    sdh_ctx_done();
}

template <std::uint32_t REG, std::uint32_t SAVE, bool ON, std::uint32_t V>
ALWI void sdh_unpack_swap_reg() {
    if constexpr (ON) {
        TTI_RDCFG(SAVE, REG);
        TTI_SETDMAREG(0, V & 0xFFFF, 0, LO_16(p_gpr_unpack::TMP0));
        TTI_SETDMAREG(0, V >> 16, 0, HI_16(p_gpr_unpack::TMP0));
        TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, REG);
    } else {
        TTI_WRCFG(SAVE, p_cfg::WRCFG_32b, REG);
    }
}

template <bool ON>
ALWI void sdh_unpack_rmax_cfg() {
    if constexpr (ON) {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);
        TTI_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
        TTI_SETADCXX(p_setadc::UNP_A, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    } else {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);
    }
    sdh_unpack_swap_reg<
        THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32,
        p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0,
        ON,
        (FACE_C_DIM * FACE_R_DIM) | ((FACE_C_DIM * FACE_R_DIM) << 16)>();
    sdh_unpack_swap_reg<
        THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1,
        p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1,
        ON,
        4 | (1 << 16)>();
}

template <std::uint32_t COLS>
ALWI void sdh_unpack_rmax(const std::uint32_t cb, const std::uint32_t cb_scale, const std::uint32_t row0) {
    const std::uint32_t id0 = get_operand_id(cb);
    const std::uint32_t id1 = get_operand_id(cb_scale);
    const std::uint32_t a0 =
        get_local_cb_interface(id0).fifo_rd_ptr - 1 + get_local_cb_interface(id0).fifo_page_size * row0;
    const std::uint32_t a1 = get_local_cb_interface(id1).fifo_rd_ptr - 1;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t* cfg = get_cfg_pointer();
    sdh_ctx_wait();
    if (unp_cfg_context == 0) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = a0;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = a1;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a0;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = a1;
    }
    sdh_ctx_post();
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#pragma GCC unroll 0
    for (std::uint32_t c = 0; c < COLS; c++) {
        TTI_UNPACR(SrcA, 0b00000001, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }
    sdh_ctx_done();
}

ALWI void sdh_unpack_rsum_init() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);
    TTI_SETADCXX(p_setadc::UNP_B, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP_A, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

ALWI void sdh_unpack_rsum(
    const std::uint32_t cb, const std::uint32_t cb_scale, const std::uint32_t i, const std::uint32_t is) {
    const std::uint32_t id0 = get_operand_id(cb);
    const std::uint32_t id1 = get_operand_id(cb_scale);
    const std::uint32_t a0 =
        get_local_cb_interface(id0).fifo_rd_ptr - 1 + get_local_cb_interface(id0).fifo_page_size * i;
    const std::uint32_t a1 =
        get_local_cb_interface(id1).fifo_rd_ptr - 1 + get_local_cb_interface(id1).fifo_page_size * is;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t* cfg = get_cfg_pointer();
    sdh_ctx_wait();
    if (unp_cfg_context == 0) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = a1;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = a0;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a1;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = a0;
    }
    sdh_ctx_post();
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    sdh_ctx_done();
}

template <EltwiseBinaryReuseDestType R>
ALWI void sdh_unpack_reuse_init() {
    static_assert(R == EltwiseBinaryReuseDestType::DEST_TO_SRCA || R == EltwiseBinaryReuseDestType::DEST_TO_SRCB);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    if constexpr (R == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
        TTI_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    } else {
        TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    }
}

template <EltwiseBinaryReuseDestType R>
ALWI void sdh_unpack_reuse(const std::uint32_t cb, const std::uint32_t i) {
    const std::uint32_t id = get_operand_id(cb);
    const std::uint32_t a = get_local_cb_interface(id).fifo_rd_ptr - 1 + get_local_cb_interface(id).fifo_page_size * i;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t* cfg = get_cfg_pointer();
    sdh_ctx_wait();
    if constexpr (R == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
        cfg[unp_cfg_context == 0 ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = a;
    } else {
        cfg[unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a;
    }
    sdh_ctx_post();
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (std::uint32_t f = 0; f < 4; f++) {
        if constexpr (R == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
            TTI_UNPACR_NOP(
                SrcA,
                0,
                0,
                p_unpacr_nop::SET_DVALID,
                0,
                p_unpacr_nop::WAIT_LIKE_UNPACR,
                0,
                0,
                p_unpacr_nop::UNP_ZEROSRC);
            TTI_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        } else {
            TTI_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_UNPACR_NOP(
                SrcB,
                0,
                0,
                p_unpacr_nop::SET_DVALID,
                0,
                p_unpacr_nop::WAIT_LIKE_UNPACR,
                0,
                0,
                p_unpacr_nop::UNP_ZEROSRC);
        }
    }
    sdh_ctx_done();
}

ALWI void sdh_unpack_rmaxg_init() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP_B, FACE_C_DIM - 1, 0x0);
}

ALWI void sdh_unpack_rmaxg(
    const std::uint32_t cb, const std::uint32_t cb_scale, const std::uint32_t i, const std::uint32_t is) {
    const std::uint32_t id0 = get_operand_id(cb);
    const std::uint32_t id1 = get_operand_id(cb_scale);
    const std::uint32_t a0 =
        get_local_cb_interface(id0).fifo_rd_ptr - 1 + get_local_cb_interface(id0).fifo_page_size * i;
    const std::uint32_t a1 =
        get_local_cb_interface(id1).fifo_rd_ptr - 1 + get_local_cb_interface(id1).fifo_page_size * is;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t* cfg = get_cfg_pointer();
    sdh_ctx_wait();
    if (unp_cfg_context == 0) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = a0;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = a1;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a0;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = a1;
    }
    sdh_ctx_post();
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (std::uint32_t f = 0; f < 4; f++) {
        TTI_UNPACR(SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }
    sdh_ctx_done();
}

ALWI void sdh_unpack_mm(
    const std::uint32_t cb_in0,
    const std::uint32_t cb_in1,
    const std::uint32_t idx0,
    const std::uint32_t idx1,
    const std::uint32_t ct,
    const std::uint32_t rt,
    const std::uint32_t kt) {
    const std::uint32_t id0 = get_operand_id(cb_in0);
    const std::uint32_t id1 = get_operand_id(cb_in1);
    const std::uint32_t base0 = get_local_cb_interface(id0).fifo_rd_ptr - 1;
    const std::uint32_t base1 = get_local_cb_interface(id1).fifo_rd_ptr - 1;
    const std::uint32_t size0 = get_local_cb_interface(id0).fifo_page_size;
    const std::uint32_t size1 = get_local_cb_interface(id1).fifo_page_size;
    volatile std::uint32_t* cfg = get_cfg_pointer();
    const bool reuse_a = ct >= rt;
    const std::uint32_t t_dim = reuse_a ? rt : ct;
    if (!reuse_a) {
        TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
    }
    for (std::uint32_t t = 0; t < t_dim; t++) {
        const std::uint32_t a0 = base0 + size0 * (idx0 + (reuse_a ? t * kt : 0));
        const std::uint32_t a1 = base1 + size1 * (idx1 + (reuse_a ? 0 : t));
        sdh_ctx_wait();
        if (unp_cfg_context == 0) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = a1;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = a0;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a1;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = a0;
        }
        sdh_ctx_post();
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        if (reuse_a) {
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            for (std::uint32_t c = 0; c < ct; c++) {
                sdh_unpack_mm_step_a();
            }
        } else {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            for (std::uint32_t r = 0; r < rt; r++) {
                sdh_unpack_mm_step_b();
            }
        }
        sdh_ctx_done();
    }
}

}  // namespace ckernel
#endif
#ifdef TRISC_PACK
namespace ckernel {

template <std::uint32_t AM, std::uint32_t LAST>
ALWI void sdh_pacr() {
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        AM,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        0,
        0,
        LAST);
}

template <bool ON>
ALWI void sdh_pack_rmask() {
    constexpr std::uint32_t edge0 = ON ? 0x00020000 : 0x0000FFFF;
    TTI_SETDMAREG(0, edge0 & 0xFFFF, 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, edge0 >> 16, 0, HI_16(p_gpr_pack::TMP0));
    if constexpr (ON) {
        TTI_SETDMAREG(0, 0x0001, 0, LO_16(p_gpr_pack::TMP_LO));
        TTI_SETDMAREG(0, 0x5555, 0, LO_16(p_gpr_pack::TMP1));
        TTI_SETDMAREG(0, 0x5555, 0, HI_16(p_gpr_pack::TMP1));
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(ON ? FACE_R_DIM : 1);
    if constexpr (!ON) {
        cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_enable_RMW>(0);
    }
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(ON ? p_gpr_pack::TMP_LO : p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(ON ? p_gpr_pack::TMP1 : p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
    if constexpr (ON) {
        cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32, 0, 0xffffffff>(0x11111111);
        cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0x100);
    }
    TTI_NOP;
    TTI_NOP;
}

template <bool OOO>
ALWI void sdh_pack(const std::uint32_t idst, const std::uint32_t cb, const std::uint32_t out_idx) {
    const std::uint8_t id = get_output_id(cb);
    auto& ci = get_local_cb_interface(id);
    std::uint32_t addr;
    if constexpr (OOO) {
        addr = ci.fifo_wr_ptr + static_cast<std::uint32_t>(ci.fifo_page_size) * out_idx - 1;
    } else {
        addr = ci.fifo_wr_ptr + ci.fifo_wr_tile_ptr - 1;
        ci.fifo_wr_tile_ptr += ci.fifo_page_size;
    }
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, idst);
    const std::uint32_t tagged = (1u << 31) | addr;
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(tagged), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
    TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_DMANOP;
#pragma GCC unroll 3
    for (int f = 0; f < 3; f++) {
        sdh_pacr<ADDR_MOD_0, 0>();
        sdh_pacr<ADDR_MOD_0, 0>();
        sdh_pacr<ADDR_MOD_0, 0>();
        sdh_pacr<ADDR_MOD_2, 0>();
    }
    sdh_pacr<ADDR_MOD_0, 0>();
    sdh_pacr<ADDR_MOD_0, 0>();
    sdh_pacr<ADDR_MOD_0, 0>();
    sdh_pacr<ADDR_MOD_1, 1>();
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

}  // namespace ckernel
#endif
#ifdef TRISC_MATH
namespace ckernel {

constexpr std::uint32_t sdh_reg_ab[8] = {
    ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32,
    ADDR_MOD_AB_SEC1_SrcAIncr_ADDR32,
    ADDR_MOD_AB_SEC2_SrcAIncr_ADDR32,
    ADDR_MOD_AB_SEC3_SrcAIncr_ADDR32,
    ADDR_MOD_AB_SEC4_SrcAIncr_ADDR32,
    ADDR_MOD_AB_SEC5_SrcAIncr_ADDR32,
    ADDR_MOD_AB_SEC6_SrcAIncr_ADDR32,
    ADDR_MOD_AB_SEC7_SrcAIncr_ADDR32};
constexpr std::uint32_t sdh_reg_d[8] = {
    ADDR_MOD_DST_SEC0_DestIncr_ADDR32,
    ADDR_MOD_DST_SEC1_DestIncr_ADDR32,
    ADDR_MOD_DST_SEC2_DestIncr_ADDR32,
    ADDR_MOD_DST_SEC3_DestIncr_ADDR32,
    ADDR_MOD_DST_SEC4_DestIncr_ADDR32,
    ADDR_MOD_DST_SEC5_DestIncr_ADDR32,
    ADDR_MOD_DST_SEC6_DestIncr_ADDR32,
    ADDR_MOD_DST_SEC7_DestIncr_ADDR32};
constexpr std::uint32_t sdh_reg_bias[8] = {
    ADDR_MOD_BIAS_SEC0_BiasIncr_ADDR32,
    ADDR_MOD_BIAS_SEC1_BiasIncr_ADDR32,
    ADDR_MOD_BIAS_SEC2_BiasIncr_ADDR32,
    ADDR_MOD_BIAS_SEC3_BiasIncr_ADDR32,
    ADDR_MOD_BIAS_SEC4_BiasIncr_ADDR32,
    ADDR_MOD_BIAS_SEC5_BiasIncr_ADDR32,
    ADDR_MOD_BIAS_SEC6_BiasIncr_ADDR32,
    ADDR_MOD_BIAS_SEC7_BiasIncr_ADDR32};

constexpr std::uint32_t sdh_s(std::uint32_t incr, std::uint32_t cr, std::uint32_t clr) {
    return incr | (cr << 6) | (clr << 7);
}
constexpr std::uint32_t sdh_d(std::uint32_t incr, std::uint32_t cr, std::uint32_t clr, std::uint32_t fid) {
    return incr | (cr << 10) | (clr << 11) | (fid << 13);
}

template <std::uint32_t M, std::uint32_t A, std::uint32_t B, std::uint32_t D>
ALWI void sdh_am() {
    TTI_SETC16(sdh_reg_ab[M], A | (B << 8));
    TTI_SETC16(sdh_reg_d[M], D);
    TTI_SETC16(sdh_reg_bias[M], 0);
}

constexpr std::uint32_t sdh_phases = to_underlying(MATH_FIDELITY);
constexpr std::uint32_t sdh_fid = sdh_phases > 0 ? 1 : 0;
static_assert(sdh_phases > 0);

inline std::uint32_t sdh_mm_mode = 2;

ALWI void sdh_mm_init(const std::uint32_t transpose, const std::uint32_t mode) {
    sdh_am<0, sdh_s(0, 0, 0), sdh_s(8, 0, 0), sdh_d(8, 0, 0, 0)>();
    sdh_am<5, sdh_s(0, 1, 1), sdh_s(0, 1, 1), sdh_d(0, 1, 1, sdh_fid)>();
    if (mode) {
        sdh_am<6, sdh_s(0, 1, 1), sdh_s(0, 1, 1), sdh_d(0, 1, 1, 4)>();
    }
    if (transpose) {
        sdh_am<1, sdh_s(32, 0, 0), sdh_s(0, 1, 0), sdh_d(8, 0, 0, 0)>();
    } else {
        sdh_am<1, sdh_s(16, 0, 0), sdh_s(0, 1, 0), sdh_d(8, 0, 0, 0)>();
    }
    sdh_am<2, sdh_s(0, 1, 0), sdh_s(32, 1, 0), sdh_d(8, 0, 0, 0)>();
    if (transpose) {
        sdh_am<4, sdh_s(16, 1, 0), sdh_s(48, 1, 0), sdh_d(0, 1, 0, 0)>();
    } else {
        sdh_am<4, sdh_s(32, 1, 0), sdh_s(48, 1, 0), sdh_d(0, 1, 0, 0)>();
    }
    if (mode) {
        TTI_REPLAY(16, 11, 0, 1);
        TTI_NOP;
        TTI_NOP;
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_NOP;
        TTI_NOP;
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_NOP;
        TTI_NOP;
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_NOP;
        TTI_NOP;
    } else {
        TTI_REPLAY(16, 16, 0, 1);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
    sdh_mm_mode = mode;
}

ALWI void sdh_mm_tile(const std::uint32_t dst, const std::uint32_t ct, const std::uint32_t rt) {
    const bool reuse_a = ct >= rt;
    const std::uint32_t t_dim = reuse_a ? rt : ct;
    const std::uint32_t r_dim = reuse_a ? ct : rt;
    for (std::uint32_t t = 0; t < t_dim; t++) {
        for (std::uint32_t r = 0; r < r_dim; r++) {
            const std::uint32_t idx = dst + (reuse_a ? ct * t + r : t + r * ct);
            TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (idx << 6) + sdo_dest_base());
            if (sdh_mm_mode) {
                for (std::uint32_t p = 0; p < sdh_phases; p++) {
                    TTI_REPLAY(16, 11, 0, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                    TTI_REPLAY(16, 11, 0, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
                    TTI_REPLAY(16, 11, 0, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                    TTI_REPLAY(16, 11, 0, 0);
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);
                }
            } else {
                for (std::uint32_t p = 0; p < sdh_phases; p++) {
                    TTI_REPLAY(16, 16, 0, 0);
                }
            }
            if (reuse_a) {
                TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
            } else {
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
            }
            if (r == r_dim - 1) {
                if (reuse_a) {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                } else {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
                }
            }
        }
    }
}

inline std::uint32_t sdh_bin_acc = 0;

template <EltwiseBinaryType T, BroadcastType B>
ALWI void sdh_bin_init(const std::uint32_t acc_to_dest) {
    constexpr std::uint32_t f = (T == EltwiseBinaryType::ELWMUL) ? sdh_fid : 0;
    constexpr std::uint32_t bi = (B == BroadcastType::NONE || B == BroadcastType::COL) ? 8 : 0;
    sdh_am<0, sdh_s(8, 0, 0), sdh_s(bi, 0, 0), sdh_d(8, 0, 0, 0)>();
    sdh_am<1, sdh_s(0, 0, 0), sdh_s(0, 0, 0), sdh_d(0, 0, 0, 0)>();
    sdh_am<2, sdh_s(0, 0, 1), sdh_s(0, 0, 1), sdh_d(0, 1, 0, f)>();
    sdh_am<3, sdh_s(0, 0, 1), sdh_s(0, 0, 1), sdh_d(8, 0, 0, 4) | (1u << 12)>();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
    sdh_bin_acc = acc_to_dest;
}

template <EltwiseBinaryType T, std::uint32_t BC>
ALWI void sdh_elw_as() {
    if (sdh_bin_acc) {
        if constexpr (T == EltwiseBinaryType::ELWADD) {
            TTI_ELWADD(0, 1, BC, ADDR_MOD_0, 0);
            TTI_ELWADD(0, 1, BC, ADDR_MOD_0, 0);
        } else {
            TTI_ELWSUB(0, 1, BC, ADDR_MOD_0, 0);
            TTI_ELWSUB(0, 1, BC, ADDR_MOD_0, 0);
        }
    } else {
        if constexpr (T == EltwiseBinaryType::ELWADD) {
            TTI_ELWADD(0, 0, BC, ADDR_MOD_0, 0);
            TTI_ELWADD(0, 0, BC, ADDR_MOD_0, 0);
        } else {
            TTI_ELWSUB(0, 0, BC, ADDR_MOD_0, 0);
            TTI_ELWSUB(0, 0, BC, ADDR_MOD_0, 0);
        }
    }
}

template <std::uint32_t BC, std::uint32_t CLR>
ALWI void sdh_elw_mul_face() {
    for (std::uint32_t p = 0; p + 1 < sdh_phases; p++) {
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_0, 0);
        TTI_ELWMUL(0, 0, BC, ADDR_MOD_2, 0);
    }
    TTI_ELWMUL(0, 0, BC, ADDR_MOD_0, 0);
    TTI_ELWMUL(CLR, 0, BC, ADDR_MOD_3, 0);
}

template <EltwiseBinaryReuseDestType R, std::uint32_t ROW>
ALWI void sdh_d2s_rows() {
    if constexpr (R == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
        TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + ROW, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, ROW);
    } else {
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + ROW, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, ROW);
    }
}

template <EltwiseBinaryType T, EltwiseBinaryReuseDestType R>
ALWI void sdh_bin_reuse() {
    constexpr std::uint32_t vld = R == EltwiseBinaryReuseDestType::DEST_TO_SRCA ? p_stall::SRCA_VLD : p_stall::SRCB_VLD;
#pragma GCC unroll 0
    for (std::uint32_t n = 0; n < 4; n++) {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | vld);
        sdh_d2s_rows<R, 0>();
        sdh_d2s_rows<R, 4>();
        sdh_d2s_rows<R, 8>();
        sdh_d2s_rows<R, 12>();
        sdh_elw_as<T, p_elwise::SRCB_NO_BCAST>();
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
    }
}

template <EltwiseBinaryType T, BroadcastType B, EltwiseBinaryReuseDestType R>
ALWI void sdh_bin(const std::uint32_t idst) {
    static_assert(B == BroadcastType::NONE || B == BroadcastType::COL);
    static_assert(
        R == EltwiseBinaryReuseDestType::NONE || (T != EltwiseBinaryType::ELWMUL && B == BroadcastType::NONE));
    constexpr std::uint32_t BC = (B == BroadcastType::COL) ? p_elwise::SRCB_BCAST_COL : p_elwise::SRCB_NO_BCAST;
    constexpr std::uint32_t CLR = (B == BroadcastType::COL) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (idst << 6) + sdo_dest_base());
    if constexpr (T == EltwiseBinaryType::ELWMUL) {
        if constexpr (B == BroadcastType::COL) {
            for (std::uint32_t r = 0; r < 2; r++) {
                sdh_elw_mul_face<BC, CLR>();
                sdh_elw_mul_face<BC, CLR>();
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        } else {
            for (std::uint32_t n = 0; n < 4; n++) {
                sdh_elw_mul_face<BC, CLR>();
            }
        }
    } else {
        if constexpr (B == BroadcastType::COL) {
            for (std::uint32_t r = 0; r < 2; r++) {
                sdh_elw_as<T, BC>();
                TTI_SETRWC(CLR, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
                sdh_elw_as<T, BC>();
                TTI_SETRWC(CLR, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
                TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
            }
        } else {
            if constexpr (R != EltwiseBinaryReuseDestType::NONE) {
                sdh_bin_reuse<T, R>();
            } else {
                for (std::uint32_t n = 0; n < 4; n++) {
                    sdh_elw_as<T, BC>();
                    TTI_SETRWC(CLR, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
                }
            }
        }
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

ALWI void sdh_rmax_init() {
    static_assert(DST_ACCUM_MODE);
    sdh_am<6, sdh_s(0, 0, 0), sdh_s(0, 0, 0), sdh_d(0, 0, 0, 4)>();
    sdh_am<1, sdh_s(16, 1, 0), sdh_s(0, 0, 1), sdh_d(0, 0, 0, 0)>();
    sdh_am<2, sdh_s(0, 0, 0), sdh_s(0, 0, 0), sdh_d(4, 1, 0, 0)>();
    sdh_am<3, sdh_s(0, 0, 0), sdh_s(0, 0, 0), sdh_d(20, 1, 0, 0)>();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    TTI_REPLAY(0, 15, 0, 1);
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
    TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_2, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_3, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 0);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 4);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 8);
    TTI_MOVB2D(0, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_6, p_movb2d::MOV_4_ROWS, 12);
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
    math::_configure_default_zero_flag_state_();
}

ALWI void sdh_rsum_init() {
    sdh_am<0, sdh_s(0, 0, 0), sdh_s(8, 0, 0), sdh_d(8, 0, 0, 0)>();
    sdh_am<1, sdh_s(0, 0, 0), sdh_s(24, 0, 0), sdh_d(24, 0, 0, 0)>();
    sdh_am<2, sdh_s(16, 0, 0), sdh_s(16, 1, 0), sdh_d(0, 1, 0, 0)>();
    sdh_am<3, sdh_s(0, 1, 1), sdh_s(0, 1, 1), sdh_d(0, 1, 1, sdh_fid)>();
    TTI_REPLAY(16, 8, 0, 1);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
}

ALWI void sdh_rsum(const std::uint32_t idst) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (idst << 6) + sdo_dest_base());
    for (std::uint32_t p = 0; p < sdh_phases; p++) {
        TTI_REPLAY(16, 8, 0, 0);
    }
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

template <std::uint32_t COLS>
ALWI void sdh_rmax(const std::uint32_t idst) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (idst << 6) + sdo_dest_base());
#pragma GCC unroll 0
    for (std::uint32_t c = 0; c < COLS; c++) {
        TTI_REPLAY(0, 2, 0, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_REPLAY(0, 2, 0, 0);
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
    }
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
    math::_configure_src_zero_flag_(true);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
    TTI_REPLAY(2, 13, 0, 0);
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
    math::_configure_src_zero_flag_(false);
}

ALWI void sdh_rmaxg_init() {
    sdh_am<0, sdh_s(0, 0, 0), sdh_s(0, 0, 0), sdh_d(0, 0, 0, 4)>();
    sdh_am<1, sdh_s(0, 0, 0), sdh_s(8, 0, 0), sdh_d(8, 0, 0, 0)>();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
}

ALWI void sdh_rmaxg(const std::uint32_t idst) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (idst << 6) + sdo_dest_base());
#pragma GCC unroll 0
    for (std::uint32_t h = 0; h < 2; h++) {
        if (h != 0) {
            for (std::uint32_t k = 0; k < 3; k++) {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
            TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_BD);
        }
        TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        TTI_GMPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        math::_configure_src_zero_flag_(true);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_TRNSPSRCB;
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_ZEROSRC(0, 1, 0, 1);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
        math::_configure_default_zero_flag_state_();
    }
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
}

ALWI void sdh_mm_block(
    const std::uint32_t dst, const std::uint32_t transpose, const std::uint32_t ct, const std::uint32_t rt) {
    const std::uint32_t mode = *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(MEM_L1_ARC_FW_SCRATCH) % 2;
    if (mode != sdh_mm_mode) {
        sdh_mm_init(transpose, mode);
    }
    sdh_mm_tile(dst, ct, rt);
}

}  // namespace ckernel
#endif
namespace ckernel {

template <EltwiseBinaryType T, BroadcastType B>
ALWI void sdh_binop_init(const std::uint32_t acc = 0) {
    MATH((sdh_bin_init<T, B>(acc)));
    UNPACK((sdh_unpack_ab_init()));
}

template <EltwiseBinaryType T, EltwiseBinaryReuseDestType R>
ALWI void sdh_reuse_init() {
    UNPACK((sdh_unpack_reuse_init<R>()));
    MATH((sdh_bin_init<T, BroadcastType::NONE>(0)));
}

ALWI void sdh_copy_tr_init(const std::uint32_t cb) {
    sdo_copy_init(cb);
    UNPACK((cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1)));
}

ALWI void sdh_reconfig_srcb(const std::uint32_t cb) {
    UNPACK((sdo_unpack_fmt_b(cb)));
    MATH((sdo_math_operands()));
}

}  // namespace ckernel
#endif
