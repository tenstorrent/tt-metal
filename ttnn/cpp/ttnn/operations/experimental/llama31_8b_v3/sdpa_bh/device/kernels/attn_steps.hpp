// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace attn {

#ifdef TRISC_UNPACK
inline void unpack_format_a(std::uint32_t src, std::uint32_t dst, std::uint32_t tile_size) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>(src == to_underlying(DataFormat::UInt8) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(src);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>(
        (src & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(dst);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(canonical_unpA_z_stride(dst));
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(canonical_unpA_y_stride(dst));
}

inline void unpack_format_b(std::uint32_t src, std::uint32_t dst, std::uint32_t tile_size) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcBUnsigned_RMW>(src == to_underlying(DataFormat::UInt8) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(src);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(
        (src & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(dst);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(datum_size_in_bytes(dst) * FACE_C_DIM * FACE_R_DIM);
}

inline void unpack_format_of_a(std::uint32_t in) {
    unpack_format_a(unpack_src_format[in], unpack_dst_format[in], get_local_cb_interface(in).fifo_page_size);
}

inline void unpack_format_of_b(std::uint32_t in) {
    unpack_format_b(unpack_src_format[in], unpack_dst_format[in], get_local_cb_interface(in).fifo_page_size);
}

template <std::uint32_t src, std::uint32_t base0, std::uint32_t base1, std::uint32_t step>
inline void unpack_mm_stream() {
    using namespace ckernel;
    TTI_UNPACR(src, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_RDCFG(p_gpr_unpack::TMP0, base0);
    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, step);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, base0);
    TTI_NOP;
    TTI_UNPACR(src, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_RDCFG(p_gpr_unpack::TMP0, base1);
    TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, step);
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, base1);
    TTI_NOP;
}

inline void unpack_mm_setup(std::uint32_t transpose, std::uint32_t ct, std::uint32_t rt, std::uint32_t kt) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TT_SETADCXX(p_setadc::UNP_A, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TT_SETDMAREG(0, LOWER_HALFWORD(kt), 0, LO_16(p_gpr_unpack::KT_DIM));
    if (ct >= rt) {
        load_replay_buf(0, 12, [] {
            unpack_mm_stream<
                SrcA,
                THCON_SEC0_REG3_Base_address_ADDR32,
                THCON_SEC0_REG3_Base_cntx1_address_ADDR32,
                p_gpr_unpack::TILE_SIZE_A>();
        });
    } else {
        load_replay_buf(0, 12, [] {
            unpack_mm_stream<
                SrcB,
                THCON_SEC1_REG3_Base_address_ADDR32,
                THCON_SEC1_REG3_Base_cntx1_address_ADDR32,
                p_gpr_unpack::TMP_LO>();
        });
    }
    ckernel_unpack_template(false, false, lltt::replay_insn(0, 6), 0, 0, 0, lltt::replay_insn(6, 6), 0, 0).program();
}

inline void unpack_mm(
    std::uint32_t in0,
    std::uint32_t in1,
    std::uint32_t tile0,
    std::uint32_t tile1,
    std::uint32_t ct,
    std::uint32_t rt,
    std::uint32_t kt) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    const std::uint32_t base0 = get_local_cb_interface(in0).fifo_rd_ptr - 1;
    const std::uint32_t base1 = get_local_cb_interface(in1).fifo_rd_ptr - 1;
    const std::uint32_t size0 = get_local_cb_interface(in0).fifo_page_size;
    const std::uint32_t size1 = get_local_cb_interface(in1).fifo_page_size;
    const bool hold_b = ct >= rt;
    const std::uint32_t steps = hold_b ? rt : ct;
    if (!hold_b) {
        TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
    }
    for (std::uint32_t t = 0; t < steps; t++) {
        const std::uint32_t a0 = base0 + size0 * (tile0 + (hold_b ? t * kt : 0));
        const std::uint32_t a1 = base1 + size1 * (tile1 + (hold_b ? 0 : t));
        wait_for_next_context(2);
        if (0 == unp_cfg_context) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = a1;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = a0;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a1;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = a0;
        }
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        if (hold_b) {
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        } else {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        }
        TT_MOP(0, (hold_b ? ct : rt) - 1, unp_cfg_context == 0 ? 0 : 0xff);
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        switch_config_context(unp_cfg_context);
    }
}

inline void unpack_copy_setup(std::uint32_t face_transpose) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(face_transpose);
    config_unpacker_x_end<p_setadc::UNP_A>(FACE_R_DIM);
    ckernel_template mop(
        4,
        1,
        TT_OP_UNPACR_NOP(
            SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC));
    mop.set_start_op(TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    mop.program();
}

inline void unpack_copy(std::uint32_t in, std::uint32_t tile) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t address =
        get_local_cb_interface(in).fifo_rd_ptr - 1 + get_local_cb_interface(in).fifo_page_size * tile;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] =
        address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

template <bool col>
inline void unpack_pair_setup() {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    config_unpacker_x_end<p_setadc::UNP_AB>(FACE_R_DIM);
    constexpr std::uint32_t a = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    constexpr std::uint32_t b = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    if constexpr (col) {
        ckernel_template mop(2, 2, a);
        mop.set_start_op(b);
        mop.set_end_ops(TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001), TT_OP_NOP);
        mop.program();
    } else {
        ckernel_template mop(2, 2, a, b);
        mop.set_end_op(TT_OP_NOP);
        mop.program();
    }
}

inline void unpack_pair(std::uint32_t in_a, std::uint32_t in_b, std::uint32_t tile_a, std::uint32_t tile_b) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t address_a =
        get_local_cb_interface(in_a).fifo_rd_ptr - 1 + get_local_cb_interface(in_a).fifo_page_size * tile_a;
    const std::uint32_t address_b =
        get_local_cb_interface(in_b).fifo_rd_ptr - 1 + get_local_cb_interface(in_b).fifo_page_size * tile_b;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

inline void unpack_cols_over_dest_setup() {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    config_unpacker_x_end<p_setadc::UNP_B>(FACE_R_DIM);
    constexpr std::uint32_t a_ready = TT_OP_UNPACR_NOP(
        SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    ckernel_template mop(2, 1, a_ready, a_ready);
    mop.set_start_op(TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    mop.set_end_op(TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001));
    mop.program();
}

inline void unpack_cols_over_dest(std::uint32_t in, std::uint32_t tile) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t address =
        get_local_cb_interface(in).fifo_rd_ptr - 1 + get_local_cb_interface(in).fifo_page_size * tile;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC1_REG3_Base_address_ADDR32 : THCON_SEC1_REG3_Base_cntx1_address_ADDR32] =
        address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

inline void unpack_rowmax_setup() {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);
    config_unpacker_x_end<p_setadc::UNP_A>(FACE_R_DIM);
    config_unpacker_x_end<p_setadc::UNP_B>(1);
    load_replay_buf(0, 2, [] {
        TTI_UNPACR(SrcA, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    });
    ckernel_template mop(1, 4, lltt::replay_insn(0, 2));
    mop.program();
}

inline void unpack_scaler_only(std::uint32_t scaler) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t address = get_local_cb_interface(scaler).fifo_rd_ptr - 1;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address;
    }
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (std::uint32_t f = 0; f < 4; f++) {
        TTI_UNPACR_NOP(
            SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR(SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
inline void math_format_ab(std::uint32_t fa, std::uint32_t fb) {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(is_int8_or_int32_format(fa) || is_int8_or_int32_format(fb));
    src_zero_flag_srca_fmt = fa;
    src_zero_flag_srcb_fmt = fb;
    _configure_default_zero_flag_state_();
}

inline void math_format_a(std::uint32_t fa) {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(is_int8_or_int32_format(fa));
    src_zero_flag_srca_fmt = fa;
    _configure_default_zero_flag_state_();
}

inline void math_format_b(std::uint32_t fb) {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(is_int8_or_int32_format(fb));
    src_zero_flag_srcb_fmt = fb;
    _configure_default_zero_flag_state_();
}

template <bool slow>
inline void math_mm_setup(std::uint32_t transpose, std::uint32_t ct, std::uint32_t rt) {
    using namespace ckernel;
    using namespace ckernel::math;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_0);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = 1, .clr = 0}}
        .set(ADDR_MOD_5);
    if constexpr (slow) {
        addr_mod_t{
            .srca = {.incr = 0, .clr = 1, .cr = 1},
            .srcb = {.incr = 0, .clr = 1, .cr = 1},
            .dest = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_6);
    }
    if (transpose) {
        addr_mod_t{.srca = {.incr = 32}, .srcb = {.incr = 0, .cr = 1}, .dest = {.incr = 8}}.set(ADDR_MOD_1);
    } else {
        addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 0, .cr = 1}, .dest = {.incr = 8}}.set(ADDR_MOD_1);
    }
    addr_mod_t{.srca = {.incr = 0, .cr = 1}, .srcb = {.incr = 32, .cr = 1}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    if (transpose) {
        addr_mod_t{.srca = {.incr = 16, .cr = 1}, .srcb = {.incr = 48, .cr = 1}, .dest = {.incr = 0, .cr = 1}}.set(
            ADDR_MOD_4);
    } else {
        addr_mod_t{.srca = {.incr = 32, .cr = 1}, .srcb = {.incr = 48, .cr = 1}, .dest = {.incr = 0, .cr = 1}}.set(
            ADDR_MOD_4);
    }
    if constexpr (slow) {
        load_replay_buf(replay_buf_offset, 11, [] {
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
        });
        ckernel_template mop(
            2, 2, lltt::replay_insn(replay_buf_offset, 11), TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0));
        mop.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0));
        mop.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0));
        mop.program();
    } else {
        load_replay_buf(replay_buf_offset, 16, [] {
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
        });
        ckernel_template mop(1, 4, lltt::replay_insn(replay_buf_offset, 16));
        if (ct >= rt) {
            mop.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        } else {
            mop.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
        mop.program();
    }
    reset_counters(p_setrwc::SET_ABD_F);
    _configure_default_zero_flag_state_();
}

template <bool slow>
inline void mm_tile_phases(bool hold_b) {
    if constexpr (slow) {
#pragma GCC unroll 0
        for (std::uint32_t phase = 0; phase < 4; phase++) {
            ckernel::ckernel_template::run();
        }
        if (hold_b) {
            TTI_SETRWC(ckernel::p_setrwc::CLR_A, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F);
        } else {
            TTI_SETRWC(ckernel::p_setrwc::CLR_B, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F);
        }
    } else {
        ckernel::ckernel_template::run();
    }
}

inline void mm_line_done(bool hold_b) {
    if (hold_b) {
        TTI_SETRWC(ckernel::p_setrwc::CLR_B, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F);
    } else {
        TTI_SETRWC(ckernel::p_setrwc::CLR_A, 0, 0, 0, 0, ckernel::p_setrwc::SET_ABD_F);
    }
}

template <bool slow>
inline void math_mm(std::uint32_t dst_index, std::uint32_t ct, std::uint32_t rt) {
    const bool hold_b = ct >= rt;
    const std::uint32_t lines = hold_b ? rt : ct;
    const std::uint32_t per_line = hold_b ? ct : rt;
    const std::uint32_t line_step = hold_b ? ct : 1;
    const std::uint32_t tile_step = hold_b ? 1 : ct;
    std::uint32_t line_start = dst_index;
    for (std::uint32_t line = 0; line < lines; line++, line_start += line_step) {
        std::uint32_t tile = line_start;
        for (std::uint32_t k = 1; k <= per_line; k++, tile += tile_step) {
            ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(tile);
            mm_tile_phases<slow>(hold_b);
            if (k == per_line) {
                mm_line_done(hold_b);
            }
        }
    }
}

inline std::uint32_t mm_slow_programmed = 0;

inline void math_mm_step(std::uint32_t dst_index, std::uint32_t transpose, std::uint32_t ct, std::uint32_t rt) {
    volatile std::uint32_t slow = *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(MEM_L1_ARC_FW_SCRATCH) % 2;
    if (slow) {
        if (mm_slow_programmed != 1) {
            math_mm_setup<true>(transpose, ct, rt);
            mm_slow_programmed = 1;
        }
        math_mm<true>(dst_index, ct, rt);
    } else {
        if (mm_slow_programmed != 0) {
            math_mm_setup<false>(transpose, ct, rt);
            mm_slow_programmed = 0;
        }
        math_mm<false>(dst_index, ct, rt);
    }
}

inline void math_copy_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
    addr_mod_t{.srca = {.incr = 1}, .srcb = {.incr = 0}, .dest = {.incr = 1}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 0}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    ckernel_template mop(4, 2, TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
    mop.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    mop.program();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    reset_counters(p_setrwc::SET_ABD_F);
}

inline void math_copy(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    clear_dst_reg_addr();
}

enum class Elw { add, sub, mul };

template <Elw op, bool col>
constexpr std::uint32_t elw_op(std::uint32_t clr, std::uint32_t acc, std::uint32_t addr_mod) {
    constexpr std::uint32_t bc = col ? ckernel::p_elwise::SRCB_BCAST_COL : ckernel::p_elwise::SRCB_NO_BCAST;
    if constexpr (op == Elw::add) {
        return TT_OP_ELWADD(clr, acc, bc, addr_mod, 0);
    } else if constexpr (op == Elw::sub) {
        return TT_OP_ELWSUB(clr, acc, bc, addr_mod, 0);
    } else {
        return TT_OP_ELWMUL(clr, acc, bc, addr_mod, 0);
    }
}

template <Elw op, bool col, bool over_dest>
inline void math_pair_setup(std::uint32_t acc) {
    using namespace ckernel;
    using namespace ckernel::math;
    constexpr std::uint32_t phase_step = op == Elw::mul ? 1 : 0;
    addr_mod_t{.srca = {.incr = MAX_FPU_ROWS}, .srcb = {.incr = MAX_FPU_ROWS}, .dest = {.incr = MAX_FPU_ROWS}}.set(
        ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_1);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = phase_step}}
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = MAX_FPU_ROWS, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
    constexpr std::uint32_t clr = col ? p_setrwc::CLR_A : p_setrwc::CLR_AB;
    if constexpr (op == Elw::mul) {
        ckernel_template mop(4, 2, elw_op<op, col>(0, 0, ADDR_MOD_0));
        mop.set_last_inner_loop_instr(elw_op<op, col>(0, 0, ADDR_MOD_2));
        mop.set_last_outer_loop_instr(elw_op<op, col>(clr, 0, ADDR_MOD_3));
        mop.program();
    } else {
        ckernel_template mop(over_dest ? 1 : (col ? 2 : 4), 2, elw_op<op, col>(0, acc, ADDR_MOD_0));
        mop.set_end_op(TT_OP_SETRWC(clr, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        mop.program();
    }
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    reset_counters(p_setrwc::SET_ABD_F);
    _configure_default_zero_flag_state_();
}

template <Elw op, bool col>
inline void math_pair(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    if constexpr (op == Elw::mul) {
#pragma GCC unroll 0
        for (std::uint32_t r = 0; r < 2; r++) {
#pragma GCC unroll 0
            for (std::uint32_t c = 0; c < 2; c++) {
                ckernel_template::run();
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        }
    } else if constexpr (col) {
#pragma GCC unroll 0
        for (std::uint32_t r = 0; r < 2; r++) {
            ckernel_template::run();
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        }
    } else {
        ckernel_template::run();
    }
    clear_dst_reg_addr();
}

inline void math_sub_cols_over_dest(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
#pragma GCC unroll 0
    for (std::uint32_t r = 0; r < 2; r++) {
#pragma GCC unroll 0
        for (std::uint32_t c = 0; c < 2; c++) {
            move_d2a_fixed_face(ADDR_MOD_1);
            ckernel_template::run();
        }
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
    }
    clear_dst_reg_addr();
}

inline void math_rowmax_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    addr_mod_t{.fidelity = {.clr = 1}}.set(ADDR_MOD_0);
    addr_mod_t{.srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_1);
    ckernel_template mop(1, 4, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_1, p_gpool::INDEX_DIS, 0));
    mop.set_last_inner_loop_instr(
        TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
    mop.set_last_outer_loop_instr(
        TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
    mop.program();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    reset_counters(p_setrwc::SET_ABD_F);
    _configure_default_zero_flag_state_();
}

template <std::uint32_t clear>
inline void row_max_face() {
    TTI_GMPOOL(clear, ckernel::p_gpool::DIM_16X16, ckernel::ADDR_MOD_0, ckernel::p_gpool::INDEX_DIS, 0);
}

inline void row_max_fold() {
    using namespace ckernel;
    using namespace ckernel::math;
    _configure_preserve_zero_flag_state_();
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_TRNSPSRCB;
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_ZEROSRC(0, 1, 0, 1);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1, 0);
    _configure_default_zero_flag_state_();
}

inline void row_max_next_rows() {
#pragma GCC unroll 3
    for (int step = 0; step < 3; step++) {
        TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, ckernel::p_setrwc::CR_D, 8, 0, 0, ckernel::p_setrwc::SET_D);
    }
    TTI_SETRWC(ckernel::p_setrwc::CLR_AB, ckernel::p_setrwc::CR_D, 8, 0, 0, ckernel::p_setrwc::SET_BD);
}

inline void math_rowmax(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    row_max_face<p_setrwc::CLR_AB>();
    row_max_face<p_setrwc::CLR_NONE>();
    row_max_fold();
    row_max_next_rows();
    row_max_face<p_setrwc::CLR_AB>();
    row_max_face<p_setrwc::CLR_NONE>();
    row_max_fold();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
}

inline void sfpu_open(std::uint32_t dst_index) {
    using namespace ckernel;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void sfpu_next_face_pair() {
    ckernel::math::inc_dst_addr<8>();
    ckernel::math::inc_dst_addr<8>();
    ckernel::math::inc_dst_addr<8>();
    ckernel::math::inc_dst_addr<8>();
}

inline void sfpu_close() { ckernel::math::clear_dst_reg_addr(); }

template <bool step_two>
inline void sfpu_base_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_SFPCONFIG(0, 0xF, 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    if constexpr (step_two) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    }
    reset_counters(p_setrwc::SET_ABD_F);
}

constexpr std::uint32_t lo16(float f) { return __builtin_bit_cast(std::uint32_t, f) & 0xFFFFu; }
constexpr std::uint32_t hi16(float f) { return __builtin_bit_cast(std::uint32_t, f) >> 16; }

template <std::uint32_t lreg, std::uint32_t bits>
inline void program_const() {
    TTI_SFPLOADI(0, 10, bits & 0xFFFFu);
    TTI_SFPLOADI(0, 8, bits >> 16);
    TTI_SFPCONFIG(0, lreg, 0);
}

inline void math_exp_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    sfpu_base_setup<false>();
    TTI_SFPCONFIG(0, 0xF, 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    reset_counters(p_setrwc::SET_ABD_F);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    program_const<12, 0x3FB8AA3Bu>();
    program_const<13, 0xBF317200u>();
    program_const<14, 0xB5BFBE8Eu>();
}

template <std::uint32_t lreg, std::uint32_t bits>
inline void load_word() {
    TTI_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFFu);
    TTI_SFPLOADI(lreg, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
}

inline void exp_coefficients() {
    using namespace ckernel;
    load_word<p_sfpu::LREG4, __builtin_bit_cast(std::uint32_t, 8.37312452e-3f)>();
    load_word<p_sfpu::LREG5, __builtin_bit_cast(std::uint32_t, 4.16695364e-2f)>();
    load_word<p_sfpu::LREG6, __builtin_bit_cast(std::uint32_t, 1.66664720e-1f)>();
    load_word<p_sfpu::LREG7, __builtin_bit_cast(std::uint32_t, 4.99999851e-1f)>();
}

inline void exp_reduce_and_poly() {
    using namespace ckernel;
    constexpr std::uint32_t scale_reg = p_sfpu::LREG11;
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
    TTI_SFPMUL(p_sfpu::LREG1, scale_reg, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATA, 0x15A5);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_NEAREST,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG3,
        p_sfpu::LREG3,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_SMAG16);
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPCAST_MOD1_INT32_TO_SM32);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    TTI_SFPMULI(0x7F80, p_sfpu::LREG0, 0);
}

inline void exp_rebuild_and_store() {
    using namespace ckernel;
    TTI_SFPEXEXP(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0xF01, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPIADD(0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
}

template <std::uint32_t scale_bits>
inline void math_exp_scaled(std::uint32_t dst_index) {
    using namespace ckernel;
    sfpu_open(dst_index);
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, scale_bits & 0xFFFFu);
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, scale_bits >> 16);
    TTI_SFPCONFIG(0, 11, 0);
    exp_coefficients();
    load_replay_buf<Exec>(0, 16, [] { exp_reduce_and_poly(); });
    exp_rebuild_and_store();
#pragma GCC unroll 8
    for (int d = 1; d < 32; d++) {
        TTI_REPLAY(0, 16, 0, 0);
        exp_rebuild_and_store();
    }
    TTI_SFPCONFIG(0, 11, 1);
    sfpu_close();
}

template <std::uint16_t scale_bf16>
inline void exp_cols_face() {
    using namespace ckernel;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    constexpr float k4 = 4.1959439860014343843000081999668024587178974865521e-2;
    constexpr float k3 = 0.16792157982882225102649214918047336097544632172075f;
    constexpr float k2 = 0.49998365704615426417337683145647067790385638465486f;
    constexpr float k1 = 0.99996228117047652035114096488703457970402030983204f;
    constexpr float k0 = 1.0000001510806179002040134468008959160576106495165f;
    constexpr float log2e = 1.44269504088896340736f;
    constexpr float neg_ln2 = -0.69314718055994530942f;
    TTI_SFPLOADI(p_sfpu::LREG3, 0xA, lo16(k4));
    TTI_SFPLOADI(p_sfpu::LREG3, 0x8, hi16(k4));
    TTI_SFPLOADI(p_sfpu::LREG4, 0xA, lo16(k3));
    TTI_SFPLOADI(p_sfpu::LREG4, 0x8, hi16(k3));
    TTI_SFPLOADI(p_sfpu::LREG5, 0xA, lo16(k2));
    TTI_SFPLOADI(p_sfpu::LREG5, 0x8, hi16(k2));
    TTI_SFPLOADI(p_sfpu::LREG6, 0xA, lo16(k1));
    TTI_SFPLOADI(p_sfpu::LREG6, 0x8, hi16(k1));
    TTI_SFPLOADI(p_sfpu::LREG7, 0xA, lo16(k0));
    TTI_SFPLOADI(p_sfpu::LREG7, 0x8, hi16(k0));
    for (int d = 0; d < 4; d++) {
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, 0, scale_bf16);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, 0xA, lo16(log2e));
        TTI_SFPLOADI(p_sfpu::LREG1, 0x8, hi16(log2e));
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, 0xA, lo16(neg_ln2));
        TTI_SFPLOADI(p_sfpu::LREG0, 0x8, hi16(neg_ln2));
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
        TT_SFPADDI(0x42fe, p_sfpu::LREG1, 0);
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 6);
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <std::uint16_t scale_bf16>
inline void math_exp_cols(std::uint32_t dst_index) {
    sfpu_open(dst_index);
#pragma GCC unroll 0
    for (int face = 0; face < 2; face++) {
        exp_cols_face<scale_bf16>();
        sfpu_next_face_pair();
    }
    sfpu_close();
}

inline void math_recip_setup() {
    sfpu_base_setup<true>();
    TTI_SFPLOADI(0, 0, 0x4000);
    TTI_SFPCONFIG(0, 12, 0);
}

inline void recip_cols_face() {
    TTI_REPLAY(0, 12, 1, 1);
    TTI_SFPLOAD(1, 0, 7, 0);
    TTI_SFPARECIP(0, 1, 0, 0);
    TTI_SFPMAD(1, 0, 12, 3, 2);
    TTI_SFPLOADI(2, 0, 32768);
    TTI_SFPMAD(3, 0, 2, 2, 1);
    TTI_SFPGT(0, 3, 9, 1);
    TTI_SFPMAD(1, 2, 12, 1, 1);
    TTI_SFPLOADI(3, 0, 32768);
    TTI_SFPMAD(2, 1, 3, 0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSTORE(0, 0, 7, 0);
    TTI_INCRWC(0, 4, 0, 0);
    TTI_REPLAY(0, 12, 0, 0);
    TTI_REPLAY(0, 12, 0, 0);
    TTI_REPLAY(0, 12, 0, 0);
}

inline void math_recip_cols(std::uint32_t dst_index) {
    sfpu_open(dst_index);
#pragma GCC unroll 0
    for (int face = 0; face < 2; face++) {
        recip_cols_face();
        sfpu_next_face_pair();
    }
    sfpu_close();
}

inline void math_max_setup() {
    sfpu_base_setup<true>();
    TTI_SFPSWAP(0, 2, 12, 9);
    TTI_SFPSHFT2(0, 0, 13, 6);
    TTI_SFPLOADI(0, 10, 0x80 | (1 << 3) | 4);
    TTI_SFPLOADI(0, 8, 0x80 | 0x40 | (3 << 3) | 5);
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPLOADI(0, 10, 0);
    TTI_SFPLOADI(0, 8, (0x40 | (2 << 3) | 3) << 8);
    TTI_SFPCONFIG(0, 5, 0);
    TTI_SFPCONFIG(0x330, 8, 1);
}

inline void max_face_pair(std::uint32_t in0, std::uint32_t in1, std::uint32_t out) {
    const std::uint32_t o0 = in0 << 6;
    const std::uint32_t o1 = in1 << 6;
    const std::uint32_t o2 = out << 6;
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        TT_SFPLOADMACRO(i & 1, 0, ckernel::ADDR_MOD_7, o0);
        TT_SFPLOAD(2, 0, ckernel::ADDR_MOD_7, o1);
        TT_SFPLOADMACRO((1 << 2) | 3, 0, ckernel::ADDR_MOD_6, o2);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

inline void math_max_cols(std::uint32_t in0, std::uint32_t in1, std::uint32_t out) {
    sfpu_open(0);
#pragma GCC unroll 0
    for (int face = 0; face < 2; face++) {
        max_face_pair(in0, in1, out);
        sfpu_next_face_pair();
    }
    sfpu_close();
}

inline void math_section_open() { ckernel::math::math_dest_wait(); }

inline void math_section_close() {
    using namespace ckernel;
    using namespace ckernel::math;
    set_math_semaphores();
    math_sync_tile_dst_index = 0;
    dest_section_flip();
}
#endif

#ifdef TRISC_PACK
inline void pack_row_mask_on(std::uint32_t out) {
    using namespace ckernel;
    using namespace ckernel::packer;
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(out);
    pck_edge_offset_u pack_edge_offset = {.val = 0};
    pack_edge_offset.f.mask = 0x0;
    pack_edge_offset.f.tile_row_set_select_pack0 = 1;
    const std::uint32_t edge_offset_sec1_mask = 0x0001;
    const std::uint32_t row_set_mapping_1 = 0x55555555;
    TTI_SETDMAREG(0, LOWER_HALFWORD(pack_edge_offset.val), 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, UPPER_HALFWORD(pack_edge_offset.val), 0, HI_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, LOWER_HALFWORD(edge_offset_sec1_mask), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_SETDMAREG(0, LOWER_HALFWORD(row_set_mapping_1), 0, LO_16(p_gpr_pack::TMP1));
    TTI_SETDMAREG(0, UPPER_HALFWORD(row_set_mapping_1), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(tensor_shape.face_r_dim);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);
    const std::uint32_t face_set_mapping = (tensor_shape.num_faces_c_dim == 1) ? 0x55555555 : 0x11111111;
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
    cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32, 0, 0xffffffff>(face_set_mapping);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0x100);
    TTI_NOP;
    TTI_NOP;
}

inline void pack_row_mask_off() {
    using namespace ckernel;
    using namespace ckernel::packer;
    constexpr std::uint32_t all_rows = 0xffff;
    TTI_SETDMAREG(0, all_rows, 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_enable_RMW>(0);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);
    TTI_NOP;
    TTI_NOP;
}

inline std::uint32_t packed_as = 0xFFFFFFFFu;

inline void pack_format(std::uint32_t out) {
    const std::uint32_t f = std::uint32_t(pack_src_format[out]) | (std::uint32_t(pack_dst_format[out]) << 8);
    if (f != packed_as) {
        ckernel::packer::reconfig_packer_data_format<true>(
            pack_src_format[out],
            pack_dst_format[out],
            get_local_cb_interface(out).fifo_page_size,
            pack_tile_c_dim[out],
            pack_tile_num_faces[out],
            false);
        packed_as = f;
    }
}

inline void pack_accumulate(std::uint32_t enable) { ckernel::packer::reconfigure_packer_l1_acc(enable); }

inline void pack_clamp_negative(bool on) {
    using namespace ckernel;
    const std::uint32_t mode =
        on ? static_cast<std::uint32_t>(ReluType::ZERO_RELU) : static_cast<std::uint32_t>(ReluType::NO_RELU);
    constexpr std::uint32_t mask = STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    t6_mutex_acquire(mutex::REG_RMW);
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, mask>(mode << STACC_RELU_ApplyRelu_SHAMT);
    t6_mutex_release(mutex::REG_RMW);
}

inline void pack_give_back_to_math() {
    ckernel::t6_semaphore_get<ckernel::p_stall::PACK>(ckernel::semaphore::MATH_PACK);
}

inline void pack_section_open() {
    TTI_SEMWAIT(
        ckernel::p_stall::STALL_TDMA | ckernel::p_stall::STALL_SYNC,
        ckernel::semaphore::t6_sem(ckernel::semaphore::MATH_PACK),
        ckernel::p_stall::STALL_ON_ZERO);
}

inline void pack_section_close() {
    using namespace ckernel;
    using namespace ckernel::packer;
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    TT_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, dest_offset_id % 2);
    t6_semaphore_get<p_stall::NONE>(semaphore::MATH_PACK);
    flip_packer_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();
}

inline void pack_tile_to(std::uint32_t dst_index, std::uint32_t address) {
    using namespace ckernel;
    using namespace ckernel::packer;
    set_dst_write_addr(dst_index);
    program_packer_destination(address);
    ckernel_template::run();
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

inline void pack_next(std::uint32_t dst_index, std::uint32_t out) {
    auto& cb = get_local_cb_interface(out);
    const std::uint32_t address = cb.fifo_wr_ptr + cb.fifo_wr_tile_ptr - 1;
    cb.fifo_wr_tile_ptr += cb.fifo_page_size;
    pack_tile_to(dst_index, address);
}

inline void pack_at(std::uint32_t dst_index, std::uint32_t out, std::uint32_t slot) {
    auto& cb = get_local_cb_interface(out);
    pack_tile_to(dst_index, cb.fifo_wr_ptr + cb.fifo_page_size * slot - 1);
}
#endif

inline void open() { MATH((math_section_open())); }
inline void close() { MATH((math_section_close())); }
inline void drain() { PACK((pack_section_open())); }
inline void release() { PACK((pack_section_close())); }
inline void out(std::uint32_t dst_index, std::uint32_t cb) { PACK((pack_next(dst_index, cb))); }
inline void out_at(std::uint32_t dst_index, std::uint32_t cb, std::uint32_t slot) {
    PACK((pack_at(dst_index, cb, slot)));
}

inline void formats(std::uint32_t a, std::uint32_t b) {
    UNPACK((unpack_format_of_a(a)));
    UNPACK((unpack_format_of_b(b)));
    MATH((math_format_ab(unpack_dst_format[a], unpack_dst_format[b])));
}
inline void format_a(std::uint32_t a) {
    UNPACK((unpack_format_of_a(a)));
    MATH((math_format_a(unpack_dst_format[a])));
}
inline void format_b(std::uint32_t b) {
    UNPACK((unpack_format_of_b(b)));
    MATH((math_format_b(unpack_dst_format[b])));
}

inline void mm_start(std::uint32_t, std::uint32_t) {
    MATH((math_mm_setup<false>(0, 1, 1)));
    UNPACK((unpack_mm_setup(0, 1, 1, 1)));
}
inline void mm_setup(
    std::uint32_t, std::uint32_t, std::uint32_t transpose, std::uint32_t ct, std::uint32_t rt, std::uint32_t kt) {
    UNPACK((unpack_mm_setup(transpose, ct, rt, kt)));
    MATH((math_mm_setup<false>(transpose, ct, rt)));
    MATH((mm_slow_programmed = 0));
}
inline void mm(
    std::uint32_t in0,
    std::uint32_t in1,
    std::uint32_t tile0,
    std::uint32_t tile1,
    std::uint32_t dst_index,
    std::uint32_t transpose,
    std::uint32_t ct,
    std::uint32_t rt,
    std::uint32_t kt) {
    UNPACK((unpack_mm(in0, in1, tile0, tile1, ct, rt, kt)));
    MATH((math_mm_step(dst_index, transpose, ct, rt)));
}

#ifndef TRISC_PACK
constexpr bool wide_word(std::uint32_t f) {
    const std::uint32_t code = f & 0x1F;
    return code == 0 || code == 8;
}
constexpr bool copies_through_srca() {
    for (std::uint32_t i = 0; i < 64; i++) {
        if (unpack_src_format[i] != 255 && wide_word(unpack_src_format[i]) && wide_word(unpack_dst_format[i])) {
            return false;
        }
    }
    return true;
}
static_assert(copies_through_srca());
#endif

inline void copy_setup(std::uint32_t in) {
    UNPACK((unpack_copy_setup(0)));
    MATH((math_copy_setup()));
    MATH((ckernel::math::_configure_copy_zero_flag_state_(unpack_dst_format[in])));
}
inline void copy_setup_face_transposed() {
    UNPACK((unpack_copy_setup(1)));
    MATH((math_copy_setup()));
}
inline void copy(std::uint32_t in, std::uint32_t tile, std::uint32_t dst_index) {
    UNPACK((unpack_copy(in, tile)));
    MATH((math_copy(dst_index)));
}

inline void add_setup(std::uint32_t, std::uint32_t, std::uint32_t acc = 0) {
    MATH((math_pair_setup<Elw::add, false, false>(acc)));
    UNPACK((unpack_pair_setup<false>()));
}
inline void sub_setup(std::uint32_t, std::uint32_t) {
    MATH((math_pair_setup<Elw::sub, false, false>(0)));
    UNPACK((unpack_pair_setup<false>()));
}
inline void sub_cols_setup(std::uint32_t, std::uint32_t) {
    MATH((math_pair_setup<Elw::sub, true, false>(0)));
    UNPACK((unpack_pair_setup<true>()));
}
inline void mul_cols_setup(std::uint32_t, std::uint32_t) {
    MATH((math_pair_setup<Elw::mul, true, false>(0)));
    UNPACK((unpack_pair_setup<true>()));
}
inline void sub_cols_over_dest_setup() {
    UNPACK((unpack_cols_over_dest_setup()));
    MATH((math_pair_setup<Elw::sub, true, true>(0)));
}
inline void add(std::uint32_t a, std::uint32_t b, std::uint32_t ta, std::uint32_t tb, std::uint32_t dst_index) {
    UNPACK((unpack_pair(a, b, ta, tb)));
    MATH((math_pair<Elw::add, false>(dst_index)));
}
inline void sub(std::uint32_t a, std::uint32_t b, std::uint32_t ta, std::uint32_t tb, std::uint32_t dst_index) {
    UNPACK((unpack_pair(a, b, ta, tb)));
    MATH((math_pair<Elw::sub, false>(dst_index)));
}
inline void sub_cols(std::uint32_t a, std::uint32_t b, std::uint32_t ta, std::uint32_t tb, std::uint32_t dst_index) {
    MATH((math_pair<Elw::sub, true>(dst_index)));
    UNPACK((unpack_pair(a, b, ta, tb)));
}
inline void mul_cols(std::uint32_t a, std::uint32_t b, std::uint32_t ta, std::uint32_t tb, std::uint32_t dst_index) {
    MATH((math_pair<Elw::mul, true>(dst_index)));
    UNPACK((unpack_pair(a, b, ta, tb)));
}
inline void sub_cols_over_dest(std::uint32_t b, std::uint32_t tb, std::uint32_t dst_index) {
    UNPACK((unpack_cols_over_dest(b, tb)));
    MATH((math_sub_cols_over_dest(dst_index)));
}

inline void rowmax_setup(std::uint32_t out) {
    UNPACK((unpack_rowmax_setup()));
    MATH((math_rowmax_setup()));
    PACK((pack_row_mask_on(out)));
}
inline void rowmax(
    std::uint32_t in, std::uint32_t scaler, std::uint32_t tile, std::uint32_t scaler_tile, std::uint32_t dst_index) {
    MATH((math_rowmax(dst_index)));
    UNPACK((unpack_pair(in, scaler, tile, scaler_tile)));
}
inline void rowmax_done() { PACK((pack_row_mask_off())); }

inline void max_setup() { MATH((math_max_setup())); }
inline void max_cols(std::uint32_t in0, std::uint32_t in1, std::uint32_t out) { MATH((math_max_cols(in0, in1, out))); }
inline void exp_setup() { MATH((math_exp_setup())); }
template <std::uint32_t scale_bits>
inline void exp_scaled(std::uint32_t dst_index) {
    MATH((math_exp_scaled<scale_bits>(dst_index)));
}

}  // namespace attn
